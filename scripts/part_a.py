import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
DB_PATH = os.path.join(ROOT, "data", "instamart.db")
SQL_PATH = os.path.join(ROOT, "sql", "queries.sql")
os.makedirs(CLEAN, exist_ok=True)

customers = pd.read_csv(os.path.join(RAW, "customers.csv"))
products = pd.read_csv(os.path.join(RAW, "products.csv"))
orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)


def audit_frame(frame):
    return {
        "shape": list(frame.shape),
        "columns": {
            column: {
                "dtype": str(frame[column].dtype),
                "missing_count": int(frame[column].isna().sum()),
                "missing_percent": round(float(frame[column].isna().mean() * 100), 2),
            }
            for column in frame.columns
        },
    }


raw_audit = {
    "customers": audit_frame(customers),
    "products": audit_frame(products),
    "orders": audit_frame(orders),
    "tickets": audit_frame(tickets),
}
raw_metrics = {
    "orders_rows": len(orders),
    "exact_duplicate_rows": int(orders.duplicated().sum()),
    "pending_rows": int((orders["status"] == "pending").sum()),
    "distinct_order_ids": int(orders["order_id"].nunique()),
    "non_numeric_amounts": int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()),
    "negative_delivery_minutes": int((pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum()),
    "blank_delivery_minutes": int(
        orders["delivery_minutes"].isna().sum()
        + orders["delivery_minutes"].astype(str).str.strip().eq("").sum()
    ),
    "blank_ages": int(
        customers["age"].isna().sum()
        + customers["age"].astype(str).str.strip().eq("").sum()
    ),
    "ages_outside_10_100": int(
        pd.to_numeric(customers["age"], errors="coerce").map(
            lambda value: pd.notna(value) and not 10 <= value <= 100
        ).sum()
    ),
    "raw_city_spellings": int(customers["city"].nunique()),
    "raw_nonblank_gender_spellings": int(
        customers.loc[customers["gender"].astype(str).str.strip().ne(""), "gender"].nunique()
    ),
    "blank_genders": int(
        customers["gender"].isna().sum()
        + customers["gender"].astype(str).str.strip().eq("").sum()
    ),
}

customers_clean = customers.copy()
customers_clean["city"] = customers_clean["city"].astype(str).str.strip().str.title()
customers_clean["gender"] = customers_clean["gender"].astype(str).str.strip().str.lower()
customers_clean["gender"] = customers_clean["gender"].replace(
    {"m": "male", "f": "female", "": "unknown"}
)
customers_clean["gender"] = customers_clean["gender"].where(
    customers_clean["gender"].isin(["male", "female"]), "unknown"
)
customers_clean["age"] = pd.to_numeric(customers_clean["age"], errors="coerce")
customers_clean["age"] = customers_clean["age"].where(
    customers_clean["age"].between(10, 100), np.nan
)
ages_replaced = int(customers_clean["age"].isna().sum())
customers_clean["age"] = customers_clean["age"].fillna(
    customers_clean.groupby("city")["age"].transform("median")
)
customers_clean["signup_date"] = pd.to_datetime(
    customers_clean["signup_date"], errors="coerce"
)
customer_changes = {
    "city_values_normalized": int(
        (customers["city"].astype(str) != customers_clean["city"]).sum()
    ),
    "gender_values_mapped": int(
        (customers["gender"].fillna("").astype(str) != customers_clean["gender"]).sum()
    ),
    "ages_replaced_then_imputed": ages_replaced,
    "signup_dates_converted": int(customers_clean["signup_date"].notna().sum()),
}

orders_clean = orders.copy()
orders_clean["order_ts"] = pd.to_datetime(orders_clean["order_ts"], errors="coerce")
timestamp_rows = int(orders_clean["order_ts"].notna().sum())
orders_clean = orders_clean.drop_duplicates()
exact_duplicates_removed = len(orders) - len(orders_clean)
rows_after_exact_deduplication = len(orders_clean)
orders_clean = orders_clean.sort_values(
    ["order_id", "order_ts"], kind="stable"
).drop_duplicates("order_id", keep="last")
superseded_rows_removed = rows_after_exact_deduplication - len(orders_clean)
orders_clean["order_amount"] = pd.to_numeric(
    orders_clean["order_amount"], errors="coerce"
)
orders_clean = orders_clean.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)
amounts_filled = int(orders_clean["order_amount"].isna().sum())
orders_clean["order_amount"] = orders_clean["order_amount"].fillna(
    orders_clean["price_inr"]
    * orders_clean["quantity"]
    * (1 - orders_clean["discount_pct"] / 100)
).round(2)
orders_clean["delivery_minutes"] = pd.to_numeric(
    orders_clean["delivery_minutes"], errors="coerce"
)
negative_durations = int((orders_clean["delivery_minutes"] < 0).sum())
orders_clean.loc[orders_clean["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
completed = orders_clean["status"].eq("completed")
completed_delivery_fills = int(
    orders_clean.loc[completed, "delivery_minutes"].isna().sum()
)
category_medians = (
    orders_clean.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)
orders_clean.loc[completed, "delivery_minutes"] = (
    orders_clean.loc[completed, "delivery_minutes"].fillna(category_medians)
)
orders_clean = orders_clean.drop(columns=["category", "price_inr"])
orders_clean = orders_clean.sort_values("order_ts").reset_index(drop=True)
orders_clean["order_ts"] = orders_clean["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
cleaning_changes = {
    **customer_changes,
    "order_timestamps_converted": timestamp_rows,
    "exact_duplicate_rows_removed": exact_duplicates_removed,
    "superseded_order_records_removed": superseded_rows_removed,
    "order_amounts_filled": amounts_filled,
    "negative_delivery_durations_set_missing": negative_durations,
    "completed_delivery_durations_imputed_by_category": completed_delivery_fills,
    "cancelled_delivery_durations_imputed": 0,
}

customers_clean.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders_clean.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

with sqlite3.connect(DB_PATH) as connection:
    customers_clean.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders_clean.to_sql("orders", connection, if_exists="replace", index=False)

queries = """
-- Q1: WHERE then HAVING; completed orders only.
SELECT customer_id, SUM(order_amount) AS spend_over_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: revenue by city and product category.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: products priced above the catalogue average.
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers without any completed order.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: customer completed spend compared with the average customer spend.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed'
    GROUP BY customer_id
), average_spend AS (
    SELECT AVG(total_spend) AS mean_spend FROM customer_spend
)
SELECT customer_id, total_spend
FROM customer_spend
WHERE total_spend > (SELECT mean_spend FROM average_spend);

-- Q6: city-local rankings of completed spend.
WITH customer_spend AS (
    SELECT c.city, c.customer_id, SUM(o.order_amount) AS total_spend
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_number
FROM customer_spend;

-- Q7: daily completed revenue and cumulative revenue.
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
"""
with open(SQL_PATH, "w", encoding="utf-8") as file:
    file.write(queries.strip() + "\n")

with sqlite3.connect(DB_PATH) as connection:
    sql_without_comments = re.sub(r"(?m)^--.*$", "", queries)
    sql_statements = [
        statement.strip()
        for statement in sql_without_comments.split(";")
        if statement.strip()
    ]
    if len(sql_statements) != 7:
        raise AssertionError(f"Expected seven SQL queries, got {len(sql_statements)}")
    q1, q2, q3, q4, q5, q6, q7 = [
        pd.read_sql_query(statement, connection)
        for statement in sql_statements
    ]

completed_orders = orders_clean.loc[orders_clean["status"].eq("completed")].copy()
completed_mask = orders_clean["status"].eq("completed")
completed_orders["order_ts"] = pd.to_datetime(completed_orders["order_ts"])
pivot = pd.pivot_table(
    completed_orders.merge(customers_clean[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id"),
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    fill_value=0,
)
city_agg = completed_orders.merge(
    customers_clean[["customer_id", "city"]], on="customer_id"
).groupby("city").agg(
    amount_sum=("order_amount", "sum"),
    amount_mean=("order_amount", "mean"),
    amount_count=("order_amount", "count"),
)
amount = completed_orders["order_amount"]
q1_amount, q3_amount = amount.quantile([0.25, 0.75])
iqr = q3_amount - q1_amount
lower_fence, upper_fence = q1_amount - 1.5 * iqr, q3_amount + 1.5 * iqr
correlations = completed_orders[
    ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
].corr()
correlation_pairs = correlations.where(
    np.triu(np.ones(correlations.shape), k=1).astype(bool)
).stack()
largest_pair = correlation_pairs.abs().idxmax()

checks = {
    "orders_rows": len(orders_clean) == 12309,
    "completed_rows": int((orders_clean["status"] == "completed").sum()) == 11110,
    "cancelled_rows": int((orders_clean["status"] == "cancelled").sum()) == 1199,
    "pending_removed": int((orders_clean["status"] == "pending").sum()) == 0,
    "amount_complete": int(orders_clean["order_amount"].isna().sum()) == 0,
    "completed_delivery_complete": int(orders_clean.loc[completed_mask, "delivery_minutes"].isna().sum()) == 0,
    "cancelled_delivery_missing": int(orders_clean.loc[orders_clean["status"].eq("cancelled"), "delivery_minutes"].isna().sum()) == 1199,
    "completed_revenue": round(float(amount.sum()), 2) == 2998668.80,
    "cities": customers_clean["city"].nunique() == 6,
    "ages_complete": int(customers_clean["age"].isna().sum()) == 0,
    "q1_customers": len(q1) == 30,
    "q2_cells": len(q2) == 30,
    "q3_products": len(q3) == 11,
    "q4_customers": len(q4) == 25,
    "q5_customers": len(q5) == 550,
    "q6_rank_ties": int((q6["spend_rank"] != q6["row_number"]).sum()) == 8,
    "q7_days": len(q7) == 183,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise AssertionError(f"Part A reference checkpoints failed: {failed}")

print("RAW_AUDIT=" + json.dumps(raw_audit, sort_keys=True))
print("RAW_METRICS=" + json.dumps(raw_metrics, sort_keys=True))
print("CLEAN_COUNTS=" + json.dumps({
    "city": customers_clean["city"].value_counts().sort_index().to_dict(),
    "gender": customers_clean["gender"].value_counts().sort_index().to_dict(),
    "age_min": int(customers_clean["age"].min()),
    "age_max": int(customers_clean["age"].max()),
    "order_rows": len(orders_clean),
    "completed": int(orders_clean["status"].eq("completed").sum()),
    "cancelled": int(orders_clean["status"].eq("cancelled").sum()),
    "completed_revenue": round(float(amount.sum()), 2),
    "category_median_fills": completed_delivery_fills,
}, sort_keys=True))
print("CLEANING_CHANGES=" + json.dumps(cleaning_changes, sort_keys=True))
print("SQL_RESULTS=" + json.dumps({
    "Q1": len(q1),
    "Q2_rows": len(q2),
    "Q2_total": round(float(q2["revenue"].sum()), 2),
    "Q2_largest_cell": q2.loc[q2["revenue"].idxmax()].to_dict(),
    "Q3": len(q3),
    "Q4": len(q4),
    "Q5": len(q5),
    "Q5_average": round(float(completed_orders.groupby("customer_id")["order_amount"].sum().mean()), 2),
    "Q6_city_top": q6.sort_values(["city", "spend_rank"]).groupby("city").head(1)[
        ["city", "customer_id", "total_spend"]
    ].to_dict(orient="records"),
    "Q6_rank_not_row_number": int((q6["spend_rank"] != q6["row_number"]).sum()),
    "Q7_rows": len(q7),
    "Q7_final_cumulative": round(float(q7["cumulative_revenue"].iloc[-1]), 2),
}, sort_keys=True))
print("PANDAS_STATS=" + json.dumps({
    "pivot_grand_total": round(float(pivot.loc["All", "All"]), 2),
    "city_aggregates": city_agg.round(2).to_dict(orient="index"),
    "amount_mean": round(float(amount.mean()), 2),
    "amount_median": round(float(amount.median()), 2),
    "amount_skew": round(float(amount.skew()), 2),
    "q1": round(float(q1_amount), 2),
    "q3": round(float(q3_amount), 2),
    "lower_fence": round(float(lower_fence), 2),
    "upper_fence": round(float(upper_fence), 2),
    "outliers_below": int((amount < lower_fence).sum()),
    "outliers_above": int((amount > upper_fence).sum()),
    "largest_correlation_pair": [largest_pair[0], largest_pair[1]],
    "largest_correlation": round(float(correlation_pairs.loc[largest_pair]), 3),
    "correlations": correlations.round(3).to_dict(),
}, sort_keys=True))
print("ALL CHECKPOINTS PASSED")
