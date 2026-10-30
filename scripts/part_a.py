import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
DB_PATH = os.path.join(ROOT, "data", "instamart.db")
SQL_PATH = os.path.join(ROOT, "sql", "queries.sql")

os.makedirs(CLEAN, exist_ok=True)

customers = pd.read_csv(os.path.join(RAW, "customers.csv"), keep_default_na=False)
products = pd.read_csv(os.path.join(RAW, "products.csv"), keep_default_na=False)
orders = pd.read_csv(os.path.join(RAW, "orders.csv"), keep_default_na=False)

# Audit summary for README and quick validation.
audit = {
    "raw_orders_rows": len(orders),
    "exact_duplicate_rows": int(orders.duplicated().sum()),
    "pending_rows": int((orders["status"] == "pending").sum()),
    "distinct_order_ids": int(orders["order_id"].nunique()),
    "non_numeric_order_amount": int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()),
    "negative_delivery_minutes": int((pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum()),
    "blank_delivery_minutes": int((orders["delivery_minutes"].astype(str).str.strip() == "").sum()),
    "blank_age": int((customers["age"].astype(str).str.strip() == "").sum()),
    "age_out_of_range": int(customers["age"].map(lambda x: str(x).strip() not in ["", "nan"] and not (10 <= float(x) <= 100)).sum()),
    "distinct_raw_city_spellings": int(customers["city"].nunique()),
    "distinct_non_blank_raw_gender_spellings": int(customers.loc[customers["gender"].astype(str).str.strip() != "", "gender"].nunique()),
    "blank_gender": int((customers["gender"].astype(str).str.strip() == "").sum()),
}

# Clean customers.
customers_clean = customers.copy()
customers_clean["city"] = customers_clean["city"].astype(str).str.strip().str.title()
customers_clean["gender"] = customers_clean["gender"].astype(str).str.strip()
customers_clean["gender"] = customers_clean["gender"].replace({
    "": "unknown",
    "male": "male",
    "Male": "male",
    "M": "male",
    "m": "male",
    "MALE": "male",
    "female": "female",
    "Female": "female",
    "F": "female",
    "f": "female",
    "FEMALE": "female",
})
customers_clean["gender"] = customers_clean["gender"].fillna("unknown")
customers_clean["age"] = pd.to_numeric(customers_clean["age"], errors="coerce")
customers_clean["age"] = customers_clean["age"].where(customers_clean["age"].between(10, 100), pd.NA)
city_medians = customers_clean.groupby("city")["age"].transform("median")
customers_clean["age"] = customers_clean["age"].fillna(city_medians)
customers_clean["signup_date"] = pd.to_datetime(customers_clean["signup_date"], errors="coerce")

# Clean orders.
orders_clean = orders.copy()
orders_clean["order_ts"] = pd.to_datetime(orders_clean["order_ts"], errors="coerce")
orders_clean = orders_clean.drop_duplicates()
orders_clean = orders_clean.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
orders_clean["order_amount"] = pd.to_numeric(orders_clean["order_amount"], errors="coerce")
orders_clean = orders_clean.merge(products[["product_id", "price_inr"]], on="product_id", how="left")
orders_clean["order_amount"] = orders_clean["order_amount"].fillna(orders_clean["price_inr"] * orders_clean["quantity"] * (1 - orders_clean["discount_pct"] / 100))
orders_clean["delivery_minutes"] = pd.to_numeric(orders_clean["delivery_minutes"], errors="coerce")
orders_clean.loc[orders_clean["status"] == "completed", "delivery_minutes"] = orders_clean.loc[orders_clean["status"] == "completed", "delivery_minutes"].fillna(
    orders_clean.loc[orders_clean["status"] == "completed"].groupby("product_id")["delivery_minutes"].transform("median")
)
orders_clean.loc[orders_clean["status"] == "cancelled", "delivery_minutes"] = pd.NA
orders_clean["order_amount"] = orders_clean["order_amount"].round(2)
orders_clean = orders_clean.drop(columns=["price_inr"])
orders_clean = orders_clean.sort_values("order_ts").reset_index(drop=True)

customers_clean.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders_clean.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

# SQLite database.
conn = sqlite3.connect(DB_PATH)
customers_clean.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

# SQL queries.
queries = '''
-- 1. Total gross revenue in completed orders
SELECT ROUND(SUM(order_amount), 2) AS revenue_completed FROM orders WHERE status = 'completed';

-- 2. Order completion rate by city
SELECT c.city, COUNT(o.order_id) AS orders, SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END) AS completed
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY completed DESC;

-- 3. Average order value by product category
SELECT p.category, ROUND(AVG(o.order_amount), 2) AS avg_order_value
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category;

-- 4. Most common late-delivery product categories
SELECT p.category, COUNT(*) AS late_orders
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed' AND o.delivery_minutes > 45
GROUP BY p.category
ORDER BY late_orders DESC;

-- 5. Top customers by lifetime value
SELECT c.customer_id, ROUND(SUM(o.order_amount), 2) AS lifetime_value
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY lifetime_value DESC
LIMIT 10;

-- 6. New customers by month
SELECT strftime('%Y-%m', signup_date) AS signup_month, COUNT(*) AS customers
FROM customers
GROUP BY signup_month
ORDER BY signup_month;

-- 7. Cancelled-order share by city
SELECT c.city, ROUND(100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS cancelled_pct
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY cancelled_pct DESC;
'''
with open(SQL_PATH, "w", encoding="utf-8") as f:
    f.write(queries.strip() + "\n")

# Basic validation for the key checkpoints.
assert len(orders_clean) == orders["order_id"].nunique(), "clean order count should be unique order ids"
assert (orders_clean["status"] == "pending").sum() == 0, "no pending orders remain in clean export"
assert (orders_clean["status"] == "completed").sum() == 11110, "completed count should be 11110"
assert (orders_clean["status"] == "cancelled").sum() == 1199, "cancelled count should be 1199"
assert orders_clean["order_amount"].isna().sum() == 0
assert orders_clean.loc[orders_clean["status"] == "completed", "delivery_minutes"].isna().sum() == 0
assert orders_clean.loc[orders_clean["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199
print("ALL CHECKPOINTS PASSED")
