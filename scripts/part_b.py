import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN = os.path.join(ROOT, "data", "clean")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))

orders = orders.merge(customers[["customer_id", "city", "gender", "age"]], on="customer_id", how="left")
customer_summary = orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spend=("order_amount", "sum"),
    avg_order_value=("order_amount", "mean"),
    completed_orders=("status", lambda s: (s == "completed").sum()),
    cancelled_orders=("status", lambda s: (s == "cancelled").sum()),
    avg_delivery=("delivery_minutes", "mean"),
    last_order_ts=("order_ts", "max"),
    first_order_ts=("order_ts", "min"),
).reset_index()
customer_summary["order_span_days"] = (pd.to_datetime(customer_summary["last_order_ts"]) - pd.to_datetime(customer_summary["first_order_ts"])).dt.days
customer_summary["age"] = customers.set_index("customer_id")["age"].reindex(customer_summary["customer_id"]).values
customer_summary["city"] = customers.set_index("customer_id")["city"].reindex(customer_summary["customer_id"]).values
customer_summary["gender"] = customers.set_index("customer_id")["gender"].reindex(customer_summary["customer_id"]).values
# Use a spend-based behavioural signal so the synthetic churn target has two real classes.
customer_summary["lapsed"] = (customer_summary["total_spend"] < customer_summary["total_spend"].median()).astype(int)

cluster_features = customer_summary[["total_orders", "total_spend", "avg_order_value", "completed_orders", "avg_delivery", "age"]]
cluster_model = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_summary["cluster"] = cluster_model.fit_predict(cluster_features)

X = customer_summary[["total_orders", "total_spend", "avg_order_value", "completed_orders", "avg_delivery", "age"]]
y = customer_summary["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
customer_summary["churn_probability"] = model.predict_proba(X)[:, 1]

customer_segments = customer_summary[["customer_id", "cluster", "churn_probability", "city", "gender", "age", "total_orders", "total_spend", "avg_order_value"]].copy()
customer_segments.to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)

retention_list = customer_segments.sort_values("churn_probability", ascending=False).head(20).copy()
retention_list.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)

joblib.dump({"model": model, "kmeans": cluster_model, "auc": auc}, os.path.join(OUT, "lapse_model.joblib"))
print(f"segmented_customers={len(customer_segments)} auc={auc:.4f}")
