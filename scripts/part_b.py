import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN = os.path.join(ROOT, "data", "clean")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")

customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
orders["order_id"] = orders["order_id"].astype(str)
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

completed = orders.loc[orders["status"].eq("completed")].copy()
completed["order_date"] = completed["order_ts"].dt.normalize()
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

rfm_features = ["recency", "frequency", "monetary"]
rfm_scaler = StandardScaler()
scaled_rfm = rfm_scaler.fit_transform(rfm[rfm_features])
cluster_metrics = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled_rfm)
    cluster_metrics.append({
        "k": k,
        "inertia": round(float(model.inertia_), 1),
        "silhouette": round(float(silhouette_score(scaled_rfm, labels)), 3),
    })

chosen_k = 4
kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(scaled_rfm)
profiles = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
)
premium_id = int(profiles["mean_monetary"].idxmax())
profiles["segment_name"] = ""
profiles["is_premium"] = False
profiles.loc[premium_id, ["segment_name", "is_premium"]] = ["Premium", True]
remaining = profiles.drop(index=premium_id).sort_values(
    ["mean_frequency", "mean_recency"], ascending=[False, True]
)
segment_names = ["Frequent Loyalists", "Steady Shoppers", "Re-engagement Needed"]
for segment_id, name in zip(remaining.index, segment_names):
    profiles.loc[segment_id, "segment_name"] = name
rfm["segment_name"] = rfm["segment_id"].map(profiles["segment_name"])
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

pca = PCA(n_components=2)
pca.fit(scaled_rfm)
demographics = customers[["customer_id", "city", "gender", "age"]]
segments = rfm.merge(demographics, on="customer_id", how="left")
segment_output = segments[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
]
segment_output.to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)

before = orders.loc[
    orders["order_ts"].lt(CUTOFF)
    & orders["status"].isin(["completed", "cancelled"])
].copy()
completed_before = before.loc[before["status"].eq("completed")].copy()
cohort_ids = completed_before["customer_id"].unique()
after_window = completed.loc[
    completed["order_ts"].ge(CUTOFF) & completed["order_ts"].lt(WINDOW_END)
]
active_ids = set(after_window["customer_id"])
labels = pd.DataFrame({"customer_id": cohort_ids})
labels["lapsed"] = (~labels["customer_id"].isin(active_ids)).astype(int)

feature_rows = []
for customer_id in cohort_ids:
    cust_orders = before.loc[before["customer_id"].eq(customer_id)]
    cust_completed = cust_orders.loc[cust_orders["status"].eq("completed")]
    product_categories = cust_completed.merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    most_frequent = product_categories["category"].mode()
    customer = customers.loc[customers["customer_id"].eq(customer_id)].iloc[0]
    feature_rows.append({
        "customer_id": customer_id,
        "recency": (CUTOFF - cust_completed["order_ts"].max()).days,
        "frequency": cust_completed["order_id"].nunique(),
        "monetary": cust_completed["order_amount"].sum(),
        "average_discount_pct": cust_completed["discount_pct"].mean(),
        "cancellation_rate": float(cust_orders["status"].eq("cancelled").sum())
        / len(cust_orders),
        "average_delivery_minutes": cust_completed["delivery_minutes"].mean(),
        "favorite_category": most_frequent.iloc[0] if not most_frequent.empty else "Unknown",
        "city": customer["city"],
        "age": customer["age"],
        "gender": customer["gender"],
        "tenure_days": (CUTOFF - customer["signup_date"]).days,
    })
feature_frame = pd.DataFrame(feature_rows).merge(labels, on="customer_id", validate="one_to_one")

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]
categorical_features = ["favorite_category", "city", "gender"]
if feature_frame[numeric_features + categorical_features + ["lapsed"]].isna().any().any():
    raise AssertionError("Lapse features and labels must not contain missing values")

X = feature_frame[numeric_features + categorical_features]
y = feature_frame["lapsed"]
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    feature_frame["customer_id"],
    test_size=0.2,
    stratify=y,
    random_state=42,
)
preprocessor = ColumnTransformer(
    [
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical_features),
    ],
    remainder="drop",
)

baseline_predictions = np.zeros(len(y_test), dtype=int)
baseline = {
    "accuracy": round(float(accuracy_score(y_test, baseline_predictions)), 3),
    "lapsed_f1": round(float(f1_score(y_test, baseline_predictions, zero_division=0)), 3),
}
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
logistic_search = GridSearchCV(
    logistic_pipeline,
    {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
logistic_search.fit(X_train, y_train)

forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])
forest_search = GridSearchCV(
    forest_pipeline,
    {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
forest_search.fit(X_train, y_train)


def evaluate(estimator):
    prediction = estimator.predict(X_test)
    return {
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "precision": round(float(precision_score(y_test, prediction, pos_label=1, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, prediction, pos_label=1, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, prediction, pos_label=1, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y_test, prediction)), 3),
    }


logistic_metrics = evaluate(logistic_search.best_estimator_)
forest_metrics = evaluate(forest_search.best_estimator_)
logistic_model = logistic_search.best_estimator_.named_steps["classifier"]
logistic_names = logistic_search.best_estimator_.named_steps[
    "preprocessor"
].get_feature_names_out()
coefficients = dict(zip(logistic_names, logistic_model.coef_[0]))
coefficient_examples = {}
for feature in ["numeric__recency", "numeric__frequency", "numeric__monetary"]:
    coefficient_examples[feature] = round(float(coefficients[feature]), 4)
selected_name, selected_pipeline, selected_metrics = (
    ("random_forest", forest_search.best_estimator_, forest_metrics)
    if forest_metrics["f1"] >= logistic_metrics["f1"]
    else ("logistic_regression", logistic_search.best_estimator_, logistic_metrics)
)
if selected_metrics["f1"] < 0.55:
    raise AssertionError("Selected lapse model must reach positive-class test F1 >= 0.55")

class_counts = y_train.value_counts()
majority_count = int(class_counts.max())
minority_label = int(class_counts.idxmin())
minority_train = X_train.loc[y_train.eq(minority_label)]
minority_labels = y_train.loc[y_train.eq(minority_label)]
needed = majority_count - len(minority_train)
sampled_positions = np.random.RandomState(42).choice(
    len(minority_train), size=needed, replace=True
)
balanced_X = pd.concat(
    [X_train, minority_train.iloc[sampled_positions]], ignore_index=True
)
balanced_y = pd.concat(
    [y_train.reset_index(drop=True), minority_labels.iloc[sampled_positions].reset_index(drop=True)],
    ignore_index=True,
)
balanced_logistic = clone(logistic_search.best_estimator_)
balanced_logistic.fit(balanced_X, balanced_y)
balanced_logistic_metrics = evaluate(balanced_logistic)

forest_model = forest_search.best_estimator_.named_steps["classifier"]
forest_names = forest_search.best_estimator_.named_steps[
    "preprocessor"
].get_feature_names_out()
importance_indices = np.argsort(forest_model.feature_importances_)[::-1][:5]
top_importances = [
    {
        "feature": str(forest_names[index]),
        "importance": round(float(forest_model.feature_importances_[index]), 4),
    }
    for index in importance_indices
]
logistic_model = logistic_search.best_estimator_.named_steps["classifier"]
joblib.dump(
    {"model": selected_pipeline, "model_name": selected_name},
    os.path.join(OUT, "lapse_model.joblib"),
)
reloaded = joblib.load(os.path.join(OUT, "lapse_model.joblib"))["model"]
test_predictions = selected_pipeline.predict(X_test)
reloaded_predictions = reloaded.predict(X_test)
if not np.array_equal(test_predictions, reloaded_predictions):
    raise AssertionError("Reloaded model predictions differ from the original")

test_ids = pd.DataFrame({
    "customer_id": id_test.to_numpy(),
    "lapse_probability": selected_pipeline.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.to_numpy(),
})
premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
retention = test_ids.loc[test_ids["customer_id"].isin(premium_ids)].sort_values(
    "lapse_probability", ascending=False
).head(10)
retention.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)

print("RFM=" + json.dumps({
    "customers": len(rfm),
    "recency_median": int(rfm["recency"].median()),
    "recency_min": int(rfm["recency"].min()),
    "recency_max": int(rfm["recency"].max()),
    "frequency_mean": round(float(rfm["frequency"].mean()), 2),
    "frequency_max": int(rfm["frequency"].max()),
    "monetary_mean": round(float(rfm["monetary"].mean()), 2),
    "monetary_max": round(float(rfm["monetary"].max()), 2),
}, sort_keys=True))
print("CLUSTER_METRICS=" + json.dumps(cluster_metrics))
print("SEGMENT_PROFILES=" + json.dumps(
    profiles.reset_index().round(2).to_dict(orient="records"), sort_keys=True
))
print("PCA_VARIANCE=" + json.dumps([round(float(x), 3) for x in pca.explained_variance_ratio_]))
print("LAPSE=" + json.dumps({
    "cohort": len(feature_frame),
    "lapsed": int(y.sum()),
    "lapsed_percent": round(float(y.mean() * 100), 2),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "baseline": baseline,
    "frequency_scaler_train_mean": round(float(
        logistic_search.best_estimator_.named_steps["preprocessor"]
        .named_transformers_["numeric"].named_steps["scaler"].mean_[
            numeric_features.index("frequency")
        ]
    ), 3),
    "frequency_full_cohort_mean": round(float(feature_frame["frequency"].mean()), 3),
    "logistic_best_params": logistic_search.best_params_,
    "logistic_cv_f1": round(float(logistic_search.best_score_), 3),
    "logistic_test": logistic_metrics,
    "forest_best_params": forest_search.best_params_,
    "forest_cv_f1": round(float(forest_search.best_score_), 3),
    "forest_test": forest_metrics,
    "selected_model": selected_name,
    "balanced_train_counts": balanced_y.value_counts().sort_index().to_dict(),
    "balanced_logistic_test": balanced_logistic_metrics,
    "forest_top_features": top_importances,
    "logistic_coefficients": coefficient_examples,
    "premium_test_retention_rows": len(retention),
    "premium_retention_true_lapsed": int(retention["actually_lapsed"].sum()),
    "reloaded_predictions_match": bool(np.array_equal(test_predictions, reloaded_predictions)),
}, sort_keys=True))
