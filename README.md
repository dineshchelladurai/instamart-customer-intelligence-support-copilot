x`# Instamart Customer Intelligence & Support Copilot

## Overview
This repository builds a customer intelligence and support copilot for a fictional Instamart scenario. The project is split into four connected parts: data cleaning and storage, customer segmentation and churn prediction, support ticket triage with guardrails, and a policy-driven refund assistant.

## Reproduce
1. `python -m venv .venv`
2. `python -m pip install --upgrade pip`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python scripts/part_a.py`
6. `python scripts/part_b.py`
7. `python scripts/part_c.py`
8. `python scripts/part_d.py`

## Part A
### Audit summary
| Metric | Value |
|---|---:|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct raw gender spellings | 10 |
| blank genders | 30 |

### Customer cleaning rules
- Remove leading/trailing whitespace and apply title case to city names.
- Map gender spellings to `male`, `female`, or `unknown`.
- Replace ages outside 10-100 with missing and impute with the city median using `groupby().transform()`.
- Convert `signup_date` to a pandas datetime.

After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; all ages between 18 and 60.

### Order cleaning rules
1. Convert `order_ts` to datetime.
2. Remove exact duplicate records.
3. Keep only the latest record per `order_id` by timestamp.
4. Convert `order_amount` to numeric; invalid values become missing; fill using the product price and quantity discount formula.
5. Set negative `delivery_minutes` to missing.
6. For completed orders, fill missing `delivery_minutes` using the median delivery time of the same product category among completed orders.
7. Cancelled orders keep `delivery_minutes` missing.

Reference outputs: 12,309 rows; 11,110 completed, 1,199 cancelled; no pending rows; sum of order_amount over completed orders = INR 2,998,668.80.

### SQLite and SQL
The cleaned tables are exported to `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv` and stored in `data/instamart.db` as `customers`, `products`, and `orders`. Seven queries are stored in `sql/queries.sql` and cover revenue distribution, order completion rates, and city-level retention patterns.

## Part B
Customer segments are generated from behaviour metrics and a churn model is trained with the cleaned customer and order data. The outputs are written to `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`.

## Part C
Support tickets are processed through the offline mock LLM classifier, with masked personal data before classification. Output files: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

## Part D
The policy assistant answers policy questions using the ground-truth documents in `data/policies/`, while the refund agent uses `data/instamart.db` to validate orders and issue refunds. Outputs: `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, and `state/`.

### Seven-scenario result table
| Scenario | Outcome |
|---|---|
| S1 | REFUND |
| S2 | ESCALATE |
| S3 | DENIED |
| S4 | DENIED |
| S5 | NOT_FOUND |
| S6 | FALLBACK |
| S7 | REFUND |

### Stop-safely demonstrations
- Stop-safe demo 1: tool call loop detection and graceful fallback.
- Stop-safe demo 2: policy assistant failed request handling and no unsafe action.

### Crash-and-resume tools
- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

## Limitations and Next Steps
This project uses synthetic data and a deterministic offline LLM stand-in. The next step would be moving the same workflow to a real data warehouse and production chat stack while keeping the policy guardrails and audit trail in place.
