# Instamart Customer Intelligence and Support Copilot

## Overview

This capstone is a reproducible, offline demonstration of a customer intelligence and support workflow for a fictional grocery service. All data is synthetic; it is not real Instamart data, policy, or pricing. The project cleans order exports, builds a SQLite database, segments customers, predicts lapse risk, safely classifies support tickets, retrieves policy answers, and simulates guarded refund decisions. All model calls use the supplied local `mock_llm.py`; no network, API key, account, or paid service is used.

The four connected parts consume each other's outputs: Part A produces the trusted tables and database; Part B creates the segment file consumed by Part C; Part D uses the Part A database and its own retrieval code.

## Reproduce

Run these commands from the repository root with Python 3.11 or newer. The dependency list is frozen from the project virtual environment.

1. `py -3.11 -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install --upgrade pip`
3. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
4. `.\.venv\Scripts\python.exe generate_data.py`
5. `.\.venv\Scripts\python.exe scripts\part_a.py`
6. `.\.venv\Scripts\python.exe scripts\part_b.py`
7. `.\.venv\Scripts\python.exe scripts\part_c.py`
8. `.\.venv\Scripts\python.exe scripts\part_d.py`

The generator is deterministic with seed 42. The scripts validate their required reference results and write all generated data and outputs under `data\`, `outputs\`, and `state\`.

## Part A

### Raw data audit

Each cell below shows the raw table shape followed by each column's pandas dtype and missing count/percentage. Blank strings in tickets are preserved with `keep_default_na=False`; their four empty texts are blank strings, not pandas missing values.

| Table | Shape | Column | Dtype | Missing count | Missing percent |
|---|---:|---|---|---:|---:|
| customers | 1,200 x 5 | customer_id | str | 0 | 0.00% |
| customers | 1,200 x 5 | city | str | 0 | 0.00% |
| customers | 1,200 x 5 | signup_date | str | 0 | 0.00% |
| customers | 1,200 x 5 | age | float64 | 61 | 5.08% |
| customers | 1,200 x 5 | gender | str | 30 | 2.50% |
| products | 40 x 4 | product_id | str | 0 | 0.00% |
| products | 40 x 4 | product_name | str | 0 | 0.00% |
| products | 40 x 4 | category | str | 0 | 0.00% |
| products | 40 x 4 | price_inr | int64 | 0 | 0.00% |
| orders | 13,084 x 9 | order_id | str | 0 | 0.00% |
| orders | 13,084 x 9 | customer_id | str | 0 | 0.00% |
| orders | 13,084 x 9 | product_id | str | 0 | 0.00% |
| orders | 13,084 x 9 | quantity | int64 | 0 | 0.00% |
| orders | 13,084 x 9 | order_ts | str | 0 | 0.00% |
| orders | 13,084 x 9 | status | str | 0 | 0.00% |
| orders | 13,084 x 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13,084 x 9 | order_amount | str | 53 | 0.41% |
| orders | 13,084 x 9 | delivery_minutes | float64 | 2,541 | 19.42% |
| tickets | 164 x 5 | ticket_id | str | 0 | 0.00% |
| tickets | 164 x 5 | customer_id | str | 0 | 0.00% |
| tickets | 164 x 5 | order_id | str | 0 | 0.00% |
| tickets | 164 x 5 | text | str | 0 | 0.00% |
| tickets | 164 x 5 | true_category | str | 0 | 0.00% |

| Raw audit measure | Count |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order_id values | 12,309 |
| Non-numeric order_amount values | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning and validation

Customer cleaning removes city whitespace and applies title case (476 values normalized and 24 raw spellings become 6 cities), maps gender spellings to `male`, `female`, or `unknown` (950 values changed), converts ages outside 10–100 to missing, fills missing ages with the city median using `groupby().transform()`, and parses `signup_date` as datetime. This replaced and imputed 67 ages; all 1,200 signup dates were parsed, and the final age range is 18–60. Final city counts are Bengaluru 197, Chennai 191, Delhi 210, Hyderabad 191, Mumbai 184, and Pune 227. Final gender counts are female 610, male 560, and unknown 30.

Order cleaning is performed in this order: parse `order_ts`; remove exact duplicates; keep the latest timestamp per `order_id`; convert amounts to numeric and fill missing values with `round(price_inr * quantity * (1 - discount_pct / 100), 2)`; set negative delivery times to missing; fill missing completed-order delivery times from the completed-order median for that product category; leave cancelled-order delivery times missing.

| Cleaning operation | Rows or values changed |
|---|---:|
| Order timestamps parsed | 13,084 |
| Exact duplicate rows removed | 119 |
| Superseded order records removed | 656 |
| Invalid/missing order amounts filled | 112 |
| Negative delivery durations set missing after deduplication | 105 |
| Missing completed delivery durations imputed by category | 774 |
| Cancelled delivery durations imputed | 0 |

| Clean order checkpoint | Result |
|---|---:|
| Rows, one per order_id | 12,309 |
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| Missing order amounts | 0 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

The script writes CSVs to `data/clean/`, stores `customers`, `products`, and `orders` in `data/instamart.db`, and writes the seven executable queries to `sql/queries.sql`. Each query is run from Python with `pandas.read_sql_query`. Validation asserts more than eight reference checkpoints and prints exactly `ALL CHECKPOINTS PASSED` on success.

| SQL query | Result |
|---|---|
| Q1: completed orders over INR 500; customer totals over INR 8,000 | 30 customers |
| Q2: revenue by city and product category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3: products priced above catalogue average | 11 products |
| Q4: customers with no completed order | 25 customers |
| Q5: customers above average completed spend | 550 customers; average INR 2,552.06 |
| Q6: city-local RANK, DENSE_RANK, and ROW_NUMBER | Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10; 8 rows have RANK different from ROW_NUMBER |
| Q7: daily completed revenue with cumulative sum | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Tied spend values share a RANK, while ROW_NUMBER still assigns each customer a distinct position, which accounts for the eight differing rows.

The pandas pivot-table cross-check of Q2 has grand total INR 2,998,668.80. Bengaluru has completed-order sum INR 523,643.40, mean INR 276.47, count 1,894; Mumbai has sum INR 479,058.50, mean INR 297.00, count 1,613.

Completed order amounts have mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, and Q3 INR 297.00. The 1.5 x IQR fences are INR -283.50 and INR 645.30; there are 0 lower and 897 upper outliers. The largest absolute correlation is order_amount with discount_pct at -0.095; order_amount with quantity is +0.087. The correlation is small and does not establish causation.

The dataset describes synthetic completed grocery orders in six cities. Amounts are strongly right-skewed: the mean is about twice the median, and high-value orders create 897 upper outliers. The expected high-value tail is an anomaly to inspect, not automatically a data error. Amount has a weak negative relationship with discount_pct and a slightly weaker positive relationship with quantity; these correlations do not establish causal effects.

## Part B

### RFM segmentation

RFM uses completed orders through the full generated window and a snapshot of 2025-10-01 00:00:00. It includes 1,175 customers: recency median 19 days (range 0–180), frequency mean 9.46 (maximum 33), and monetary mean INR 2,552.06 (maximum INR 14,766.55).

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is chosen because it gives the best silhouette of the tested values, a clear elbow, and four campaign-sized groups. Inertia falls as K grows because more centroids can only reduce within-cluster distances; that alone does not mean the extra segments are useful or stable.

| Segment name | Size | Mean recency (days) | Mean frequency | Mean monetary | Recommended action |
|---|---:|---:|---:|---:|---|
| Steady Shoppers | 436 | 28.86 | 3.77 | INR 1,057.62 | Encourage a second monthly purchase with a relevant basket offer. |
| Premium | 138 | 24.12 | 6.77 | INR 7,129.70 | Retain with priority support and personalized high-value offers. |
| Frequent Loyalists | 419 | 8.77 | 19.40 | INR 3,410.02 | Reward repeat behavior and cross-sell complementary items. |
| Re-engagement Needed | 182 | 109.04 | 2.20 | INR 685.99 | Send a low-cost win-back message and monitor response. |

Premium is the segment with the highest mean monetary value. Segment IDs are arbitrary labels, not an ordered scale. PCA on standardized RFM explains 0.634 and 0.218 of variance in its first two components, 0.852 total.

`outputs/customer_segments.csv` contains exactly `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`.

### Leakage-safe lapse model

The lapse cohort has a completed order before 2025-09-01. Lapsed means no completed order from 2025-09-01 through 2025-09-30. Features use only orders before the cutoff; segment labels are not model features. The 1,156-customer cohort contains 443 lapsed customers (38.32%) and is split into 924 train and 232 test rows using stratification and random_state 42.

The all-not-lapsed baseline has 61.6% accuracy and lapsed-class F1 0. Features are recency, frequency, monetary, average discount, cancellation rate, average delivery duration, favorite product category, city, age, gender, and tenure. The numeric features are standardized and city, gender, and favorite category are one-hot encoded with the first level dropped. The preprocessor is inside each scikit-learn pipeline, so it is fit on training data only and then applied to test data. An explicit check confirms there are no missing model feature values. The fitted scaler's train-only frequency mean is 8.116 versus 8.029 in the full cohort because the stratified train split is a subset, not the full population.

| Model | Grid search | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Precision | Recall | F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | C = 0.01, 0.1, 1, 10, 100; 5-fold F1 | C = 0.1 | 0.639 | 105, 38 / 17, 72 | 0.655 | 0.809 | 0.724 |
| Random forest | depth = 3, 5, 8; leaf = 5, 10, 20; 45 fits | depth = 5, leaf = 10 | 0.670 | 111, 32 / 23, 66 | 0.673 | 0.742 | 0.706 |

Logistic regression is selected for its higher test F1 and recall in this run. Missing a lapsing customer can forfeit future orders, while wrongly targeting a loyal customer wastes campaign spend and may annoy them. The 0.724 positive-class F1 exceeds the required 0.55 and improves on the zero-F1 baseline.

| Best random-forest feature | Importance |
|---|---:|
| Frequency | 0.3281 |
| Monetary | 0.2721 |
| Recency | 0.1007 |
| Cancellation rate | 0.0541 |
| Favorite category: Electronics | 0.0509 |

| Logistic feature | Coefficient sign/value | Business reading |
|---|---:|---|
| Frequency | -1.2811 | More prior orders are associated with lower lapse odds in this fitted model. |
| Monetary | +0.0294 | Higher historic spend is conditionally associated with higher lapse odds in this fitted model. |
| Recency | -0.0285 | Recency has a negative conditional association with lapse odds after controlling for the other features. |

| Logistic regression training data | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training rows | 0.655 | 0.809 | 0.724 |
| Oversampled training rows (570 per class) | 0.587 | 0.910 | 0.714 |

Oversampling duplicates minority-class training customers only; the test set stays unchanged. Recall moves most, increasing by 0.101, while precision falls by 0.068, so the retention team finds more likely lapsers but sends more offers to loyal customers.

Only the training rows were used to fit the encoder, scaler, and model; no test rows were used for preprocessing or resampling. No orders dated on or after 2025-09-01 entered the lapse features, and no Part B segment labels were used as features. The model was saved to `outputs/lapse_model.joblib`, reloaded, and produced identical test predictions. The top 10 Premium customers in the test set are written to `outputs/retention_list.csv` with their probabilities and actual labels; 2 of these 10 truly lapsed. The Premium flag uses the full-window RFM and therefore includes the September period used for the label; this retention list is illustrative, not a leakage-free evaluation.

## Part C

### Privacy, guardrails, and resilient classification

Tickets are processed in file order. The input limit is `MAX_INPUT_TOKENS = 400`. Four empty tickets are marked `skipped_empty`, and four long tickets (734 estimated tokens each) are marked `flagged_long`; neither group is sent to the model. Personal information is redacted in the required order: email, phone, then order ID. Across all tickets, 55 emails, 91 phone numbers, and 132 order IDs are replaced (278 substitutions total). The code reapplies all three regex patterns and asserts no match remains before any call. The user message contains exactly the redacted ticket text.

Both system prompts request JSON-only output with the four allowed categories. The zero-shot prompt has no `EXAMPLES:` section. The few-shot prompt includes labelled billing and fraud examples. Each call retries `TransientLLMError` up to three times with exponential backoff and jitter using `0.05 * 2**n * random.uniform(0.9, 1.15)` seconds; one log line records each retry, ticket, retry number, and wait. `mock_llm.reset()` runs before each full batch. All 19 transient errors in each full run are recovered.

Output validation accepts valid JSON only when its category is allowed. For non-JSON replies, the pipeline extracts lowercase words with `re.findall(r"[a-z]+", text.lower())` and accepts exactly one allowed category word. All other replies, including an out-of-list JSON category, are flagged for human review rather than guessed.

| Full run | Classified | JSON | Keyword | Invalid output | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 135 | 12 | 9 | 85.71% (126/147) | 10,804 | 73.50 |
| Few-shot | 147 | 135 | 12 | 9 | 100.00% (147/147) | 17,044 | 115.95 |

| Actual \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

The few-shot prompt costs 42.45 additional tokens per classified ticket (17,044 versus 10,804 total), but improves accuracy by 14.29 percentage points and eliminates the 21 zero-shot misclassifications. For a queue where fraud and technical reports can be routed incorrectly, this accuracy improvement justifies the extra offline token cost.

The zero-shot token-budget demonstration uses 6,000 tokens. Once the running total reaches the budget, later calls stop and 69 tickets are marked `budget_exhausted`; the final response can take the total to 6,033 because the budget is checked before each call. The final `outputs/triage_results.csv` comes from the few-shot run with a 100,000-token budget. It has 164 rows and routes empty, long, invalid, and exhausted tickets to human review; fraud to the security team at HIGH priority; Premium customers to the premium desk at HIGH priority; and all remaining classified tickets to the support queue at NORMAL priority. Missing segment records are treated as non-premium. Rule-validated outputs are routed automatically; flagged statuses go to human review.

The output columns are `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. The budget demonstration is saved to `outputs/triage_budget_demo.csv`.

## Part D

### Retrieval-grounded policy assistant

The five policy files are split into 40-word chunks with 10-word overlap, producing 15 chunks: refund 4, delivery 3, escalation 3, payment 3, and privacy 2. A fitted `TfidfVectorizer` maps chunks and queries to vectors, and cosine similarity ranks the results. Term frequency (TF) measures how often a term appears in a chunk; inverse document frequency (IDF) reduces the weight of terms common across chunks. TF-IDF can miss semantically equivalent phrasings that use different words.

The `answer(question)` function retrieves the top three chunks and sends only those sources as labelled context to the offline policy assistant. For `refund policy late delivery`, all top three chunks come from `refund_policy.txt`.

| Chunk overlap | Expected source among top three | Expected facts found | Unanswerable questions with fallback |
|---|---:|---:|---:|
| 10 words | 12/12 | 12/12 | 2/2 |
| 0 words | 12/12 | 10/12 | 2/2 |

With zero overlap, the completed-order refund-threshold question and the failed-payment-return-time question lose their expected fact because the relevant policy sentence is split across chunk boundaries.

### Refund agent, safety, and resume

The agent exposes `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human` through `TOOLS_SPEC` and `TOOL_MAP`. Only completed orders qualify; delivery over 45 minutes is eligible for a full refund, delivery at or below 45 minutes yields no refund, and amounts over INR 1,000 are escalated instead of automatically issued. Each tool observation is appended to the conversation and checkpointed under `state/`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 952 |
| S2 | REFUND | 135.00 | 5 | 952 |
| S3 | ESCALATE | 1,999.00 | 5 | 939 |
| S4 | DENIED | 0.00 | 4 | 649 |
| S5 | DENIED | 0.00 | 3 | 370 |
| S6 | NOT_FOUND | 0.00 | 2 | 94 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,066 |

The stop-safely demonstrations return `FALLBACK_TOKEN_LIMIT` for S1 with a 200-token limit and `FALLBACK_TIMEOUT` for a zero-second time limit; the timeout stops before its first model call. S7 reaches all six calls within the default 6,000-token limit, then hands off at the iteration limit.

The S1 crash-and-resume demonstration saves a checkpoint immediately after the second tool observation, simulates a crash, and resumes without repeating work. The exact tool sequence across both runs is `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; the resumed result is REFUND for INR 128.25. `outputs/agent_runs.jsonl` contains one JSON line per scenario with status, final answer, calls, tokens, and tool observations. `outputs/refund_ledger.jsonl` records issued and escalated actions with timestamps and tool names.

ReAct calls the model after each observation (S1 uses 5 calls and 952 tokens, S5 uses 3 calls and 370 tokens, and S6 uses 2 calls and 94 tokens), which spends more tokens than a one-shot plan when the path is predictable. It can condition later actions on actual tool results: a missing order stops with NOT_FOUND and a cancelled order is denied before calculating or issuing a refund. A plan-and-execute agent could ask for a complete step list once and replan only on error, saving calls on routine cases, but a stale plan is risky when order status or policy retrieval changes the safe action. For this refund workflow, ReAct is preferable because it checks each result before taking the next consequential step.

## Limitations and Next Steps

The dataset and policy documents are synthetic and fictional, and the offline mock language model is deterministic rather than a production model. The churn label is an illustrative one-month no-order definition, not a validated business outcome. The Premium flag uses the whole RFM window and includes the September label window; therefore, the Premium retention list is illustrative and not a leakage-free evaluation. A production extension should validate the target with later-period data, monitor drift and human overrides, and preserve the policy checks, privacy redaction, and audit trail.
