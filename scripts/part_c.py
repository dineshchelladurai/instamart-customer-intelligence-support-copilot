import json
import os
import random
import re
import sys
import time

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mock_llm


RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "outputs")
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")
WORD_PATTERN = re.compile(r"[a-z]+")
os.makedirs(OUT, exist_ok=True)

ZERO_SHOT_PROMPT = (
    "CLASSIFY each support ticket into exactly one allowed category: billing, delivery, "
    "technical, fraud. Reply with JSON only in the form "
    '{"category": "...", "reason": "..."} and use only the allowed category names.'
)
FEW_SHOT_PROMPT = (
    "CLASSIFY each support ticket into exactly one allowed category: billing, delivery, "
    "technical, fraud. Reply with JSON only in the form "
    '{"category": "...", "reason": "..."} and use only the allowed category names.\n'
    "EXAMPLES:\n"
    'Ticket: "I was charged twice for one order." '
    'JSON: {"category": "billing", "reason": "duplicate charge"}\n'
    'Ticket: "Someone hacked my wallet and made an unauthorized payment." '
    'JSON: {"category": "fraud", "reason": "unauthorized access"}\n'
)


def redact(text):
    counts = {}
    text, counts["email"] = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, counts["phone"] = PHONE_PATTERN.subn("[PHONE]", text)
    text, counts["order_id"] = ORDER_PATTERN.subn("[ORDER_ID]", text)
    return text, counts


def contains_unredacted_pii(text):
    return any(
        pattern.search(text)
        for pattern in (EMAIL_PATTERN, PHONE_PATTERN, ORDER_PATTERN)
    )


def call_with_retries(messages, ticket_id, rng):
    retries = 0
    while True:
        try:
            return mock_llm.chat(messages), retries
        except mock_llm.TransientLLMError:
            if retries >= 3:
                raise
            wait = BASE_DELAY * (2**retries) * rng.uniform(0.9, 1.15)
            print(
                f"retry ticket={ticket_id} retry={retries} wait={wait:.6f}s",
                flush=True,
            )
            time.sleep(wait)
            retries += 1


def parse_response(content):
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        words = WORD_PATTERN.findall((content or "").lower())
        categories = set(words) & ALLOWED_CATEGORIES
        if len(categories) == 1:
            return next(iter(categories)), "classified_keyword"
        return "", "flagged_invalid_output"
    category = result.get("category") if isinstance(result, dict) else None
    if category in ALLOWED_CATEGORIES:
        return category, "classified_json"
    return "", "flagged_invalid_output"


def run_pipeline(tickets, prompt, token_budget=None, reset=True):
    if reset:
        mock_llm.reset()
    rng = random.Random(42)
    running_tokens = 0
    retry_count = 0
    results = []
    for row in tickets.to_dict(orient="records"):
        ticket_id = row["ticket_id"]
        original_text = row["text"] or ""
        redacted, _ = redact(original_text)
        if original_text.strip() == "":
            status, category, ticket_tokens = "skipped_empty", "", 0
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status, category, ticket_tokens = "flagged_long", "", 0
        elif token_budget is not None and running_tokens >= token_budget:
            status, category, ticket_tokens = "budget_exhausted", "", 0
        else:
            assert not contains_unredacted_pii(redacted)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": redacted},
            ]
            assert messages[-1]["content"] == redacted
            assert not contains_unredacted_pii(messages[-1]["content"])
            response, retries = call_with_retries(messages, ticket_id, rng)
            retry_count += retries
            ticket_tokens = int(response.usage.total_tokens)
            running_tokens += ticket_tokens
            category, status = parse_response(
                response.choices[0].message.content or ""
            )
        results.append({
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "status": status,
            "category": category,
            "true_category": row["true_category"],
            "tokens_used": ticket_tokens,
        })
    return results, running_tokens, retry_count


def classify_summary(results, total_tokens, retries):
    classified = [
        row for row in results
        if row["status"] in {"classified_json", "classified_keyword"}
    ]
    accuracy = (
        sum(row["category"] == row["true_category"] for row in classified)
        / len(classified)
        if classified
        else 0
    )
    counts = pd.Series([row["status"] for row in results]).value_counts().to_dict()
    return {
        "status_counts": counts,
        "classified": len(classified),
        "accuracy": round(accuracy * 100, 2),
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": round(
            total_tokens / len(classified), 2
        ) if classified else 0,
        "retries": retries,
    }


tickets = pd.read_csv(
    os.path.join(RAW, "tickets.csv"), keep_default_na=False
)
redaction_totals = {"email": 0, "phone": 0, "order_id": 0}
for text in tickets["text"]:
    redacted_text, counts = redact(text)
    for key, value in counts.items():
        redaction_totals[key] += value
    if contains_unredacted_pii(redacted_text):
        raise AssertionError("PII remains after applying the three redaction patterns")
redaction_totals["total"] = sum(redaction_totals.values())

zero_results, zero_tokens, zero_retries = run_pipeline(
    tickets, ZERO_SHOT_PROMPT
)
few_results, few_tokens, few_retries = run_pipeline(
    tickets, FEW_SHOT_PROMPT, token_budget=100000
)
budget_limit = 6000
budget_results, budget_tokens, budget_retries = run_pipeline(
    tickets, ZERO_SHOT_PROMPT, token_budget=6000
)
if not any(row["status"] == "budget_exhausted" for row in budget_results):
    budget_limit = 5000
    budget_results, budget_tokens, budget_retries = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, token_budget=budget_limit
    )

segments = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))
segment_lookup = segments.set_index("customer_id").to_dict(orient="index")


def route_results(results, output_path):
    routed = []
    for row in results:
        segment = segment_lookup.get(row["customer_id"])
        segment_name = segment["segment_name"] if segment else ""
        if row["status"] in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route, priority = "human_review", "NORMAL"
        elif row["category"] == "fraud":
            route, priority = "security_team", "HIGH"
        elif segment and bool(segment["is_premium"]):
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        routed.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": row["status"],
            "category": row["category"],
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"],
        })
    pd.DataFrame(routed).to_csv(output_path, index=False)
    return routed


final_routed = route_results(
    few_results, os.path.join(OUT, "triage_results.csv")
)
budget_routed = route_results(
    budget_results, os.path.join(OUT, "triage_budget_demo.csv")
)
zero_classified = [
    row for row in zero_results
    if row["status"] in {"classified_json", "classified_keyword"}
]
confusion = pd.crosstab(
    pd.Series([row["true_category"] for row in zero_classified], name="actual"),
    pd.Series([row["category"] for row in zero_classified], name="predicted"),
).reindex(
    index=["billing", "delivery", "technical", "fraud"],
    columns=["billing", "delivery", "technical", "fraud"],
    fill_value=0,
)

assert len(final_routed) == 164
assert redaction_totals == {
    "email": 55,
    "phone": 91,
    "order_id": 132,
    "total": 278,
}
assert classify_summary(zero_results, zero_tokens, zero_retries)["status_counts"] == {
    "classified_json": 135,
    "flagged_invalid_output": 9,
    "classified_keyword": 12,
    "skipped_empty": 4,
    "flagged_long": 4,
}
assert classify_summary(few_results, few_tokens, few_retries)["accuracy"] == 100.0
assert zero_retries == 19 and few_retries == 19
assert any(row["status"] == "budget_exhausted" for row in budget_results)

print("REDACTION_COUNTS=" + json.dumps(redaction_totals, sort_keys=True))
print("ZERO_SHOT=" + json.dumps(
    classify_summary(zero_results, zero_tokens, zero_retries), sort_keys=True
))
print("FEW_SHOT=" + json.dumps(
    classify_summary(few_results, few_tokens, few_retries), sort_keys=True
))
print("ZERO_SHOT_CONFUSION=" + json.dumps(confusion.to_dict()))
print("BUDGET=" + json.dumps({
    "token_budget": budget_limit,
    "budget_exhausted": sum(
        row["status"] == "budget_exhausted" for row in budget_results
    ),
    "tokens_consumed": budget_tokens,
    "retry_count": budget_retries,
}))
print("FINAL_STATUS_COUNTS=" + json.dumps(
    pd.Series([row["status"] for row in final_routed]).value_counts().to_dict(),
    sort_keys=True,
))
