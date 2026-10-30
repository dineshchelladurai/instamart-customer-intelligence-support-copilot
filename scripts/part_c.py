import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mock_llm

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def mask_email_phone(text):
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    return text


tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
triage_rows = []
for row in tickets.to_dict(orient="records"):
    text = mask_email_phone(row["text"] or "")
    payload = [
        {"role": "system", "content": "CLASSIFY ticket by issue type. Return JSON with category and reason."},
        {"role": "user", "content": text},
    ]
    try:
        response = mock_llm.chat(payload)
        content = response.choices[0].message.content
        data = json.loads(content)
        pred = data.get("category", "technical")
    except Exception:
        pred = "technical"
    triage_rows.append({
        "ticket_id": row["ticket_id"],
        "customer_id": row["customer_id"],
        "order_id": row["order_id"],
        "true_category": row["true_category"],
        "predicted_category": pred,
        "masked_text": text,
        "status": "routed",
    })

triage_df = pd.DataFrame(triage_rows)
triage_df.to_csv(os.path.join(OUT, "triage_results.csv"), index=False)

budget_demo = pd.DataFrame([
    {"budget": 1000, "tickets_per_day": 500, "avg_resolution_minutes": 17, "accuracy": 0.81},
    {"budget": 2500, "tickets_per_day": 1200, "avg_resolution_minutes": 11, "accuracy": 0.89},
    {"budget": 5000, "tickets_per_day": 2500, "avg_resolution_minutes": 8, "accuracy": 0.93},
])
budget_demo.to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
print(f"triaged_tickets={len(triage_df)}")
