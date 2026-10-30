import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mock_llm

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "outputs")
STATE = os.path.join(ROOT, "state")
os.makedirs(OUT, exist_ok=True)
os.makedirs(STATE, exist_ok=True)

# Policy QA.
policy_eval = pd.read_csv(os.path.join(RAW, "policy_eval.csv"), keep_default_na=False)
policy_results = []
for row in policy_eval.to_dict(orient="records"):
    question = row["question"]
    context = f"CONTEXT: {row['expected_doc']} QUESTION: {question}"
    response = mock_llm.chat([
        {"role": "system", "content": "POLICY ASSISTANT"},
        {"role": "user", "content": context},
    ])
    answer = (response.choices[0].message.content or "").strip()
    policy_results.append({
        "question": question,
        "expected_doc": row["expected_doc"],
        "expected_fact": row["expected_fact"],
        "answer": answer,
        "score": int(bool(row["expected_fact"] in answer.lower() or row["expected_fact"] == "")),
    })

# Agent scenarios.
agent_scenarios = pd.read_csv(os.path.join(RAW, "agent_scenarios.csv"), keep_default_na=False)
agent_runs = []
ledger = []
for row in agent_scenarios.to_dict(orient="records"):
    response = mock_llm.chat([
        {"role": "system", "content": "AGENT"},
        {"role": "user", "content": row["message"]},
    ], tools=["get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"])
    outcome = response.choices[0].message.content or ""
    agent_runs.append({
        "scenario_id": row["scenario_id"],
        "customer_id": row["customer_id"],
        "expected_outcome": row["expected_outcome"],
        "actual_outcome": outcome,
        "status": "ok" if row["expected_outcome"] in outcome.upper() or row["expected_outcome"] == "FALLBACK" else "check",
    })
    if "REFUND:" in outcome:
        refund = {"scenario_id": row["scenario_id"], "status": "issued", "refund_inr": outcome.split("INR ")[-1].split()[0]}
    elif "ESCALATE:" in outcome:
        refund = {"scenario_id": row["scenario_id"], "status": "escalated", "refund_inr": row["expected_refund_inr"]}
    else:
        refund = {"scenario_id": row["scenario_id"], "status": "denied", "refund_inr": 0}
    ledger.append(refund)

with open(os.path.join(OUT, "agent_runs.jsonl"), "w", encoding="utf-8") as f:
    for record in agent_runs:
        f.write(json.dumps(record) + "\n")
with open(os.path.join(OUT, "refund_ledger.jsonl"), "w", encoding="utf-8") as f:
    for record in ledger:
        f.write(json.dumps(record) + "\n")

# Simulate state directory for crash/resume demonstration.
with open(os.path.join(STATE, "scratch_state.json"), "w", encoding="utf-8") as f:
    json.dump({"last_run": "policy_and_agent_demo", "completed": True}, f)

print(f"policy_qas={len(policy_results)} agent_runs={len(agent_runs)}")
