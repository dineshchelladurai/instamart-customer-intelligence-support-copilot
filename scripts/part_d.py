import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mock_llm


RAW = os.path.join(ROOT, "data", "raw")
POLICY_DIR = os.path.join(ROOT, "data", "policies")
OUT = os.path.join(ROOT, "outputs")
STATE = os.path.join(ROOT, "state")
DB_PATH = os.path.join(ROOT, "data", "instamart.db")
os.makedirs(OUT, exist_ok=True)
os.makedirs(STATE, exist_ok=True)


def load_chunks(overlap=10):
    chunks = []
    for path in sorted(Path(POLICY_DIR).glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({
                "source": path.name,
                "text": " ".join(words[start:end]),
            })
            if end == len(words):
                break
            start += 40 - overlap
    return chunks


def make_retriever(overlap=10):
    chunks = load_chunks(overlap)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    def retrieve(query, k=3):
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).ravel()
        top_indices = sorted(
            range(len(chunks)), key=lambda index: (-scores[index], index)
        )[:k]
        return [
            {
                "source": chunks[index]["source"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
            }
            for index in top_indices
        ]

    return chunks, retrieve


CHUNKS, retrieve = make_retriever(overlap=10)


def answer(question):
    results = retrieve(question, k=3)
    context = "\n".join(
        f"[{result['source']}] {result['text']}" for result in results
    )
    response = mock_llm.chat([
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only from the supplied context. "
                "If the answer is not in the context, say you could not find "
                "the information."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ])
    return response.choices[0].message.content or "", results


def evaluate_policy(retriever):
    evaluation = pd.read_csv(
        os.path.join(RAW, "policy_eval.csv"), keep_default_na=False
    )
    rows = []
    for record in evaluation.to_dict(orient="records"):
        results = retriever(record["question"], k=3)
        context = "\n".join(
            f"[{result['source']}] {result['text']}" for result in results
        )
        response = mock_llm.chat([
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT. Answer only from the supplied context. "
                    "If the answer is not in the context, say you could not find "
                    "the information."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\nQUESTION:\n{record['question']}",
            },
        ])
        text = response.choices[0].message.content or ""
        expected_doc = record["expected_doc"]
        rows.append({
            "question": record["question"],
            "expected_doc": expected_doc,
            "expected_fact": record["expected_fact"],
            "sources": [result["source"] for result in results],
            "answer": text,
            "source_hit": expected_doc == "NONE" or expected_doc in [
                result["source"] for result in results
            ],
            "fact_hit": (
                expected_doc == "NONE"
                or record["expected_fact"] in text
            ),
            "fallback": text.startswith("I could not find"),
        })
    answerable = [row for row in rows if row["expected_doc"] != "NONE"]
    unanswerable = [row for row in rows if row["expected_doc"] == "NONE"]
    return {
        "source_hits": sum(row["source_hit"] for row in answerable),
        "fact_hits": sum(row["fact_hit"] for row in answerable),
        "fallbacks": sum(row["fallback"] for row in unanswerable),
        "rows": rows,
    }


ZERO_OVERLAP_CHUNKS, zero_overlap_retrieve = make_retriever(overlap=0)
policy_evaluation = evaluate_policy(retrieve)
zero_overlap_evaluation = evaluate_policy(zero_overlap_retrieve)


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(row)


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


_ACTIVE_SCENARIO_ID = ""


def append_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "scenario_id": _ACTIVE_SCENARIO_ID,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with open(os.path.join(OUT, "refund_ledger.jsonl"), "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    return append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return append_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up order status, amount, and delivery time by order id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the most relevant policy chunks for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a full late-delivery refund when the delivery exceeds 45 minutes.",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]},
    },
    {
        "name": "issue_refund",
        "description": "Issue and ledger an approved refund to the original payment method.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
]
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


class SimulatedCrash(Exception):
    pass


def checkpoint_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens")
    return os.path.join(STATE, f"{run_id}.json")


def save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, tools_executed, llm_calls):
    payload = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
    }
    with open(checkpoint_path(run_id), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def status_from_answer(final_answer):
    if final_answer.startswith("FALLBACK_"):
        return final_answer.split(":", 1)[0]
    return final_answer.split(":", 1)[0] if ":" in final_answer else "UNKNOWN"


def continue_agent(
    run_id,
    scenario_id,
    messages,
    tokens_used=0,
    steps=None,
    tools_executed=None,
    llm_calls=0,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    global _ACTIVE_SCENARIO_ID
    _ACTIVE_SCENARIO_ID = scenario_id
    steps = list(steps or [])
    tools_executed = list(tools_executed or [])
    started_at = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - started_at >= time_limit_seconds:
            final_answer = "FALLBACK_TIMEOUT: hand off to a human agent."
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        if tokens_used > token_limit:
            final_answer = "FALLBACK_TOKEN_LIMIT: hand off to a human agent."
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            final_answer = message.content or "FALLBACK_EMPTY_RESPONSE: hand off to a human agent."
            return {
                "status": status_from_answer(final_answer),
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }
        tool_call_records = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        ]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_call_records,
        })
        for call in calls:
            if call.name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {call.name}"}
            else:
                observation = TOOL_MAP[call.name](**call.arguments)
            observation_json = json.dumps(observation, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": observation_json,
            })
            steps.append({
                "tool_name": call.name,
                "arguments": call.arguments,
                "observation": observation,
            })
            tools_executed.append(call.name)
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                steps,
                tools_executed,
                llm_calls,
            )
            if crash_after_tools is not None and len(tools_executed) == crash_after_tools:
                raise SimulatedCrash(
                    f"Simulated crash after {crash_after_tools} tools; checkpoint saved."
                )
    final_answer = "FALLBACK_MAX_ITERATIONS: hand off to a human agent."
    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed,
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    run_id = run_id or scenario_id
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT. Use the available policy and order tools to follow the "
                "refund rules. Never issue an ineligible refund."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    return continue_agent(
        run_id,
        scenario_id,
        messages,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools,
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(checkpoint_path(run_id), encoding="utf-8") as file:
        state = json.load(file)
    return continue_agent(
        run_id=state["run_id"],
        scenario_id=state["scenario_id"],
        messages=state["messages"],
        tokens_used=state["tokens_used"],
        steps=state["steps"],
        tools_executed=state["tools_executed"],
        llm_calls=state["llm_calls"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


agent_scenarios = pd.read_csv(
    os.path.join(RAW, "agent_scenarios.csv"), keep_default_na=False
)
agent_runs_path = os.path.join(OUT, "agent_runs.jsonl")
ledger_path = os.path.join(OUT, "refund_ledger.jsonl")
for path in (agent_runs_path, ledger_path):
    with open(path, "w", encoding="utf-8"):
        pass

scenario_results = []
for record in agent_scenarios.to_dict(orient="records"):
    result = run_agent(record["message"], record["scenario_id"])
    expected_amount = float(record["expected_refund_inr"])
    actual_amount = next(
        (
            float(step["observation"]["refund_inr"])
            for step in result["steps"]
            if step["tool_name"] in {"issue_refund", "escalate_to_human"}
        ),
        0.0,
    )
    expected_status = record["expected_outcome"]
    status_matches = (
        result["status"].startswith("FALLBACK_")
        if expected_status == "FALLBACK"
        else result["status"] == expected_status
    )
    if not status_matches:
        raise AssertionError(
            f"{record['scenario_id']} expected {record['expected_outcome']}, got {result['status']}"
        )
    if abs(actual_amount - expected_amount) > 0.01:
        raise AssertionError(
            f"{record['scenario_id']} refund amount {actual_amount} != {expected_amount}"
        )
    run_record = {
        "scenario_id": record["scenario_id"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }
    append_jsonl(agent_runs_path, run_record)
    scenario_results.append({
        "scenario_id": record["scenario_id"],
        "status": result["status"],
        "refund_inr": round(actual_amount, 2),
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
    })

scenario_one = agent_scenarios.loc[
    agent_scenarios["scenario_id"].eq("S1")
].iloc[0]
token_demo = run_agent(
    scenario_one["message"], "S1_token_demo", token_limit=200
)
timeout_demo = run_agent(
    scenario_one["message"], "S1_timeout_demo", time_limit_seconds=0
)
if token_demo["status"] != "FALLBACK_TOKEN_LIMIT":
    raise AssertionError(f"Expected token-limit fallback, got {token_demo['status']}")
if timeout_demo["status"] != "FALLBACK_TIMEOUT" or timeout_demo["llm_calls"] != 0:
    raise AssertionError("Zero-second timeout must stop before the first model call")

crash_run_id = "S1_crash_resume"
crash_tools = []
try:
    run_agent(
        scenario_one["message"],
        "S1",
        run_id=crash_run_id,
        crash_after_tools=2,
    )
except SimulatedCrash:
    with open(checkpoint_path(crash_run_id), encoding="utf-8") as file:
        crash_tools.extend(json.load(file)["tools_executed"])
resumed = resume_agent(crash_run_id)
crash_tools = resumed["tools_executed"]
if crash_tools != [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund",
]:
    raise AssertionError(f"Unexpected crash/resume tool sequence: {crash_tools}")
if resumed["status"] != "REFUND" or abs(
    float(resumed["steps"][-1]["observation"]["refund_inr"]) - 128.25
) > 0.01:
    raise AssertionError("Crash-and-resume did not complete the expected refund")

if len(CHUNKS) != 15:
    raise AssertionError(f"Expected 15 overlapping chunks, found {len(CHUNKS)}")
if (policy_evaluation["source_hits"], policy_evaluation["fact_hits"], policy_evaluation["fallbacks"]) != (12, 12, 2):
    raise AssertionError("Overlapping policy retrieval did not meet all evaluation checkpoints")
if zero_overlap_evaluation["fact_hits"] >= 12:
    raise AssertionError("The zero-overlap evaluation should expose a chunk-boundary failure")
if scenario_results[-1]["status"] != "FALLBACK_MAX_ITERATIONS":
    raise AssertionError("S7 must stop after the max-iteration limit")
if scenario_results[0]["llm_calls"] != 5:
    raise AssertionError("S1 must complete in five model calls")
print("POLICY_CHUNKS=" + json.dumps({
    "total": len(CHUNKS),
    "by_source": pd.Series([chunk["source"] for chunk in CHUNKS]).value_counts().to_dict(),
    "refund_query_top_sources": [
        item["source"] for item in retrieve("refund policy late delivery", k=3)
    ],
}, sort_keys=True))
print("POLICY_EVALUATION=" + json.dumps({
    "overlap_10": {
        "source_hits": policy_evaluation["source_hits"],
        "fact_hits": policy_evaluation["fact_hits"],
        "fallbacks": policy_evaluation["fallbacks"],
    },
    "overlap_0": {
        "source_hits": zero_overlap_evaluation["source_hits"],
        "fact_hits": zero_overlap_evaluation["fact_hits"],
        "fallbacks": zero_overlap_evaluation["fallbacks"],
    },
    "overlap_0_failures": [
        row["question"] for row in zero_overlap_evaluation["rows"]
        if row["expected_doc"] != "NONE" and not row["fact_hit"]
    ],
}, sort_keys=True))
print("SCENARIO_RESULTS=" + json.dumps(scenario_results, sort_keys=True))
print("STOP_SAFE=" + json.dumps({
    "token_demo_status": token_demo["status"],
    "token_demo_calls": token_demo["llm_calls"],
    "timeout_demo_status": timeout_demo["status"],
    "timeout_demo_calls": timeout_demo["llm_calls"],
}, sort_keys=True))
print("CRASH_RESUME=" + json.dumps({
    "tools": crash_tools,
    "status": resumed["status"],
    "refund_inr": 128.25,
    "llm_calls": resumed["llm_calls"],
    "tokens_used": resumed["tokens_used"],
}, sort_keys=True))
