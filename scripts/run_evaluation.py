"""
Offline evaluation runner for the AI Ticket Router agent.

This script loads a small labeled dataset of customer messages and runs them
through the full LangGraph pipeline, then reports basic quality metrics:

- Intent classification accuracy and per-intent precision/recall
- Routing action accuracy (auto_respond vs route_to_human)

Usage:
    python scripts/run_evaluation.py

Requirements:
    - GROQ_API_KEY and OPENAI_API_KEY set in the environment
    - Knowledge base seeded (e.g. `make seed` or `make seed-docker`)
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import build_graph
from src.agent.state import TicketState
from src.db.database import close_db, init_db
from src.rag.vectorstore import VectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "evaluation" / "dataset.jsonl"


Intent = Literal["billing", "technical", "shipping", "general", "complaint", "refund"]
Action = Literal["auto_respond", "route_to_human"]


@dataclass
class Example:
    id: str
    message: str
    expected_intent: Intent
    expected_action: Action


async def load_graph() -> Any:
    """Initialize DB, vector store, and compiled graph for evaluation."""
    await init_db()
    vectorstore = VectorStore()
    await vectorstore.initialize()
    graph = build_graph(vectorstore)
    return graph, vectorstore


def load_dataset() -> List[Example]:
    """Load labeled evaluation examples from JSONL file."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    examples: List[Example] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                Example(
                    id=data["id"],
                    message=data["message"],
                    expected_intent=data["expected_intent"],
                    expected_action=data["expected_action"],
                )
            )
    return examples


async def run_single_example(graph: Any, example: Example) -> Dict[str, Any]:
    """Run a single example through the agent graph and capture predictions."""
    initial_state: TicketState = {
        "raw_message": example.message,
        "customer_id": "eval",
        "channel": "email",
        "timestamp": datetime.now(),
        "agent_trace": [],
    }
    result = await graph.ainvoke(initial_state)
    return {
        "id": example.id,
        "expected_intent": example.expected_intent,
        "predicted_intent": result.get("intent"),
        "expected_action": example.expected_action,
        "predicted_action": result.get("action"),
        "kb_confidence": result.get("kb_confidence"),
    }


def compute_classification_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute accuracy and per-class precision/recall for intent and action."""
    intent_tp = Counter()
    intent_fp = Counter()
    intent_fn = Counter()

    action_tp = Counter()
    action_fp = Counter()
    action_fn = Counter()

    total_intent = 0
    correct_intent = 0
    total_action = 0
    correct_action = 0

    for r in results:
        # Intent metrics
        true_intent = r["expected_intent"]
        pred_intent = r["predicted_intent"]
        if pred_intent is not None:
            total_intent += 1
            if pred_intent == true_intent:
                correct_intent += 1
                intent_tp[true_intent] += 1
            else:
                intent_fp[pred_intent] += 1
                intent_fn[true_intent] += 1

        # Action metrics
        true_action = r["expected_action"]
        pred_action = r["predicted_action"]
        if pred_action is not None:
            total_action += 1
            if pred_action == true_action:
                correct_action += 1
                action_tp[true_action] += 1
            else:
                action_fp[pred_action] += 1
                action_fn[true_action] += 1

    def prf(tp: Counter, fp: Counter, fn: Counter) -> Dict[str, Any]:
        per_class: Dict[str, Any] = {}
        for label in sorted({*tp.keys(), *fp.keys(), *fn.keys()}):
            t = tp[label]
            f_p = fp[label]
            f_n = fn[label]
            precision = t / (t + f_p) if t + f_p > 0 else 0.0
            recall = t / (t + f_n) if t + f_n > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            per_class[label] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "support": t + f_n,
            }
        return per_class

    metrics = {
        "intent": {
            "accuracy": round(correct_intent / total_intent, 3) if total_intent else 0.0,
            "per_class": prf(intent_tp, intent_fp, intent_fn),
        },
        "action": {
            "accuracy": round(correct_action / total_action, 3) if total_action else 0.0,
            "per_class": prf(action_tp, action_fp, action_fn),
        },
        "counts": {
            "total_examples": len(results),
            "evaluated_intents": total_intent,
            "evaluated_actions": total_action,
        },
    }
    return metrics


async def main() -> None:
    print("Loading evaluation dataset...")
    examples = load_dataset()
    print(f"Loaded {len(examples)} examples from {DATASET_PATH}")

    print("Initializing agent graph and vector store...")
    graph, vectorstore = await load_graph()

    try:
        print("Running evaluation...")
        results: List[Dict[str, Any]] = []
        for example in examples:
            r = await run_single_example(graph, example)
            results.append(r)

        metrics = compute_classification_metrics(results)

        print("\n=== Evaluation Results ===")
        print(json.dumps(metrics, indent=2))

        # Also print a compact summary for quick reading
        print("\nSummary:")
        print(f"- Intent accuracy: {metrics['intent']['accuracy']:.3f}")
        print(f"- Action accuracy: {metrics['action']['accuracy']:.3f}")

    finally:
        await close_db()
        await vectorstore.close()


if __name__ == "__main__":
    asyncio.run(main())

