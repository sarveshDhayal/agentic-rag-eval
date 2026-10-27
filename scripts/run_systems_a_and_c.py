"""
Interactive runner and smoke evaluation for System A (Prompting) and System C (Basic RAG).
Accepts --smoke flag to test quickly on 5 representative questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.system_a_prompting import SystemAPrompting
from src.systems.system_c_rag import SystemCRAG


def run_ac_comparison(num_questions: int = 5) -> List[Dict[str, Any]]:
    eval_path = PROJECT_ROOT / "data/eval/golden_eval_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    # Pick 1 from each category for representative smoke test
    categories = ["simple_factual", "multi_hop", "numeric_calculation", "out_of_scope", "adversarial"]
    smoke_items = []
    for cat in categories:
        for item in eval_items:
            if item["category"] == cat:
                smoke_items.append(item)
                break

    if len(smoke_items) < num_questions:
        smoke_items = eval_items[:num_questions]

    print("=" * 80)
    print(f"RUNNING SYSTEMS A & C SMOKE COMPARISON ({len(smoke_items)} Questions)")
    print("=" * 80)

    sys_a = SystemAPrompting()
    sys_c = SystemCRAG()

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(smoke_items, 1):
        q = item["question"]
        gt = item["ground_truth"]
        cat = item["category"]

        print(f"\n[{idx}/{len(smoke_items)}] Category: {cat.upper()}")
        print(f"Question: {q}")
        print(f"Ground Truth: {gt}")

        res_a = sys_a.answer(q)
        res_c = sys_c.answer(q)

        print("-" * 40)
        print(f"System A (Prompting): {res_a.response[:160]}...")
        print(f"  Latency: {res_a.latency_seconds:.3f}s | Tokens: {res_a.total_tokens}")
        print("-" * 40)
        print(f"System C (Basic RAG): {res_c.response[:160]}...")
        print(f"  Latency: {res_c.latency_seconds:.3f}s | Tokens: {res_c.total_tokens} | Chunks: {len(res_c.retrieved_chunks)}")
        print("=" * 80)

        results.append({
            "id": item["id"],
            "category": cat,
            "question": q,
            "ground_truth": gt,
            "system_a": res_a.to_dict(),
            "system_c": res_c.to_dict(),
        })

    out_path = PROJECT_ROOT / "results/smoke_systems_ac.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved smoke comparison outputs to: {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run System A and System C comparison")
    parser.add_argument("--smoke", action="store_true", default=True, help="Run on 5 smoke questions")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to test")
    args = parser.parse_args()

    run_ac_comparison(num_questions=args.count if not args.smoke else 5)


if __name__ == "__main__":
    main()

