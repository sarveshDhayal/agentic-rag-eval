"""
Unified Evaluation Runner & Comparative Benchmarking Engine.
Executes Systems A, B, C, and D over data/eval/golden_eval_set.json,
computes LLM-as-a-Judge metrics, runs 1000-resample Bootstrap Confidence Intervals,
and exports results to results/eval_results.csv and results/eval_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap_ci import compute_bootstrap_ci
from src.evaluation.judge import EvaluationJudge
from src.systems.base import BaseSystem
from src.systems.system_a_prompting import SystemAPrompting
from src.systems.system_b_finetuned import SystemBFinetuned
from src.systems.system_c_rag import SystemCRAG
from src.systems.system_d_agentic import SystemDAgentic
from src.utils.config_loader import load_config


class EvaluationRunner:
    """
    Coordinates multi-system evaluation over the student handbook golden dataset.
    """

    def __init__(
        self,
        systems: Optional[Dict[str, BaseSystem]] = None,
        judge: Optional[EvaluationJudge] = None,
    ) -> None:
        self.config = load_config()
        self.judge = judge or EvaluationJudge()

        # Initialize the 4 comparative systems
        if systems is not None:
            self.systems = systems
        else:
            print("Initializing Comparative Systems (A, B, C, D)...")
            self.systems = {
                "System A (Prompting)": SystemAPrompting(),
                "System B (Fine-Tuned)": SystemBFinetuned(),
                "System C (Basic RAG)": SystemCRAG(),
                "System D (Agentic RAG)": SystemDAgentic(),
            }

    def load_dataset(self, smoke: bool = False, count: Optional[int] = None) -> List[Dict[str, Any]]:
        eval_path = PROJECT_ROOT / self.config["paths"]["golden_eval"]
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if smoke:
            # Select 2 questions per category for a representative 10-question smoke run
            categories = ["simple_factual", "multi_hop", "numeric_calculation", "out_of_scope", "adversarial"]
            smoke_set = []
            for cat in categories:
                cat_items = [item for item in data if item["category"] == cat]
                smoke_set.extend(cat_items[:2])
            return smoke_set[: (count or 10)]

        if count is not None:
            return data[:count]

        return data

    def run_evaluation(
        self,
        smoke: bool = False,
        count: Optional[int] = None,
        target_systems: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        dataset = self.load_dataset(smoke=smoke, count=count)
        active_systems = {
            name: sys_inst
            for name, sys_inst in self.systems.items()
            if target_systems is None or any(t.lower() in name.lower() for t in target_systems)
        }

        print("=" * 80)
        print(f"STARTING EVALUATION RUN: {len(dataset)} Questions | {len(active_systems)} Systems")
        print(f"Smoke Mode: {smoke} | Systems: {list(active_systems.keys())}")
        print("=" * 80)

        detailed_rows: List[Dict[str, Any]] = []
        system_metric_accumulator: Dict[str, Dict[str, List[float]]] = {
            s_name: {
                "faithfulness": [],
                "hallucination_rate": [],
                "answer_correctness": [],
                "abstention_accuracy": [],
                "context_precision": [],
                "context_recall": [],
                "latency_seconds": [],
                "total_tokens": [],
            }
            for s_name in active_systems
        }

        category_breakdown: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            s_name: {} for s_name in active_systems
        }

        for q_idx, item in enumerate(dataset, 1):
            q_id = item["id"]
            question = item["question"]
            ground_truth = item["ground_truth"]
            category = item["category"]
            answerable = item["answerable"]

            print(f"\n[{q_idx}/{len(dataset)}] Category: {category} | Q: {question[:60]}...")

            for s_name, sys_inst in active_systems.items():
                if category not in category_breakdown[s_name]:
                    category_breakdown[s_name][category] = {
                        "faithfulness": [],
                        "correctness": [],
                        "abstention": [],
                    }

                # 1. Run system inference
                t_start = time.time()
                resp = sys_inst.answer(question)
                inf_latency = time.time() - t_start

                # 2. Evaluate with independent Judge LLM
                eval_res = self.judge.evaluate_response(
                    question=question,
                    ground_truth=ground_truth,
                    prediction=resp.response,
                    category=category,
                    answerable=answerable,
                    retrieved_chunks=resp.retrieved_chunks,
                )

                # 3. Accumulate metrics
                acc = system_metric_accumulator[s_name]
                acc["faithfulness"].append(eval_res.faithfulness)
                acc["hallucination_rate"].append(eval_res.hallucination_rate)
                acc["answer_correctness"].append(eval_res.answer_correctness)
                acc["abstention_accuracy"].append(eval_res.abstention_accuracy)
                acc["context_precision"].append(eval_res.context_precision)
                acc["context_recall"].append(eval_res.context_recall)
                acc["latency_seconds"].append(resp.latency_seconds if resp.latency_seconds > 0 else inf_latency)
                acc["total_tokens"].append(float(resp.total_tokens))

                # Category breakdown accumulation
                cb = category_breakdown[s_name][category]
                cb["faithfulness"].append(eval_res.faithfulness)
                cb["correctness"].append(eval_res.answer_correctness)
                cb["abstention"].append(eval_res.abstention_accuracy)

                # Record detailed row
                detailed_rows.append({
                    "question_id": q_id,
                    "category": category,
                    "answerable": answerable,
                    "question": question,
                    "ground_truth": ground_truth,
                    "system_name": s_name,
                    "prediction": resp.response,
                    "retrieved_chunks_count": len(resp.retrieved_chunks),
                    "faithfulness": eval_res.faithfulness,
                    "hallucination_rate": eval_res.hallucination_rate,
                    "answer_correctness": eval_res.answer_correctness,
                    "abstention_accuracy": eval_res.abstention_accuracy,
                    "context_precision": eval_res.context_precision,
                    "context_recall": eval_res.context_recall,
                    "latency_seconds": round(resp.latency_seconds if resp.latency_seconds > 0 else inf_latency, 4),
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.total_tokens,
                    "judge_reasoning": eval_res.reasoning,
                })

                print(f"  {s_name:<24} -> Faith: {eval_res.faithfulness} | Corr: {eval_res.answer_correctness} | Lat: {acc['latency_seconds'][-1]:.3f}s | Tok: {resp.total_tokens}")

        # Compute summary with 95% Bootstrap CIs
        summary_results: Dict[str, Any] = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions_evaluated": len(dataset),
                "smoke_mode": smoke,
                "bootstrap_resamples": 1000,
                "confidence_level": 0.95,
            },
            "systems": {},
        }

        for s_name, metrics in system_metric_accumulator.items():
            sys_summary: Dict[str, Any] = {}
            for m_name, val_list in metrics.items():
                if m_name in ["latency_seconds", "total_tokens"]:
                    sys_summary[m_name] = {
                        "mean": round(float(np.mean(val_list)), 4) if val_list else 0.0,
                        "p50": round(float(np.percentile(val_list, 50)), 4) if val_list else 0.0,
                        "p95": round(float(np.percentile(val_list, 95)), 4) if val_list else 0.0,
                    }
                else:
                    ci_stats = compute_bootstrap_ci(val_list, n_resamples=1000, ci_level=0.95)
                    sys_summary[m_name] = ci_stats

            # Per-category summary
            cat_summary: Dict[str, Any] = {}
            for c_name, c_metrics in category_breakdown[s_name].items():
                cat_summary[c_name] = {
                    "count": len(c_metrics["correctness"]),
                    "faithfulness": round(float(np.mean(c_metrics["faithfulness"])), 4) if c_metrics["faithfulness"] else 0.0,
                    "correctness": round(float(np.mean(c_metrics["correctness"])), 4) if c_metrics["correctness"] else 0.0,
                    "abstention_accuracy": round(float(np.mean(c_metrics["abstention"])), 4) if c_metrics["abstention"] else 0.0,
                }
            sys_summary["category_breakdown"] = cat_summary

            summary_results["systems"][s_name] = sys_summary

        # Export outputs
        self._export_results(detailed_rows, summary_results)

        return summary_results

    def _export_results(
        self,
        detailed_rows: List[Dict[str, Any]],
        summary_results: Dict[str, Any],
    ) -> None:
        res_dir = PROJECT_ROOT / "results"
        res_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export CSV
        csv_path = res_dir / "eval_results.csv"
        if detailed_rows:
            fieldnames = list(detailed_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detailed_rows)
            print(f"\n[✓] Detailed question-level results exported to: {csv_path}")

        # 2. Export JSON summary
        json_path = res_dir / "eval_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"[✓] Summary metrics & 95% Bootstrap CIs exported to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG Multi-System Evaluation Runner")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (10 questions)")
    parser.add_argument("--count", type=int, default=None, help="Specific number of questions to evaluate")
    parser.add_argument("--all", action="store_true", help="Run full 105-question golden benchmark")
    parser.add_argument("--system", type=str, default=None, help="Filter specific system: A, B, C, D, or all")
    args = parser.parse_args()

    # Default to smoke if neither --all nor --count provided
    is_smoke = args.smoke or (not args.all and args.count is None)

    runner = EvaluationRunner()
    target_sys = [args.system] if args.system and args.system.lower() != "all" else None

    summary = runner.run_evaluation(
        smoke=is_smoke,
        count=args.count,
        target_systems=target_sys,
    )

    print("\n" + "=" * 80)
    print("EVALUATION HARNESS BENCHMARK SUMMARY")
    print("=" * 80)
    for s_name, stats in summary["systems"].items():
        f = stats["faithfulness"]
        h = stats["hallucination_rate"]
        c = stats["answer_correctness"]
        a = stats["abstention_accuracy"]
        lat = stats["latency_seconds"]
        tok = stats["total_tokens"]

        print(f"\nSystem: {s_name}")
        print(f"  Faithfulness:        {f['mean']:.2%} (95% CI: [{f['ci_lower']:.2%}, {f['ci_upper']:.2%}])")
        print(f"  Hallucination Rate:  {h['mean']:.2%} (95% CI: [{h['ci_lower']:.2%}, {h['ci_upper']:.2%}])")
        print(f"  Answer Correctness:  {c['mean']:.2%} (95% CI: [{c['ci_lower']:.2%}, {c['ci_upper']:.2%}])")
        print(f"  Abstention Accuracy: {a['mean']:.2%} (95% CI: [{a['ci_lower']:.2%}, {a['ci_upper']:.2%}])")
        print(f"  Latency (s):         Mean: {lat['mean']:.3f}s | p95: {lat['p95']:.3f}s")
        print(f"  Token Usage:         Mean: {tok['mean']:.1f} tokens")
    print("=" * 80)


if __name__ == "__main__":
    main()
