"""
Empirical Ablation Experiments:
1. Chunking Strategy Ablation: Fixed-Size vs Section-Aware Chunking on RAG.
2. Top-k Retrieval Parameter Sweep: k in {2, 4, 8} on retrieval hit rate and token consumption.
3. Component Ablation: Agentic RAG With vs. Without Query Rewriting.

Saves results directly to results/ablations_summary.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import HandbookRetriever
from src.retrieval.retrieval_eval import evaluate_retrieval_strategy
from src.systems.system_c_rag import SystemCRAG
from src.systems.system_d_agentic import SystemDAgentic
from src.evaluation.judge import EvaluationJudge


def run_topk_ablation(eval_items: List[Dict[str, Any]], k_values: List[int] = [2, 4, 8]) -> Dict[str, Any]:
    print("Running Top-k Retrieval Ablation Sweep...")
    results: Dict[str, Any] = {}
    judge = EvaluationJudge()

    # Use answerable questions
    answerable_set = [item for item in eval_items if item.get("answerable")][:15]

    for k in k_values:
        print(f"  Evaluating Top-k = {k}...")
        rag_sys = SystemCRAG(top_k=k, strategy="section_aware")
        
        correct_count = 0
        total_tokens = 0
        latencies = []

        for item in answerable_set:
            q = item["question"]
            gt = item["ground_truth"]
            cat = item["category"]

            t0 = time.time()
            resp = rag_sys.answer(q)
            latencies.append(time.time() - t0)
            total_tokens += resp.total_tokens

            eval_res = judge.evaluate_response(
                question=q,
                ground_truth=gt,
                prediction=resp.response,
                category=cat,
                answerable=True,
                retrieved_chunks=resp.retrieved_chunks,
            )
            if eval_res.answer_correctness >= 0.5:
                correct_count += 1

        accuracy = correct_count / len(answerable_set)
        avg_tokens = total_tokens / len(answerable_set)
        avg_latency = sum(latencies) / len(latencies)

        results[f"k_{k}"] = {
            "top_k": k,
            "evaluated_queries": len(answerable_set),
            "answer_correctness": round(accuracy, 4),
            "mean_token_usage": round(avg_tokens, 1),
            "mean_latency_seconds": round(avg_latency, 4),
        }
        print(f"    k={k} -> Correctness: {accuracy:.2%} | Mean Tokens: {avg_tokens:.1f} | Latency: {avg_latency:.3f}s")

    return results


def run_rewriter_ablation(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("Running Agentic RAG Query-Rewriter Component Ablation...")
    judge = EvaluationJudge()
    multi_hop_items = [item for item in eval_items if item.get("category") == "multi_hop"][:10]

    # System D WITH rewriter
    agent_with_rewriter = SystemDAgentic()

    # System D WITHOUT rewriter (mock/bypass rewriter tool)
    agent_no_rewriter = SystemDAgentic()
    # Bypass rewriter by returning identity
    agent_no_rewriter.rewriter_tool.run = lambda q: q

    correct_with = 0
    correct_without = 0
    tokens_with = 0
    tokens_without = 0

    for item in multi_hop_items:
        q = item["question"]
        gt = item["ground_truth"]

        # With rewriter
        res_with = agent_with_rewriter.answer(q)
        tokens_with += res_with.total_tokens
        ev_with = judge.evaluate_response(q, gt, res_with.response, "multi_hop", True, res_with.retrieved_chunks)
        if ev_with.answer_correctness >= 0.5:
            correct_with += 1

        # Without rewriter
        res_without = agent_no_rewriter.answer(q)
        tokens_without += res_without.total_tokens
        ev_without = judge.evaluate_response(q, gt, res_without.response, "multi_hop", True, res_without.retrieved_chunks)
        if ev_without.answer_correctness >= 0.5:
            correct_without += 1

    total = len(multi_hop_items)
    acc_with = correct_with / total
    acc_without = correct_without / total

    print(f"  Agentic RAG WITH Rewriting:    Correctness: {acc_with:.2%} | Avg Tokens: {tokens_with/total:.1f}")
    print(f"  Agentic RAG WITHOUT Rewriting: Correctness: {acc_without:.2%} | Avg Tokens: {tokens_without/total:.1f}")

    return {
        "multi_hop_queries_evaluated": total,
        "with_query_rewriter": {
            "answer_correctness": round(acc_with, 4),
            "mean_tokens": round(tokens_with / total, 1),
        },
        "without_query_rewriter": {
            "answer_correctness": round(acc_without, 4),
            "mean_tokens": round(tokens_without / total, 1),
        },
        "delta_accuracy": round(acc_with - acc_without, 4),
    }


def run_chunking_ablation(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("Running Chunking Strategy Ablation (Fixed-Size vs Section-Aware)...")
    sec_retriever = HandbookRetriever(strategy="section_aware", auto_index=True)
    fix_retriever = HandbookRetriever(strategy="fixed_size", auto_index=True)

    sec_metrics = evaluate_retrieval_strategy(sec_retriever, eval_items)
    fix_metrics = evaluate_retrieval_strategy(fix_retriever, eval_items)

    return {
        "section_aware": {
            "hit@1": sec_metrics["hit@1"],
            "hit@3": sec_metrics["hit@3"],
            "hit@5": sec_metrics["hit@5"],
            "mrr": sec_metrics["mrr"],
        },
        "fixed_size": {
            "hit@1": fix_metrics["hit@1"],
            "hit@3": fix_metrics["hit@3"],
            "hit@5": fix_metrics["hit@5"],
            "mrr": fix_metrics["mrr"],
        },
        "hit@1_delta": round(sec_metrics["hit@1"] - fix_metrics["hit@1"], 4),
        "mrr_delta": round(sec_metrics["mrr"] - fix_metrics["mrr"], 4),
    }


def main() -> None:
    eval_path = PROJECT_ROOT / "data/eval/golden_eval_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("=" * 75)
    print("STARTING EMPIRICAL ABLATION EXPERIMENTS")
    print("=" * 75)

    chunking_ablation = run_chunking_ablation(eval_items)
    topk_ablation = run_topk_ablation(eval_items, k_values=[2, 4, 8])
    rewriter_ablation = run_rewriter_ablation(eval_items)

    ablations_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunking_strategy_ablation": chunking_ablation,
        "top_k_parameter_sweep": topk_ablation,
        "query_rewriter_component_ablation": rewriter_ablation,
    }

    out_file = PROJECT_ROOT / "results/ablations_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablations_summary, f, indent=2)

    print("=" * 75)
    print(f"[✓] Saved complete ablation study results to: {out_file}")


if __name__ == "__main__":
    main()

