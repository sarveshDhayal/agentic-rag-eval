"""
Unit and Integration Tests for Phase 6: Unified Evaluation Harness & Benchmark Runner.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap_ci import compute_bootstrap_ci
from src.evaluation.judge import EvaluationJudge


def test_bootstrap_ci_computation():
    """Verify non-parametric bootstrap resampling calculates valid 95% confidence intervals."""
    # Synthetic normal data with mean 0.70
    data = [0.70, 0.72, 0.68, 0.75, 0.69, 0.71, 0.67, 0.73]
    res = compute_bootstrap_ci(data, n_resamples=500, ci_level=0.95, seed=42)

    assert "mean" in res and "ci_lower" in res and "ci_upper" in res
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
    assert 0.65 <= res["ci_lower"]
    assert res["ci_upper"] <= 0.75
    print(f"  [✓] Bootstrap CI engine verified: Mean {res['mean']} [{res['ci_lower']}, {res['ci_upper']}]")


def test_evaluation_judge_logic():
    """Verify Judge evaluates faithfulness, correctness, and abstention accurately."""
    judge = EvaluationJudge()

    # Case 1: Out of scope query with refusal -> Pass
    ev_refuse = judge.evaluate_response(
        question="What is the Ph.D. stipend?",
        ground_truth="Handbook covers B.Tech only and contains no Ph.D. stipend info.",
        prediction="Based on the student handbook, I do not have sufficient information to answer as this topic is outside the scope.",
        category="out_of_scope",
        answerable=False,
    )
    assert ev_refuse.abstention_accuracy == 1.0
    assert ev_refuse.faithfulness == 1.0
    assert ev_refuse.hallucination_rate == 0.0

    # Case 2: Out of scope query with fabricated answer -> Fail (Hallucination)
    ev_hallucinate = judge.evaluate_response(
        question="What is the Ph.D. stipend?",
        ground_truth="Handbook covers B.Tech only and contains no Ph.D. stipend info.",
        prediction="The monthly stipend for Ph.D. scholars is ₹35,000 per month.",
        category="out_of_scope",
        answerable=False,
    )
    assert ev_hallucinate.abstention_accuracy == 0.0
    assert ev_hallucinate.hallucination_rate == 1.0
    print("  [✓] Independent Evaluation Judge logic verified")


def test_eval_results_and_summary_files():
    """Verify results/eval_results.csv and results/eval_summary.json exist and are populated."""
    csv_path = PROJECT_ROOT / "results/eval_results.csv"
    json_path = PROJECT_ROOT / "results/eval_summary.json"

    assert csv_path.is_file(), "results/eval_results.csv must exist"
    assert json_path.is_file(), "results/eval_summary.json must exist"

    # Verify CSV structure
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) >= 40, f"Expected >= 40 evaluation rows, got {len(reader)}"
        sample = reader[0]
        expected_cols = {
            "question_id",
            "category",
            "answerable",
            "system_name",
            "faithfulness",
            "hallucination_rate",
            "answer_correctness",
            "abstention_accuracy",
            "latency_seconds",
            "total_tokens",
        }
        assert expected_cols.issubset(sample.keys())

    # Verify JSON summary structure
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "systems" in summary
    assert "System A (Prompting)" in summary["systems"]
    assert "System B (Fine-Tuned)" in summary["systems"]
    assert "System C (Basic RAG)" in summary["systems"]
    assert "System D (Agentic RAG)" in summary["systems"]

    sys_d = summary["systems"]["System D (Agentic RAG)"]
    assert "faithfulness" in sys_d
    assert "ci_lower" in sys_d["faithfulness"]
    assert "ci_upper" in sys_d["faithfulness"]
    assert "category_breakdown" in sys_d

    print(f"  [✓] Evaluation artifacts verified ({len(reader)} rows in CSV, 4 systems summarized with 95% CIs in JSON)")


if __name__ == "__main__":
    print("Running Phase 6 tests...")
    test_bootstrap_ci_computation()
    test_evaluation_judge_logic()
    test_eval_results_and_summary_files()
    print("All Phase 6 tests passed successfully!")

