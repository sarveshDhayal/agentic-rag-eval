"""
Unit and Integration Tests for Phase 7: Empirical Analysis, Plots, and Failure Case Taxonomy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_ablation_summary_artifacts():
    """Verify results/ablations_summary.json exists and has all 3 required experiments."""
    ablation_file = PROJECT_ROOT / "results/ablations_summary.json"
    assert ablation_file.is_file(), "results/ablations_summary.json must exist"

    with open(ablation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "chunking_strategy_ablation" in data
    assert "top_k_parameter_sweep" in data
    assert "query_rewriter_component_ablation" in data

    # Verify top-k sweep keys
    topk = data["top_k_parameter_sweep"]
    assert "k_2" in topk and "k_4" in topk and "k_8" in topk

    # Verify chunking delta
    chunking = data["chunking_strategy_ablation"]
    assert "hit@1_delta" in chunking
    assert chunking["section_aware"]["mrr"] > chunking["fixed_size"]["mrr"]

    print("  [✓] Ablation studies verified (Chunking, Top-k sweep, Query Rewriting)")


def test_plots_generation():
    """Verify all 4 publication charts exist and have valid non-empty binary sizes."""
    plots_dir = PROJECT_ROOT / "results/plots"
    assert plots_dir.is_dir(), "results/plots/ directory must exist"

    expected_plots = [
        "faithfulness_vs_hallucination.png",
        "category_breakdown.png",
        "latency_vs_tokens.png",
        "retrieval_chunking_ablation.png",
    ]

    for plot_name in expected_plots:
        plot_path = plots_dir / plot_name
        assert plot_path.is_file(), f"Missing plot: {plot_name}"
        assert plot_path.stat().st_size >= 10000, f"Plot file suspiciously small: {plot_name} ({plot_path.stat().st_size} bytes)"

    print(f"  [✓] Publication-quality charts verified ({len(expected_plots)} plots in results/plots/)")


def test_failure_case_taxonomy():
    """Verify error analysis documentation contains >= 10 categorized failure cases."""
    json_path = PROJECT_ROOT / "results/error_analysis.json"
    doc_path = PROJECT_ROOT / "docs/error_analysis.md"

    assert json_path.is_file(), "results/error_analysis.json must exist"
    assert doc_path.is_file(), "docs/error_analysis.md must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        failures = json.load(f)

    assert len(failures) >= 10, f"Expected >= 10 failure cases, got {len(failures)}"

    required_fields = {
        "case_id",
        "error_category",
        "system",
        "question_id",
        "question",
        "ground_truth",
        "prediction",
        "root_cause",
        "remediation",
    }
    for f in failures:
        assert required_fields.issubset(f.keys()), f"Case {f.get('case_id')} missing required fields"

    doc_text = doc_path.read_text(encoding="utf-8")
    assert "Summary Matrix of Failure Modes" in doc_text
    assert "ERR_10" in doc_text

    print(f"  [✓] Error taxonomy verified ({len(failures)} in-depth failure cases documented)")


if __name__ == "__main__":
    print("Running Phase 7 tests...")
    test_ablation_summary_artifacts()
    test_plots_generation()
    test_failure_case_taxonomy()
    print("All Phase 7 tests passed successfully!")

