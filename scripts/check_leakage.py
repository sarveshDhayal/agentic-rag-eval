"""
Train/Eval Data Contamination & Leakage Audit Script.
Rigorously checks for exact and fuzzy overlap between the fine-tuning training set
and the golden evaluation benchmark. Generates results/leakage_report.json.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_text(text: str) -> str:
    """Lowercases text and strips non-alphanumeric characters."""
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9\s]", "", text)


def compute_token_jaccard(s1: str, s2: str) -> float:
    """Computes Jaccard similarity between word token sets."""
    tokens1: Set[str] = set(normalize_text(s1).split())
    tokens2: Set[str] = set(normalize_text(s2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def compute_sequence_similarity(s1: str, s2: str) -> float:
    """Computes string similarity ratio using difflib.SequenceMatcher."""
    norm1 = normalize_text(s1)
    norm2 = normalize_text(s2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def audit_leakage(
    eval_path: Path | None = None,
    train_path: Path | None = None,
    fuzzy_threshold: float = 0.82,
) -> Dict[str, Any]:
    """
    Audits eval and train datasets for exact and fuzzy contamination.
    """
    if eval_path is None:
        eval_path = PROJECT_ROOT / "data/eval/golden_eval_set.json"
    if train_path is None:
        train_path = PROJECT_ROOT / "data/processed/train_qa_pairs.jsonl"

    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation file missing: {eval_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file missing: {train_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    train_items: List[Dict[str, Any]] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_items.append(json.loads(line))

    exact_matches: List[Dict[str, Any]] = []
    suspicious_fuzzy_matches: List[Dict[str, Any]] = []
    max_similarity_overall = 0.0
    per_eval_max_sims: List[float] = []

    train_normalized = [
        (idx, item.get("id", f"train_{idx}"), item.get("instruction", ""), normalize_text(item.get("instruction", "")))
        for idx, item in enumerate(train_items)
    ]

    for eval_item in eval_items:
        e_id = eval_item["id"]
        e_q = eval_item["question"]
        e_norm = normalize_text(e_q)

        eval_max_sim = 0.0
        best_match_info = None

        for t_idx, t_id, t_inst, t_norm in train_normalized:
            # Check exact match
            if e_norm == t_norm:
                exact_matches.append({
                    "eval_id": e_id,
                    "eval_question": e_q,
                    "train_id": t_id,
                    "train_instruction": t_inst,
                    "type": "exact_match",
                })

            # Check fuzzy similarity
            seq_sim = SequenceMatcher(None, e_norm, t_norm).ratio()
            jaccard = compute_token_jaccard(e_q, t_inst)
            composite_sim = (0.6 * seq_sim) + (0.4 * jaccard)

            if composite_sim > eval_max_sim:
                eval_max_sim = composite_sim
                best_match_info = {
                    "eval_id": e_id,
                    "eval_question": e_q,
                    "train_id": t_id,
                    "train_instruction": t_inst,
                    "sequence_similarity": round(seq_sim, 4),
                    "jaccard_similarity": round(jaccard, 4),
                    "composite_similarity": round(composite_sim, 4),
                }

        per_eval_max_sims.append(eval_max_sim)
        if eval_max_sim > max_similarity_overall:
            max_similarity_overall = eval_max_sim

        if eval_max_sim >= fuzzy_threshold and best_match_info:
            suspicious_fuzzy_matches.append(best_match_info)

    avg_max_similarity = (
        sum(per_eval_max_sims) / len(per_eval_max_sims) if per_eval_max_sims else 0.0
    )

    passed = (len(exact_matches) == 0) and (len(suspicious_fuzzy_matches) == 0)

    report = {
        "audit_status": "PASSED" if passed else "FAILED",
        "eval_questions_count": len(eval_items),
        "train_pairs_count": len(train_items),
        "exact_matches_count": len(exact_matches),
        "suspicious_fuzzy_matches_count": len(suspicious_fuzzy_matches),
        "fuzzy_threshold_used": fuzzy_threshold,
        "max_similarity_overall": round(max_similarity_overall, 4),
        "average_max_similarity_per_eval_q": round(avg_max_similarity, 4),
        "exact_matches": exact_matches,
        "suspicious_fuzzy_matches": suspicious_fuzzy_matches,
    }

    report_path = PROJECT_ROOT / "results/leakage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    print("Running Train/Eval Data Leakage Audit...")
    report = audit_leakage()
    print("=" * 60)
    print(f"Audit Status:                     {report['audit_status']}")
    print(f"Eval Questions Audited:           {report['eval_questions_count']}")
    print(f"Train Pairs Audited:              {report['train_pairs_count']}")
    print(f"Exact Matches (Leakage):          {report['exact_matches_count']}")
    print(f"Suspicious Fuzzy Matches:         {report['suspicious_fuzzy_matches_count']}")
    print(f"Maximum Cross-Dataset Similarity: {report['max_similarity_overall']:.2%}")
    print(f"Average Max Similarity:           {report['average_max_similarity_per_eval_q']:.2%}")
    print("=" * 60)
    if report["audit_status"] == "PASSED":
        print("✓ Zero leakage detected. Evaluation set is statistically decoupled from training data.")
    else:
        print("⚠️ Warning: Potential contamination detected. Inspect results/leakage_report.json.")


if __name__ == "__main__":
    main()

