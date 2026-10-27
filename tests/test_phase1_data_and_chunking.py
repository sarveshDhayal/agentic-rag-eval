"""
Unit and Integration Tests for Phase 1: Data Preparation, Chunking & Leakage Audit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import FixedSizeChunker, SectionAwareChunker, get_chunker


def test_student_handbook_integrity():
    """Verify raw handbook exists, meets page/word thresholds, and covers key topics."""
    handbook_path = PROJECT_ROOT / "data/raw/student_handbook.md"
    assert handbook_path.is_file(), "Raw handbook must exist at data/raw/student_handbook.md"

    text = handbook_path.read_text(encoding="utf-8")
    assert len(text) >= 20000, f"Handbook text too short ({len(text)} chars); expected >= 20000"

    key_topics = [
        "Apex Institute of Technology",
        "Section 3.1: Minimum Attendance Mandate",
        "Section 3.2: Medical and Co-Curricular Condonation Slabs",
        "Section 4.2: 10-Point Absolute Letter Grading System",
        "Section 5.1: Annual Tuition and Institutional Fee Schedule",
        "Section 6.3: Curfew Timings, Late Slips, and Disciplinary Measures",
        "Section 7.1: Institutional Merit-Cum-Means (MCM) Scholarship",
        "Section 8.1: Central Library Circulation Rules",
        "Section 9.1: Plagiarism Limits in Capstone Projects and Theses",
    ]
    for topic in key_topics:
        assert topic in text, f"Handbook missing critical required topic: {topic}"


def test_golden_eval_dataset():
    """Verify evaluation dataset has >= 100 questions, correct schema, and 5 categories."""
    eval_path = PROJECT_ROOT / "data/eval/golden_eval_set.json"
    assert eval_path.is_file(), "Evaluation dataset missing"

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 100, f"Expected min 100 eval questions, got {len(data)}"

    required_keys = {"id", "question", "ground_truth", "supporting_section", "category", "answerable"}
    category_counts = {}

    for item in data:
        assert required_keys.issubset(item.keys()), f"Item {item.get('id')} missing required fields"
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if cat in ("out_of_scope", "adversarial"):
            assert item["answerable"] is False, f"Item {item['id']} in {cat} must have answerable=False"
        else:
            assert item["answerable"] is True, f"Item {item['id']} in {cat} must have answerable=True"

    expected_categories = {"simple_factual", "multi_hop", "numeric_calculation", "out_of_scope", "adversarial"}
    assert set(category_counts.keys()) == expected_categories, f"Category mismatch: {category_counts.keys()}"
    print(f"  [✓] Verified {len(data)} evaluation questions across all 5 categories")


def test_train_dataset_and_leakage():
    """Verify training dataset has 500-1000 items and passes the leakage audit."""
    train_path = PROJECT_ROOT / "data/processed/train_qa_pairs.jsonl"
    assert train_path.is_file(), "Training dataset missing"

    train_count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assert "instruction" in item and "output" in item and "section" in item
                train_count += 1

    assert 500 <= train_count <= 1000, f"Expected 500-1000 train items, got {train_count}"
    print(f"  [✓] Verified {train_count} fine-tuning training pairs")

    # Leakage report check
    leakage_path = PROJECT_ROOT / "results/leakage_report.json"
    assert leakage_path.is_file(), "Leakage report missing"

    with open(leakage_path, "r", encoding="utf-8") as f:
        leakage = json.load(f)

    assert leakage["audit_status"] == "PASSED", "Leakage audit failed!"
    assert leakage["exact_matches_count"] == 0, "Exact matches detected between train and eval sets!"
    assert leakage["suspicious_fuzzy_matches_count"] == 0, "Suspicious fuzzy matches detected!"
    print("  [✓] Leakage audit confirmed: 0 exact matches, 0 suspicious fuzzy matches")


def test_chunking_strategies():
    """Verify both chunkers function properly and section-aware preserves tables."""
    handbook_path = PROJECT_ROOT / "data/raw/student_handbook.md"
    text = handbook_path.read_text(encoding="utf-8")

    fixed_chunker = get_chunker("fixed_size", {"fixed_size": {"chunk_size": 500, "chunk_overlap": 50}})
    fixed_chunks = fixed_chunker.chunk(text)
    assert len(fixed_chunks) > 20, "Fixed chunker produced too few chunks"
    for c in fixed_chunks:
        assert c.char_count <= 500

    section_chunker = get_chunker("section_aware", {"section_aware": {"min_chunk_length": 50, "max_chunk_length": 1500}})
    section_chunks = section_chunker.chunk(text)
    assert len(section_chunks) > 10, "Section-aware chunker produced too few chunks"

    # Check breadcrumb structure and table preservation
    table_chunks = [c for c in section_chunks if c.metadata.get("has_table")]
    assert len(table_chunks) >= 5, "Section-aware chunker failed to detect table sections"
    for c in section_chunks:
        assert c.text.startswith("["), "Section-aware chunks must begin with context breadcrumb"

    print(f"  [✓] Chunkers verified: Fixed ({len(fixed_chunks)} chunks), Section-Aware ({len(section_chunks)} chunks, {len(table_chunks)} tables intact)")


if __name__ == "__main__":
    print("Running Phase 1 validation tests...")
    test_student_handbook_integrity()
    test_golden_eval_dataset()
    test_train_dataset_and_leakage()
    test_chunking_strategies()
    print("All Phase 1 tests passed successfully!")

