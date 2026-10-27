"""
Phase 9 Unit Tests: Final Documentation, Capstone Report, Presentation Slides, Viva Q&A, and README.
Verifies completeness, presence of required sections/slides/Q&As, and strict adherence to empirical results.
"""

import os
import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_readme_completeness() -> None:
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md does not exist!"
    content = readme_path.read_text(encoding="utf-8")
    assert len(content) > 2000, "README.md is too short!"

    # Check key sections
    assert "System A" in content, "System A missing in README"
    assert "System B" in content, "System B missing in README"
    assert "System C" in content, "System C missing in README"
    assert "System D" in content, "System D missing in README"
    assert "Mermaid" in content or "flowchart" in content, "Mermaid diagram missing in README"
    assert "Streamlit" in content, "Streamlit section missing in README"

    # Verify exact empirical benchmark numbers from results/eval_summary.json
    eval_summary_path = PROJECT_ROOT / "results" / "eval_summary.json"
    assert eval_summary_path.exists()
    with open(eval_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    sys_a_faith = summary["systems"]["System A (Prompting)"]["faithfulness"]["mean"]
    sys_d_faith = summary["systems"]["System D (Agentic RAG)"]["faithfulness"]["mean"]
    sys_c_faith = summary["systems"]["System C (Basic RAG)"]["faithfulness"]["mean"]

    # Verify formatted percentages appear in README (handles LaTeX or plain format)
    val_a = f"{sys_a_faith * 100:.2f}"
    val_d = f"{sys_d_faith * 100:.2f}"
    val_c = f"{sys_c_faith * 100:.2f}"

    assert val_a in content, f"{val_a} missing in README"
    assert val_d in content, f"{val_d} missing in README"
    assert val_c in content, f"{val_c} missing in README"
    print("  [✓] README.md verified with exact empirical metrics")


def test_final_report_sections() -> None:
    report_path = PROJECT_ROOT / "docs" / "final_report.md"
    assert report_path.exists(), "docs/final_report.md does not exist!"
    content = report_path.read_text(encoding="utf-8")
    assert len(content) > 5000, "docs/final_report.md is too short!"

    required_sections = [
        "Abstract",
        "Introduction",
        "Related Work",
        "Dataset Construction",
        "System Architectures",
        "Evaluation Methodology",
        "Empirical Results",
        "Error Analysis",
        "Limitations",
        "Future Work",
        "References"
    ]
    for sec in required_sections:
        assert sec.lower() in content.lower(), f"Required section '{sec}' missing in docs/final_report.md"

    # Verify references list has formal papers
    assert "Brown, T." in content or "Lewis, P." in content, "Formal academic references missing"
    assert "QLoRA" in content or "Dettmers" in content, "QLoRA reference missing"
    assert "Yao, S." in content or "ReAct" in content, "ReAct reference missing"
    print("  [✓] docs/final_report.md verified across all 10 academic sections")


def test_presentation_slides_completeness() -> None:
    slides_path = PROJECT_ROOT / "docs" / "presentation_slides.md"
    assert slides_path.exists(), "docs/presentation_slides.md does not exist!"
    content = slides_path.read_text(encoding="utf-8")

    # Verify all 12 slides exist
    for slide_num in range(1, 13):
        assert f"Slide {slide_num}:" in content, f"Slide {slide_num} missing in presentation_slides.md"

    # Verify speaker script and visual suggestions are present
    assert "Speaker Script" in content, "Speaker scripts missing from presentation slides"
    assert "Visual Layout" in content or "Visual" in content, "Visual layout suggestions missing"
    print("  [✓] docs/presentation_slides.md verified with all 12 slides and speaker scripts")


def test_viva_qa_completeness() -> None:
    viva_path = PROJECT_ROOT / "docs" / "viva_qa.md"
    assert viva_path.exists(), "docs/viva_qa.md does not exist!"
    content = viva_path.read_text(encoding="utf-8")

    # Verify all 20 questions exist
    for q_num in range(1, 21):
        assert f"Q{q_num}:" in content, f"Question Q{q_num} missing in viva_qa.md"

    # Verify structural components
    assert "Technical Answer" in content, "Technical answers missing in viva_qa.md"
    assert "In My Project" in content, "Project connections missing in viva_qa.md"
    assert "Examiner Follow-Up Trap" in content, "Follow-up examiner traps missing in viva_qa.md"
    print("  [✓] docs/viva_qa.md verified with all 20 technical questions and examiner traps")


if __name__ == "__main__":
    print("Running Phase 9 documentation tests...")
    test_readme_completeness()
    test_final_report_sections()
    test_presentation_slides_completeness()
    test_viva_qa_completeness()
    print("All Phase 9 tests passed successfully!")
