"""
Unit and Integration Tests for Phase 8: Streamlit Interactive Demonstration Dashboard.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_app_syntax_and_structure():
    """Verify app.py parses cleanly without any syntax errors and contains core tabs."""
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.is_file(), "app.py must exist"

    code = app_file.read_text(encoding="utf-8")
    tree = ast.parse(code, filename="app.py")
    assert isinstance(tree, ast.Module), "app.py failed AST parsing"

    # Verify essential components
    required_features = [
        "st.set_page_config",
        "Single System Lab",
        "Side-by-Side Comparison",
        "Agentic ReAct Trace Inspector",
        "Evaluation Benchmark Dashboard",
        "Failure Case Taxonomy Explorer",
        "Student Handbook Browser",
        "load_systems",
    ]
    for feat in required_features:
        assert feat in code, f"app.py missing feature: {feat}"

    print("  [✓] Streamlit app.py syntax and modular tabs verified")


if __name__ == "__main__":
    print("Running Phase 8 tests...")
    test_app_syntax_and_structure()
    print("All Phase 8 tests passed successfully!")
