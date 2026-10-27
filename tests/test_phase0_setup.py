"""
Smoke tests for Phase 0: Project Foundation and Environment Setup.
Verifies directory structure, configuration integrity, and documentation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.utils.config_loader import load_config, PROJECT_ROOT


def test_directory_structure():
    """Verify that all foundational directories are properly created."""
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/eval",
        "data/vectorstore",
        "docs",
        "notebooks",
        "results",
        "results/plots",
        "results/traces",
        "src/chunking",
        "src/retrieval",
        "src/systems",
        "src/evaluation",
        "src/utils",
        "tests",
    ]
    for rel_dir in expected_dirs:
        dir_path = PROJECT_ROOT / rel_dir
        assert dir_path.is_dir(), f"Expected directory missing: {rel_dir}"


def test_configuration_loading():
    """Verify config.yaml can be loaded and has all essential sections."""
    cfg = load_config()
    required_sections = [
        "project",
        "paths",
        "embedding",
        "chunking",
        "vectorstore",
        "retrieval",
        "llm",
        "agent",
        "evaluation",
    ]
    for section in required_sections:
        assert section in cfg, f"Missing required configuration section: {section}"

    assert cfg["project"]["seed"] == 42
    assert cfg["embedding"]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert cfg["llm"]["generator"]["provider"] != ""
    assert cfg["llm"]["judge"]["provider"] != ""
    # Generator and Judge must be configured differently
    assert (
        cfg["llm"]["generator"]["model"] != cfg["llm"]["judge"]["model"]
        or cfg["llm"]["generator"]["provider"] != cfg["llm"]["judge"]["provider"]
    ), "Generator and Judge LLM must not be the identical model/provider"


def test_git_and_env_templates():
    """Verify .gitignore and .env.example are in place to protect credentials."""
    env_example = PROJECT_ROOT / ".env.example"
    gitignore = PROJECT_ROOT / ".gitignore"

    assert env_example.is_file(), ".env.example must exist"
    assert gitignore.is_file(), ".gitignore must exist"

    gitignore_content = gitignore.read_text(encoding="utf-8")
    assert ".env" in gitignore_content, ".env must be ignored in .gitignore"


def test_documentation_files():
    """Verify syllabus mapping and architecture documentation exist."""
    syllabus_doc = PROJECT_ROOT / "docs/syllabus_mapping.md"
    arch_doc = PROJECT_ROOT / "docs/architecture.md"

    assert syllabus_doc.is_file(), "docs/syllabus_mapping.md must exist"
    assert arch_doc.is_file(), "docs/architecture.md must exist"

    syllabus_text = syllabus_doc.read_text(encoding="utf-8")
    assert "Syllabus Topic" in syllabus_text
    assert "System A" in syllabus_text
    assert "System D" in syllabus_text

    arch_text = arch_doc.read_text(encoding="utf-8")
    assert "mermaid" in arch_text
    assert "AgentState" in arch_text


if __name__ == "__main__":
    print("Running Phase 0 unit tests...")
    test_directory_structure()
    print("  [✓] Directory structure verified")
    test_configuration_loading()
    print("  [✓] Configuration schema verified")
    test_git_and_env_templates()
    print("  [✓] Environment template & security rules verified")
    test_documentation_files()
    print("  [✓] Documentation and syllabus mapping verified")
    print("All Phase 0 tests passed successfully!")
