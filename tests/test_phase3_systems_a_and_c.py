"""
Unit and Integration Tests for Phase 3: Systems A and C (Prompting Baseline & Basic RAG).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_cache import LLMCache
from src.utils.llm_client import LLMClient
from src.systems.system_a_prompting import SystemAPrompting
from src.systems.system_c_rag import SystemCRAG


def test_llm_cache_lifecycle(tmp_path: Path):
    """Test SQLite disk cache initialization, insertion, lookup, and stats."""
    db_path = tmp_path / "test_cache.sqlite"
    cache = LLMCache(db_path=db_path)

    key = cache.make_key(
        provider="groq",
        model="llama-3.1-8b-instant",
        temperature=0.1,
        prompt="Test prompt",
    )

    # Miss check
    assert cache.get(key) is None

    # Set and hit check
    cache.set(
        cache_key=key,
        provider="groq",
        model="llama-3.1-8b-instant",
        response_text="Test cached answer",
        latency=0.15,
        prompt_tokens=5,
        completion_tokens=4,
    )

    cached_item = cache.get(key)
    assert cached_item is not None
    assert cached_item["response_text"] == "Test cached answer"
    assert cached_item["cache_hit"] is True

    stats = cache.stats()
    assert stats["total_cached_entries"] == 1
    assert stats["session_hits"] == 1
    assert stats["session_misses"] == 1
    print("  [✓] LLM disk cache verified")


def test_system_a_prompting():
    """Verify System A generates prompt-only response without retrieved chunks."""
    sys_a = SystemAPrompting()
    q = "What is the minimum passing grade at AIT?"
    res = sys_a.answer(q)

    assert res.system_name == "System A (Prompting Baseline)"
    assert res.query == q
    assert len(res.response) > 0
    assert len(res.retrieved_chunks) == 0, "System A must NOT retrieve external chunks"
    assert res.total_tokens > 0
    print(f"  [✓] System A (Prompting Baseline) verified (Tokens: {res.total_tokens})")


def test_system_c_rag():
    """Verify System C retrieves document chunks and generates context-grounded response."""
    sys_c = SystemCRAG(top_k=3, strategy="section_aware")
    q = "What is the medical condonation fee?"
    res = sys_c.answer(q)

    assert res.system_name == "System C (Basic RAG)"
    assert res.query == q
    assert len(res.response) > 0
    assert len(res.retrieved_chunks) == 3, f"Expected 3 retrieved chunks, got {len(res.retrieved_chunks)}"
    assert any("Section" in c["section_title"] for c in res.retrieved_chunks)
    assert res.total_tokens > 0
    print(f"  [✓] System C (Basic RAG) verified (Retrieved {len(res.retrieved_chunks)} chunks)")


def test_smoke_results_file():
    """Verify smoke test outputs exist in results/."""
    smoke_path = PROJECT_ROOT / "results/smoke_systems_ac.json"
    assert smoke_path.is_file(), "results/smoke_systems_ac.json missing"

    with open(smoke_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 5
    for item in data:
        assert "system_a" in item
        assert "system_c" in item
        assert item["system_a"]["retrieved_chunks"] == []
        assert len(item["system_c"]["retrieved_chunks"]) > 0
    print(f"  [✓] Verified smoke results record for {len(data)} test questions")


if __name__ == "__main__":
    print("Running Phase 3 tests...")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_llm_cache_lifecycle(Path(td))
    test_system_a_prompting()
    test_system_c_rag()
    test_smoke_results_file()
    print("All Phase 3 tests passed successfully!")
