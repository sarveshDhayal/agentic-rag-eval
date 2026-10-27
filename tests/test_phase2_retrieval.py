"""
Unit and Integration Tests for Phase 2: Embeddings, Vector Store, and Retrieval Evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.vectorstore import FAISSVectorStore
from src.retrieval.retriever import HandbookRetriever
from src.chunking.base import Chunk


def test_embedding_pipeline():
    """Verify BGE embedding pipeline produces normalized 384-dim vectors."""
    ep = EmbeddingPipeline(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    assert ep.dimension == 384, f"Expected 384 dimensions, got {ep.dimension}"

    sample_texts = [
        "What is the minimum attendance required at Apex Institute of Technology?",
        "Students must maintain at least 75% attendance.",
    ]
    embs = ep.embed_documents(sample_texts)
    assert embs.shape == (2, 384), f"Unexpected embedding shape: {embs.shape}"

    # Verify L2 normalization: norm should be approximately 1.0
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)

    q_emb = ep.embed_query("What is the hostel curfew?")
    assert q_emb.shape == (1, 384)
    print("  [✓] Embedding pipeline verified (384-dim, L2-normalized)")


def test_faiss_vectorstore_lifecycle(tmp_path: Path):
    """Test FAISS index creation, searching, and disk persistence."""
    ep = EmbeddingPipeline()
    vs = FAISSVectorStore(embedding_pipeline=ep, metric="cosine")

    chunks = [
        Chunk(
            chunk_id="test_001",
            text="Hostel curfew on weekdays is strictly 9:30 PM.",
            section_title="Hostel Rules",
            strategy="test",
        ),
        Chunk(
            chunk_id="test_002",
            text="Tuition fee must be paid before July 15th.",
            section_title="Fee Schedule",
            strategy="test",
        ),
    ]

    vs.build_from_chunks(chunks)
    results = vs.search("What time do I have to return to the hostel?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "test_001", "Expected curfew chunk to rank first"
    assert results[0].score > results[1].score

    # Test save and load
    save_dir = tmp_path / "test_vs"
    vs.save(save_dir)
    assert (save_dir / "index.faiss").is_file()
    assert (save_dir / "metadata.json").is_file()

    loaded_vs = FAISSVectorStore(embedding_pipeline=ep)
    loaded_vs.load(save_dir)
    loaded_results = loaded_vs.search("hostel curfew", top_k=1)
    assert loaded_results[0].chunk_id == "test_001"
    print("  [✓] FAISS Vector Store build, search, and persistence verified")


def test_handbook_retriever_e2e():
    """Verify high-level retriever retrieves context from actual handbook."""
    retriever = HandbookRetriever(strategy="section_aware", auto_index=True)
    chunks = retriever.retrieve("What is the medical condonation fee?", top_k=3)

    assert len(chunks) > 0
    assert any("1,500" in c.text or "condonation" in c.text.lower() for c in chunks)

    context_str = retriever.get_context_string("medical condonation fee", top_k=2)
    assert "Document Chunk 1" in context_str
    assert "Source Section:" in context_str
    print("  [✓] HandbookRetriever end-to-end integration verified")


def test_retrieval_metrics_results():
    """Verify retrieval evaluation results were computed and section-aware wins."""
    metrics_path = PROJECT_ROOT / "results/retrieval_metrics.json"
    assert metrics_path.is_file(), "results/retrieval_metrics.json missing"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sec = data["benchmark_summary"]["section_aware"]
    fix = data["benchmark_summary"]["fixed_size"]

    assert sec["hit@1"] >= 0.80, f"Expected Section-Aware Hit@1 >= 0.80, got {sec['hit@1']}"
    assert sec["hit@3"] >= 0.90, f"Expected Section-Aware Hit@3 >= 0.90, got {sec['hit@3']}"
    assert sec["mrr"] > fix["mrr"], "Section-Aware MRR should strictly exceed Fixed-Size MRR"
    print(f"  [✓] Retrieval benchmark verified: Section-Aware Hit@1={sec['hit@1']:.2%}, MRR={sec['mrr']:.4f}")


if __name__ == "__main__":
    print("Running Phase 2 tests...")
    test_embedding_pipeline()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_faiss_vectorstore_lifecycle(Path(td))
    test_handbook_retriever_e2e()
    test_retrieval_metrics_results()
    print("All Phase 2 tests passed successfully!")

