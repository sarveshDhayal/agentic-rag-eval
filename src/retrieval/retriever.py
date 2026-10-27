"""
High-level handbook retriever module.
Provides clean retrieval interface for Basic RAG (System C) and Agentic RAG (System D).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking import get_chunker
from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.vectorstore import FAISSVectorStore, RetrievedChunk
from src.utils.config_loader import PROJECT_ROOT, load_config


class HandbookRetriever:
    """
    Retriever engine connecting chunking strategies, embedding pipeline,
    and FAISS vector indexing.
    """

    def __init__(
        self,
        strategy: str = "section_aware",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        persist_dir: Path | str | None = None,
        auto_index: bool = True,
    ) -> None:
        self.strategy = strategy
        self.config = load_config()
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
        )
        self.vector_store = FAISSVectorStore(
            embedding_pipeline=self.embedding_pipeline,
            metric="cosine",
        )

        if persist_dir is None:
            self.persist_dir = PROJECT_ROOT / f"data/vectorstore/{strategy}"
        else:
            self.persist_dir = Path(persist_dir)

        if auto_index:
            self._initialize_index()

    def _initialize_index(self) -> None:
        """
        Loads persisted index if available; otherwise chunks handbook and builds new index.
        """
        index_file = self.persist_dir / "index.faiss"
        meta_file = self.persist_dir / "metadata.json"

        if index_file.is_file() and meta_file.is_file():
            try:
                self.vector_store.load(self.persist_dir)
                return
            except Exception as e:
                print(f"[HandbookRetriever] Failed to load cached index, rebuilding: {e}")

        # Build fresh index from raw handbook
        handbook_path = PROJECT_ROOT / self.config["paths"]["raw_handbook"]
        text = handbook_path.read_text(encoding="utf-8")

        chunker = get_chunker(self.strategy, self.config["chunking"])
        chunks = chunker.chunk(text)

        self.vector_store.build_from_chunks(chunks)
        self.vector_store.save(self.persist_dir)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[RetrievedChunk]:
        """
        Retrieves top-k most relevant chunks for the query.
        """
        return self.vector_store.search(query=query, top_k=top_k, similarity_threshold=threshold)

    def get_context_string(self, query: str, top_k: int = 3, threshold: float = 0.0) -> str:
        """
        Retrieves chunks and formats them into a clean prompt context block with citations.
        """
        chunks = self.retrieve(query=query, top_k=top_k, threshold=threshold)
        if not chunks:
            return "No relevant sections found in the student handbook."

        context_blocks: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"--- [Document Chunk {idx}] ---\n"
                f"Source Section: {chunk.section_title}\n"
                f"Similarity Score: {chunk.score:.4f}\n"
                f"Content:\n{chunk.text.strip()}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)
