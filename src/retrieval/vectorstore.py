"""
FAISS Vector Store wrapper with Cosine Similarity indexing, metadata storage,
and disk persistence for student handbook chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from src.chunking.base import Chunk
from src.retrieval.embeddings import EmbeddingPipeline


@dataclass
class RetrievedChunk:
    """
    Representation of a retrieved chunk with similarity score and metadata.
    """
    chunk_id: str
    text: str
    score: float
    section_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(float(self.score), 4),
            "section_title": self.section_title,
            "metadata": self.metadata,
        }


class FAISSVectorStore:
    """
    In-memory and persisted FAISS vector index using normalized inner product (cosine similarity).
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        metric: str = "cosine",
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.metric = metric
        self.dimension = embedding_pipeline.dimension
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        """
        Extracts texts from chunks, computes dense embeddings, and initializes FAISS index.
        """
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunk list.")

        texts = [c.text for c in chunks]
        embeddings = self.embedding_pipeline.embed_documents(texts)
        self.dimension = embeddings.shape[1]

        # Inner Product on L2-normalized vectors represents Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self.chunks_metadata = [c.to_dict() for c in chunks]

    def search(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Searches index for top_k most similar chunks for the given query.
        """
        if self.index is None or len(self.chunks_metadata) == 0:
            return []

        q_emb = self.embedding_pipeline.embed_query(query)
        if q_emb.shape[1] != self.dimension:
            # Fallback re-dimension if needed
            self.dimension = q_emb.shape[1]

        top_k = min(top_k, len(self.chunks_metadata))
        scores, indices = self.index.search(q_emb, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            if score < similarity_threshold:
                continue

            meta = self.chunks_metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    score=float(score),
                    section_title=meta["section_title"],
                    metadata=meta.get("metadata", {}),
                )
            )

        return results

    def save(self, persist_dir: Path | str) -> None:
        """
        Saves the FAISS index and metadata to disk.
        """
        p_dir = Path(persist_dir)
        p_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(p_dir / "index.faiss"))

        with open(p_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": self.metric,
                    "dimension": self.dimension,
                    "chunks": self.chunks_metadata,
                },
                f,
                indent=2,
            )

    def load(self, persist_dir: Path | str) -> None:
        """
        Loads the FAISS index and metadata from disk.
        """
        p_dir = Path(persist_dir)
        index_file = p_dir / "index.faiss"
        meta_file = p_dir / "metadata.json"

        if not index_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(f"Missing vector store files in: {p_dir}")

        self.index = faiss.read_index(str(index_file))

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.metric = data.get("metric", "cosine")
            self.dimension = data.get("dimension", self.dimension)
            self.chunks_metadata = data.get("chunks", [])

