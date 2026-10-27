"""
Dense semantic embedding pipeline using BAAI/bge-small-en-v1.5.
Maps text chunks and user queries into dense 384-dimensional vector space.
"""

from __future__ import annotations

import os
from typing import List, Optional
import numpy as np

# Optional import with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class EmbeddingPipeline:
    """
    Manages embedding generation for documents and queries.
    Standardized on BAAI/bge-small-en-v1.5 (384 dimensions).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.dimension = 384
        self._model = None

        if HAS_SENTENCE_TRANSFORMERS:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            try:
                # Try loading with local_files_only first
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
                self.dimension = (
                    self._model.get_embedding_dimension()
                    if hasattr(self._model, "get_embedding_dimension")
                    else self._model.get_sentence_embedding_dimension()
                )
            except Exception:
                try:
                    # Fallback to standard load
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self.dimension = (
                        self._model.get_embedding_dimension()
                        if hasattr(self._model, "get_embedding_dimension")
                        else self._model.get_sentence_embedding_dimension()
                    )
                except Exception as e:
                    print(f"[EmbeddingPipeline] Could not load '{model_name}': {e}. Using deterministic fallback.")
                    self._model = None

    @property
    def is_transformer_loaded(self) -> bool:
        return self._model is not None

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of document chunk texts.
        Returns float32 numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        else:
            return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single search query.
        For BGE models, query encoding can include an optional search prefix.
        """
        # BGE recommended instruction prefix for short queries
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        
        if self._model is not None:
            # Try with query prefix; if failed, use raw query
            try:
                emb = self._model.encode(
                    bge_query,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
            except Exception:
                emb = self._model.encode(
                    query,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
            return np.ascontiguousarray(emb, dtype=np.float32).reshape(1, -1)
        else:
            return self._fallback_embed([query])

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic TF-IDF/Hashing fallback embedding for offline environments.
        Ensures unit testing and pipeline execution succeed even if Hugging Face is unreachable.
        """
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(n_features=self.dimension, alternate_sign=False, norm="l2")
        sparse_mat = vectorizer.transform(texts)
        dense_mat = sparse_mat.toarray().astype(np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense_mat / norms
