"""
Retrieval package initialization.
"""

from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.vectorstore import FAISSVectorStore, RetrievedChunk
from src.retrieval.retriever import HandbookRetriever
from src.retrieval.retrieval_eval import run_retrieval_benchmark

__all__ = [
    "EmbeddingPipeline",
    "FAISSVectorStore",
    "RetrievedChunk",
    "HandbookRetriever",
    "run_retrieval_benchmark",
]

