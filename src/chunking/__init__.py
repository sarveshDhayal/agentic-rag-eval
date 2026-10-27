"""
Chunking module initialization and factory function.
"""

from typing import Any, Dict
from src.chunking.base import BaseChunker, Chunk
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.section_aware import SectionAwareChunker


def get_chunker(strategy_name: str, config: Dict[str, Any] | None = None) -> BaseChunker:
    """
    Factory to instantiate chunker based on configuration strategy.
    """
    if config is None:
        config = {}

    strategy = strategy_name.lower().strip()
    if strategy == "fixed_size":
        fixed_cfg = config.get("fixed_size", {})
        return FixedSizeChunker(
            chunk_size=fixed_cfg.get("chunk_size", 500),
            chunk_overlap=fixed_cfg.get("chunk_overlap", 50),
        )
    elif strategy == "section_aware":
        sec_cfg = config.get("section_aware", {})
        return SectionAwareChunker(
            min_chunk_length=sec_cfg.get("min_chunk_length", 50),
            max_chunk_length=sec_cfg.get("max_chunk_length", 1500),
        )
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy_name}'. Choose 'fixed_size' or 'section_aware'.")


__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "SectionAwareChunker",
    "get_chunker",
]

