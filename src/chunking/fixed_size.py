"""
Fixed-size text chunker implementation with configurable overlap.
Standard baseline chunking strategy commonly used in simple RAG pipelines.
"""

from __future__ import annotations

from typing import Any, Dict
from src.chunking.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """
    Splits text by character count with a sliding window overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def chunk(self, text: str, source_doc: str = "student_handbook.md") -> list[Chunk]:
        """
        Generates fixed-size text chunks from input text.
        """
        chunks: list[Chunk] = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks

        start = 0
        idx = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_slice = text[start:end].strip()
            
            if chunk_slice:
                chunk_id = f"fixed_{idx:04d}"
                metadata: Dict[str, Any] = {
                    "source_doc": source_doc,
                    "start_char": start,
                    "end_char": end,
                    "chunk_index": idx,
                }
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_slice,
                        section_title=f"Offset_{start}_{end}",
                        strategy="fixed_size",
                        char_count=len(chunk_slice),
                        metadata=metadata,
                    )
                )
                idx += 1

            if end >= text_length:
                break
            start += self.step_size

        return chunks

