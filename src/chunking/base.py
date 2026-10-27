"""
Base chunking interface and data models for document ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    """
    Standard chunk data representation across all chunking strategies.
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_title: str = "General"
    strategy: str = "base"
    char_count: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_count:
            # Rule of thumb for English text: ~4 characters per token
            self.token_count = max(1, len(self.text.split()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "section_title": self.section_title,
            "strategy": self.strategy,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """
    Abstract Base Class for text chunkers.
    """

    @abstractmethod
    def chunk(self, text: str, source_doc: str = "student_handbook.md") -> list[Chunk]:
        """Split text into a sequence of Chunk objects."""
        pass
