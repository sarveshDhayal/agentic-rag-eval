"""
Base architecture and data classes for experimental systems A, B, C, and D.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SystemResponse:
    """
    Standardized response container across all 4 experimental systems.
    """
    system_name: str
    query: str
    response: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "query": self.query,
            "response": self.response,
            "retrieved_chunks": self.retrieved_chunks,
            "latency_seconds": round(self.latency_seconds, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


class BaseSystem(ABC):
    """
    Abstract Base Class for all benchmarked QA systems.
    """

    def __init__(self, system_name: str) -> None:
        self.system_name = system_name

    @abstractmethod
    def answer(self, query: str) -> SystemResponse:
        """
        Processes a student query and returns a standardized SystemResponse.
        """
        pass
