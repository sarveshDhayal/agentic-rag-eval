"""
SQLite-backed disk cache for LLM API calls.
Prevents duplicate API calls, preserves rate limits, ensures determinism,
and provides latency and token count metrics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import PROJECT_ROOT, load_config


class LLMCache:
    """
    Persistent SQLite key-value cache for LLM completions.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            config = load_config()
            rel_path = config.get("paths", {}).get("llm_cache", "results/.llm_cache.sqlite")
            self.db_path = PROJECT_ROOT / rel_path
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completions (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    latency REAL NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(provider: str, model: str, temperature: float, prompt: Any) -> str:
        """
        Creates a deterministic SHA-256 hash key from LLM parameters and input.
        """
        if isinstance(prompt, (list, dict)):
            serialized = json.dumps(prompt, sort_keys=True)
        else:
            serialized = str(prompt)

        raw = f"{provider.lower()}|{model.lower()}|{temperature:.2f}|{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached completion record if present.
        """
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT response_text, latency, prompt_tokens, completion_tokens FROM completions WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if row is not None:
                self.hits += 1
                return {
                    "response_text": row["response_text"],
                    "latency": float(row["latency"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "completion_tokens": int(row["completion_tokens"]),
                    "cache_hit": True,
                }
            self.misses += 1
            return None

    def set(
        self,
        cache_key: str,
        provider: str,
        model: str,
        response_text: str,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """
        Stores completion in SQLite cache.
        """
        prompt_hash = cache_key[:16]
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO completions 
                (cache_key, provider, model, prompt_hash, response_text, latency, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    provider,
                    model,
                    prompt_hash,
                    response_text,
                    latency,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) as total FROM completions")
            total = cur.fetchone()["total"]
        return {
            "total_cached_entries": total,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses), 4) if (self.hits + self.misses) > 0 else 0.0,
        }

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM completions")
            conn.commit()
