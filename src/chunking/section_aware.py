"""
Section-aware hierarchical markdown chunker.
Splits documents along structural boundaries (chapters, sections, subheadings),
preserves complete Markdown tables, and prepends hierarchical breadcrumbs
to retain semantic context for dense vector retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from src.chunking.base import BaseChunker, Chunk


class SectionAwareChunker(BaseChunker):
    """
    Parses Markdown headers (#, ##, ###) and splits content semantically,
    ensuring tables remain intact and context breadcrumbs are preserved.
    """

    def __init__(self, min_chunk_length: int = 50, max_chunk_length: int = 1500) -> None:
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

    def chunk(self, text: str, source_doc: str = "student_handbook.md") -> list[Chunk]:
        lines = text.splitlines()
        chunks: list[Chunk] = []
        
        current_doc_title = "Apex Institute Student Handbook"
        current_chapter = "General"
        current_section = "General"
        
        current_block_lines: List[str] = []
        chunk_idx = 0

        def flush_current_block() -> None:
            nonlocal chunk_idx, current_block_lines
            if not current_block_lines:
                return

            raw_body = "\n".join(current_block_lines).strip()
            has_tbl = any("|" in l and "---" in l for l in raw_body.splitlines())
            if len(raw_body) < self.min_chunk_length and not has_tbl:
                current_block_lines = []
                return

            # Check if block exceeds max_chunk_length and contains multiple paragraphs
            sub_blocks: List[str] = []
            if len(raw_body) > self.max_chunk_length and "\n\n" in raw_body and not has_tbl:
                paragraphs = raw_body.split("\n\n")
                current_sub: List[str] = []
                current_len = 0
                for p in paragraphs:
                    if current_len + len(p) > self.max_chunk_length and current_sub:
                        sub_blocks.append("\n\n".join(current_sub).strip())
                        current_sub = [p]
                        current_len = len(p)
                    else:
                        current_sub.append(p)
                        current_len += len(p)
                if current_sub:
                    sub_blocks.append("\n\n".join(current_sub).strip())
            else:
                sub_blocks = [raw_body]

            for s_idx, body in enumerate(sub_blocks):
                # Construct contextual breadcrumb for dense retrieval
                breadcrumb = f"[{current_chapter} > {current_section}]"
                contextual_text = f"{breadcrumb}\n{body}"
                
                chunk_id = f"sec_{chunk_idx:04d}"
                metadata: Dict[str, Any] = {
                    "source_doc": source_doc,
                    "doc_title": current_doc_title,
                    "chapter": current_chapter,
                    "section": current_section,
                    "has_table": any("|" in l and "---" in l for l in body.splitlines()),
                    "sub_index": s_idx,
                }
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=contextual_text,
                        section_title=f"{current_chapter} - {current_section}",
                        strategy="section_aware",
                        char_count=len(contextual_text),
                        metadata=metadata,
                    )
                )
                chunk_idx += 1

            current_block_lines = []

        in_table = False

        for line in lines:
            stripped = line.strip()

            # Detect Markdown Table boundaries
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                current_block_lines.append(line)
                continue
            elif in_table:
                # Exiting a table
                in_table = False

            # Header detection
            if stripped.startswith("# ") and not stripped.startswith("## "):
                flush_current_block()
                current_doc_title = stripped.lstrip("# ").strip()
            elif stripped.startswith("## "):
                flush_current_block()
                current_chapter = stripped.lstrip("# ").strip()
                current_section = current_chapter
            elif stripped.startswith("### "):
                flush_current_block()
                current_section = stripped.lstrip("# ").strip()
            else:
                current_block_lines.append(line)

        flush_current_block()
        return chunks
