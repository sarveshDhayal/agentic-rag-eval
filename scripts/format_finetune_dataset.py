"""
Dataset formatter for Supervised Fine-Tuning (SFT).
Converts data/processed/train_qa_pairs.jsonl into ChatML and Alpaca SFT format
with a deterministic 90/10 train/validation split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SYSTEM_PROMPT = (
    "You are the official academic and student affairs advisor at Apex Institute of Technology (AIT). "
    "Answer the student's question accurately and concisely based strictly on official institute regulations."
)


def format_chatml(instruction: str, context: str, output: str) -> str:
    """Formats QA pair into standard Qwen / ChatML format."""
    user_content = f"{instruction}\n\nContext: {context}" if context else instruction
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def prepare_sft_splits(val_ratio: float = 0.10, seed: int = 42) -> Dict[str, Any]:
    raw_train_file = PROJECT_ROOT / "data/processed/train_qa_pairs.jsonl"
    if not raw_train_file.is_file():
        raise FileNotFoundError(f"Missing train data at: {raw_train_file}")

    items: List[Dict[str, Any]] = []
    with open(raw_train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(items)

    val_size = max(1, int(len(items) * val_ratio))
    val_items = items[:val_size]
    train_items = items[val_size:]

    processed_train: List[Dict[str, Any]] = []
    processed_val: List[Dict[str, Any]] = []

    for item in train_items:
        inst = item["instruction"]
        inp = item.get("input", "")
        out = item["output"]
        chatml_text = format_chatml(inst, inp, out)
        processed_train.append({
            "id": item["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{inst}\n\n{inp}".strip()},
                {"role": "assistant", "content": out},
            ],
            "text": chatml_text,
            "instruction": inst,
            "input": inp,
            "output": out,
            "section": item.get("section", ""),
        })

    for item in val_items:
        inst = item["instruction"]
        inp = item.get("input", "")
        out = item["output"]
        chatml_text = format_chatml(inst, inp, out)
        processed_val.append({
            "id": item["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{inst}\n\n{inp}".strip()},
                {"role": "assistant", "content": out},
            ],
            "text": chatml_text,
            "instruction": inst,
            "input": inp,
            "output": out,
            "section": item.get("section", ""),
        })

    train_out = PROJECT_ROOT / "data/processed/sft_train.jsonl"
    val_out = PROJECT_ROOT / "data/processed/sft_val.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for item in processed_train:
            f.write(json.dumps(item) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for item in processed_val:
            f.write(json.dumps(item) + "\n")

    stats = {
        "total_examples": len(items),
        "train_examples": len(processed_train),
        "val_examples": len(processed_val),
        "train_path": str(train_out),
        "val_path": str(val_out),
    }

    return stats


def main() -> None:
    print("Formatting SFT datasets for QLoRA training...")
    stats = prepare_sft_splits()
    print("=" * 60)
    print(f"Total Source Examples:      {stats['total_examples']}")
    print(f"SFT Training Set Size:      {stats['train_examples']} (Saved to {stats['train_path']})")
    print(f"SFT Validation Set Size:    {stats['val_examples']} (Saved to {stats['val_path']})")
    print("=" * 60)
    print("Ready for QLoRA training on Google Colab T4 GPU.")


if __name__ == "__main__":
    main()
