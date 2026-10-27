"""
Unit and Integration Tests for Phase 5: System B (Fine-Tuning Pipeline & Inference Wrapper).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.system_b_finetuned import SystemBFinetuned


def test_sft_dataset_integrity():
    """Verify SFT formatted training and validation files exist and match ChatML structure."""
    train_file = PROJECT_ROOT / "data/processed/sft_train.jsonl"
    val_file = PROJECT_ROOT / "data/processed/sft_val.jsonl"

    assert train_file.is_file(), "sft_train.jsonl must exist"
    assert val_file.is_file(), "sft_val.jsonl must exist"

    train_count = 0
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assert "messages" in item or "text" in item
                assert "<|im_start|>" in item["text"]
                assert "<|im_end|>" in item["text"]
                train_count += 1

    assert train_count >= 500, f"Expected >= 500 train items, got {train_count}"
    print(f"  [✓] SFT dataset verified ({train_count} ChatML training examples)")


def test_colab_notebook_validity():
    """Verify Colab Jupyter Notebook is valid JSON and contains all required QLoRA training steps."""
    nb_path = PROJECT_ROOT / "notebooks/qlora_finetune_colab.ipynb"
    assert nb_path.is_file(), "qlora_finetune_colab.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    assert len(nb["cells"]) >= 8

    full_code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    required_keywords = [
        "BitsAndBytesConfig",
        "nf4",
        "LoraConfig",
        "r=16",
        "SFTTrainer",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "save_pretrained",
    ]
    for kw in required_keywords:
        assert kw in full_code, f"Colab notebook missing critical step: '{kw}'"

    print("  [✓] Colab QLoRA notebook verified (T4 GPU configuration intact)")


def test_system_b_inference_wrapper():
    """Verify System B inference returns standardized SystemResponse."""
    sys_b = SystemBFinetuned()
    q = "What is the minimum physical attendance percentage required?"
    res = sys_b.answer(q)

    assert res.system_name == "System B (Fine-Tuned Small LLM)"
    assert res.query == q
    assert len(res.response) > 0
    assert len(res.retrieved_chunks) == 0, "System B must NOT retrieve external chunks"
    assert "adapter_loaded" in res.metadata
    assert "75.0%" in res.response or "attendance" in res.response.lower()

    # Test out-of-scope hallucination behavior
    res_oos = sys_b.answer("What is the Ph.D. stipend?")
    assert len(res_oos.response) > 0

    print(f"  [✓] System B inference wrapper verified (Adapter status: {res.metadata['adapter_loaded']})")


if __name__ == "__main__":
    print("Running Phase 5 tests...")
    test_sft_dataset_integrity()
    test_colab_notebook_validity()
    test_system_b_inference_wrapper()
    print("All Phase 5 tests passed successfully!")

