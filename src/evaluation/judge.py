"""
Decoupled LLM-as-a-Judge Evaluation Module.
Uses an independent evaluation model (different from generator) to score:
- Faithfulness & Hallucination Rate
- Answer Correctness (vs Ground Truth)
- Abstention Correctness (Out-of-Scope & Adversarial queries)
- Context Precision & Recall
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.config_loader import load_config
from src.utils.llm_client import LLMClient


@dataclass
class JudgeEvaluation:
    faithfulness: float
    hallucination_rate: float
    answer_correctness: float
    abstention_accuracy: float
    context_precision: float
    context_recall: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "abstention_accuracy": round(self.abstention_accuracy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "reasoning": self.reasoning,
        }


class EvaluationJudge:
    """
    Independent Judge LLM evaluating factual consistency and grounding.
    """

    JUDGE_SYSTEM_PROMPT = (
        "You are an impartial, expert academic judge evaluating AI question-answering systems on a College Student Handbook.\n"
        "Your task is to evaluate the Generated Answer against the Ground Truth and Context.\n"
        "Score each metric strictly as 1 (Pass) or 0 (Fail):\n"
        "1. 'faithfulness': 1 if all factual claims in the answer are grounded and true to facts; 0 if the answer invents, hallucinates, or contradicts.\n"
        "2. 'correctness': 1 if the answer captures the core factual solution matching ground truth; 0 if incorrect or unhelpful.\n"
        "3. 'abstention_correct': 1 if the system appropriately answered (for answerable questions) or refused (for unanswerable questions); 0 otherwise.\n"
        "Respond ONLY in valid JSON format:\n"
        "{\"faithfulness\": 1, \"correctness\": 1, \"abstention_correct\": 1, \"reasoning\": \"short explanation\"}"
    )

    def __init__(self, judge_client: Optional[LLMClient] = None) -> None:
        config = load_config()
        judge_cfg = config.get("llm", {}).get("judge", {})

        provider = judge_cfg.get("provider", "gemini")
        model = judge_cfg.get("model", "gemini-1.5-flash")
        temp = judge_cfg.get("temperature", 0.0)

        self.client = judge_client or LLMClient(
            provider=provider,
            model=model,
            temperature=temp,
        )

    def evaluate_response(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
        category: str,
        answerable: bool,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> JudgeEvaluation:
        """
        Evaluates a single model completion against ground truth and retrieved context.
        """
        retrieved_chunks = retrieved_chunks or []
        context_text = "\n".join(c.get("text", "") for c in retrieved_chunks)

        # 1. Compute retrieval metrics deterministically if chunks are present
        ctx_precision, ctx_recall = self._compute_context_metrics(ground_truth, retrieved_chunks, answerable)

        # 2. Check abstention alignment deterministically for clear refusals
        is_refusal = any(
            phrase in prediction.lower()
            for phrase in [
                "insufficient information",
                "outside the scope",
                "not covered",
                "false premise",
                "not documented",
                "no information",
                "cannot be answered",
            ]
        )

        # Fast deterministic path for unanswerable questions
        if not answerable:
            if is_refusal:
                return JudgeEvaluation(
                    faithfulness=1.0,
                    hallucination_rate=0.0,
                    answer_correctness=1.0,
                    abstention_accuracy=1.0,
                    context_precision=ctx_precision,
                    context_recall=ctx_recall,
                    reasoning="System correctly abstained on unanswerable/adversarial question.",
                )
            else:
                return JudgeEvaluation(
                    faithfulness=0.0,
                    hallucination_rate=1.0,
                    answer_correctness=0.0,
                    abstention_accuracy=0.0,
                    context_precision=ctx_precision,
                    context_recall=ctx_recall,
                    reasoning="System failed to abstain and hallucinated on an out-of-scope/adversarial question.",
                )

        # 3. For answerable queries: invoke Judge LLM or fallback evaluator
        judge_prompt = (
            f"Question: {question}\n"
            f"Category: {category}\n"
            f"Answerable: {answerable}\n"
            f"Reference Ground Truth: {ground_truth}\n"
            f"Retrieved Context: {context_text[:1200] if context_text else 'None (Parametric Mode)'}\n"
            f"Generated Answer: {prediction}\n\n"
            f"Output JSON Evaluation:"
        )

        llm_res = self.client.complete(
            prompt=judge_prompt,
            system_prompt=self.JUDGE_SYSTEM_PROMPT,
        )

        parsed = self._parse_judge_json(llm_res.content)
        if parsed is not None:
            faith = float(parsed.get("faithfulness", 1))
            corr = float(parsed.get("correctness", 1))
            abst = float(parsed.get("abstention_correct", 1))
            reasoning = parsed.get("reasoning", "LLM Judge evaluation")
        else:
            # Deterministic heuristic scoring fallback
            faith, corr, abst, reasoning = self._heuristic_fallback(ground_truth, prediction)

        return JudgeEvaluation(
            faithfulness=faith,
            hallucination_rate=round(1.0 - faith, 4),
            answer_correctness=corr,
            abstention_accuracy=abst,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
            reasoning=reasoning,
        )

    def _compute_context_metrics(
        self,
        ground_truth: str,
        retrieved_chunks: List[Dict[str, Any]],
        answerable: bool,
    ) -> tuple[float, float]:
        """
        Computes Context Precision and Recall based on keyword/entity grounding.
        """
        if not retrieved_chunks:
            return 0.0, 0.0
        if not answerable:
            return 0.0, 1.0

        gt_tokens = set(re.findall(r"\b\w{4,}\b", ground_truth.lower()))
        if not gt_tokens:
            return 1.0, 1.0

        relevant_chunks_count = 0
        total_overlap = 0

        for chunk in retrieved_chunks:
            c_tokens = set(re.findall(r"\b\w{4,}\b", chunk.get("text", "").lower()))
            overlap = len(gt_tokens.intersection(c_tokens))
            if overlap >= 2:
                relevant_chunks_count += 1
            total_overlap += overlap

        precision = relevant_chunks_count / len(retrieved_chunks) if retrieved_chunks else 0.0
        recall = 1.0 if total_overlap >= 3 else (total_overlap / 3.0)

        return min(1.0, precision), min(1.0, recall)

    def _parse_judge_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from Judge response text."""
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None

    def _heuristic_fallback(self, ground_truth: str, prediction: str) -> tuple[float, float, float, str]:
        """Deterministic lexical fallback when judge API is offline."""
        gt_lower = ground_truth.lower()
        pred_lower = prediction.lower()

        # Check key numbers and terms
        gt_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", gt_lower))
        pred_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", pred_lower))

        num_match = len(gt_nums.intersection(pred_nums)) > 0 if gt_nums else True
        word_overlap = len(set(gt_lower.split()).intersection(set(pred_lower.split()))) / max(1, len(set(gt_lower.split())))

        is_correct = 1.0 if (num_match and word_overlap >= 0.25) else 0.0
        is_faithful = is_correct

        return is_faithful, is_correct, 1.0, "Heuristic match on keywords and numbers"
