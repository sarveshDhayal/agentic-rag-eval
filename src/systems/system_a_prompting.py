"""
System A: Prompting-Only Baseline.
Answers student questions directly via standard prompting without retrieval context.
Evaluates baseline parametric memory and hallucination frequency.
"""

from __future__ import annotations

from typing import Optional

from src.systems.base import BaseSystem, SystemResponse
from src.utils.config_loader import load_config
from src.utils.llm_client import LLMClient


class SystemAPrompting(BaseSystem):
    """
    Zero-shot / Few-shot domain prompted baseline without retrieval.
    """

    SYSTEM_PROMPT = (
        "You are an academic and student affairs advisor at Apex Institute of Technology (AIT).\n"
        "Answer the student's inquiry regarding academic regulations, attendance, fees, grading, hostels, and scholarships.\n"
        "Provide factual, direct answers.\n"
        "If you do not know the answer or the question refers to something outside institute policy, state:\n"
        "'I do not have sufficient information to answer this question.'"
    )

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(system_name="System A (Prompting Baseline)")
        config = load_config()
        gen_cfg = config.get("llm", {}).get("generator", {})

        p = provider or gen_cfg.get("provider", "groq")
        m = model or gen_cfg.get("model", "llama-3.1-8b-instant")
        temp = gen_cfg.get("temperature", 0.1)

        self.client = llm_client or LLMClient(
            provider=p,
            model=m,
            temperature=temp,
        )

    def answer(self, query: str) -> SystemResponse:
        """
        Executes zero-shot prompted completion without document context.
        """
        llm_res = self.client.complete(
            prompt=query,
            system_prompt=self.SYSTEM_PROMPT,
        )

        return SystemResponse(
            system_name=self.system_name,
            query=query,
            response=llm_res.content.strip(),
            retrieved_chunks=[],
            latency_seconds=llm_res.latency,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            metadata={
                "provider": llm_res.provider,
                "model": llm_res.model,
                "cache_hit": llm_res.cache_hit,
            },
        )


if __name__ == "__main__":
    sys_a = SystemAPrompting()
    q = "What is the minimum attendance required at AIT?"
    res = sys_a.answer(q)
    print("Query:", res.query)
    print("Response:", res.response)
    print(f"Latency: {res.latency_seconds:.3f}s | Tokens: {res.total_tokens} | Cache hit: {res.metadata.get('cache_hit')}")

