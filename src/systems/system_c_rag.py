"""
System C: Basic Retrieval-Augmented Generation (RAG).
Follows standard retrieve-then-read architecture:
1. Queries dense FAISS vector index (BAAI/bge-small-en-v1.5)
2. Injects top-k retrieved chunks into augmented prompt
3. Generates grounded response with explicit citations
"""

from __future__ import annotations

from typing import List, Optional

from src.retrieval.retriever import HandbookRetriever
from src.retrieval.vectorstore import RetrievedChunk
from src.systems.base import BaseSystem, SystemResponse
from src.utils.config_loader import load_config
from src.utils.llm_client import LLMClient


class SystemCRAG(BaseSystem):
    """
    Standard sequential RAG pipeline: Retrieve -> Augment Context -> Generate.
    """

    RAG_SYSTEM_PROMPT = (
        "You are an academic and student affairs advisor at Apex Institute of Technology (AIT).\n"
        "Answer the student's question based strictly on the provided student handbook excerpts.\n"
        "Guidelines:\n"
        "1. Ground your answer completely on the provided excerpts.\n"
        "2. Cite the supporting section name in your answer (e.g., [Section 3.2]).\n"
        "3. If the excerpts do not contain the answer, or if the question asks for details not documented in the text, state:\n"
        "   'Based on the provided student handbook excerpts, I do not have sufficient information to answer this question.'\n"
        "4. Never invent numbers, fine amounts, or exception policies."
    )

    def __init__(
        self,
        retriever: Optional[HandbookRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 3,
        strategy: str = "section_aware",
    ) -> None:
        super().__init__(system_name="System C (Basic RAG)")
        self.config = load_config()
        self.top_k = top_k or self.config.get("retrieval", {}).get("top_k", 3)
        self.strategy = strategy

        self.retriever = retriever or HandbookRetriever(
            strategy=self.strategy,
            auto_index=True,
        )

        gen_cfg = self.config.get("llm", {}).get("generator", {})
        self.client = llm_client or LLMClient(
            provider=gen_cfg.get("provider", "groq"),
            model=gen_cfg.get("model", "llama-3.1-8b-instant"),
            temperature=gen_cfg.get("temperature", 0.1),
        )

    def answer(self, query: str) -> SystemResponse:
        """
        Retrieves top-k chunks and generates grounded answer with citation metadata.
        """
        # Step 1: Dense Vector Retrieval
        retrieved_chunks: List[RetrievedChunk] = self.retriever.retrieve(query=query, top_k=self.top_k)

        # Step 2: Context Formatting
        if retrieved_chunks:
            context_blocks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_blocks.append(
                    f"--- Excerpt {i} [Section: {chunk.section_title}] ---\n{chunk.text.strip()}"
                )
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No relevant handbook excerpts retrieved."

        user_prompt = (
            f"Context Excerpts from Official Handbook:\n{context_str}\n\n"
            f"Student Inquiry: {query}\n\n"
            f"Advisor Response:"
        )

        # Step 3: LLM Generation
        llm_res = self.client.complete(
            prompt=user_prompt,
            system_prompt=self.RAG_SYSTEM_PROMPT,
        )

        serialized_chunks = [c.to_dict() for c in retrieved_chunks]

        return SystemResponse(
            system_name=self.system_name,
            query=query,
            response=llm_res.content.strip(),
            retrieved_chunks=serialized_chunks,
            latency_seconds=llm_res.latency,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            metadata={
                "provider": llm_res.provider,
                "model": llm_res.model,
                "cache_hit": llm_res.cache_hit,
                "top_k": self.top_k,
                "chunking_strategy": self.strategy,
            },
        )


if __name__ == "__main__":
    sys_c = SystemCRAG()
    q = "What is the fee for medical attendance condonation?"
    res = sys_c.answer(q)
    print("Query:", res.query)
    print("Response:\n", res.response)
    print(f"\nRetrieved Chunks: {len(res.retrieved_chunks)}")
    for c in res.retrieved_chunks:
        print(f"  - [{c['score']:.4f}] {c['section_title']}")
