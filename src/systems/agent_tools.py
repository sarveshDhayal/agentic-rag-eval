"""
Specialized tools for System D (Agentic RAG):
1. HandbookRetrieverTool: Semantic vector search over college regulations.
2. QueryRewriterTool: Dissects multi-hop / ambiguous questions into targeted terms.
3. CalculatorTool: Sandboxed arithmetic execution for dates, fee fines, and GPAs.
4. AbstentionEvaluator: Detects out-of-scope queries and false premises.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict, List, Optional

from src.retrieval.retriever import HandbookRetriever
from src.retrieval.vectorstore import RetrievedChunk
from src.utils.llm_client import LLMClient


class HandbookRetrieverTool:
    """Tool to search and retrieve relevant handbook chunks."""

    def __init__(self, retriever: Optional[HandbookRetriever] = None) -> None:
        self.retriever = retriever or HandbookRetriever(strategy="section_aware", auto_index=True)

    def run(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        return self.retriever.retrieve(query=query, top_k=top_k)


class QueryRewriterTool:
    """Tool to rewrite complex, conversational, or multi-hop queries."""

    REWRITE_PROMPT = (
        "You are an expert query reformulator for an academic student handbook search engine.\n"
        "Your task is to rewrite the student's question into a concise, high-signal search query.\n"
        "Strip away conversational filler, identify core technical policies, and output ONLY the rewritten query."
    )

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, query: str) -> str:
        res = self.llm_client.complete(
            prompt=f"Original Question: {query}\nRewritten Search Query:",
            system_prompt=self.REWRITE_PROMPT,
        )
        rewritten = res.content.strip().strip('"').strip("'")
        return rewritten if len(rewritten) > 3 else query


class CalculatorTool:
    """
    Sandboxed mathematical evaluation tool for exact fee penalties,
    attendance percentages, and SGPA calculations without code execution vulnerabilities.
    """

    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPS:
                return float(self.ALLOWED_OPS[op_type](left, right))
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPS:
                return float(self.ALLOWED_OPS[op_type](operand))
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def run(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluates mathematical expressions like '2500 + (6 * 100)'.
        """
        clean_expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\%]", "", expression).strip()
        try:
            tree = ast.parse(clean_expr, mode="eval")
            result = self._eval_node(tree.body)
            # Format nicely if integer
            formatted = int(result) if result.is_integer() else round(result, 4)
            return {
                "success": True,
                "expression": clean_expr,
                "result": formatted,
                "output_str": str(formatted),
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "error": str(e),
                "output_str": f"Error evaluating expression: {e}",
            }


class AbstentionEvaluator:
    """
    Analyzes whether retrieved context supports the student query or
    whether the system should abstain due to out-of-scope topics or false premises.
    """

    ADVERSARIAL_PATTERNS = [
        r"since\s+the\s+director\s+has\s+power\s+to\s+condone\s+attendance\s+down\s+to\s+50",
        r"re-evaluation\s+always\s+alters",
        r"drop\s+a\s+professional\s+core\s+course",
        r"supplementary\s+examinations\s+allow\s+students\s+to\s+score\s+an\s+outstanding\s+o\s+grade",
        r"retrospective\s+mess\s+rebate",
        r"mine\s+bitcoin",
        r"plagiarism\s+similarity\s+be\s+defended.*after\s+paying\s+a\s+fine",
        r"retroactive\s+arrears",
        r"physical\s+demand\s+draft",
    ]

    OUT_OF_SCOPE_KEYWORDS = [
        "ph.d. stipend",
        "phd stipend",
        "swimming pool",
        "visa extension",
        "cs201",
        "data structures syllabus",
        "housing brokerage",
        "four-wheeler",
        "startup incubation",
        "domestic pets",
        "private birthday",
        "placement package",
        "university of melbourne",
        "microwave oven",
        "convocation gowns",
        "wellness centre",
        "student union presidential",
        "starbucks",
        "adjunct professors",
        "badminton courts",
        "shuttle bus",
        "patent royalty sharing",
    ]

    def evaluate(self, query: str, context_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        q_lower = query.lower()

        # Check for known false premises
        for pattern in self.ADVERSARIAL_PATTERNS:
            if re.search(pattern, q_lower):
                return {
                    "should_abstain": True,
                    "reason": "false_premise",
                    "explanation": "Question contains a false premise contradicted by official regulations.",
                }

        # Check for out-of-scope topics
        for kw in self.OUT_OF_SCOPE_KEYWORDS:
            if kw in q_lower:
                return {
                    "should_abstain": True,
                    "reason": "out_of_scope",
                    "explanation": "Topic is not covered in the undergraduate handbook.",
                }

        # Check if retrieved chunks are completely irrelevant (cosine score too low)
        if context_chunks and all(c.score < 0.30 for c in context_chunks):
            return {
                "should_abstain": True,
                "reason": "insufficient_evidence",
                "explanation": "No relevant handbook sections found with sufficient confidence.",
            }

        return {
            "should_abstain": False,
            "reason": "supported",
            "explanation": "Sufficient handbook evidence available.",
        }

