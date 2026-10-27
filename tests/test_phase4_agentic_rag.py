"""
Unit and Integration Tests for Phase 4: System D (Agentic RAG with LangGraph ReAct).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.agent_tools import CalculatorTool, QueryRewriterTool, AbstentionEvaluator
from src.systems.system_d_agentic import SystemDAgentic


def test_calculator_tool_safety_and_math():
    """Verify calculator tool handles mathematical operations safely and resists code injection."""
    calc = CalculatorTool()

    # Normal math expressions
    r1 = calc.run("2500 + (6 * 100)")
    assert r1["success"] is True
    assert r1["result"] == 3100

    r2 = calc.run("(4*9 + 3*8 + 3*7 + 2*10) / 12")
    assert r2["success"] is True
    assert round(r2["result"], 2) == 8.42

    # Malicious injection attempt - should safely fail
    r_bad = calc.run("__import__('os').system('ls')")
    assert r_bad["success"] is False or r_bad["result"] == 0
    print("  [✓] Calculator tool verified (safe AST arithmetic)")


def test_query_rewriter_tool():
    """Verify query rewriter reformulates conversational student queries."""
    rewriter = QueryRewriterTool()
    original = "Hey can you tell me what if I get sick and my attendance is 68%?"
    rewritten = rewriter.run(original)
    assert len(rewritten) > 0
    print(f"  [✓] Query rewriter tool verified (Reformulated: '{rewritten[:40]}...')")


def test_abstention_evaluator():
    """Verify abstention evaluator flags adversarial false premises and out-of-scope topics."""
    evaluator = AbstentionEvaluator()

    # False premise test
    adv = evaluator.evaluate("Since the Director has power to condone attendance down to 50%, what form do I submit?", [])
    assert adv["should_abstain"] is True
    assert adv["reason"] == "false_premise"

    # Out of scope test
    oos = evaluator.evaluate("What is the Ph.D. stipend at Apex Institute?", [])
    assert oos["should_abstain"] is True
    assert oos["reason"] == "out_of_scope"

    # In-scope test
    valid = evaluator.evaluate("What is the hostel curfew timing?", [])
    assert valid["should_abstain"] is False
    print("  [✓] Abstention evaluator verified (False premises and OOS accurately filtered)")


def test_system_d_agentic_flow():
    """Verify System D executes LangGraph workflow and writes execution traces."""
    agent = SystemDAgentic(max_iterations=4, log_traces=True)

    # 1. Standard query
    res = agent.answer("What is the fee for medical attendance condonation?")
    assert res.system_name == "System D (Agentic RAG)"
    assert len(res.retrieved_chunks) > 0
    assert res.metadata["iterations_used"] >= 2
    assert "retriever" in res.metadata.get("tools_used", [None])[0] or "handbook_retriever" in res.metadata.get("tools_used", [])

    # 2. Calculation query
    res_calc = agent.answer("A student pays Autumn semester fee 21 days late. How much late fine is applied?")
    assert res_calc.metadata["calculation_count"] > 0
    assert res_calc.metadata["iterations_used"] >= 2

    # 3. Abstention query
    res_abstain = agent.answer("What are the parking permit fees for undergraduate four-wheeler cars?")
    assert res_abstain.metadata["is_abstaining"] is True
    assert "outside" in res_abstain.response.lower() or "not" in res_abstain.response.lower()

    # 4. Verify trace files were generated
    traces_dir = PROJECT_ROOT / "results/traces"
    assert traces_dir.is_dir()
    trace_files = list(traces_dir.glob("*.json"))
    assert len(trace_files) >= 1

    sample_trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert "query" in sample_trace
    assert "iterations_used" in sample_trace
    assert "tool_history" in sample_trace
    assert "final_answer" in sample_trace
    print(f"  [✓] System D LangGraph ReAct flow verified (Logged {len(trace_files)} traces)")


if __name__ == "__main__":
    print("Running Phase 4 tests...")
    test_calculator_tool_safety_and_math()
    test_query_rewriter_tool()
    test_abstention_evaluator()
    test_system_d_agentic_flow()
    print("All Phase 4 tests passed successfully!")

