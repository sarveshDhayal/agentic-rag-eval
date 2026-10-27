"""
Streamlit Web Demonstration for Agentic RAG with Evaluation Harness.
Features:
- Single System Interactive Lab
- Side-by-Side 4-System Comparison (A vs B vs C vs D)
- Agentic ReAct Thought Trace Visualizer
- Empirical Benchmark Dashboard with 95% Bootstrap CIs and Plots
- Failure Case Taxonomy Explorer
- Raw Student Handbook Inspector
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.system_a_prompting import SystemAPrompting
from src.systems.system_b_finetuned import SystemBFinetuned
from src.systems.system_c_rag import SystemCRAG
from src.systems.system_d_agentic import SystemDAgentic
from src.utils.config_loader import load_config

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Benchmark Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 8px;
    }
    .trace-step {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        border-left: 3px solid #10B981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_systems():
    """Cache system instances to avoid reloading models on every interaction."""
    sys_a = SystemAPrompting()
    sys_b = SystemBFinetuned()
    sys_c = SystemCRAG()
    sys_d = SystemDAgentic()
    return {
        "System A (Prompting Baseline)": sys_a,
        "System B (Fine-Tuned Small LLM)": sys_b,
        "System C (Basic RAG)": sys_c,
        "System D (Agentic RAG)": sys_d,
    }


@st.cache_data
def load_eval_data():
    eval_file = PROJECT_ROOT / "data/eval/golden_eval_set.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_summary_data():
    summary_file = PROJECT_ROOT / "results/eval_summary.json"
    if summary_file.is_file():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_error_cases():
    err_file = PROJECT_ROOT / "results/error_analysis.json"
    if err_file.is_file():
        with open(err_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# Initialize systems
systems = load_systems()
golden_eval_set = load_eval_data()
summary_data = load_summary_data()
error_cases = load_error_cases()

# Header
st.markdown('<div class="main-header">🎓 Agentic RAG with Evaluation Harness</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">B.Tech CS & AI Capstone | Comparing Prompting vs. Fine-Tuning vs. Basic RAG vs. Agentic RAG on Hallucination Rate & Faithfulness</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation & Controls")
mode = st.sidebar.radio(
    "Choose Demonstration Mode:",
    [
        "🔍 Single System Lab",
        "⚔️ Side-by-Side Comparison",
        "🧠 Agentic ReAct Trace Inspector",
        "📊 Evaluation Benchmark Dashboard",
        "⚠️ Failure Case Taxonomy Explorer",
        "📖 Student Handbook Browser",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture Overview:**")
st.sidebar.info(
    "- **System A**: Parametric Prompting (Zero-shot)\n"
    "- **System B**: QLoRA Small LLM (Qwen2.5-1.5B)\n"
    "- **System C**: Basic RAG (BGE-Small + FAISS)\n"
    "- **System D**: Agentic ReAct (LangGraph State Graph)"
)

# -----------------------------------------------------------------------------
# MODE 1: SINGLE SYSTEM INTERACTIVE LAB
# -----------------------------------------------------------------------------
if mode == "🔍 Single System Lab":
    st.subheader("Interactive Single System Query Lab")
    st.write("Test individual systems with custom questions or golden evaluation queries.")

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_sys_name = st.selectbox("Select System to Evaluate:", list(systems.keys()), index=3)
        sample_questions = [
            "-- Custom Input --",
            "What is the minimum physical attendance percentage required in each registered course?",
            "What is the penalty for failure to display an RFID badge upon spot inspection?",
            "A student pays their Autumn semester fee 21 days late. How much late fine is applied?",
            "If two students tie for the Chairman's Academic Excellence Award, how is the winner determined?",
            "What is the monthly stipend provided to full-time Ph.D. scholars at AIT?",
            "Since the Director has statutory power to condone attendance down to 50%, what form do I submit?",
        ]
        chosen_sample = st.selectbox("Or choose a golden benchmark query:", sample_questions)

    with col2:
        if chosen_sample != "-- Custom Input --":
            query_input = st.text_area("Student Inquiry:", value=chosen_sample, height=90)
        else:
            query_input = st.text_area("Student Inquiry:", value="What is the condonation fee for medical attendance shortage?", height=90)

        run_btn = st.button("Generate Answer", type="primary")

    if run_btn and query_input.strip():
        active_system = systems[selected_sys_name]
        with st.spinner(f"Running inference through {selected_sys_name}..."):
            t0 = time.time()
            res = active_system.answer(query_input.strip())
            lat = time.time() - t0

        st.markdown("### Response")
        st.success(res.response)

        # Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{res.latency_seconds:.3f} s")
        m2.metric("Total Tokens", f"{res.total_tokens}")
        m3.metric("Retrieved Chunks", f"{len(res.retrieved_chunks)}")
        m4.metric("Abstention Status", "Abstained" if res.metadata.get("is_abstaining") else "Answered")

        # Retrieved Chunks Explorer
        if res.retrieved_chunks:
            with st.expander(f"📚 View Retrieved Chunks ({len(res.retrieved_chunks)} Chunks)"):
                for idx, c in enumerate(res.retrieved_chunks, 1):
                    st.markdown(f"**Chunk {idx} | Section: `{c['section_title']}` (Similarity: `{c['score']:.4f}`)**")
                    st.info(c["text"])

        # Agentic Trace (If System D)
        if selected_sys_name == "System D (Agentic RAG)":
            with st.expander("🧠 Agentic ReAct Reasoning Trace", expanded=True):
                st.write(f"**Iterations Executed:** {res.metadata.get('iterations_used', 1)} | **Tools Invoked:** `{res.metadata.get('tools_used', [])}`")
                traces_dir = PROJECT_ROOT / "results/traces"
                if traces_dir.is_dir():
                    trace_files = sorted(traces_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
                    if trace_files:
                        with open(trace_files[0], "r", encoding="utf-8") as tf:
                            t_data = json.load(tf)
                        st.json(t_data.get("tool_history", []))

# -----------------------------------------------------------------------------
# MODE 2: SIDE-BY-SIDE COMPARISON
# -----------------------------------------------------------------------------
elif mode == "⚔️ Side-by-Side Comparison":
    st.subheader("Side-by-Side Comparative Evaluation (All 4 Systems)")
    st.write("Compare the generative behavior, grounding, and latency of all 4 paradigms on identical input.")

    preset_q = st.selectbox(
        "Select a representative challenge question:",
        [
            "A student pays their Autumn semester fee 21 days late. How much late fine is applied?",
            "What is the monthly stipend provided to full-time Ph.D. scholars at Apex Institute?",
            "Since the Director has statutory power to condone attendance down to 50%, what form do I submit?",
            "What is the fee for medical attendance condonation between 65% and 74.9%?",
        ],
    )
    compare_btn = st.button("Execute Side-by-Side Run", type="primary")

    if compare_btn:
        with st.spinner("Executing queries across Systems A, B, C, and D..."):
            res_a = systems["System A (Prompting Baseline)"].answer(preset_q)
            res_b = systems["System B (Fine-Tuned Small LLM)"].answer(preset_q)
            res_c = systems["System C (Basic RAG)"].answer(preset_q)
            res_d = systems["System D (Agentic RAG)"].answer(preset_q)

        colA, colB, colC, colD = st.columns(4)

        with colA:
            st.markdown("#### System A: Prompting")
            st.caption("Parametric Memory Only")
            st.write(res_a.response)
            st.markdown(f"⏱️ **Latency:** `{res_a.latency_seconds:.3f}s`")
            st.markdown(f"🪙 **Tokens:** `{res_a.total_tokens}`")
            st.markdown("🔍 **Chunks:** `0`")

        with colB:
            st.markdown("#### System B: Fine-Tuned")
            st.caption("QLoRA Adapter (Qwen2.5-1.5B)")
            st.write(res_b.response)
            st.markdown(f"⏱️ **Latency:** `{res_b.latency_seconds:.3f}s`")
            st.markdown(f"🪙 **Tokens:** `{res_b.total_tokens}`")
            st.markdown("🔍 **Chunks:** `0`")

        with colC:
            st.markdown("#### System C: Basic RAG")
            st.caption("Retrieve -> Ingest -> Generate")
            st.write(res_c.response)
            st.markdown(f"⏱️ **Latency:** `{res_c.latency_seconds:.3f}s`")
            st.markdown(f"🪙 **Tokens:** `{res_c.total_tokens}`")
            st.markdown(f"🔍 **Chunks:** `{len(res_c.retrieved_chunks)}`")

        with colD:
            st.markdown("#### System D: Agentic RAG")
            st.caption("LangGraph ReAct with Tools")
            st.write(res_d.response)
            st.markdown(f"⏱️ **Latency:** `{res_d.latency_seconds:.3f}s`")
            st.markdown(f"🪙 **Tokens:** `{res_d.total_tokens}`")
            st.markdown(f"🔍 **Chunks:** `{len(res_d.retrieved_chunks)}`")

# -----------------------------------------------------------------------------
# MODE 3: AGENTIC REACT TRACE INSPECTOR
# -----------------------------------------------------------------------------
elif mode == "🧠 Agentic ReAct Trace Inspector":
    st.subheader("Agentic ReAct Trace Visualizer (System D Observability)")
    st.write("Inspect how the LangGraph ReAct agent cycles through reasoning, tool invocation, and evidence synthesis.")

    traces_dir = PROJECT_ROOT / "results/traces"
    if traces_dir.is_dir():
        trace_files = sorted(traces_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if trace_files:
            file_options = {f.name: f for f in trace_files[:20]}
            selected_file = st.selectbox("Select Recent Execution Trace:", list(file_options.keys()))

            with open(file_options[selected_file], "r", encoding="utf-8") as f:
                trace_json = json.load(f)

            st.markdown(f"### Query: *\"{trace_json.get('query')}\"*")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Iterations Used", f"{trace_json.get('iterations_used')}")
            c2.metric("Abstention Triggered", str(trace_json.get("is_abstaining")))
            c3.metric("Latency", f"{trace_json.get('latency_seconds', 0):.3f}s")
            c4.metric("Tokens Consumed", f"{trace_json.get('total_tokens', 0)}")

            st.markdown("#### Step-by-Step ReAct Execution Sequence:")
            for item in trace_json.get("tool_history", []):
                step = item.get("step", 0)
                if "thought" in item:
                    st.markdown(f"🤔 **Step {step} [Reasoner Thought]:** {item['thought']}")
                elif "tool" in item:
                    st.markdown(f"⚡ **Step {step} [Action: `{item['tool']}`]:** Input: `{item.get('input_query') or item.get('input') or item.get('expression')}`")
                    if "result" in item:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **Observation:** `{item['result']}`")
                elif "event" in item:
                    st.markdown(f"🏁 **Step {step} [Event]:** {item['event']}")

            st.markdown("#### Final Synthesized Output:")
            st.success(trace_json.get("final_answer"))
        else:
            st.info("No trace files found in results/traces/. Run a query with System D to generate traces.")
    else:
        st.info("Traces directory results/traces/ does not exist yet.")

# -----------------------------------------------------------------------------
# MODE 4: EVALUATION BENCHMARK DASHBOARD
# -----------------------------------------------------------------------------
elif mode == "📊 Evaluation Benchmark Dashboard":
    st.subheader("Comprehensive Benchmark Dashboard (All 4 Systems)")
    st.write("Quantitative evaluation comparing all 4 systems across 105 benchmark questions with 95% Bootstrap Confidence Intervals.")

    if summary_data:
        sys_data = summary_data.get("systems", {})
        table_rows = []
        for s_name, stats in sys_data.items():
            f = stats["faithfulness"]
            h = stats["hallucination_rate"]
            c = stats["answer_correctness"]
            a = stats["abstention_accuracy"]
            lat = stats["latency_seconds"]
            tok = stats["total_tokens"]

            table_rows.append({
                "System": s_name,
                "Faithfulness (%)": f"{f['mean']*100:.1f}% [{f['ci_lower']*100:.1f}%, {f['ci_upper']*100:.1f}%]",
                "Hallucination (%)": f"{h['mean']*100:.1f}% [{h['ci_lower']*100:.1f}%, {h['ci_upper']*100:.1f}%]",
                "Correctness (%)": f"{c['mean']*100:.1f}% [{c['ci_lower']*100:.1f}%, {c['ci_upper']*100:.1f}%]",
                "Abstention (%)": f"{a['mean']*100:.1f}% [{a['ci_lower']*100:.1f}%, {a['ci_upper']*100:.1f}%]",
                "P95 Latency (s)": f"{lat['p95']:.3f}s",
                "Mean Tokens": f"{tok['mean']:.1f}",
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Publication-Quality Empirical Charts")
        p_col1, p_col2 = st.columns(2)
        p1 = PROJECT_ROOT / "results/plots/faithfulness_vs_hallucination.png"
        p2 = PROJECT_ROOT / "results/plots/category_breakdown.png"
        p3 = PROJECT_ROOT / "results/plots/latency_vs_tokens.png"
        p4 = PROJECT_ROOT / "results/plots/retrieval_chunking_ablation.png"

        with p_col1:
            if p1.is_file():
                st.image(str(p1), caption="Figure 1: Faithfulness vs Hallucination Rate across 4 Systems")
            if p3.is_file():
                st.image(str(p3), caption="Figure 3: Efficiency Frontier (Tokens vs Latency)")
        with p_col2:
            if p2.is_file():
                st.image(str(p2), caption="Figure 2: Per-Category Accuracy Breakdown")
            if p4.is_file():
                st.image(str(p4), caption="Figure 4: Chunking Strategy Ablation (Hit@K and MRR)")
    else:
        st.warning("No summary data found at results/eval_summary.json. Run `python3 src/evaluation/eval_runner.py --all` first.")

# -----------------------------------------------------------------------------
# MODE 5: FAILURE CASE TAXONOMY EXPLORER
# -----------------------------------------------------------------------------
elif mode == "⚠️ Failure Case Taxonomy Explorer":
    st.subheader("Failure Case Taxonomy & Qualitative Error Analysis")
    st.write("Browse 10 categorized empirical failure cases extracted from the benchmark with root causes and architectural remediations.")

    if error_cases:
        filter_sys = st.selectbox("Filter by System:", ["All"] + list({f["system"] for f in error_cases}))
        filtered = [f for f in error_cases if filter_sys == "All" or f["system"] == filter_sys]

        for fc in filtered:
            with st.expander(f"🔴 [{fc['case_id']}] {fc['error_category']} — {fc['system']}"):
                st.markdown(f"**Question ID**: `{fc['question_id']}`")
                st.markdown(f"**Student Inquiry**: *\"{fc['question']}\"*")
                st.markdown(f"**Expected Ground Truth**: *\"{fc['ground_truth']}\"*")
                st.markdown(f"**Observed System Prediction**: *\"{fc['prediction']}\"*")
                st.markdown(f"**Root Cause Analysis**: {fc['root_cause']}")
                st.markdown(f"**Architectural Remediation**: {fc['remediation']}")
    else:
        st.info("No failure cases found in results/error_analysis.json. Run scripts/extract_error_analysis.py first.")

# -----------------------------------------------------------------------------
# MODE 6: STUDENT HANDBOOK BROWSER
# -----------------------------------------------------------------------------
elif mode == "📖 Student Handbook Browser":
    st.subheader("Official Student Handbook Document Browser")
    st.write("Browse the fictional *Apex Institute of Technology* academic handbook governing this project.")

    handbook_file = PROJECT_ROOT / "data/raw/student_handbook.md"
    if handbook_file.is_file():
        hb_text = handbook_file.read_text(encoding="utf-8")
        st.markdown(hb_text)
    else:
        st.error("Handbook file not found at data/raw/student_handbook.md")
