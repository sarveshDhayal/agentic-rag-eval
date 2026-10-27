# System Architecture & Technical Specifications

This document outlines the architectural blueprints of the **Agentic RAG with Evaluation Harness** framework. It details the structural differences between Prompting, Fine-Tuning, Basic RAG, and Agentic RAG, along with the LangGraph state machine and evaluation harness workflows.

---

## 1. High-Level System Architecture

The following diagram illustrates how student queries flow into each of the four experimental paradigms, followed by standardized evaluation via the independent Judge LLM.

```mermaid
flowchart TD
    subgraph DataLayer ["Data & Ingestion Pipeline"]
        RawHandbook["Raw Student Handbook (Markdown + Tables)"]
        Chunker1["Fixed-Size Chunker (500 chars, 50 overlap)"]
        Chunker2["Section-Aware Chunker (Preserves Tables & Headers)"]
        RawHandbook --> Chunker1
        RawHandbook --> Chunker2
        
        EmbedModel["Embedding Model: BAAI/bge-small-en-v1.5 (384-dim)"]
        VectorDB[("FAISS Vector Index (Cosine Metric)")]
        Chunker1 -.-> EmbedModel
        Chunker2 --> EmbedModel --> VectorDB
    end

    subgraph Systems ["Comparative Paradigms"]
        Q["Student Query"]
        
        subgraph SysA ["System A: Prompting Baseline"]
            A_Prompt["System Prompt + Zero-Shot Query"]
            A_LLM["Generator LLM (Parametric Only)"]
            A_Prompt --> A_LLM
        end
        
        subgraph SysB ["System B: QLoRA Fine-Tuning"]
            B_Prompt["Instruction Prompt"]
            B_LLM["Qwen2.5-1.5B + QLoRA Adapter (NF4, r=16)"]
            B_Prompt --> B_LLM
        end
        
        subgraph SysC ["System C: Basic RAG"]
            C_Ret["Top-k Retrieval (k=3)"]
            C_Prompt["Context-Augmented Prompt"]
            C_LLM["Generator LLM"]
            C_Ret --> C_Prompt --> C_LLM
        end
        
        subgraph SysD ["System D: Agentic RAG (LangGraph)"]
            D_State["Typed AgentState"]
            D_ReAct["ReAct Reasoning Loop"]
            D_Tools["Tools: Retriever | Rewriter | Calc | Abstain"]
            D_State --> D_ReAct <--> D_Tools
        end
        
        Q --> SysA
        Q --> SysB
        Q --> SysC
        Q --> SysD
        VectorDB -.-> C_Ret
        VectorDB -.-> D_Tools
    end

    subgraph EvaluationLayer ["Unified Evaluation Harness"]
        SysA --> Runner["Evaluation Runner & Cache"]
        SysB --> Runner
        SysC --> Runner
        SysD --> Runner
        
        GoldenSet[("Golden Eval Set (100+ Qs across 5 Categories)")] --> Runner
        
        Judge["Independent Judge LLM (Distinct Model)"]
        Runner <--> Judge
        
        subgraph Metrics ["Benchmarked Metrics"]
            M_Faith["Faithfulness & Hallucination Rate"]
            M_Corr["Answer Correctness (vs Ground Truth)"]
            M_Ret["Context Precision & Recall"]
            M_Abs["Abstention Accuracy (Out-of-Scope)"]
            M_Perf["Latency (p50, p95) & Token Usage"]
            M_CI["95% Bootstrap Confidence Intervals"]
        end
        
        Judge --> Metrics
        Metrics --> Artifacts["CSV Reports, JSON Summaries & Plots"]
    end
```

---

## 2. System D: Agentic RAG State Machine (LangGraph)

System D models reasoning as a cyclic state graph with deterministic tool execution and recursion guardrails:

```mermaid
stateDiagram-v2
    [*] --> InitializeState : Student Query Arrives
    InitializeState --> AgentReasoner : Set query & iteration_count = 0
    
    state AgentReasoner <<choice>>
    AgentReasoner --> QueryRewriter : Query Ambiguous / Multi-Hop
    AgentReasoner --> HandbookRetriever : Information Needed
    AgentReasoner --> CalculatorTool : Formula / Fee / Date Arithmetic Needed
    AgentReasoner --> FormulateAnswer : Sufficient Evidence Gathered
    AgentReasoner --> Abstain : Out-of-Scope / Counterfactual Detected
    
    QueryRewriter --> UpdateState : New targeted sub-query generated
    HandbookRetriever --> UpdateState : Top chunks retrieved & appended
    CalculatorTool --> UpdateState : Python math evaluated safely
    
    UpdateState --> CheckIterationGuard
    
    state CheckIterationGuard <<choice>>
    CheckIterationGuard --> AgentReasoner : iteration_count < max_iterations (4)
    CheckIterationGuard --> FallbackAbstain : iteration_count >= max_iterations
    
    FormulateAnswer --> FinalizeTrace : Grounded answer with citations
    Abstain --> FinalizeTrace : Explicit "I do not have sufficient information"
    FallbackAbstain --> FinalizeTrace : Max iterations exceeded fallback
    
    FinalizeTrace --> [*] : Return AgentOutput to Harness
```

### AgentState Typed Schema
```python
class AgentState(TypedDict):
    messages: list[dict[str, str]]
    original_query: str
    current_query: str
    retrieved_chunks: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    iteration_count: int
    max_iterations: int
    is_abstaining: bool
    final_answer: str | None
    citations: list[str]
```

---

## 3. Evaluation Harness Pipeline

To prevent self-evaluation bias and ensure reproducible evaluation, generation and judging are decoupled:

```mermaid
sequenceDiagram
    autonumber
    actor Harness as Evaluation Runner
    participant Cache as LLM Disk Cache (SQLite)
    participant Gen as Generator LLM (e.g. Groq Llama-3.1-8b)
    participant Judge as Judge LLM (e.g. Gemini-1.5-Flash)
    participant Stats as Bootstrap CI Engine

    Harness->>Cache: Query cached system response?
    alt Cache Hit
        Cache-->>Harness: Return cached prediction & latency
    else Cache Miss
        Harness->>Gen: Generate prediction for question
        Gen-->>Harness: Return prediction, latency, tokens
        Harness->>Cache: Store prediction
    end

    Harness->>Judge: Prompt Judge with (Question, Context, Ground Truth, Prediction)
    Judge-->>Harness: Output JSON {faithfulness: 0/1, correctness: 0/1, reasoning: str}
    
    Harness->>Stats: Run 1000-sample bootstrap resampling
    Stats-->>Harness: Return Mean, 95% Lower CI, 95% Upper CI
    Harness->>Harness: Save results to results/eval_results.csv & results/eval_summary.json
```

---

## 4. Key Architectural Design Decisions

1. **Decoupled Generator & Judge**: Generator and Judge models must have distinct model weights and providers to prevent systemic self-confirmation bias.
2. **Deterministic Calculator Tool**: LLMs consistently struggle with multi-tier fee penalties and date differences. The ReAct agent delegates all arithmetic to a sandboxed Python execution tool.
3. **Explicit Abstention Gate**: Rather than forcing a hallucinated answer when a query lies outside the handbook, System D includes an explicit decision node to abstain gracefully.
4. **Transparent SQLite LLM Cache**: Prevents duplicate API calls during repeated metric runs, ensuring rate limits are preserved and latency measurements remain reproducible.

