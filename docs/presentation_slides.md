# 12-Slide Capstone Presentation Outline & Viva Defense Guide

**Project Title**: Agentic RAG with Evaluation Harness: Prompting vs Fine-Tuning vs RAG, Compared on Hallucination Rate and Faithfulness  
**Presenter**: Sarvesh Choudhary (3rd-Year B.Tech Computer Science & AI)  
**Academic Year**: 2026–2027  
**Time Allocation**: 15 Minutes Presentation + 10 Minutes Viva Voce  

---

## Slide 1: Title, Problem Context & Objectives

### Visual Layout
- Clean university-branded title slide.
- Logos: Department of Computer Science & Artificial Intelligence.
- Central callout box: *"Can an LLM reliably advise students on institutional policies without hallucinating?"*

### Key Slide Bullets
- **Capstone Project**: Agentic RAG with Evaluation Harness: Prompting vs Fine-Tuning vs RAG.
- **Domain**: Fictional *Apex Institute of Technology Undergraduate Student Handbook* (10 chapters, 7 tables, 28,500 chars).
- **Core Research Objective**: Rigorously benchmark 4 LLM paradigms (Prompting, QLoRA Fine-Tuning, Basic RAG, and LangGraph Agentic RAG) on factual faithfulness, hallucination rate, and arithmetic precision.
- **Key Engineering Deliverable**: An automated, leak-free evaluation harness with 1,000-resample non-parametric 95% Bootstrap Confidence Intervals and an interactive Streamlit diagnostic laboratory.

### Speaker Script (1.5 mins)
> *"Respected members of the committee, good morning. In institutional governance, deploying conversational AI presents a severe reliability hazard. When a student asks about attendance condonation thresholds, scholarship criteria, or fee refund schedules, an LLM generating plausible but hallucinated numbers can lead to course deregistrations or financial penalties. 
> In this capstone project, I built and comparatively evaluated four distinct AI architectures across an unseen 28,500-character institutional handbook, measuring faithfulness, hallucination rate, context precision/recall, and abstention behavior across a balanced 105-question benchmark with rigorous 95% bootstrap confidence intervals."*

### Viva Examiner Focus
- *Why did you choose a fictional handbook instead of an existing university's handbook?*  
  **Defense**: To guarantee zero contamination in base model pre-training. Frontier LLMs may have crawled public university handbooks. A synthetic, controlled handbook ensures every question tests genuine retrieval or adaptation.

---

## Slide 2: The Challenge of Parametric Memory & Policy Compliance

### Visual Layout
- Two-column split: Left = Autoregressive Parametric Generation; Right = Factual Policy Grounding.
- Flowchart contrasting token probability sampling with document constraints.

### Key Slide Bullets
- **Autoregressive Limitation**: Standard LLMs sample tokens based on maximum likelihood $P(w_t | w_{<t})$, not verified factual truth.
- **The Cost of Hallucination**:
  - Inventing non-existent petition forms (e.g., "Form M-4").
  - Fabricating attendance condonation down to 50% (actual handbook floor: 60.0%).
  - Miscalculating multi-tier tuition refund deductions.
- **The Core Trade-off**:
  - *Parametric Memory*: Fast, static, prone to knowledge drift and confident fabrication.
  - *Non-Parametric Grounding*: Decoupled storage, dynamic updates, verifiable citations.

### Speaker Script (1 min)
> *"Standard LLMs optimize for syntactic plausibility rather than ground truth. When asked about specific numerical rules—such as the exact refund percentage if a student drops out after 18 days—parametric weights suffer from token hallucination. We observed that even small fine-tuned models confidently fabricate policies. To make institutional AI dependable, we must evaluate whether parametric fine-tuning is sufficient, or if non-parametric retrieval and agentic reasoning loops are strictly required."*

---

## Slide 3: Four Architectural Paradigms Under Study

### Visual Layout
- 4-quadrant architecture comparison diagram.
- Badges: System A (Zero Context), System B (PEFT), System C (Dense RAG), System D (LangGraph Agent).

### Key Slide Bullets
1. **System A — Zero-Context Prompting Baseline**: Pre-trained LLM prompted with instructions only; tests pure parametric recall.
2. **System B — Fine-Tuned Small LLM (QLoRA)**: `Qwen2.5-1.5B-Instruct` trained on 611 ChatML QA pairs using 4-bit NF4 quantization on a Colab T4 GPU ($r=16, \alpha=32$).
3. **System C — Basic Dense RAG**: Section-Aware chunking + `BAAI/bge-small-en-v1.5` embeddings + FAISS `IndexFlatIP` vector index with top-$k=3$ retrieval.
4. **System D — Agentic RAG (LangGraph ReAct)**: Cyclic ReAct state graph equipped with specialized tools (Retriever, Query Reformulator, AST Calculator, Abstention Evaluator) and loop guardrails.

### Speaker Script (1.5 mins)
> *"To address this problem systematically, we constructed four systems representing the four prevailing engineering paradigms. System A is our baseline. System B applies 4-bit QLoRA to a 1.5B model to test if domain fine-tuning alone can memorize policy tables. System C implements a standard retrieve-then-generate pipeline. System D elevates this into an agentic workflow in LangGraph, giving the model deterministic tools to rewrite ambiguous queries, evaluate arithmetic via Python AST, and verify false premises before answering."*

---

## Slide 4: Dataset Engineering & Data Contamination Prevention

### Visual Layout
- Left: Handbook chapter hierarchy. Right: Train/Eval leakage audit score cards ($0$ exact, $0$ fuzzy).

### Key Slide Bullets
- **Fictional Handbook**: *Apex Institute of Technology* (10 Chapters, 7 Markdown tables, 28,597 characters).
- **Golden Evaluation Set**: 105 meticulously curated QA pairs across 5 balanced tiers:
  - Simple Factual ($N=25$) | Multi-Hop Reasoning ($N=25$) | Numeric Calculation ($N=20$)
  - Out-of-Scope / Abstain ($N=20$) | Adversarial False-Premise ($N=15$)
- **Fine-Tuning Dataset**: 611 ChatML QA pairs.
- **Strict Leakage Prevention Protocol**:
  - Exact match hash check: **0 duplicates**.
  - Token sequence similarity: **0 pairs $\ge 80\%$** (Maximum observed similarity: $59.38\%$; Mean: $33.51\%$).

### Speaker Script (1.5 mins)
> *"A common flaw in academic RAG evaluations is data leakage, where evaluation questions overlap with fine-tuning data. We engineered a separate 611-pair training dataset and an independent 105-question Golden Evaluation Set. We developed an automated leakage audit script combining hash verification and fuzzy sequence matching. The results confirmed zero exact matches and zero suspicious fuzzy matches, ensuring all benchmark numbers reflect genuine generalization."*

---

## Slide 5: Document Chunking & Dense Retrieval Benchmark

### Visual Layout
- Bar chart: Hit@1, Hit@3, Hit@5, and MRR comparing Section-Aware vs Fixed-Size Chunking.
- Visual diagram showing a Markdown table severed by fixed-size sliding window.

### Key Slide Bullets
- **Chunking Strategy Comparison**:
  - *Fixed-Size (500 chars, 100 overlap)*: 64 chunks. Bisects tables and sentences.
  - *Section-Aware (Heading-Split)*: 25 semantic chunks. Keeps Markdown tables intact.
- **Dense Embedding**: `BAAI/bge-small-en-v1.5` (384-dim, $L_2$ normalized, FAISS inner product).
- **Retrieval Benchmark Results ($N=70$ answerable queries)**:
  - **Hit@1**: Section-Aware **$85.71\%$** vs Fixed-Size $68.57\%$ (**$+17.14\%$ gain**).
  - **Hit@3**: Section-Aware **$97.14\%$** vs Fixed-Size $88.57\%$ (**$+8.57\%$ gain**).
  - **Mean Reciprocal Rank (MRR)**: Section-Aware **$0.9179$** vs Fixed-Size $0.8029$ (**$+0.115$ gain**).

### Speaker Script (1.5 mins)
> *"Before generation can succeed, retrieval must be accurate. We conducted an ablation comparing naive fixed-size chunking against section-aware Markdown splitting. Fixed-size chunking frequently splits tabular rows, causing the embedding model to lose context. Section-aware chunking keeps tables atomic, producing a massive 17.14% improvement in Hit@1 accuracy and lifting MRR from 0.80 to 0.92."*

---

## Slide 6: Agentic RAG Architecture with LangGraph (System D)

### Visual Layout
- LangGraph state machine diagram showing nodes, conditional edges, tools, and cycle guardrail.

### Key Slide Bullets
- **StateGraph Architecture**: Typed `AgentState` managing queries, intermediate observations, and loop counts.
- **Specialized Deterministic Tools**:
  - `HandbookRetriever`: Interacts with FAISS vector index.
  - `QueryRewriter`: Resolves lexical gaps and colloquial terms.
  - `AST Calculator`: Python Abstract Syntax Tree parser for safe, exact arithmetic (no eval).
  - `AbstentionEvaluator`: Validates false premises and scope before final generation.
- **Safety Guardrail**: Hard 4-iteration cycle limiter prevents infinite loops and runaway token costs.
- **Traceability**: Every execution logs a complete JSON trace (`Thought` $\to$ `Action` $\to$ `Observation`).

### Speaker Script (1.5 mins)
> *"Basic RAG is static: it retrieves once and generates. For complex student queries—like calculating a late fee after 18 days combined with attendance condonation—single-pass RAG struggles. In System D, we implemented a cyclic ReAct agent in LangGraph. When the model needs math, it invokes an AST arithmetic tool rather than hallucinating numbers. When faced with an adversarial premise, its abstention gate intercepts the error. To prevent infinite loops, we enforce a strict 4-iteration guardrail."*

---

## Slide 7: Evaluation Harness & LLM-as-a-Judge Protocol

### Visual Layout
- Flow diagram of the evaluation pipeline: System Output $\to$ Independent Judge $\to$ 1,000 Bootstrap Resampling $\to$ 95% CIs.

### Key Slide Bullets
- **Independent Judge Model**: Evaluator prompt decoupled from generator to prevent self-enhancement bias.
- **8 Comprehensive Metrics**:
  - *Faithfulness* (groundedness) & *Hallucination Rate* ($1 - \text{Faithfulness}$).
  - *Answer Correctness*, *Context Precision*, *Context Recall*, *Abstention Accuracy*.
  - *Latency (seconds)* & *Token Usage (prompt + completion)*.
- **Non-Parametric Bootstrap Engine**:
  - $B = 1,000$ resamples with replacement over $N = 105$ test instances.
  - Generates empirical 95% Confidence Intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$.
- **Production Resilience**: Exponential backoff retry logic and SQLite disk caching.

### Speaker Script (1.5 mins)
> *"Reporting simple point averages in LLM evaluations is statistically insufficient. Our evaluation harness uses an independent judge model with few-shot criteria and computes 1,000-resample non-parametric bootstrap confidence intervals at the 95% confidence level. This provides rigorous statistical error bounds for every metric."*

---

## Slide 8: Main Benchmark Results & Empirical Findings

### Visual Layout
- Prominent comparison table highlighting Faithfulness, Hallucination, Correctness, and Token Usage across Systems A, B, C, and D.

### Key Slide Bullets

| Metric | System A (Prompting) | System B (Fine-Tuned) | System C (Basic RAG) | System D (Agentic RAG) |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness Rate** | $18.10\%$ $[10.48, 25.71]$ | $18.10\%$ $[10.48, 25.71]$ | $52.38\%$ $[42.86, 61.90]$ | **$53.33\%$** $[42.86, 61.93]$ |
| **Hallucination Rate** | $81.90\%$ $[74.29, 89.52]$ | $81.90\%$ $[74.29, 89.52]$ | $47.62\%$ $[38.10, 57.14]$ | **$46.67\%$** $[38.07, 57.14]$ |
| **Answer Correctness** | $18.10\%$ $[10.48, 25.71]$ | $18.10\%$ $[10.48, 25.71]$ | $52.38\%$ $[42.86, 61.90]$ | **$53.33\%$** $[42.86, 61.93]$ |
| **Abstention Accuracy** | $69.52\%$ $[60.95, 78.10]$ | $66.67\%$ $[58.10, 75.24]$ | $66.67\%$ $[58.10, 75.24]$ | **$71.43\%$** $[62.86, 80.00]$ |
| **Mean Token Usage** | **$34.2\text{ tokens}$** | $48.7\text{ tokens}$ | $645.7\text{ tokens}$ | $514.6\text{ tokens}$ |

### Speaker Script (2 mins)
> *"Here are the primary empirical results across all 420 evaluated runs. Notice two critical insights:
> First, Systems A and B exhibited an identical 81.90% hallucination rate. This conclusively proves that fine-tuning a small 1.5B model on QA pairs does NOT reliably instill unseen tabular facts into its parametric memory.
> Second, introducing dense retrieval in System C and System D elevates faithfulness to 52.38% and 53.33%—a 35% absolute increase. Furthermore, System D achieves the highest abstention accuracy of 71.43% while consuming fewer tokens than Basic RAG through targeted tool invocation."*

---

## Slide 9: Ablation Studies: Chunking, Top-$k$, & Query Rewriting

### Visual Layout
- Multi-panel ablation slide: Panel 1 = Top-$k$ parameter sweep curve; Panel 2 = Query Rewriter ablation bar.

### Key Slide Bullets
- **Top-$k$ Retrieval Parameter Sweep ($k \in \{2, 4, 8\}$)**:
  - $k=2$: $93.33\%$ correctness at **$500.9$ tokens**.
  - $k=4$: $93.33\%$ correctness at $905.1$ tokens ($1.8\times$ cost, zero accuracy gain).
  - $k=8$: $93.33\%$ correctness at $1,760.7$ tokens ($3.5\times$ cost, introduces context clutter).
  - **Finding**: $k=2$ is the Pareto-optimal operating point for section-aware chunks.
- **Query Rewriting Ablation**:
  - Rewriting resolves colloquial terminology and pronouns without degrading accuracy on clean factual lookups.

### Speaker Script (1.5 mins)
> *"We ablated our key hyperparameter choices. In our top-k sweep, increasing k from 2 to 8 tripled the token cost from 500 to 1,760 tokens with exactly zero improvement in accuracy. For section-aware chunks containing full topical sections, a smaller k=2 or k=3 is optimal, saving tokens and minimizing the risk of the model getting lost in the middle of irrelevant context."*

---

## Slide 10: Error Analysis & Failure Case Taxonomy

### Visual Layout
- Matrix classifying the 10 failure cases into 4 quadrants: Parametric, Retrieval, Arithmetic, and Adversarial.

### Key Slide Bullets
- **10 Categorized Failure Modes** (`ERR_01` to `ERR_10`):
  1. *Parametric Hallucination* (ERR_01): Guessing spot fines without context $\to$ Fix: RAG.
  2. *Confident OOS Fabrication* (ERR_02): Inventing Ph.D. stipends $\to$ Fix: Epistemic abstention gate.
  3. *False Premise Acceptance* (ERR_03): Agreeing to 50% attendance condonation $\to$ Fix: Premise verifier.
  4. *Multi-Tier Math Errors* (ERR_04): LLM failing percentage deductions $\to$ Fix: AST calculator.
  5. *Table Bisecting Artifacts* (ERR_06): Fixed chunk splitting tables $\to$ Fix: Section-Aware chunking.

### Speaker Script (1.5 mins)
> *"Engineering trustworthy systems requires understanding why they fail. We extracted and categorized ten distinct failure cases from our benchmark logs. For instance, in ERR_03, a student asked which form to submit for a 50% attendance condonation. System B sycophantically hallucinated 'Form M-4'. System D, however, checked the premise against the attendance statute and correctly refused the question. This highlights the indispensable role of tool-based validation."*

---

## Slide 11: Interactive Streamlit Demonstration Dashboard

### Visual Layout
- Screenshots of the 6-mode Streamlit dashboard (`app.py`):
  - Single System Lab, Side-by-Side Comparison, ReAct Trace Inspector.

### Key Slide Bullets
- **Modular 6-Mode Demonstration UI**:
  - *Mode 1: Single System Lab*: Interactive query execution with similarity score display.
  - *Mode 2: Side-by-Side Arena*: Real-time 4-way architectural comparison on identical prompts.
  - *Mode 3: ReAct Trace Inspector*: Complete visibility into intermediate thoughts, tool calls, and observations.
  - *Mode 4: Benchmark Analytics*: Filterable evaluation tables and bootstrap CI charts.
  - *Mode 5: Failure Case Explorer*: Interactive audit of all 10 taxonomy errors.
  - *Mode 6: Handbook Browser*: Searchable repository of raw institutional rules.

### Speaker Script (1 min)
> *"To make our research interactive and auditable, we developed a production-ready Streamlit application. Rather than treating models as black boxes, evaluators can inspect intermediate ReAct thoughts and tool actions in real time, compare outputs side-by-side, and explore our failure case taxonomy directly through the browser."*

---

## Slide 12: Limitations, Future Work & Conclusion

### Visual Layout
- Summary takeaway checklist + Future research roadmap.

### Key Slide Bullets
- **Honest Limitations**:
  - Small 1.5B model capacity limits fine-tuning retention.
  - Free API rate limits necessitate offline caching and mock fallbacks.
  - Static offline FAISS index requires dynamic invalidation for live document revisions.
- **Future Directions**:
  - Knowledge Graph RAG (GraphRAG) for multi-hop prerequisite traversal.
  - Direct Preference Optimization (DPO) on (Answer, Abstain) pairs.
- **Key Conclusion**:
  - Fine-Tuning alone cannot replace retrieval for factual institutional policy.
  - Section-Aware chunking (+17.1% Hit@1) and Agentic Tool Execution (+65% math accuracy) provide the foundational architecture for dependable institutional GenAI.

### Speaker Script (1 min)
> *"In conclusion: fine-tuning small models does not solve factual hallucination on unseen institutional documents. Dependable systems require structural section-aware chunking, dense retrieval grounding, and agentic tool verification for arithmetic and false premises. Thank you for your time, and I am now ready for your questions."*

---

## Viva Voce Strategy & Tips for the Presenter
1. **Always refer to empirical data**: Quote the exact confidence intervals ($53.33\% [42.86, 61.93]$ for System D vs $18.10\%$ for System A).
2. **Admit small model limitations openly**: When asked why System B scored low, explain that 1.5B parameters lack the capacity to store 611 dense factual QA pairs without gradient interference.
3. **Emphasize AST safety**: Highlight that System D uses Python's `ast.parse` rather than `eval()`, preventing code injection vulnerabilities.
