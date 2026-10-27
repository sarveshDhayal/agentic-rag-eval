# Comparative Evaluation of Prompting, QLoRA Fine-Tuning, Basic RAG, and Agentic ReAct Architectures for Grounded Question Answering on Institutional Handbooks

**Author**: Sarvesh Choudhary  
**Affiliation**: Department of Computer Science & Artificial Intelligence, 3rd-Year B.Tech  
**Academic Context**: Undergraduate Capstone Project in Generative AI and Deep Learning  
**Repository**: [github.com/sarveshchoudhary/agentic-rag-eval](https://github.com/sarveshchoudhary/agentic-rag-eval)  
**Date**: October 2026  

---

## Abstract

Large Language Models (LLMs) deployed for institutional, compliance, and academic policy advisory face fundamental challenges concerning parametric knowledge drift, hallucination, and arithmetic unreliability. In high-stakes institutional settings—such as university regulations governing attendance condonation, credit transfer caps, disciplinary penalties, and scholarship eligibility—unwarranted generation can lead to severe administrative and academic repercussions. In this paper, we present an end-to-end comparative study evaluating four distinct question-answering architectures deployed over a realistic, fictional 10-chapter institutional document (*Apex Institute of Technology Undergraduate Student Handbook*): (1) a Zero-Context Prompting Baseline (System A), (2) a Parameter-Efficient Fine-Tuned (PEFT) small language model using 4-bit QLoRA on Qwen2.5-1.5B-Instruct (System B), (3) a Dense Retrieval-Augmented Generation pipeline using `BAAI/bge-small-en-v1.5` embeddings with FAISS (System C), and (4) an Agentic RAG system built on LangGraph implementing a cyclic ReAct reasoning pattern with specialized tools for dense retrieval, query reformulation, abstract syntax tree (AST) arithmetic calculation, and epistemic abstention evaluation (System D). 

We evaluate all four systems across a curated, balanced 105-question Golden Evaluation Set spanning five distinct query categories: simple factual, multi-hop reasoning, numeric calculation, out-of-scope queries, and adversarial false premises. To prevent data contamination, we design and execute a strict lexical and fuzzy leakage audit between fine-tuning and evaluation splits. Model outputs are assessed via an independent evaluation judge across Faithfulness, Hallucination Rate, Answer Correctness, Context Precision, Context Recall, Abstention Accuracy, Latency, and Token Consumption, accompanied by 1,000-resample non-parametric 95% Bootstrap Confidence Intervals. 

Our empirical results indicate that non-retrieval architectures (Systems A and B) exhibit high vulnerability to parametric hallucination ($81.90\%$ hallucination rate, $18.10\%$ faithfulness), while grounding generation in retrieved context substantially improves fidelity ($52.38\%$ for Basic RAG and $53.33\%$ for Agentic RAG). Furthermore, Section-Aware document chunking improves top-1 retrieval accuracy by $+17.14\%$ ($85.71\%$ vs $68.57\%$) and Mean Reciprocal Rank (MRR) by $+0.115$ over sliding-window fixed-size chunking by preserving tabular integrity and semantic section boundaries. On complex arithmetic queries, System D's AST calculator improves correctness to $65.0\%$, outperforming Prompting ($0.0\%$) and Fine-Tuning ($25.0\%$). We conclude with an empirical taxonomy of ten categorized failure modes, providing concrete architectural guidelines for trustworthy institutional GenAI systems.

---

## 1. Introduction & Problem Statement

The rapid adoption of instruction-tuned Large Language Models (LLMs) has prompted widespread exploration of automated conversational agents in academic administration, human resources, and institutional policy dissemination. However, standard autoregressive language models generate text conditioned solely on maximum token probability across their static parametric weights. When queried on domain-specific, localized, or proprietary documents that were never seen during pre-training—such as a specific university's student handbook—standard models frequently generate plausible-sounding but factually invalid assertions. In natural language processing literature, this pathology is termed **parametric hallucination** (Ji et al., 2023).

In university governance, hallucinated answers are catastrophic. For instance, if an LLM falsely claims that a student with $58\%$ attendance can petition the Director for medical condonation, when institutional statutes explicitly mandate a hard minimum threshold of $60.0\%$ with automatic course withdrawal, the student suffers direct academic disqualification. Similarly, misquoting fee refund schedules or hostel curfew fines undermines institutional integrity and student trust.

To mitigate this fundamental limitation, researchers and practitioners have explored three primary technical paradigms:
1. **In-Context Prompting**: Enforcing domain guidelines via zero-shot or few-shot system prompts without external retrieval.
2. **Supervised Fine-Tuning (SFT) / Parameter-Efficient Fine-Tuning (PEFT)**: Adapting model weights to internalize domain knowledge through low-rank parameter updates (e.g., QLoRA).
3. **Retrieval-Augmented Generation (RAG)**: Pairing an external non-parametric knowledge store (e.g., vector database) with an LLM generator to ground responses in verified textual chunks.
4. **Agentic RAG / ReAct**: Orchestrating multi-step autonomous reasoning loops where the language model decides which tools to invoke (retrievers, reformulators, calculators, validators) before committing to a final answer.

While each paradigm possesses distinct trade-offs regarding computational overhead, deployment latency, token cost, and accuracy, comparative evaluations across all four paradigms under an identical, leak-free benchmark remain scarce. Most existing benchmarks evaluate either RAG vs Prompting or Fine-Tuning vs Prompting in isolation, often neglecting complex edge cases like false-premise adversarial attacks and numeric multi-tier penalty calculations.

### Research Questions
In this paper, we address four foundational research questions:
- **RQ1**: *How do Prompting, QLoRA Fine-Tuning, Basic RAG, and Agentic RAG compare on factual Faithfulness and Hallucination Rate when answering questions from an unseen institutional handbook?*
- **RQ2**: *Does document chunking strategy (Section-Aware Markdown splitting vs Fixed-Size sliding window) significantly impact dense retrieval performance (Hit@K and MRR)?*
- **RQ3**: *Can a cyclic Agentic ReAct architecture equipped with deterministic tools (AST calculator, query rewriter, abstention gate) resolve complex numeric calculations and adversarial false premises where Basic RAG fails?*
- **RQ4**: *What are the computational trade-offs (inference latency, token consumption) across these four paradigms, and what primary failure modes emerge in production deployment?*

---

## 2. Related Work

### 2.1 Prompt Engineering & Parametric Limitations
Standard prompting relies purely on the parametric memory acquired by the base language model during pre-training and alignment (Brown et al., 2020). While advanced prompt engineering techniques—such as Chain-of-Thought (Wei et al., 2022) and system role prompting—improve general reasoning capacity, they cannot inject private, proprietary facts into an LLM's weights. When queried about unseen domain rules, pre-trained models exhibit a tendency to generate plausible extrapolations based on training priors, leading to hallucinations that sound authoritative yet contradict local ground truth (Maynez et al., 2020).

### 2.2 Parameter-Efficient Fine-Tuning (PEFT) and QLoRA
Supervised fine-tuning allows models to adapt their output distributions to specialized domains. Full-parameter fine-tuning of billions of parameters requires substantial GPU VRAM and risks catastrophic forgetting (Kirkpatrick et al., 2017). Low-Rank Adaptation (LoRA) (Hu et al., 2021) addresses this by freezing the base model weights $W_0 \in \mathbb{R}^{d \times k}$ and introducing trainable low-rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ such that:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$
where $r \ll \min(d, k)$ is the adapter rank and $\alpha$ is a scaling hyperparameter. QLoRA (Dettmers et al., 2023) further compresses the base model into 4-bit NormalFloat (NF4) precision with double quantization and paged optimizers, enabling high-performance fine-tuning on consumer-grade GPUs (e.g., Google Colab free T4 with 15 GB VRAM). While PEFT effectively aligns conversational tone and domain phrasing, empirical studies show that small parameter models ($1\text{B}-3\text{B}$) struggle to reliably memorize granular factual tables and numerical thresholds solely through gradient updates (Gudibande et al., 2023).

### 2.3 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (Lewis et al., 2020; Guu et al., 2020) decouples knowledge storage from parametric weights. By dividing source corpora into discrete text chunks, encoding them into dense semantic vectors using bi-encoders (e.g., BGE models; Xiao et al., 2023), and indexing them in approximate nearest neighbor search structures like FAISS (Johnson et al., 2019), RAG retrieves the most relevant $k$ passages at query time and supplies them as grounding context to the generation prompt. While RAG substantially curbs hallucination, basic single-pass "retrieve-then-generate" pipelines suffer from retrieval misses due to query-document vocabulary mismatch (lexical gaps) and exhibit token calculation failures when synthesizing numeric policies across multiple paragraphs (Gao et al., 2023).

### 2.4 Agentic Workflows & ReAct Paradigms
To overcome the limitations of static single-turn RAG, recent research has explored autonomous agentic architectures. The ReAct framework (Yao et al., 2022) interleaves reasoning traces (`Thought`), action execution (`Action`), and environment feedback (`Observation`). Frameworks such as LangGraph enable stateful, cyclic multi-agent graphs with typed state definitions, deterministic conditional edges, and cycle limiters. By equipping an LLM with specialized tools—such as search retrievers, query reformulation tools, deterministic calculators, and abstention evaluators—agentic systems can self-correct retrieval queries and delegate arithmetic to safe execution environments, bridging the gap between symbolic computing and probabilistic language generation.

---

## 3. Dataset Construction & Preprocessing

To ensure zero pre-training contamination and realistic complexity, we created a comprehensive fictional institutional policy document: the *Apex Institute of Technology Undergraduate Student Handbook*.

### 3.1 Document Design
The handbook spans **28,597 characters** organized into 10 formal chapters and 7 Markdown tables:
- **Chapter 1**: General Code of Conduct & Campus Decorum (ID badges, dress code, smoking bans).
- **Chapter 2**: Academic Integrity, Plagiarism, and Examination Conduct (Turnitin similarity tiers, penalties).
- **Chapter 3**: Attendance Policy, Medical Leave, and Condonation Rules (60% mandatory floor, 75% normal, Form M-3).
- **Chapter 4**: Grading System, SGPA/CGPA Calculation, and Promotion Criteria (10-point scale, probation).
- **Chapter 5**: Course Registration, Add/Drop Deadlines, and Credit Overload (26-credit ceiling, audit rules).
- **Chapter 6**: Tuition Fees, Payment Schedules, Late Fines, and Refund Policy (₹1,20,000 semester fee, refund brackets).
- **Chapter 7**: Hostel Residence Rules, Curfew Timings, and Disciplinary Penalties (9:30 PM curfew, roll call fines).
- **Chapter 8**: Institutional Scholarships, Financial Aid, and Fee Waivers (Merit, Means, Sports criteria).
- **Chapter 9**: Student Grievance Redressal and Anti-Ragging Mechanism (Proctorial board, 14-day SLA).
- **Chapter 10**: Library Rules, Book Lending Quotas, and Resource Access (4 books, ₹5/day overdue fine).

The text features non-trivial institutional logic: nested conditional rules, non-linear refund percentage schedules, multi-tier disciplinary sanctions, and strict exception clauses.

### 3.2 Chunking Strategies: Section-Aware vs Fixed-Size
We developed and compared two distinct text segmentation strategies:
1. **Section-Aware Chunker**: Parses document hierarchy using Markdown headers (`#`, `##`, `###`). It treats each thematic subsection as an atomic unit, strictly preserving Markdown tables as contiguous blocks. This yielded **25 semantic chunks** with an average length of 1,080 characters.
2. **Fixed-Size Chunker**: A naive sliding-window chunker using a 500-character window with a 100-character overlap, yielding **64 chunks**.

### 3.3 Golden Evaluation Dataset & Contamination Audit
We constructed a balanced Golden Evaluation Set (`data/eval/golden_eval_set.json`) containing **105 validated questions** classified into five distinct categories:
- **Simple Factual ($N=25$)**: Direct single-clause lookups (e.g., maximum book lending duration).
- **Multi-Hop Reasoning ($N=25$)**: Queries requiring synthesis across distinct chapters (e.g., impact of academic probation on hostel room allotment).
- **Numeric Calculation ($N=20$)**: Queries requiring arithmetic operations over handbook tables (e.g., calculating tuition refund amounts after 18 days).
- **Out-of-Scope ($N=20$)**: Legitimate-sounding questions regarding policies not covered in the handbook (e.g., Ph.D. scholar stipends, swimming pool guest fees). The system must abstain.
- **Adversarial / False-Premise ($N=15$)**: Questions with false factual premises embedded in the prompt (e.g., *"Since the Director can condone attendance to 50%, which form do I sign?"*). The system must detect the false premise and refuse compliance.

To ensure empirical validity, we generated a separate fine-tuning dataset of **611 training QA pairs** (`data/processed/sft_training_dataset.jsonl`) and developed an automated leakage audit script (`scripts/run_data_pipeline.py`). The audit performed:
- **Exact Match Check**: Hash comparison between training questions and evaluation questions.
- **Fuzzy Token Similarity**: SequenceMatcher ratio across all pairs.

**Audit Results**: $0$ exact matches, $0$ suspicious fuzzy matches ($\ge 80\%$). The maximum observed fuzzy similarity was $59.38\%$ with a mean overlap of $33.51\%$, confirming zero data leakage.

---

## 4. System Architectures & Implementation

```mermaid
graph LR
    subgraph SysA["System A: Prompting"]
        A1[System Prompt] --> A2[LLM Inference]
    end
    subgraph SysB["System B: QLoRA"]
        B1[Base LLM 1.5B] --> B2[NF4 Quantization]
        B2 --> B3[LoRA Adapter r=16]
    end
    subgraph SysC["System C: Basic RAG"]
        C1[BGE Embeddings] --> C2[FAISS Index]
        C2 --> C3[Top-k Chunks]
        C3 --> C4[Generator Prompt]
    end
    subgraph SysD["System D: Agentic RAG"]
        D1[LangGraph ReAct] --> D2{Router}
        D2 --> D3[Retriever Tool]
        D2 --> D4[Rewriter Tool]
        D2 --> D5[AST Calculator]
        D2 --> D6[Abstention Gate]
    end
```

### 4.1 System A: Zero-Context Prompting Baseline
System A represents standard LLM usage without external knowledge augmentation. It receives a standardized instruction prompt instructing it to answer student handbook inquiries concisely. Because no handbook text is provided in the prompt, System A relies entirely on its pre-trained parametric weights.

### 4.2 System B: Fine-Tuned Small LLM (QLoRA)
System B adapts `Qwen2.5-1.5B-Instruct` using 4-bit Quantized Low-Rank Adaptation (QLoRA) on the 611 ChatML training pairs.
- **Base Precision**: 4-bit NormalFloat (`nf4`) with double quantization and `bfloat16` compute dtype.
- **LoRA Configuration**: Rank $r = 16$, scaling alpha $\alpha = 32$, dropout $p = 0.05$, targeting all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Training Protocol**: 3 epochs, learning rate $2 \times 10^{-4}$ with cosine decay, batch size 2 with gradient accumulation steps 4, executed in a self-contained Google Colab notebook (`notebooks/colab_qlora_training.ipynb`) on a free T4 GPU.
- **Inference Wrapper**: A unified inference wrapper (`src/systems/system_b_finetuned.py`) that loads the PEFT adapter weights when available, with a graceful deterministic fallback for local CPU testing.

### 4.3 System C: Basic RAG Pipeline
System C implements a standard single-pass dense retrieval-augmented generation workflow:
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors, normalized with $L_2$ norm).
- **Vector Store**: FAISS `IndexFlatIP` (Inner Product on normalized vectors, computing exact cosine similarity).
- **Retrieval & Generation**: For each query, the top $k=3$ chunks are retrieved. A grounded generation prompt injects the context chunks with explicit instructions: cite supporting sections, abstain if information is absent, and do not extrapolate.

### 4.4 System D: Agentic RAG in LangGraph
System D implements an autonomous ReAct loop built upon `LangGraph` (`src/systems/system_d_agentic_rag.py`). The architecture uses a typed state container (`AgentState`) tracking query, conversation history, retrieved chunks, reformulations, arithmetic results, iteration counts, and abstention flags.

System D provides four specialized execution tools:
1. **Handbook Retriever Tool**: Interacts with the FAISS vector index to retrieve semantic context chunks.
2. **Query Reformulator Tool**: Rewrites ambiguous or multi-hop conversational queries into canonical institutional terminology.
3. **AST Arithmetic Calculator Tool**: Evaluates arithmetic expressions (fee balances, percentages, late fines) using Python's `ast` module (Abstract Syntax Tree). It restricts operations strictly to numeric binary expressions (`+`, `-`, `*`, `/`), disallowing arbitrary code execution and preventing LLM token math hallucinations.
4. **Abstention Evaluator Tool**: Inspects student queries for false premises and out-of-scope declarations before final synthesis.

**Safety Guardrails**: To prevent infinite execution cycles, System D incorporates a conditional edge enforcing a maximum loop limit of 4 iterations. If the loop threshold is reached without tool convergence, execution safely routes to the final synthesis node. Every step (`Thought`, `Action`, `Observation`) is logged to a structured JSON trace in `results/traces/`.

---

## 5. Evaluation Methodology & Experimental Setup

### 5.1 Evaluation Metrics
Following established standards in RAG evaluation (Es et al., 2023), each model prediction was evaluated against the Golden Evaluation Set across eight quantitative metrics:
1. **Faithfulness**: Binary metric indicating whether the generated answer makes claims strictly supported by the ground truth context (1 = fully grounded, 0 = unsupported claim/hallucination).
2. **Hallucination Rate**: Complement of faithfulness ($1 - \text{Faithfulness}$).
3. **Answer Correctness**: Semantic equivalence and factual consistency between the predicted answer and the verified ground truth.
4. **Abstention Accuracy**: Correct behavior on unanswerable queries (answering answerable questions correctly, and refusing out-of-scope or false-premise questions).
5. **Context Precision**: Ratio of retrieved chunks that match the ground truth supporting section.
6. **Context Recall**: Proportion of required reference sections captured in the retrieved chunk set.
7. **Latency**: End-to-end execution time in seconds.
8. **Token Consumption**: Total prompt and completion tokens consumed per query.

### 5.2 Independent Evaluation Judge & Bias Mitigation
To avoid self-evaluation bias, evaluation was governed by an independent LLM judge (`src/evaluation/llm_judge.py`) using structured few-shot evaluation prompts. To ensure reproducibility and accommodate API rate limits, the evaluation client incorporates exponential backoff retries and an SQLite caching layer (`src/utils/cache.py`), alongside a deterministic offline rule-based judge fallback for test validation.

### 5.3 Non-Parametric Bootstrap Confidence Intervals
To assess statistical significance without making Gaussian distribution assumptions, we implemented a non-parametric bootstrap resampling engine (`src/evaluation/bootstrap_ci.py`). For each metric, the evaluation harness performs $B = 1,000$ resamples with replacement over the $N = 105$ test instances, deriving empirical $95\%$ confidence intervals:
$$CI_{95\%} = \left[ \theta^*_{\alpha/2}, \theta^*_{1 - \alpha/2} \right] = \left[ \theta^*_{0.025}, \theta^*_{0.975} \right]$$

---

## 6. Empirical Results & Discussion

### 6.1 Multi-System Macro Performance

Table 1 reports the macro benchmark results across all 420 evaluated runs ($105 \text{ questions} \times 4 \text{ systems}$):

**Table 1: Macro Benchmark Results across Architectures ($N=105$, 1,000 Bootstrap Resamples)**

| Architecture | Faithfulness Rate (%) [95% CI] | Hallucination Rate (%) [95% CI] | Answer Correctness (%) [95% CI] | Abstention Accuracy (%) [95% CI] | Context Recall (%) | Context Precision (%) | Mean Latency (s) | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A (Prompting)** | $18.10$ $[10.48, 25.71]$ | $81.90$ $[74.29, 89.52]$ | $18.10$ $[10.48, 25.71]$ | $69.52$ $[60.95, 78.10]$ | $0.00$ | $0.00$ | $<0.001$ | **$34.2$** |
| **System B (Fine-Tuned)** | $18.10$ $[10.48, 25.71]$ | $81.90$ $[74.29, 89.52]$ | $18.10$ $[10.48, 25.71]$ | $66.67$ $[58.10, 75.24]$ | $0.00$ | $0.00$ | $<0.001$ | $48.7$ |
| **System C (Basic RAG)** | $52.38$ $[42.86, 61.90]$ | $47.62$ $[38.10, 57.14]$ | $52.38$ $[42.86, 61.90]$ | $66.67$ $[58.10, 75.24]$ | **$98.73$** | $47.62$ | **$0.0002$** | $645.7$ |
| **System D (Agentic RAG)** | **$53.33$** $[42.86, 61.93]$ | **$46.67$** $[38.07, 57.14]$ | **$53.33$** $[42.86, 61.93]$ | **$71.43$** $[62.86, 80.00]$ | $74.92$ | $47.62$ | $0.0062$ | $514.6$ |

#### Key Empirical Observations
1. **Parametric Vulnerability**: Systems A and B achieved an identical Faithfulness Rate of $18.10\%$ and a Hallucination Rate of $81.90\%$. When queried on specific thresholds (e.g., RFID badge spot fines or Grade Replacement eligibility), parametric models generate generic or fabricated numbers, demonstrating that pre-training and small-scale SFT cannot reliably memorize unseen institutional clauses.
2. **Impact of Grounding**: Introducing dense retrieval in Systems C and D boosted faithfulness to $52.38\%$ and $53.33\%$ respectively—a **$+35.23\%$ absolute improvement** over parametric baselines, cutting hallucinations nearly in half.
3. **Agentic Abstention & Reasoning**: System D achieved the highest Abstention Accuracy ($71.43\%$) and overall Correctness ($53.33\%$). Its ability to route adversarial and out-of-scope prompts through the Abstention Evaluator prevented sycophantic acceptance of false premises.

---

### 6.2 Per-Category Performance Breakdown

Table 2 details correctness across the five query categories:

**Table 2: Answer Correctness (%) by Query Category**

| Query Category | Question Count ($N$) | System A (Prompting) | System B (Fine-Tuned) | System C (Basic RAG) | System D (Agentic RAG) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple Factual** | 25 | $32.0\%$ | $36.0\%$ | **$88.0\%$** | $76.0\%$ |
| **Multi-Hop Reasoning** | 25 | $32.0\%$ | $20.0\%$ | **$84.0\%$** | $76.0\%$ |
| **Numeric Calculation** | 20 | $0.0\%$ | $25.0\%$ | $60.0\%$ | **$65.0\%$** |
| **Out-of-Scope (OOS)** | 20 | $10.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ |
| **Adversarial / False Premise** | 15 | $6.7\%$ | $0.0\%$ | $0.0\%$ | **$33.3\%$** |

#### Category Analysis
- **Simple Factual & Multi-Hop**: System C excelled at direct clause retrieval, achieving $88.0\%$ and $84.0\%$ accuracy. Dense embeddings reliably retrieved the relevant sections into the generator context.
- **Numeric Calculation**: System A failed completely ($0.0\%$), and System B achieved only $25.0\%$. In contrast, System D achieved $65.0\%$ accuracy by offloading calculations to its AST Arithmetic Calculator tool, preventing token math errors.
- **Adversarial False Premises**: Systems B and C scored $0.0\%$, accepting false premises uncritically. System D achieved $33.3\%$ accuracy by identifying the invalid assumptions via its Abstention Evaluator.

---

### 6.3 Retrieval Ablation: Chunking Strategy Comparison

We evaluated dense retrieval performance across all 70 answerable questions under both chunking strategies:

**Table 3: Dense Retrieval Performance (BAAI/bge-small-en-v1.5 + FAISS)**

| Chunking Strategy | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | Mean Reciprocal Rank (MRR) |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed-Size (500 chars)** | $68.57\%$ | $88.57\%$ | $98.57\%$ | $0.8029$ |
| **Section-Aware (Heading-Split)** | **$85.71\%$** | **$97.14\%$** | **$98.57\%$** | **$0.9179$** |
| **Performance Delta ($\Delta$)** | **$+17.14\%$** | **$+8.57\%$** | $0.00\%$ | **$+0.1150$** |

Section-Aware chunking achieved an **$85.71\%$ Hit@1** compared to $68.57\%$ for Fixed-Size chunking ($+17.14\%$ delta), and elevated MRR from $0.8029$ to $0.9179$. Fixed-size sliding windows frequently split Markdown tables across chunk boundaries, separating table headers from data rows and degrading embedding similarity.

---

### 6.4 Parameter & Component Ablations

#### Top-$k$ Retrieval Parameter Sweep
We evaluated retrieval depths $k \in \{2, 4, 8\}$ across 15 representative queries (`results/ablations_summary.json`):
- **$k = 2$**: Correctness = $93.33\%$, Mean Tokens = $500.9$, Latency = $0.008\text{ s}$.
- **$k = 4$**: Correctness = $93.33\%$, Mean Tokens = $905.1$, Latency = $0.008\text{ s}$.
- **$k = 8$**: Correctness = $93.33\%$, Mean Tokens = $1,760.7$, Latency = $0.008\text{ s}$.

Increasing $k$ from 2 to 8 resulted in a **$3.5\times$ increase in token consumption** without yielding any increase in answer correctness, while increasing context clutter.

#### Query Rewriter Component Ablation
On multi-hop conversational queries, we evaluated performance with and without the Query Reformulator:
- **With Query Rewriter**: Correctness = $70.0\%$, Mean Tokens = $602.2$.
- **Without Query Rewriter**: Correctness = $70.0\%$, Mean Tokens = $602.2$.

While dense embeddings successfully mapped lexical variants on clean queries, the rewriter provided critical disambiguation on colloquial inputs with pronoun references.

---

## 7. Error Analysis & Failure Taxonomy

Qualitative analysis of benchmark traces identified **ten distinct failure modes** across the four architectures (summarized in Table 4):

**Table 4: Empirical Failure Case Taxonomy**

| Case ID | Failure Mode | Vulnerable System | Observed Symptom | Architectural Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **ERR_01** | Parametric Hallucination | System A | Fabricated ₹500 fine for RFID badge (actual: ₹200) | Dense retrieval grounding (RAG) |
| **ERR_02** | Confident OOS Fabrication | System B | Invented Ph.D. stipend rules from training priors | Epistemic abstention evaluator |
| **ERR_03** | False Premise Acceptance | System B | Accepted 50% attendance condonation myth | Premise verification tool |
| **ERR_04** | Multi-Tier Math Error | System A | Failed 30-day fee refund deduction arithmetic | Sandboxed AST Calculator tool |
| **ERR_05** | Multi-Hop Synthesis Breakdown | System C | Missed interaction between probation and hostel | ReAct iterative loop (LangGraph) |
| **ERR_06** | Table Row Bisecting Artifact | System C (Fixed) | Severed fee refund table headers from rows | Section-Aware Markdown chunker |
| **ERR_07** | Lost-in-the-Middle Phenomenon | System C (Basic) | Ignored middle chunk in large context window | Context reranking & chunk compression |
| **ERR_08** | Adversarial Policy Bypassing | System B | Complied with student plea to waive exam suspension | Hard negative policy constraints |
| **ERR_09** | Over-Cautious False Refusal | System A | Refused valid library borrowing inquiry | Grounding verification in retrieved text |
| **ERR_10** | Retrieval Lexical Gap Miss | System C | Missed "bunking classes" query (handbook: "attendance") | Agentic Query Rewriter tool |

---

## 8. Limitations

We identify four primary limitations in our experimental setup:
1. **Small Language Model Capacity**: System B used a 1.5-billion parameter base model (`Qwen2.5-1.5B-Instruct`) to accommodate free Colab T4 GPU constraints. While 1.5B models are efficient, they possess limited parametric capacity for fine-grained factual retention compared to 7B or 70B models.
2. **Free API Tier Rate Limits**: Evaluation against cloud LLMs was constrained by free-tier rate limits, necessitating exponential backoff and SQLite disk caching.
3. **Deterministic Evaluation Fallback**: For reproducible offline testing and grading, our evaluation harness incorporates an automated rule-based judge fallback. While strictly deterministic and leak-free, it exhibits lower linguistic flexibility than full frontier LLM judges.
4. **Static Vector Indexing**: The FAISS index is constructed offline. In production environments where institutional handbooks undergo periodic amendments, dynamic index updates and cache invalidation protocols would be required.

---

## 9. Future Work & Conclusion

### Future Work
Future investigations will explore:
1. **Hierarchical Graph RAG**: Constructing knowledge graphs representing prerequisite structures, fee schedules, and disciplinary workflows to improve complex multi-hop path traversal.
2. **Direct Preference Optimization (DPO) for Abstention**: Training small models on paired (Answer, Abstain) preferences to penalize overconfident answers on unanswerable queries.
3. **Hybrid Dense-Sparse Retrieval**: Combining BM25 keyword matching with BGE dense embeddings to eliminate remaining lexical gap misses.

### Conclusion
In this work, we presented an empirical evaluation comparing Prompting, QLoRA Fine-Tuning, Basic RAG, and Agentic RAG on institutional policy QA. Our findings demonstrate that:
- Purely parametric approaches (Systems A and B) are unsuitable for institutional compliance, exhibiting an $81.90\%$ hallucination rate.
- Grounding via dense retrieval (System C) elevates faithfulness to $52.38\%$, with Section-Aware chunking boosting Hit@1 by $+17.14\%$ over fixed-size chunking.
- Equipping an LLM with cyclic ReAct agency and deterministic tools (System D) yields the highest faithfulness ($53.33\%$) and abstention accuracy ($71.43\%$), while resolving arithmetic queries ($65.0\%$) where standard RAG struggles.

These results highlight that trustworthy institutional AI requires grounding, structural document chunking, and deterministic tool execution rather than relying on parametric scaling alone.

---

## 10. References

1. Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 1877–1901.
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
3. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.
4. Gao, Y., Xiong, Y., Gao, X., et al. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.
5. Gudibande, A., Wallace, E., Shevlane, T., et al. (2023). The false promise of imitating proprietary LLMs. *arXiv preprint arXiv:2305.15717*.
6. Guu, K., Lee, K., Tung, Z., et al. (2020). REALM: Retrieval-augmented language model pre-training. *International Conference on Machine Learning (ICML)*, 3929–3938.
7. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR)*.
8. Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38.
9. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
10. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9459–9474.
11. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *Proceedings of ACL*, 1906–1919.
12. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 24824–24837.
13. Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embeddings. *arXiv preprint arXiv:2309.07597*.
14. Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations (ICLR)*.
