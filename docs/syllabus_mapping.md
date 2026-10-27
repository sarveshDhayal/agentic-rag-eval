# Syllabus Mapping & Academic Alignment

**Course Context**: B.Tech in Computer Science & Artificial Intelligence (3rd Year)  
**Primary Subjects**: Introduction to Generative AI & Deep Learning (with NLP & Multi-Agent Systems)

This document provides an explicit mapping between the engineering components developed in this project and the theoretical topics covered in undergraduate CS & AI curricula. It serves as direct evidence of syllabus fulfillment for project reviews, evaluations, and oral examinations (viva voce).

---

## Comprehensive Component-to-Syllabus Mapping

| Project Module / Component | Primary Subject | Syllabus Topic | Theoretical Concept | Practical Implementation in Project |
| :--- | :--- | :--- | :--- | :--- |
| **System A (Baseline)** | Intro to GenAI | LLM Foundations & Prompt Engineering | Parametric knowledge, Zero-shot & Few-shot prompting, Prompt sensitivity, Instruction following | `src/systems/system_a_prompting.py`: Direct persona prompting without external knowledge context. Evaluates baseline parametric hallucination. |
| **System B (Fine-Tuning)** | Intro to GenAI & Deep Learning | Parameter-Efficient Fine-Tuning (PEFT) | Low-Rank Adaptation (LoRA), QLoRA (NF4 4-bit quantization), Double Quantization, Paged Optimizers, Catastrophic Forgetting | `notebooks/qlora_finetune_colab.ipynb` & `src/systems/system_b_finetuned.py`: SFT on Qwen2.5-1.5B/Llama-3.2-1B with low-rank adapters ($r=16, \alpha=32$). |
| **Embedding Pipeline** | Deep Learning | Dense Semantic Representations | Dense vector embeddings, Contrastive learning, Bi-encoders vs Cross-encoders, Cosine similarity vs Euclidean distance | `src/retrieval/embeddings.py`: Generates dense representations using `BAAI/bge-small-en-v1.5` (384 dimensions) with $L_2$ normalization. |
| **Chunking Strategies** | Deep Learning & NLP | Text Preprocessing & Tokenization | Context window limitations, Semantic boundary preservation, Lost-in-the-Middle problem, Information density | `src/chunking/`: Compares `FixedSizeChunker` (character sliding window) vs `SectionAwareChunker` (hierarchical markdown structure preserving tables). |
| **Vector Indexing & Search** | Deep Learning | Approximate Nearest Neighbor (ANN) Search | Vector indexing, IVF, HNSW, Exact vs Approximate search, Inner Product / Cosine similarity | `src/retrieval/vectorstore.py`: FAISS index storing chunk embeddings with metadata filtering and top-$k$ nearest-neighbor retrieval. |
| **Retrieval Evaluation** | Deep Learning & IR | Information Retrieval Metrics | Top-$k$ Retrieval, Hit Rate @ $k$, Mean Reciprocal Rank (MRR), Precision@$k$, Recall@$k$ | `src/retrieval/retrieval_eval.py`: Quantitative benchmark testing whether golden supporting sections appear in top-$k$ retrieved chunks. |
| **System C (Basic RAG)** | Intro to GenAI | Retrieval-Augmented Generation (RAG) | Non-parametric memory injection, Grounded generation, In-context learning, Context window utilization | `src/systems/system_c_rag.py`: Retrieve-then-read pipeline passing retrieved chunks into an augmented prompt with citation constraints. |
| **System D (Agentic RAG)** | Intro to GenAI & AI Agents | ReAct Pattern & Agentic Workflows | Reasoning + Acting (ReAct), Thought-Action-Observation loops, Finite State Machines, Tool use, Dynamic query reformulation | `src/systems/system_d_agentic.py`: LangGraph state graph with typed state, tool dispatch (`handbook_retriever`, `query_rewriter`, `calculator`, `answer_or_abstain`), and cycle guardrails. |
| **Evaluation Harness: Faithfulness & Hallucination** | Intro to GenAI | Hallucination Detection & LLM-as-a-Judge | Factuality evaluation, Groundedness vs Extrapolation, Self-consistency, Inter-model evaluation bias | `src/evaluation/faithfulness_eval.py`: Independent Judge LLM evaluating whether generated claims are strictly entailed by the context or ground truth. |
| **Evaluation Harness: Abstention Accuracy** | Intro to GenAI | Out-of-Distribution & Adversarial Robustness | Epistemic uncertainty, Rejection of unanswerable queries, Adversarial false premise detection | `src/evaluation/abstention_eval.py`: Evaluates whether the system correctly refrains from answering out-of-scope or counterfactual student queries. |
| **Statistical Rigor & Significance** | Deep Learning & Stats | Statistical Estimation & Hypothesis Testing | Non-parametric Bootstrap Confidence Intervals, Empirical distribution estimation, Sampling with replacement | `src/evaluation/bootstrap_ci.py`: Computes 95% bootstrap confidence intervals ($B=1000$) over accuracy, faithfulness, and latency metrics. |
| **Interactive Dashboard & Observability** | Software Engineering & AI Tools | Human-in-the-Loop & System Observability | Execution tracing, Step-by-step reasoning visualization, Comparative UI evaluation | `app.py`: Streamlit application enabling real-time inspection of retrieved chunks, tool invocation traces, and side-by-side model outputs. |

---

## Laboratory & Viva Voce Quick-Reference Matrix

| Topic Area | Viva Questions Covered | Core Project Artifact |
| :--- | :--- | :--- |
| **Prompting vs Fine-Tuning vs RAG** | When should you fine-tune vs build RAG? How does knowledge updating differ between parametric and non-parametric systems? | Comparison table in `docs/final_report.md` & `results/eval_summary.json` |
| **LoRA & Quantization** | Why does LoRA reduce trainable parameters from $d \times k$ to $r(d + k)$? What is NormalFloat4 (NF4) in QLoRA? | `notebooks/qlora_finetune_colab.ipynb` |
| **Dense Embeddings & Retrieval** | How does BGE-small map text into semantic space? Why does Section-Aware chunking outperform Fixed-Size chunking on tabular data? | `src/chunking/` & `results/plots/chunking_ablation.png` |
| **Agentic State Machines** | Why use LangGraph over sequential chains? How do conditional edges and recursion limits prevent infinite reasoning loops? | `src/systems/system_d_agentic.py` & `docs/architecture.md` |
| **Hallucination Evaluation** | How is faithfulness mathematically separated from answer correctness? Why must the judge LLM differ from the generator? | `src/evaluation/eval_harness.py` |
