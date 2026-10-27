# 20 In-Depth Viva Voce Questions & Answers Guide

**Project Title**: Agentic RAG with Evaluation Harness: Prompting vs Fine-Tuning vs RAG  
**Candidate**: Sarvesh Choudhary (3rd-Year B.Tech Computer Science & AI)  
**Curriculum Mapping**: Generative AI, Deep Learning, Natural Language Processing, and Distributed Systems  

---

## Part 1: Core GenAI, Parametric Memory & Hallucinations

### Q1: What is the theoretical root cause of hallucination in autoregressive Large Language Models?
- **Technical Answer**: Autoregressive language models generate text by sampling from a conditional probability distribution over vocabulary tokens:
  $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1}; \theta)$$
  where $\theta$ represents the static model parameters. The training objective (cross-entropy loss over internet-scale corpora) encourages the model to generate syntactically fluent, statistically probable sequences. Crucially, the model has no internal objective function for factual epistemic truth or awareness of its own ignorance. When queried about facts absent from $\theta$ (or facts subject to knowledge drift), the model samples plausible tokens based on linguistic patterns, resulting in **parametric hallucination**.
- **In My Project**: In System A (Prompting Baseline), when asked about the penalty for not displaying an RFID badge (`eval_002`), the model had never seen the *Apex Institute* handbook. It produced a generic statement about campus decorum rather than the specific ₹200 spot fine, scoring an $81.90\%$ hallucination rate.
- **Examiner Follow-Up Trap**: *"Can you eliminate hallucinations entirely by setting temperature to 0?"*  
  **Defense**: *"No. Setting temperature to 0 ($T=0$) makes decoding strictly greedy (argmax token selection), eliminating sampling randomness. However, if the argmax token in parametric memory represents an incorrect fact, the model simply generates the hallucination deterministically."*

---

### Q2: What is the fundamental difference between parametric and non-parametric memory in AI systems?
- **Technical Answer**: **Parametric memory** consists of the learned continuous weight matrices $\theta$ inside the neural network (e.g., self-attention query-key-value projections and feed-forward layers). Updating parametric memory requires backpropagation, computational compute (GPUs), and carries risks of catastrophic forgetting. **Non-parametric memory** consists of an external, structured or unstructured knowledge corpus (e.g., text chunks in FAISS, documents in Elasticsearch, or relations in a knowledge graph) that is queried dynamically at inference time without modifying model weights.
- **In My Project**: Systems A and B rely purely on parametric memory. Systems C and D decouple storage: the handbook resides in a non-parametric FAISS index, allowing instant policy updates simply by rebuilding the vector index in under 50 milliseconds without retraining the LLM.
- **Examiner Follow-Up Trap**: *"Why not just use a massive context window (e.g., 2M tokens in Gemini 1.5 Pro) instead of non-parametric RAG?"*  
  **Defense**: *"While long-context windows allow stuffing an entire document into the prompt, it incurs three major engineering penalties: (1) linear-to-quadratic latency and cost scaling with prompt length, (2) susceptibility to the 'Lost-in-the-Middle' phenomenon where attention degrades for facts placed mid-context, and (3) privacy/governance issues where entire documents must be repeatedly transmitted to external APIs."*

---

### Q3: Why did Fine-Tuning (System B) fail to outperform Basic RAG on factual accuracy in your benchmark?
- **Technical Answer**: Fine-tuning an autoregressive model on domain QA pairs adapts the model's stylistic tone, instruction adherence, and high-level latent representations. However, memorizing exact, granular numeric facts (e.g., fee schedules, dates, tabular numbers) into weight parameters requires deep memorization that small language models ($1.5\text{B}$ parameters) struggle to achieve without gradient interference across layers. Empirical research (Gudibande et al., 2023; Gekhman et al., 2024) shows that small model SFT often increases overconfidence on unanswerable queries rather than true factual retention.
- **In My Project**: In our benchmark, System B achieved a Faithfulness Rate of $18.10\%$, identical to zero-context prompting. In `ERR_02`, when asked about Ph.D. stipends, System B confidently fabricated a ₹31,000 stipend because its weights generalized from public Indian university norms rather than knowing the handbook only covers undergraduate programs.
- **Examiner Follow-Up Trap**: *"Would fine-tuning for 50 epochs instead of 3 epochs solve this?"*  
  **Defense**: *"No, training for 50 epochs on 611 examples would lead to severe overfitting, loss of conversational fluency, and catastrophic forgetting of the base model's general reasoning abilities."*

---

### Q4: What is the "Lost-in-the-Middle" phenomenon in LLM context processing?
- **Technical Answer**: Discovered by Liu et al. (2023), the Lost-in-the-Middle phenomenon refers to the non-uniform degradation of attention mechanisms in Transformer architectures. LLMs recall information significantly better when relevant evidence appears at the very beginning (primacy effect) or the very end (recency effect) of the context window. Information buried in the middle $40\%\text{--}70\%$ of long context windows suffers from diffused self-attention weights.
- **In My Project**: We observed this in `ERR_07` and in our Top-$k$ ablation. When $k$ was expanded to 8 chunks ($1,760$ tokens), the generator LLM occasionally ignored key clauses in chunk 4 and 5, whereas setting $k=2$ or $3$ ensured the relevant passage occupied a high-attention position in the prompt.
- **Examiner Follow-Up Trap**: *"How did your architecture mitigate Lost-in-the-Middle?"*  
  **Defense**: *"We implemented two mitigations: (1) Section-Aware chunking that kept chunks concise and topically self-contained, and (2) Top-$k$ parameter tuning where $k=2$ achieved maximum accuracy ($93.3\%$) with minimal context padding."*

---

## Part 2: PEFT & QLoRA Mechanics

### Q5: Explain the mathematical foundation of LoRA (Low-Rank Adaptation).
- **Technical Answer**: During fine-tuning, standard backpropagation calculates a weight update matrix $\Delta W \in \mathbb{R}^{d \times k}$ for a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$. Aghajanyan et al. (2020) demonstrated that the intrinsic rank of these adaptation matrices is very low. LoRA (Hu et al., 2021) freezes $W_0$ and parameterizes $\Delta W$ as the product of two low-rank matrices:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  where $A \in \mathbb{R}^{r \times k}$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, $B \in \mathbb{R}^{d \times r}$ is initialized to zero (ensuring $\Delta W = 0$ at step 0), $r \ll \min(d, k)$ is the rank, and $\alpha$ is a constant scaling hyperparameter. The number of trainable parameters drops from $d \times k$ to $r(d + k)$, typically reducing parameters by over $99\%$.
- **In My Project**: For `Qwen2.5-1.5B`, targeting all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r=16$ and $\alpha=32$ reduced trainable parameters to approximately $18\text{M}$ out of $1.5\text{B}$ ($<1.2\%$).
- **Examiner Follow-Up Trap**: *"Why is $B$ initialized to zero while $A$ is Gaussian?"*  
  **Defense**: *"If both were Gaussian, $\Delta W$ would be non-zero at the start of training, distorting the pre-trained model's representations before the first gradient step. Initializing $B=0$ guarantees that $\Delta W = 0$, so training begins exactly at the pre-trained model's baseline performance."*

---

### Q6: How does QLoRA achieve 4-bit quantization without destroying model performance?
- **Technical Answer**: QLoRA (Dettmers et al., 2023) introduces three algorithmic innovations:
  1. **NF4 (NormalFloat4) Data Type**: An information-theoretically optimal quantile quantization scheme for zero-mean, unit-variance normally distributed weights, allocating more bit states to regions where weights concentrate.
  2. **Double Quantization (DQ)**: Quantizing the quantization constants themselves, compressing the 32-bit fp32 scaling constants to 8-bit integers, saving an additional $0.37$ bits per parameter.
  3. **Paged Optimizers**: Managing GPU memory spikes by leveraging CUDA Unified Memory to page optimizer states between GPU VRAM and CPU RAM during long sequence backward passes.
- **In My Project**: We configured QLoRA in `notebooks/colab_qlora_training.ipynb` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, and `bnb_4bit_compute_dtype=torch.bfloat16`, enabling the 1.5B model to be trained in just $4.2\text{ GB}$ of VRAM on a free Google Colab T4 GPU.
- **Examiner Follow-Up Trap**: *"Are calculations performed in 4-bit precision during forward pass?"*  
  **Defense**: *"No. The weights are stored in 4-bit NF4 in memory, but during forward and backward passes, they are dynamically dequantized into 16-bit brain floating-point (`bfloat16`) for matrix multiplications."*

---

### Q7: What is Catastrophic Forgetting, and how does PEFT mitigate it compared to full fine-tuning?
- **Technical Answer**: Catastrophic forgetting occurs when a neural network trained sequentially on new data overwrites weight representations critical for previously learned tasks (McCloskey & Cohen, 1989). In full fine-tuning, every parameter in $\theta$ is subject to gradient updates $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, which can drift significantly from pre-training points. PEFT mitigates this by freezing $W_0$ entirely. Because the pre-trained weights remain untouched, the base model's general syntax, reasoning, and world knowledge are permanently preserved.
- **In My Project**: Freezing the base weights of `Qwen2.5-1.5B-Instruct` ensured the model maintained its instruction-following format (`ChatML`) and grammar throughout the 3 epochs of training on the handbook dataset.

---

### Q8: What is ChatML format and why is it essential for fine-tuning conversational LLMs?
- **Technical Answer**: ChatML (Chat Markup Language) is a structured tokenization syntax that wraps conversational turns in explicit role boundary tokens:
  ```text
  <|im_start|>system
  {system_message}<|im_end|>
  <|im_start|>user
  {user_query}<|im_end|>
  <|im_start|>assistant
  {assistant_response}<|im_end|>
  ```
  These special tokens prevent prompt injection attacks (where user inputs attempt to spoof system instructions) and teach the model to distinguish between instructional guidance, user queries, and assistant generations.
- **In My Project**: Our dataset pipeline (`src/chunking/` and `scripts/run_data_pipeline.py`) formatted all 611 training pairs into ChatML JSONL records in `data/processed/sft_training_dataset.jsonl`.

---

## Part 3: Embeddings & Vector Search

### Q9: Why did you choose `BAAI/bge-small-en-v1.5` over older models like Word2Vec or TF-IDF?
- **Technical Answer**: Word2Vec produces static, context-independent vectors (e.g., the word "bank" has the same vector regardless of financial or river context). TF-IDF is a sparse bag-of-words representation that relies on exact lexical matches and cannot capture semantic equivalence (e.g., "penalties" vs "fines"). `BAAI/bge-small-en-v1.5` is a contextual transformer bi-encoder producing 384-dimensional dense vectors. Trained with contrastive learning on massive query-passage pairs, it maps syntactically dissimilar but semantically equivalent sentences into proximate points in vector space.
- **In My Project**: BGE-small maps colloquial student queries like *"What happens if I get caught copying in lab exam?"* to handbook clauses titled *"Academic Integrity and Examination Malpractice"* with high cosine similarity ($>0.78$), bridging the lexical gap.
- **Examiner Follow-Up Trap**: *"Why choose BGE-small over BGE-large or OpenAI text-embedding-3-large?"*  
  **Defense**: *"BGE-small has only 33M parameters and produces 384-dimensional vectors. It runs in under 10ms on CPU, requires zero cloud API cost, and achieved an outstanding 85.71% Hit@1 on our benchmark, making it ideal for low-cost, resource-constrained deployments."*

---

### Q10: What is the mathematical relationship between Euclidean Distance, Inner Product, and Cosine Similarity in FAISS?
- **Technical Answer**: The cosine similarity between vectors $\mathbf{u}$ and $\mathbf{v}$ is defined as:
  $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  If both vectors are $L_2$-normalized such that $\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$, then:
  $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \text{InnerProduct}(\mathbf{u}, \mathbf{v})$$
  Furthermore, the squared Euclidean distance relates to inner product by:
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2(\mathbf{u} \cdot \mathbf{v})$$
  Thus, maximizing cosine similarity on $L_2$-normalized vectors is mathematically equivalent to maximizing inner product and minimizing Euclidean distance.
- **In My Project**: In `src/retrieval/embeddings.py`, we apply `normalize_embeddings=True` in SentenceTransformers, and in `src/retrieval/vector_store.py` we use `faiss.IndexFlatIP` (Inner Product). This computes exact cosine similarity with maximum computational efficiency without sqrt division during search.

---

### Q11: What is the difference between FAISS `IndexFlatIP` and an approximate index like `IndexHNSW`?
- **Technical Answer**:
  - `IndexFlatIP`: An exact brute-force search index that computes the dot product of the query against every vector in the index. It guarantees $100\%$ recall (zero quantization error) with $O(N \cdot d)$ query complexity.
  - `IndexHNSW` (Hierarchical Navigable Small World): An approximate nearest neighbor (ANN) graph index. It organizes vectors into multi-layer proximity graphs, achieving $O(\log N)$ search time at the cost of higher memory overhead and $<100\%$ recall.
- **In My Project**: Since our handbook corpus comprises 25 section chunks (or 64 fixed-size chunks), $N$ is small. Brute-force `IndexFlatIP` executes in under $0.2\text{ milliseconds}$ while guaranteeing zero retrieval approximation loss.

---

### Q12: Define Hit@K and Mean Reciprocal Rank (MRR) mathematically.
- **Technical Answer**:
  - **Hit@K**: A binary indicator averaged across $Q$ queries measuring whether the ground truth section appears within the top-$k$ retrieved chunks:
    $$\text{Hit@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le K)$$
  - **Mean Reciprocal Rank (MRR)**: Measures the position of the first relevant chunk, rewarding higher ranks:
    $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
    where $\text{rank}_q$ is the rank of the first relevant document for query $q$ (and 0 if not found).
- **In My Project**: Across 70 answerable benchmark queries, Section-Aware chunking achieved $\text{Hit@1} = 85.71\%$ and $\text{MRR} = 0.9179$, outperforming Fixed-Size chunking ($\text{Hit@1} = 68.57\%$, $\text{MRR} = 0.8029$).

---

## Part 4: Basic RAG & Chunking Strategies

### Q13: Why did Section-Aware chunking achieve $+17.14\%$ higher Hit@1 than Fixed-Size chunking?
- **Technical Answer**: Fixed-size chunking divides text based on arbitrary character or token counts (e.g., 500 characters) regardless of document syntax. This leads to two critical failure modes:
  1. **Table Bisecting Artifacts**: Splitting Markdown table headers from row values, transforming structured numeric data into fragmented noise that weakens dense vector representations.
  2. **Boundary Chopping**: Splitting conditional clauses (e.g., separating an exception from its rule).
  Section-Aware chunking respects document semantics: each section (`## Chapter`, `### Section`) is processed as a unified thematic entity, preserving tabular integrity and contextual co-occurrence.
- **In My Project**: In `scripts/compare_chunking.py`, Section-Aware chunking preserved all 7 handbook tables intact, resulting in an $85.71\%$ Hit@1 vs $68.57\%$ for 500-char fixed chunking.

---

### Q14: What are Context Precision and Context Recall in RAG evaluation?
- **Technical Answer**:
  - **Context Recall**: Measures whether all the ground-truth information required to answer the question was successfully retrieved:
    $$\text{Context Recall} = \frac{|\text{Retrieved Relevant Chunks} \cap \text{Ground Truth Sections}|}{|\text{Ground Truth Sections}|}$$
  - **Context Precision**: Measures how clean the retrieved context is—specifically, whether relevant chunks appear at the highest ranks without irrelevant noise:
    $$\text{Context Precision} = \frac{\sum_{k=1}^K (\text{Precision}@k \times v_k)}{\text{Total Relevant Chunks}}$$
    where $v_k \in \{0, 1\}$ indicates relevance at rank $k$.
- **In My Project**: System C achieved an outstanding Context Recall of $98.73\%$ with Context Precision of $47.62\%$, demonstrating that while almost all relevant facts were retrieved, some irrelevant chunks were present in the top-3 window.

---

### Q15: Why is prompt injection or false premise a threat to Basic RAG?
- **Technical Answer**: Basic RAG is a purely feed-forward, single-pass pipeline: it retrieves top-$k$ chunks based on query similarity and injects them alongside the user query into the generator prompt. When an adversarial prompt contains a false premise (e.g., *"Since the Director can condone attendance to 50%, what is the form?"*), the generator's alignment to be helpful (sycophancy) often leads it to accept the user's premise as true and hallucinate an answer, ignoring the contradictory source text.
- **In My Project**: In `ERR_03`, Basic RAG scored $0.0\%$ accuracy on false-premise questions. Only System D (with its dedicated Abstention Evaluator node) successfully caught the contradiction and abstained, scoring $33.3\%$ on adversarial queries.

---

## Part 5: Agentic RAG, LangGraph & ReAct Tooling

### Q16: Explain the ReAct (Reasoning + Acting) paradigm implemented in System D.
- **Technical Answer**: Proposed by Yao et al. (2022), ReAct synergizes natural language reasoning traces with concrete execution actions:
  $$\text{Query} \to \text{Thought}_1 \to \text{Action}_1 \to \text{Observation}_1 \to \text{Thought}_2 \dots \to \text{Final Answer}$$
  By verbalizing a reasoning thought before taking an action, the model plans sub-goals. By observing execution output from external tools (retrievers, calculators), the model conditions subsequent reasoning on verifiable facts rather than internal hallucinated assumptions.
- **In My Project**: In System D (`src/systems/system_d_agentic_rag.py`), when asked: *"If a student has a ₹1,20,000 fee and withdraws on Day 18, how much refund do they receive?"*, the agent emits:
  - `Thought`: *"I need to retrieve the refund schedule from Chapter 6."*
  - `Action`: `handbook_retriever("tuition refund schedule withdrawal")`
  - `Observation`: *"Withdrawal between 16-30 days: 50% refund minus ₹5,000 processing fee."*
  - `Thought`: *"I will use the AST calculator to compute 120000 * 0.50 - 5000."*
  - `Action`: `ast_calculator("120000 * 0.50 - 5000")`
  - `Observation`: `55000`
  - `Thought`: *"The refund is ₹55,000. Synthesizing final response."*

---

### Q17: Why did you implement a safe AST Calculator tool instead of using Python's `eval()`?
- **Technical Answer**: Using `eval()` executes arbitrary Python code strings within the host process environment. If a malicious user injects prompt commands (e.g., `eval("__import__('os').system('rm -rf /')")`), the host system is compromised. Our AST Calculator parses expressions using Python's built-in `ast` module into an Abstract Syntax Tree. It recursively inspects nodes, strictly allowing only numeric literals (`ast.Constant`), unary operators (`ast.UAdd`, `ast.USub`), and binary operators (`ast.Add`, `ast.Sub`, `ast.Mult`, `ast.Div`). Any variable assignment, function call, or import raises a `ValueError`.
- **In My Project**: Implemented in `src/systems/system_d_agentic_rag.py` in the `SafeCalculator` class. It guarantees deterministic, secure arithmetic without any vulnerability to remote code execution.

---

### Q18: How do you prevent infinite reasoning loops in LangGraph agentic workflows?
- **Technical Answer**: In a cyclic state graph, an agent can get trapped in infinite reasoning loops if a tool repeatedly returns unexpected output or fails to satisfy a completion predicate. To guarantee termination, LangGraph workflows must implement:
  1. **Typed State with Iteration Counters**: Storing `iteration_count: int` inside the graph state.
  2. **Deterministic Conditional Edges**: A router function that checks if `state["iteration_count"] >= max_iterations`. If the threshold is exceeded, the edge overrides the agent's action and unconditionally forces a transition to the final synthesis or abstention node.
- **In My Project**: We enforced a hard guardrail of `max_iterations = 4` in `should_continue()` in `src/systems/system_d_agentic_rag.py`. If tool calls do not converge in 4 cycles, the system gracefully synthesizes the best available response, preventing runaway latency and API billing exhaustion.

---

## Part 6: Evaluation Harness & Bootstrap Statistics

### Q19: What is the LLM-as-a-Judge protocol and how did you prevent position/self-enhancement bias?
- **Technical Answer**: LLM-as-a-Judge (Zheng et al., 2023) uses an instruction-tuned LLM as an automated evaluator by providing it with a scoring rubric, the query, ground truth, and candidate answer. However, LLM judges can suffer from:
  - *Position Bias*: Favoring the first option in pairwise evaluations.
  - *Verbosity Bias*: Favoring longer responses regardless of conciseness.
  - *Self-Enhancement Bias*: Favoring outputs generated by the same model family.
- **In My Project**: We mitigated these biases through:
  1. **Independent Evaluation Judge**: Using a distinct model family for evaluation from the generator.
  2. **Single-Answer Pointwise Rubrics**: Scoring each answer independently against ground-truth facts rather than pairwise comparisons.
  3. **Binary Extraction Prompts**: Demanding structured JSON scores (`{"faithfulness": 1, "correctness": 1}`) with mandatory justification strings.
  4. **Deterministic Fallback**: Providing an automated rule-based judge for leak-free local grading and testing.

---

### Q20: Explain the statistical principle behind your 1,000-resample non-parametric Bootstrap Confidence Intervals.
- **Technical Answer**: Standard parametric confidence intervals assume that sample means follow a normal distribution governed by the Central Limit Theorem ($\bar{x} \pm 1.96 \frac{s}{\sqrt{n}}$). However, evaluation metrics like Faithfulness and Correctness are discrete, binary, and highly skewed. Non-parametric bootstrap resampling (Efron & Tibshirani, 1993) makes no distributional assumptions. Given a sample of size $N=105$, we draw $B=1,000$ bootstrap replicates $\mathbf{x}^{*b}$ with replacement, each of size $N$. We compute the metric estimate $\hat{\theta}^{*b}$ for each replicate. The $95\%$ bootstrap confidence interval is given by the empirical $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles:
  $$CI_{95\%} = \left[ \hat{\theta}^*_{(25)}, \hat{\theta}^*_{(975)} \right]$$
- **In My Project**: Implemented in `src/evaluation/bootstrap_ci.py`. For System D, the empirical bootstrap interval for Faithfulness was $[42.86\%, 61.93\%]$, providing rigorous statistical proof that System D significantly outperforms System A ($[10.48\%, 25.71\%]$) with non-overlapping confidence bounds.

