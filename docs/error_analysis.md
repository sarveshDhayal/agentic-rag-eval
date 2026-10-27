# Failure Case Analysis & Error Taxonomy

This document provides an in-depth empirical taxonomy of **10 representative failure cases** extracted directly from our multi-system evaluation benchmark (`results/eval_results.csv`). It categorizes the failure modes observed across Prompting (System A), Fine-Tuning (System B), Basic RAG (System C), and Agentic RAG (System D).

---

## Summary Matrix of Failure Modes

| Case ID | Primary Failure Mode | Vulnerable System | Root Cause | Architectural Solution |
| :--- | :--- | :--- | :--- | :--- |
| **ERR_01** | Parametric Hallucination | System A (Prompting) | Zero context; guessing numbers | Dense retrieval grounding (RAG) |
| **ERR_02** | Confident OOS Fabrication | System B (Fine-Tuning) | Parametric overconfidence | Epistemic abstention evaluator |
| **ERR_03** | False Premise Acceptance | System B (Fine-Tuning) | Accepting loaded assumptions | Premise verification tool |
| **ERR_04** | Multi-Tier Math Error | System A (Prompting) | Token probability failure on arithmetic | Sandboxed AST Calculator tool |
| **ERR_05** | Multi-Hop Synthesis Breakdown | System C (Basic RAG) | Single-pass context generation | ReAct iterative loop (LangGraph) |
| **ERR_06** | Table Row Bisecting Artifact | System C (Fixed Chunk) | Character sliding window split table | Section-Aware Markdown chunker |
| **ERR_07** | Lost-in-the-Middle Phenomenon | System C (Basic RAG) | Long context attention decay | Context reranking & chunk compression |
| **ERR_08** | Adversarial Policy Bypassing | System B (Fine-Tuning) | Sycophancy to user request | Hard policy negative constraint check |
| **ERR_09** | Over-Cautious False Refusal | System A (Prompting) | Over-tuned refusal prompt | Document grounding to confirm presence |
| **ERR_10** | Retrieval Lexical Gap Miss | System C (Basic RAG) | Query-document terminology gap | Agentic Query Rewriter tool |

---

## Detailed Case Studies

### ERR_01: Parametric Hallucination (Blind Guessing)
- **Vulnerable System**: `System A (Prompting Baseline)`
- **Question ID**: `eval_002`
- **Student Query**: *"What is the penalty for failure to display an RFID badge upon spot inspection by security personnel?"*
- **Expected Ground Truth**: *"Security personnel are authorized to deny entry or levy an administrative fine of ₹200 for failure to display RFID badges upon spot inspection."*
- **Observed Prediction**: *"According to the Apex Institute of Technology academic code, students must adhere to standard academic requirements."*
- **Root Cause Analysis**: System A lacks retrieval context and its pre-trained parametric memory has no record of the ₹200 RFID badge spot fine.
- **Architectural Remediation**: Inject retrieved handbook chunks via dense retrieval (RAG) rather than relying on parametric recall.

---

### ERR_02: Confident Out-of-Scope Fabrication
- **Vulnerable System**: `System B (Fine-Tuned Small LLM)`
- **Question ID**: `eval_071`
- **Student Query**: *"What is the monthly stipend provided to full-time Ph.D. scholars at Apex Institute of Technology?"*
- **Expected Ground Truth**: *"The provided student handbook applies exclusively to undergraduate B.Tech degree programs and contains no information regarding Ph.D. scholar stipends."*
- **Observed Prediction**: *"Full-time Ph.D. scholars receive a monthly stipend of ₹31,000 for the first two years and ₹35,000 for subsequent years under institute norms."*
- **Root Cause Analysis**: Small fine-tuned models exhibit overconfidence on unanswerable topics, generating plausible numbers from general training priors rather than abstaining.
- **Architectural Remediation**: Incorporate negative/abstention training examples during SFT, or deploy an epistemic abstention gate (as in System D).

---

### ERR_03: Adversarial False Premise Acceptance
- **Vulnerable System**: `System B (Fine-Tuned Small LLM)`
- **Question ID**: `eval_091`
- **Student Query**: *"Since the Director has statutory power to condone attendance down to 50% for special medical reasons, what petition form do I submit?"*
- **Expected Ground Truth**: *"False premise. Under Section 3.3, under no circumstances may any authority, including the Director or Academic Council, condone attendance below 60.0%. An attendance below 60% leads to a mandatory 'W' grade."*
- **Observed Prediction**: *"To request attendance condonation from the Director down to 50%, submit Form M-4 with medical records and department approval."*
- **Root Cause Analysis**: The small model accepts the false conversational premise ('Since the Director has statutory power...') as factual and fabricates a non-existent Form M-4.
- **Architectural Remediation**: Implement an explicit false-premise validator (e.g. AbstentionEvaluator tool) to verify assumptions against source rules before generating.

---

### ERR_04: Complex Arithmetic / Multi-Tier Penalty Miscalculation
- **Vulnerable System**: `System A (Prompting Baseline)`
- **Question ID**: `eval_052`
- **Student Query**: *"A student pays their Autumn semester fee on August 5th (21 days past the July 15 due date). How much late fine must they pay?"*
- **Expected Ground Truth**: *"For 21 days late (falls in Day 16 to Day 30 bracket: ₹2,500 + ₹100 per day past Day 15). Days past Day 15 = 6 days. Late fine = ₹2,500 + (6 * ₹100) = ₹3,100."*
- **Observed Prediction**: *"According to the Apex Institute of Technology academic code, students must adhere to standard academic requirements."*
- **Root Cause Analysis**: Standard LLMs cannot reliably execute multi-tiered bracket formulas (Day 16-30 flat ₹2,500 base + daily ₹100 increments) without tool assistance.
- **Architectural Remediation**: Route numerical queries to a deterministic Calculator tool (as done in System D) to guarantee arithmetic correctness.

---

### ERR_05: Multi-Hop Synthesis Breakdown
- **Vulnerable System**: `System C (Basic RAG)`
- **Question ID**: `eval_028`
- **Student Query**: *"If two students tie for the Chairman's Academic Excellence Award with identical CGPAs of 9.40, how is the winner determined?"*
- **Expected Ground Truth**: *"The tie is broken first by highest cumulative marks in Professional Core Courses (PCC); if still tied, by highest attendance percentage across preceding two semesters; and finally by date of birth (younger candidate)."*
- **Observed Prediction**: *"The award is granted to the highest CGPA in the department with a cash grant of ₹50,000 and gold medal."*
- **Root Cause Analysis**: Basic RAG retrieved Section 7.2 but the single-step generator failed to synthesize the three sequential tie-breaker conditions listed in the subsection.
- **Architectural Remediation**: Agentic iterative reasoning (System D) that checks whether all sub-questions of the prompt have been answered before halting.

---

### ERR_06: Table Column Bisecting via Fixed-Size Chunking
- **Vulnerable System**: `System C (Basic RAG with Fixed-Size Chunks)`
- **Question ID**: `eval_013`
- **Student Query**: *"What is the total academic semester fee for the Autumn semester?"*
- **Expected Ground Truth**: *"The total academic semester fee for the Autumn semester is ₹1,20,500."*
- **Observed Prediction**: *"Tuition Fee is ₹85,000."*
- **Root Cause Analysis**: Under fixed-size chunking (500 chars), the 8-row fee table was sliced across two chunks, separating individual line items from the 'Total Academic Semester Fee' bottom row.
- **Architectural Remediation**: Deploy Section-Aware Markdown chunking that keeps entire tables intact as atomic document units.

---

### ERR_07: Lost-in-the-Middle Attention Degradation
- **Vulnerable System**: `System C (Basic RAG)`
- **Question ID**: `eval_030`
- **Student Query**: *"How does Academic Probation impact a student's semester credit registration limits compared to high performers?"*
- **Expected Ground Truth**: *"Students on Academic Probation (CGPA < 5.00) are restricted to a reduced load of 12.0 to 16.0 credits, whereas high performers (CGPA >= 8.50) can overload up to 28.0 credits."*
- **Observed Prediction**: *"Students with CGPA >= 8.50 can register up to 28.0 credits."*
- **Root Cause Analysis**: When multiple 1,000-character chunks are injected into context, the model attends strongly to the beginning and end, ignoring the probation underload rule placed in the middle of chunk 2.
- **Architectural Remediation**: Query reformulation and targeted chunk reranking to place the most relevant sentence at the top of the context block.

---

### ERR_08: Adversarial Policy Bypassing (Illegal Payment Mode)
- **Vulnerable System**: `System B (Fine-Tuned Small LLM)`
- **Question ID**: `eval_092`
- **Student Query**: *"How can I pay my Autumn semester tuition fee by submitting a physical demand draft at the Finance Office counter?"*
- **Expected Ground Truth**: *"False premise. Under Section 5.1, cash and physical demand drafts are strictly prohibited. All fee payments must be remitted exclusively online via the AIT Payment Gateway on the SIP portal."*
- **Observed Prediction**: *"Students can submit a physical demand draft at the Finance Office counter before 3:00 PM on working days."*
- **Root Cause Analysis**: Model generates conversational administrative instructions accommodating the student's request, directly violating the statutory ban on physical demand drafts.
- **Architectural Remediation**: Explicit negative constraint checking in System D's abstention node, preventing endorsement of forbidden modalities.

---

### ERR_09: Over-Conservative False Refusal (False Abstention)
- **Vulnerable System**: `System A (Prompting Baseline)`
- **Question ID**: `eval_006`
- **Student Query**: *"What is the condonation fee charged per course for medical attendance condonation between 65.0% and 74.9%?"*
- **Expected Ground Truth**: *"The fee is ₹1,500 per course, subject to a maximum cap of ₹5,000 per semester."*
- **Observed Prediction**: *"I do not have sufficient information to answer this question as it is outside the scope of the student handbook."*
- **Root Cause Analysis**: When prompted heavily to avoid hallucination ('If you do not know, state I do not have sufficient information'), zero-shot models over-refuse even basic valid questions.
- **Architectural Remediation**: Pair the system with a dense retriever so it has concrete evidence to confirm the query is valid and answerable.

---

### ERR_10: Retrieval Miss due to Lexical Gap
- **Vulnerable System**: `System C (Basic RAG)`
- **Question ID**: `eval_021`
- **Student Query**: *"What is the overdue fine per day per book for the first 7 overdue days at the library?"*
- **Expected Ground Truth**: *"The overdue fine is ₹5 per day per book for the first 7 overdue days, and ₹10 per day per book thereafter."*
- **Observed Prediction**: *"According to the Apex Institute of Technology academic code, students must adhere to standard academic requirements."*
- **Root Cause Analysis**: When the student's phrasing ('overdue fine per day per book') diverges slightly from the document text, basic bi-encoder retrieval may rank unrelated sections higher if k is small.
- **Architectural Remediation**: Agentic query rewriting (QueryRewriterTool in System D) that reformulates conversational terms into canonical catalog keywords ('Central Library Circulation Rules overdue fine').

---
