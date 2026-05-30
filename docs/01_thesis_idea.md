# Thesis Idea Document — Final Version
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised AI Assistants
# An Intent-Guided Retrieval-Augmented Generation Approach

**Student:** Jai Prabhas Malluri (24310875)
**Programme:** MSc in Open Data Practice, National College of Ireland
**Module:** Research Practicum (MSCODP1)

---

## 1. What Problem I Am Solving

The thesis addresses a gap that exists at the intersection of three real engineering problems in personalised AI assistants. I explain each problem separately and then show how my system solves all three together.

### Problem 1 — Token Exhaustion in Context Loading

When an AI assistant answers a question, the naive approach loads everything it knows about the user into the prompt before generating a response. For a real personal assistant that knows about a user's flights, bills, medication reminders, dietary preferences, work schedule, and upcoming events, this context becomes large very quickly. Every single API call to a large language model like Gemini charges per token. Sending 400-700 tokens of context for a question that only needs 2 relevant facts is wasteful and does not scale.

This problem was identified practically by a tool called Graphify, which demonstrated that AI coding assistants using brute-force context loading exhaust token budgets rapidly. Graphify solved this by building a Knowledge Graph of the codebase once and then traversing only the relevant part of the graph per query, reducing token use by orders of magnitude.

My thesis takes this engineering insight and applies it rigorously to a personalised AI assistant setting. I measure whether structured graph retrieval (2 targeted nodes per query) actually uses fewer tokens than flat document retrieval (3 randomly similar documents) while maintaining or improving response quality.

### Problem 2 — Dynamic User Memory Cannot Be Fine-Tuned Into a Model

A personalised assistant needs to know things that change all the time. A user books a flight. A complaint gets resolved. A dietary preference changes from vegetarian to vegan. A monthly rent amount increases.

Traditional personalisation approaches embed these facts into model weights through fine-tuning. This approach has two fundamental problems. First, retraining a large language model costs significant compute. Second, and more importantly, you cannot update one fact without risking the model forgetting others, a phenomenon known as catastrophic forgetting.

An Editable Memory Graph (EMG) stores user memory in graph nodes that support insert, update, and delete operations. When the user's phone number changes, one node is updated in under 30 milliseconds. The next query immediately retrieves the corrected fact. No model is retrained.

### Problem 3 — Flat Retrieval Fails When Intent Differs But Text Looks Similar

Standard retrieval-augmented generation retrieves documents based on semantic text similarity. In a personal assistant context, queries that share vocabulary can require completely different contextual information.

For example, "what is my account balance" and "can you report a fraud on my account" both mention account but need different information. The balance query needs the AccountState node with current balance. The fraud query needs the Event node with the open dispute case and the FAQ node with the fraud reporting process.

Flat similarity search returns overlapping document sets for both queries, mixing irrelevant context into the prompt, wasting tokens, and degrading response quality. The solution is to first classify the intent of the query, then use that intent to select which region of the memory graph to search.

---

## 2. My Proposed System

The system has three cooperating layers, each solving one of the three problems.

**Layer 1 — Intent Classification (solves Problem 3)**
Three models are trained to classify the intent of a user query: Logistic Regression, Random Forest, and Bidirectional LSTM. The predicted intent is not just a label. It is a routing signal that selects which node types in the memory graph are searched.

**Layer 2 — Editable Memory Graph (solves Problem 2)**
A typed knowledge graph stores user memory in four node categories:
- FAQ nodes: general policy and domain knowledge (static, rarely changes)
- Preference nodes: user-specific settings (communication channel, dietary needs, commute route)
- Event nodes: time-stamped personal events (flights booked, bills due, medication doses, appointments)
- AccountState nodes: current user profile facts (balance range, PTO days, health insurance plan)

Nodes can be inserted, updated, or deleted without retraining any model. The graph is built once from the user's profile and updated in real time as the user's life changes.

**Layer 3 — Token-Aware RAG with Gemini (solves Problem 1)**
After the intent classifier selects which node types to search, the system retrieves 2 targeted nodes from the relevant graph region. These are passed as context to Gemini 2.5 Flash. The actual token count sent per query is measured using Gemini's count_tokens API and compared against two baselines: a direct LLM call with no context, and a flat RAG system that retrieves 3 documents without intent filtering. The thesis proves that targeted graph retrieval sends fewer tokens while producing better quality responses.

---

## 3. Research Evolution — How I Got to This Design

My thesis went through three versions as the dataset choice and problem framing became clearer. I document this progression because it represents the actual research process of narrowing from a broad idea to a precise contribution.

### Version 1 — Banking77 (initial, abandoned)

The first version used Banking77, a dataset of 13,083 customer service queries across 77 banking intents. I trained Logistic Regression, Random Forest, and LSTM classifiers and combined them with a ChromaDB-backed RAG pipeline using Gemini.

**Why I moved on:** Banking77 covers only one domain, banking. A personalised AI assistant is fundamentally a multi-domain system. Using a single-domain benchmark meant the EMG could not demonstrate its core value: routing different life-domain queries to different typed graph regions. A system that only handles banking queries does not need a multi-typed memory graph. The research contribution was not visible with this dataset.

### Version 2 — CLINC150 (upgraded, used for classification benchmarking)

The second version switched to CLINC150, a benchmark of 22,500 queries across 150 intents spanning 10 life domains: banking, travel, auto, calendar, home, dining, health, work, utility, and meta. This directly matches the multi-domain personal assistant problem.

**What CLINC150 contributes:** CLINC150 is a standard academic benchmark with published baseline accuracies above 92 percent from dual-encoder models (Casanueva et al. 2020, Cho et al. 2025). Training the intent classifiers on CLINC150 gives academically comparable results and proves the system's routing signal is reliable on a real-world query distribution.

**What CLINC150 cannot do:** CLINC150 provides intent labels but no personalised context and no ground truth answers tied to a specific user's memory. It cannot test whether the EMG returns the right nodes for a specific user's situation, whether CRUD updates are immediately reflected in responses, or whether intent-guided retrieval uses fewer tokens than flat retrieval. The personalised evaluation component requires a different approach.

### Version 3 — Hybrid (final: CLINC150 for classification + Synthetic for RAG evaluation)

The final version uses two data sources with different roles:

**CLINC150 for intent classification training and benchmarking:**
- Trains LR, RF, and BiLSTM on 18,350 real queries across 150 intents
- Accuracy compared against published dual-encoder baselines
- Proves the routing signal generalises to real-world query patterns
- Academically grounded, reproducible, peer-reviewed benchmark

**Synthetic data for personalised RAG evaluation:**
- 20 evaluation pairs created to specifically test the three core problems
- Each pair has: a query, a target intent, a user-personalised ground truth answer
- Ground truth answers are grounded in the EMG node content for a specific synthetic user (Priya)
- Covers all 6 life domains: Finance, Travel, Health, Calendar, Work, Lifestyle
- Designed to show precisely where EMG-RAG beats flat RAG and why

---

## 4. Why Synthetic Data for the RAG Evaluation Is Justified

This is an important point to address clearly because examiners may ask about it.

**Reason 1 — No existing dataset captures personalised memory evaluation.**
RAG evaluation benchmarks (TriviaQA, Natural Questions, MS-MARCO) test retrieval from static knowledge bases. Memory agent benchmarks (LoCoMo, LongMemEval) test long-context dialogue. No published dataset tests retrieval from a dynamic typed user memory graph with ground truth answers that reflect a specific person's live data. The evaluation scenario is genuinely novel, requiring a novel evaluation instrument.

**Reason 2 — Synthetic evaluation is used in the closest prior work.**
Wang et al. (2024), the EMNLP paper that introduced Editable Memory Graphs, evaluated their system using simulated user scenarios rather than a public benchmark. This is because the problem of personalised memory evaluation inherently requires user-specific ground truth that cannot come from a generic corpus.

**Reason 3 — The synthetic evaluation is reproducible and transparent.**
The ground truth answers are generated directly from the EMG node content in the code. Any evaluator can read the node content, read the expected answer, and verify that the answer is grounded in the retrieved context. This is more transparent than benchmark datasets with opaque annotation processes.

**Reason 4 — The two-dataset design is a methodological strength, not a weakness.**
Using CLINC150 for classification gives credibility through comparison with published baselines. Using synthetic data for the personalised evaluation gives control and precision in demonstrating the three core problems. The separation of these two evaluation concerns makes the contribution clearer, not weaker.

---

## 5. Research Questions

**Primary:** Does intent-guided Knowledge Graph retrieval (EMG-RAG) produce higher quality personalised responses with fewer input tokens compared to flat RAG and direct LLM generation?

**Secondary 1:** Which intent classifier (LR, RF, BiLSTM) achieves the highest accuracy on CLINC150, and how does each compare to published baselines?

**Secondary 2:** Can graph CRUD operations update user memory in real time without retraining any model, and what is the latency of each operation type?

---

## 6. What the Thesis Proves

Four measurable claims, each verified by the code:

| Claim | Measurement | Target |
|-------|-------------|--------|
| Intent classification quality | Accuracy on CLINC150 test set | Best model > 90% |
| EMG-RAG response quality | ROUGE-L and BLEU vs ground truth | Condition C > Condition B |
| Token efficiency | Gemini count_tokens API per query | Condition C tokens < Condition B |
| Memory update speed | CRUD operation latency in milliseconds | All operations < 100ms |

---

## 7. Novel Contribution (Summary)

Three things that are not done together in any existing paper:

1. Intent classification output used as a typed retrieval routing signal in an EMG-RAG pipeline (Wang et al. 2024 use flat similarity retrieval with no intent layer)

2. Token cost measured as an evaluation metric in a personalised AI assistant (Graphify identified the problem in practice; this thesis measures and proves the claim academically)

3. Typed editable memory graph evaluated on a two-dataset hybrid design: real benchmark for classification accuracy, synthetic personalised scenarios for RAG quality and token efficiency

---

*Jai Prabhas Malluri - 24310875 - Thesis Idea v3 Final*
