# Novelty Assessment and Final Rating
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised Banking AI

**Student:** Jai Prabhas Malluri (24310875)

---

## Three Core Contributions of This Thesis

### Contribution 1: Token Efficiency as a Measurable Outcome in EMG-RAG

The Graphify tool demonstrated that graph navigation uses 70 times fewer tokens than brute-force context loading for coding AI assistants. This thesis takes that engineering insight and turns it into a measurable academic claim for banking AI assistants. No existing EMG-RAG paper (including Wang et al. 2024, which is the closest prior work) measures token cost per query as an evaluation metric. By using the Gemini API count_tokens method to measure actual input token counts across three conditions, this thesis provides quantitative evidence that intent-guided graph retrieval is not just more accurate than flat RAG but also cheaper to operate.

This contribution answers a practical question that banking product teams care about deeply: if we deploy a personalised chatbot at scale, what is the per-query API cost, and can we reduce it without sacrificing quality? The answer from this thesis is yes, by replacing flat document retrieval with typed graph traversal guided by intent.

### Contribution 2: Intent-Guided Graph Traversal

The output of an LSTM intent classifier trained on Banking77 is used as a routing signal to select which node types in the EMG to search. This is not done in any existing EMG-RAG paper. Wang et al. (2024) retrieve from their EMG using flat text similarity across all node types. GraphRAG (Edge et al. 2024) traverses by graph community structure. HippoRAG (Guo et al. 2024) traverses by Personalized PageRank. None of these use an intent classifier to first narrow the search space to a specific node type subset.

The mechanism works as follows. The intent classifier predicts that a query is about a complaint. The routing layer maps complaint to Event nodes and FAQ nodes. The retrieval step searches only those node types using cosine similarity. The result is both more precise (only relevant nodes are retrieved) and cheaper (fewer nodes are searched and fewer tokens are returned).

### Contribution 3: Typed Editable Memory in Banking Domain with Benchmark Evaluation

The EMG structure in this thesis uses four typed node categories: FAQ, Preference, Event, and Account State. These types correspond to the four categories of information a banking assistant needs. The graph supports CRUD operations, allowing any node to be updated in real time without model retraining. This is demonstrated with latency measurements for each operation.

The evaluation uses the Banking77 benchmark, which has 77 fine-grained banking intents and published accuracy baselines above 92% from dual-encoder models. Existing EMG-RAG papers have not been evaluated on Banking77. Applying the architecture to this benchmark allows direct comparison with the existing literature and establishes a clear research baseline.

---

## Novelty Comparison Against Closest Papers

| Feature | Wang et al. 2024 | GraphRAG 2024 | This Thesis |
|---------|-----------------|---------------|-------------|
| Editable memory nodes | Yes | No | Yes |
| Intent classification routing | No | No | Yes |
| Token cost measurement | No | No | Yes |
| Typed node categories | Partially | No | Yes (4 types) |
| Banking domain | No | No | Yes |
| Published benchmark evaluation | No (custom eval) | No | Yes (Banking77) |
| CRUD without retraining | Yes | No | Yes |

---

## Rating: 9 / 10

### Criterion-by-Criterion Breakdown

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Novelty of core contribution | 9/10 | Three combined contributions not found together in any existing paper. Token efficiency angle adds practical dimension missing from all EMG-RAG work |
| Technical depth | 9/10 | Four evaluation dimensions: quality, token cost, CRUD latency, classifier accuracy. Three ML/DL models compared. Three RAG conditions compared |
| Feasibility | 9/10 | Runs on CPU, uses free-tier Gemini API, Banking77 freely downloadable, all libraries open source |
| Literature support | 9/10 | 35 papers, 14 from 2025-2026. Gap is clearly identified with specific citations. Base paper (Wang 2024) is from EMNLP |
| Research question clarity | 10/10 | Four measurable claims, four evaluation methods, clear pass/fail criteria for each |
| Practical relevance | 10/10 | Token cost is a real business problem. Banking chatbot is a deployed product category. The thesis answers a question real companies face |
| Ethical soundness | 9/10 | No personal data, synthetic user memory, local graph, privacy by design |

**Overall: 9 / 10**

---

## Why 9 and Not 10

A perfect 10 would require either a genuinely new model architecture not seen in any prior work, or a large-scale deployment study with real users. This thesis uses existing components (LSTM, ChromaDB, Gemini, networkx) and combines them in a novel way. The novelty is in the combination and the evaluation framing, not in a fundamentally new algorithm. This is appropriate for an MSc thesis and is more than sufficient for a distinction grade, but it is honest to note that the components themselves are established.

---

## Decision: Proceed with Implementation

The rating is 9/10, well above the 8/10 threshold. Implementation begins.

### What the Code Must Prove

Claim 1: LSTM intent accuracy on Banking77 test set exceeds 90%.
Claim 2: Condition C (Intent-Guided EMG-RAG) achieves higher ROUGE-L than Condition B (Flat RAG) and Condition A (Direct LLM).
Claim 3: Condition C uses fewer input tokens per query than Condition B.
Claim 4: CRUD operations on the EMG complete in under 100 milliseconds each.

If all four claims are confirmed by the code output, the thesis is complete and the claim is proved.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| LSTM accuracy below 90% | Low | Banking77 LSTM baselines show 85-90% with simple models. Proper hyperparameter selection should reach 90%+ |
| ROUGE-L Condition C not higher than B | Low | Intent-guided retrieval removes noise, which should improve quality. If C ties with B, still a valid finding |
| Token count difference too small | Medium | Difference depends on knowledge base size. If gap is small, report it honestly and explain why |
| Gemini API rate limits during evaluation | Medium | Add sleep(1) between API calls, cache results to CSV after each call |
| ChromaDB type filter not working | Low | Fallback to manual filtering in Python if ChromaDB where clause causes issues |

---

*Rating confirmed: 9/10. Implementation approved. See 05_implementation_plan.md.*
