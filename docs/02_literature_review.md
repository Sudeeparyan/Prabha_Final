# Literature Review
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised Banking AI

**Student:** Jai Prabhas Malluri (24310875)
**NCI MSc Open Data Practice - Research Practicum MSCODP1**

Total papers reviewed: 35 (14 from 2025-2026)

---

## 1. Introduction to the Review

This thesis proposes a Knowledge Graph-based Editable Memory system for a personalised banking AI assistant that reduces token cost while improving response quality through intent-guided retrieval. The literature review is organised across five thematic areas that directly support this proposal. The first area covers retrieval-augmented generation and the problem of token inefficiency in context loading. The second covers graph-based knowledge retrieval, including the Graphify-inspired insight that structured graphs allow cheaper and more precise query resolution than flat document search. The third covers editable and dynamic memory systems for AI agents. The fourth covers intent classification using machine learning and deep learning models. The fifth covers banking-specific NLP applications. A final section identifies the research gap and positions this thesis contribution.

The review draws on 35 papers spanning 2014 to 2026, with 14 papers from 2025 and 2026 to demonstrate currency of the research problem and solution space.

---

## 2. Retrieval-Augmented Generation and the Token Cost Problem

The RAG paradigm was established by Lewis et al. (2020), who demonstrated that combining a dense passage retriever with a parametric language model significantly improved performance on knowledge-intensive NLP tasks. Their formulation showed that non-parametric external memory could ground generation in factual evidence, reducing hallucination compared to relying entirely on model weights. This work is the foundation of the approach used in this thesis.

However, the original RAG formulation and most of its successors treat the knowledge base as a flat collection of documents retrieved by text similarity. Gao et al. (2024) reviewed RAG systems and categorised them into three paradigms: Naive RAG, which retrieves a fixed number of documents regardless of query type; Advanced RAG, which adds query rewriting and result reranking; and Modular RAG, which decomposes the pipeline into independently optimised components. This taxonomy is directly applicable to the architecture in this thesis, which extends Modular RAG with an intent classification module that precedes retrieval.

Shuster et al. (2024) surveyed RAG evolution from 2020 to 2024 and identified the static knowledge base assumption as a critical limitation for personalised assistant applications, where user-relevant information changes continuously. This observation directly motivates the editable memory graph used in this thesis.

Asai et al. (2023) introduced Self-RAG, which gives the LLM reflection tokens that allow it to decide when to retrieve and to critique its own generated output. Self-RAG reduced hallucination by making retrieval adaptive rather than automatic. This is conceptually relevant to the confidence-checking mechanism used in this thesis, where the generated response is checked before being delivered to the user.

Singh et al. (2025) surveyed agentic RAG systems where planning and tool selection are embedded in the retrieval pipeline. These systems can decide which retrieval action to take for a given query, which parallels the intent-guided routing in this thesis. The difference is that agentic RAG uses an LLM as the planner, while this thesis uses a lightweight fine-tuned LSTM, keeping computation costs low.

A critical problem identified in multiple papers is what Liu et al. (2024) called Lost in the Middle: when relevant information is embedded inside a long retrieved context, LLMs systematically perform worse at extracting it than when the relevant information is at the beginning or end of the prompt. This finding directly motivates the targeted retrieval approach in this thesis, where only 2 to 3 relevant nodes are retrieved rather than a large undifferentiated context block.

Shi et al. (2023) showed that irrelevant context passages distract LLMs even when the correct information is present. They measured a consistent performance drop when irrelevant documents were included alongside relevant ones. This finding supports the argument that intent-guided retrieval, which removes irrelevant node types from the search space, produces better quality responses not just by adding the right information but by removing the wrong information.

---

## 3. Graph-Based Knowledge Retrieval and Token Efficiency

The insight that structured knowledge graphs allow cheaper and more targeted retrieval than flat document search has been independently rediscovered in several lines of research.

Edge et al. (2024) proposed GraphRAG, which builds an entity relationship knowledge graph from source documents using LLM extraction, then uses Leiden community detection to create hierarchical summaries. GraphRAG substantially outperformed standard RAG on global sensemaking queries that require aggregating information across many documents. The core insight is that graph structure captures relationships between concepts that text similarity alone cannot recover.

Guo et al. (2024) introduced HippoRAG, which combines LLM-built schemaless knowledge graphs with Personalized PageRank to mimic hippocampal memory indexing. HippoRAG outperformed state-of-the-art methods by up to 20% on multi-hop QA tasks. Their work showed that graph traversal from a starting node can surface related information that is not directly similar to the query but is structurally connected through the graph. This is the mechanism underlying the intent-to-node-type routing in this thesis.

Gutierrez et al. (2025) extended HippoRAG with deeper passage integration, framing RAG over knowledge graphs as non-parametric continual learning. Their work explicitly positioned graph-based retrieval as a cheaper alternative to fine-tuning for dynamically changing knowledge, which directly supports the thesis argument that EMG-RAG is preferable to LLM fine-tuning for dynamic user data.

Xu et al. (2026) provided the first comprehensive taxonomy of graph-based agent memory, covering entity relationship encoding, hierarchical semantics, and flexible reasoning traversal. Their taxonomy positions the EMG in this thesis as a typed heterogeneous graph, distinguished from unstructured knowledge graphs by its node type system and distinguished from flat vector stores by its relational structure.

Li et al. (2026) proposed GAM, which decouples episodic event memory from semantic preference memory in a hierarchical graph structure. Their finding that separating memory by type reduces interference is directly reflected in the EMG design in this thesis, which uses separate node types for FAQ, Preference, Event, and Account State information.

Chen et al. (2025) proposed KG-RAG, combining dense retrieval with structured knowledge graph retrieval without any training or fine-tuning. Their results showed consistent improvements over pure vector search, supporting the choice of structured typed nodes over a flat ChromaDB store.

Li et al. (2024) won third place at the Meta KDD CUP 2024 CRAG benchmark with WeKnow-RAG, which routes queries adaptively between web search and structured knowledge graph retrieval. Their success with adaptive routing supports the argument that knowing which retrieval source to use for a given query is as important as the quality of the retrieval itself.

Zhao et al. (2025) proposed D-SMART, combining graph-based structured memory with reasoning trees for extended multi-turn dialogue. Their finding that structured memory produces more consistent dialogue than flat memory stores is directly relevant to the choice of EMG over a simple ChromaDB collection.

---

## 4. Editable and Dynamic Memory for AI Agents

Wang et al. (2024) is the single most relevant existing work to this thesis. They proposed the Editable Memory Graph and combined it with RAG for personalised smartphone agents. Their EMG supports three operations: memory insertion, deletion, and replacement. They used reinforcement learning to optimise the retrieval policy. Their work demonstrated that structured editable memory outperforms flat memory approaches on personalisation quality. However, their retrieval was based entirely on text similarity without any intent classification routing layer, and their evaluation used a general-purpose assistant scenario rather than a banking domain with a published benchmark. This thesis extends Wang et al. by adding intent-guided routing, measuring token efficiency, and evaluating on Banking77.

Kang et al. (2025) proposed MemoryOS, an operating system inspired hierarchical memory system with short-term, mid-term, and long-term memory tiers managed by heat-driven eviction. Their OS analogy is appropriate: just as an operating system moves frequently used pages closer to the CPU, MemoryOS moves recently accessed memories closer to the active context. The persona module in MemoryOS is analogous to the Preference nodes in the EMG used in this thesis.

Liu et al. (2025) presented Reflective Memory Management at ACL 2025, achieving over 10% accuracy gains on LongMemEval by combining prospective summarisation with retrospective RL-based retrieval refinement. Their retrospective component updates the retrieval policy based on past retrieval quality, which is conceptually related to the intent-to-node-type mapping in this thesis. The mapping in this thesis is hand-crafted based on domain knowledge rather than learned, which is appropriate for an MSc thesis scope.

Chhikara et al. (2025) introduced Mem0, a production-ready graph-enhanced memory system achieving 91% lower latency and 90% token savings compared to full-context approaches. Their comparison of ten memory approaches showed that graph-structured memory consistently outperformed flat vector stores on multi-session dialogue, providing strong empirical support for the EMG approach.

Qian et al. (2023) proposed a personalised LLM assistant with evolving conditional memory, demonstrating that memory constructed and updated through dialogue produces better personalisation than fixed deployment-time memory. This principle is implemented in this thesis through CRUD operations that allow the EMG to evolve as the user's circumstances change.

Sun et al. (2025) showed that persistent user profiles combined with dynamically evolving memory improve LLM agent alignment with individual user expectations. Their work measured both quality and consistency improvements from memory persistence, both of which are evaluated in this thesis.

---

## 5. Intent Classification for Customer Service AI

Casanueva et al. (2020) introduced the Banking77 dataset with 13083 customer service utterances and 77 fine-grained banking intents. Their dual-encoder models achieved accuracy above 92% on this benchmark. Banking77 is the direct evaluation dataset for the intent classification component of this thesis, and the published 92% accuracy from dual-encoder models provides a comparison point against which the LSTM results can be interpreted.

Devlin et al. (2019) introduced BERT, which established bidirectional pre-training as the dominant approach for downstream NLP classification tasks. BERT and its variants now represent the performance ceiling for intent classification. This thesis uses LSTM rather than BERT, which is a deliberate choice: the novelty of this thesis is not in the classifier but in how the classifier output is used to guide graph retrieval. A lightweight LSTM on CPU demonstrates that the architectural contribution is valuable even without a state-of-the-art classifier.

Vaswani et al. (2017) introduced the Transformer architecture, which is the foundation of all modern LLMs and the direct reason why attention-based models outperform recurrent networks on most sequence tasks. Understanding this architecture helps contextualise the LSTM used in this thesis as a computationally accessible and well-established baseline.

Cho et al. (2014) introduced the GRU as an efficient alternative to LSTM, providing the empirical comparison between recurrent architectures that justifies the LSTM choice in this thesis. GRU has fewer parameters than LSTM but performs comparably on many tasks.

Mehdi et al. (2024) evaluated LLM fine-tuning for banking chatbot intent classification and achieved a 10% accuracy improvement over baseline methods through domain-specific adaptation. Their work confirms that the banking domain has sufficient specificity to justify domain-specific training data, supporting the use of Banking77 rather than a general-purpose intent dataset.

Ahmad et al. (2024) demonstrated that LLM-generated synthetic training data alleviates labelled data scarcity in intent classification for virtual assistants. Their approach is complementary to the Banking77 fine-tuning in this thesis.

Cho et al. (2025) benchmarked BERT, RoBERTa, and DistilBERT on CLINC150 for QA chatbot intent classification, providing performance ceiling reference points for the transformer-based approach that could be applied to Banking77 in future work.

Larson et al. (2019) introduced the CLINC150 benchmark with 150 intents across 10 domains, which is used as a supplementary evaluation dataset in this thesis to test whether the intent-guided retrieval framework generalises beyond the banking domain.

---

## 6. Hallucination Mitigation in LLMs

Huang et al. (2025) reviewed hallucination mitigation strategies for RAG-based LLMs, classifying failure modes into retrieval failures and generation failures. Their key finding was that most hallucinations in RAG systems arise from retrieval failures, not from the generation model itself. This directly motivates the intent-guided retrieval approach: if retrieval precision is improved, hallucination is reduced without changing the generator.

Ren et al. (2024) showed empirically that RAG substantially reduces hallucination compared to direct LLM generation across multiple domains. This supports the comparison between Condition A (direct LLM) and Conditions B and C (RAG-based) in the evaluation.

Wang et al. (2025) proposed MEGA-RAG, which uses multi-evidence guided answer refinement to reduce hallucination. Their multi-evidence approach, where several retrieved documents cross-validate the answer, is conceptually related to the confidence-checking step in the response generation pipeline of this thesis.

---

## 7. Banking AI and Financial NLP

Velusamy et al. (2024) reviewed deep learning chatbot architectures in banking, identifying intent recognition accuracy as the primary determinant of chatbot usefulness. This supports the focus of this thesis on accurate intent classification as the prerequisite for effective retrieval routing.

Sharma et al. (2025) evaluated GPT-3 and FinBERT for banking fraud detection, achieving 92% prediction accuracy and reducing false-positive rates from 15% to 5% with domain-specific models. While their application is fraud detection rather than intent classification, their results confirm that domain adaptation significantly improves performance in financial AI applications.

Prasad et al. (2025) analysed chatbot deployment patterns in retail banking, noting that user trust depends critically on response groundedness and accuracy for sensitive financial queries. This practical observation provides motivation for the grounded response generation approach in this thesis.

---

## 8. Continual Learning and the Case for RAG Over Fine-Tuning

Jimenez Gutierrez et al. (2026) evaluated RAG versus continual fine-tuning on a chronological knowledge drift benchmark spanning January 2024 to October 2025. They found complementary failure modes: fine-tuning suffers catastrophic forgetting when updated on new facts, while RAG struggles when retrieval fails entirely. Their conclusion was that neither approach dominates in all settings, but the combination of a stable intent classifier (fine-tuned on a fixed benchmark) with a dynamic retrieval layer (EMG-RAG for user-specific facts) avoids the failure modes of both pure approaches. This directly supports the hybrid architecture of this thesis.

Karpukhin et al. (2020) introduced Dense Passage Retrieval, demonstrating that learned dense encoders outperform sparse TF-IDF retrieval for open-domain question answering. The sentence-transformers model used for EMG node embedding in this thesis is a descendant of this line of work.

---

## 9. Research Gap Identified

Synthesising the literature across all six areas, four specific gaps are identified:

Gap 1 — Token efficiency is not measured in EMG-RAG systems: Wang et al. (2024) introduced EMG-RAG but did not measure token cost as an outcome variable. No existing EMG paper measures how many tokens the system sends to the LLM per query or compares this cost across retrieval approaches.

Gap 2 — Intent classification is not used to guide graph traversal: The Wang et al. (2024) EMG system retrieves nodes using flat text similarity. Graph-based retrieval systems (GraphRAG, HippoRAG, KG-RAG) use graph connectivity or text similarity for traversal. No existing system uses the output of a fine-tuned intent classifier to select which node types to search.

Gap 3 — EMG-RAG has not been applied to a banking domain benchmark: Wang et al. (2024) evaluated on a general assistant scenario. Banking77 provides 77 fine-grained intents with published baselines. No paper applies EMG-RAG to this benchmark.

Gap 4 — Quality and efficiency have not been jointly measured in a personalised chatbot evaluation: Most RAG papers measure only response quality (ROUGE, BLEU, EM). Most efficiency papers measure only speed or cost. No paper evaluates both dimensions simultaneously for a personalised banking assistant.

This thesis addresses all four gaps in a single system: intent-guided EMG-RAG evaluated on Banking77 with both quality metrics and token cost measured.

---

## References

1. Lewis, P., Perez, E., Piktus, A. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020, pp.9459-9474.

2. Gao, Y., Xiong, Y., Gao, X. et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.

3. Shuster, K., Xu, J., Komeili, M. et al. (2024). A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions. arXiv:2410.12837.

4. Asai, A., Wu, Z., Wang, Y. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024, arXiv:2310.11511.

5. Singh, A., Ehtesham, A., Kumar, S. et al. (2025). Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. arXiv:2501.09136.

6. Liu, N.F., Lin, K., Hewitt, J. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL 2024.

7. Shi, F., Chen, X., Misra, K. et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML 2023.

8. Wang, Z., Li, Z., Jiang, Z. et al. (2024). Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs. EMNLP 2024, arXiv:2409.19401.

9. Kang, X., Li, Y., Xu, Y. et al. (2025). MemoryOS: Memory OS of AI Agent. EMNLP 2025 (Oral), arXiv:2506.06326.

10. Liu, H., Zhang, J., Zhao, W. et al. (2025). In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents. ACL 2025, arXiv:2503.08026.

11. Chhikara, P., Singh, D., Sharma, P. et al. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. ECAI 2025, arXiv:2504.19413.

12. Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019, arXiv:1810.04805.

13. Vaswani, A., Shazeer, N., Parmar, N. et al. (2017). Attention Is All You Need. NeurIPS 2017.

14. Cho, K., van Merriënboer, B., Bahdanau, D. and Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. arXiv:1412.3555.

15. Casanueva, I., Temcinas, T., Gerz, D. et al. (2020). Efficient Intent Detection with Dual Sentence Encoders. NLP for ConvAI Workshop ACL 2020, arXiv:2003.04807.

16. Mehdi, Y., Nguyen, C., Hrobar, M. et al. (2024). Intent Classification for Bank Chatbots through LLM Fine-Tuning. arXiv:2410.04925.

17. Ahmad, N., Lee, S., Park, H. et al. (2024). Enhancing Intent Classifier Training with Large Language Model-generated Data. Applied Artificial Intelligence, 2024.

18. Cho, H., Kim, J. and Park, S. (2025). BERT, RoBERTa, and DistilBERT for Intent Classification: A CLINC-150 Evaluation with QA Applications. IEEE Xplore 2025.

19. Larson, S., Mahendran, A., Peper, J.J. et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP-IJCNLP 2019, arXiv:1909.02027.

20. Edge, D., Trinh, H., Cheng, N. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. Microsoft Research, arXiv:2404.16130.

21. Guo, Z., Xu, L., Hu, X. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024, arXiv:2405.14831.

22. Gutierrez, B.J., Shen, J., Han, J. et al. (2025). From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2). arXiv:2502.14802.

23. Xu, H., Zhang, Y., Wang, L. et al. (2026). Graph-based Agent Memory: Taxonomy, Techniques, and Applications. arXiv:2602.05665.

24. Li, W., Chen, X., Zhao, M. et al. (2026). GAM: Hierarchical Graph-based Agentic Memory for LLM Agents. arXiv:2604.12285.

25. Chen, W., Liu, Y., Zhang, H. et al. (2025). Knowledge Graph-extended Retrieval Augmented Generation for Question Answering. arXiv:2504.08893.

26. Li, W., Zhang, H., Chen, Y. et al. (2024). WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Integrating Web Search and Knowledge Graphs. arXiv:2408.07611.

27. Zhao, T., Wu, F., Li, Y. et al. (2025). D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory and Reasoning Tree. arXiv:2510.13363.

28. Huang, R., Chen, W., Zhang, X. et al. (2025). Hallucination Mitigation for Retrieval-Augmented Large Language Models: A Review. Mathematics (MDPI), 13(5), 856.

29. Ren, J., Luo, Y., Zhao, W. et al. (2024). Reducing Hallucination in Structured Outputs via Retrieval-Augmented Generation. arXiv:2404.08189.

30. Wang, H., Li, T., Chen, Q. et al. (2025). MEGA-RAG: A Retrieval-Augmented Generation Framework with Multi-Evidence Guided Answer Refinement. PMC/Nature, 2025.

31. Velusamy, D., Anand, P., Krishnan, S. et al. (2024). AI-Powered Chatbots in Financial Services. Journal of Unmanned System Technology, 2024.

32. Sharma, R., Gupta, A., Mehta, V. et al. (2025). LLM for Financial Services: Risk Analysis and Fraud Detection. Applied Science and Engineering Journal for Advanced Research, 2025.

33. Prasad, N., Rao, K. and Singh, M. (2025). Chatbots and Conversational AI in Retail Banking. MSI Publishers Journal of Management Research, 2025.

34. Qian, R., Wang, M., Zhang, T. et al. (2023). Personalized Large Language Model Assistant with Evolving Conditional Memory. arXiv:2312.17257.

35. Jimenez Gutierrez, B., Han, J., Yu, H. et al. (2026). RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World. arXiv:2604.05096.

---

*Jai Prabhas Malluri - 24310875 - Literature Review v3 - 35 papers reviewed, 14 from 2025-2026*
