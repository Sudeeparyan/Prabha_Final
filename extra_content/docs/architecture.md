# System Architecture — Intent-Guided EMG-RAG
## Knowledge Graph-Based Editable Memory for Token-Efficient Personalised AI Assistants

---

## Full System Flowchart (Mermaid)

```mermaid
flowchart TD
    Q(["🗣️ User Natural Language Query"])

    Q --> PP

    subgraph CLF ["🔵  Layer 1 — Intent Classification   |   CLINC150 Benchmark: 150 intents · 10 life domains · 22,500 queries"]
        direction TB
        PP["Text Preprocessing\nlowercase · remove punctuation · normalize whitespace"]
        PP --> TF
        PP --> WS

        TF["TF-IDF Vectorizer\n8,000 terms · 1-2 grams\n→ sparse feature matrix"]
        WS["Word Sequence Encoder\nvocab 12k · maxlen 30\n→ integer token sequences"]

        TF  --> LR_M["Logistic Regression\nAccuracy: 82.78%  ✓ Best\nlbfgs · C=1.0 · max_iter=1000"]
        TF  --> RF_M["Random Forest\nAccuracy: 78.44%\n200 estimators"]
        WS  --> BL_M["Bidirectional LSTM\nAccuracy: 75.53%\nembed 128d · hidden 128 · dropout 0.3"]

        LR_M & RF_M & BL_M --> INT(["🏷️ Predicted Intent Label\n+ Confidence Score"])
    end

    INT --> RT

    subgraph ROUTE ["🟠  Layer 2 — Intent-to-Node-Type Router   |   Novel Contribution: classifier output as typed retrieval signal"]
        RT["CLINC150 Intent → EMG Node Type Mapping\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nbalance        → [AccountState, Event]\nreport_fraud   → [Event, FAQ]\nmeal_suggestion→ [Preference, FAQ]\ncalendar       → [Event]\nimprove_credit → [FAQ, AccountState]\n... (150 intents mapped)"]
        RT --> TYPES(["📌 1–2 Target Node Types Selected"])
    end

    TYPES --> FILTER

    subgraph EMG ["🟢  Layer 3 — Editable Memory Graph   |   NetworkX (graph structure) + ChromaDB (vector store) + all-MiniLM-L6-v2 (embeddings)"]
        direction TB

        FILTER["ChromaDB Filtered Vector Search\nCosine similarity — restricted to selected node types\nSolves Problem 3: Intent-Context Mismatch"]

        FILTER --> FAQ_N["📄 FAQ Nodes  (10)\nDomain policy knowledge\nRarely changes"]
        FILTER --> PREF_N["⚙️ Preference Nodes  (4)\nUser-specific settings\nEditable anytime"]
        FILTER --> EVT_N["📅 Event Nodes  (8)\nTimestamped personal events\nHigh-change frequency"]
        FILTER --> ACC_N["💳 AccountState Nodes  (3)\nCurrent profile facts\nPeriodically updated"]

        FAQ_N & PREF_N & EVT_N & ACC_N --> TOP_K(["🎯 Top-2 Typed Nodes\nMost relevant context retrieved\nDocument nodes also searched if uploaded"])

        CRUD["⚡ CRUD Operations\n────────────────────\nINSERT new node  < 21 ms\nUPDATE node content  < 21 ms\nDELETE node  < 8 ms\n────────────────────\n✅ No model retraining\n✅ Instant effect on next query\n(Solves Problem 2: Dynamic Memory)"]

        CRUD -.->|live updates| FAQ_N
        CRUD -.->|live updates| PREF_N
        CRUD -.->|live updates| EVT_N
        CRUD -.->|live updates| ACC_N
    end

    TOP_K --> PB
    Q     --> PB
    INT   --> PB

    subgraph GEN ["🔴  Layer 4 — Response Generation   |   Gemini 2.5 Flash via Google AI SDK"]
        PB["Prompt Builder\n[System: You are Jarvis, intent={intent}]\n[Context: {node_type_1} {content_1}]\n[Context: {node_type_2} {content_2}]\n[Query: {user_query}]"]

        PB --> TC["📊 Token Counter\nGemini count_tokens API\n━━━━━━━━━━━━━━━━━━━━━━━━\nCond A (no context):  28.4 avg\nCond B (flat RAG):   157.1 avg\nCond C (EMG-RAG):    145.4 avg\n→ 7.5% fewer than Flat RAG\n(Solves Problem 1: Token Exhaustion)"]

        PB --> GEMINI["🤖 Gemini 2.5 Flash\nGrounded generation from retrieved evidence\nReduces hallucination vs parametric recall"]

        GEMINI --> CONF{"Confidence\nCheck"}
        CONF -->|"≥ threshold"| RESP(["✅ Grounded Personalised Response\nROUGE-L: 0.587  BLEU: 0.264"])
        CONF -->|"< threshold"| FALL(["⚠️ Template Fallback\nSafe generic answer"])
    end

    style CLF   fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style ROUTE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style EMG   fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style GEN   fill:#fce4ec,stroke:#880e4f,stroke-width:2px
```

---

## Three-Problem Solution Map

```mermaid
flowchart LR
    subgraph P1 ["Problem 1: Token Exhaustion"]
        direction TB
        B1["❌ Naive: send ALL user data\n~157 tokens per query"]
        S1["✅ Fix: Intent-Guided Graph\n→ retrieve only 2 typed nodes\n~145 tokens per query\n7.5% reduction"]
        B1 --> S1
    end

    subgraph P2 ["Problem 2: Dynamic Memory"]
        direction TB
        B2["❌ Fine-tuning: encode facts in weights\n→ retraining takes hours/days\n→ catastrophic forgetting risk"]
        S2["✅ Fix: EMG CRUD Operations\n→ INSERT / UPDATE / DELETE\n→ < 21 ms · zero retraining\n→ instant effect on next query"]
        B2 --> S2
    end

    subgraph P3 ["Problem 3: Intent-Context Mismatch"]
        direction TB
        B3["❌ Flat RAG: cosine similarity only\n'account balance' and 'fraud report'\nboth match 'account' nodes"]
        S3["✅ Fix: Typed Routing\nbalance → AccountState+Event\nreport_fraud → Event+FAQ\n→ correct context each time"]
        B3 --> S3
    end

    style P1 fill:#fff3e0,stroke:#e65100
    style P2 fill:#e8f5e9,stroke:#2e7d32
    style P3 fill:#e3f2fd,stroke:#1565c0
```

---

## Dataset Design Rationale

```mermaid
flowchart LR
    subgraph CLINC ["CLINC150 (Real Benchmark)"]
        C1["22,500 queries\n150 intents\n10 life domains"]
        C2["Peer-reviewed\nPublished baselines\n~92% dual-encoder"]
        C3["→ Intent Classifier\nTraining + Benchmarking"]
    end

    subgraph SYN ["Synthetic Scenarios"]
        S1["20 evaluation\nquery-answer pairs\n6 life domains"]
        S2["User profile: Priya\n25-node EMG\nComplete life data"]
        S3["→ EMG-RAG Evaluation\nPersonalised memory test"]
    end

    subgraph WHY ["Why Two Datasets?"]
        W1["No existing dataset has:\n(query, user-memory-context,\nground-truth-answer) triples"]
        W2["Wang et al. 2024 (closest\nprior work) also used\nsimulated scenarios"]
    end

    CLINC --> WHY
    SYN --> WHY
```

---

## Novelty Over Prior Work

| Feature | Lewis et al. 2020 (RAG) | Wang et al. 2024 (EMG-RAG) | **This Thesis** |
|---|---|---|---|
| Retrieval method | Dense vector similarity | Flat text similarity | **Intent-guided typed retrieval** |
| Token cost measured | ❌ | ❌ | **✅ Gemini count_tokens API** |
| Memory updateable | ❌ (static doc store) | ✅ CRUD | **✅ CRUD < 25 ms** |
| Intent routing layer | ❌ | ❌ | **✅ 150-intent CLINC150 router** |
| Multi-domain benchmark | General QA | General assistant | **✅ CLINC150 (10 domains)** |
| Evaluation dataset | Real | Simulated | **✅ Hybrid (real + synthetic)** |

---

## Research Questions Answered

1. **Does intent-guided KG retrieval produce higher quality responses with fewer tokens than flat RAG?**
   → YES: ROUGE-L 0.587 vs 0.532 (Cond C > B), tokens 145.4 vs 157.1 (7.5% fewer)

2. **Which intent classifier achieves highest accuracy on CLINC150?**
   → Logistic Regression: 82.78% (vs RF 78.44%, BiLSTM 75.53%)

3. **Can graph CRUD operations update memory in real time without retraining?**
   → YES: INSERT 16.88 ms, UPDATE 20.15 ms, DELETE 7.52 ms — all < 25 ms, no model modified
