# System Architecture
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised Banking AI

**Student:** Jai Prabhas Malluri (24310875)

---

## Architecture Philosophy

The architecture is designed around three problems. Token exhaustion is solved by graph traversal (only 2-3 nodes retrieved per query). Dynamic memory is solved by editable graph nodes (CRUD without retraining). Intent-context mismatch is solved by the routing layer (intent classifier controls which node types are searched). Each layer solves one problem.

---

## Diagram 1 - Full System Architecture with Token Cost Awareness

```mermaid
flowchart TB
    subgraph USER["User Input"]
        UQ["User Query"]
    end

    subgraph PREPROCESS["Text Preprocessing"]
        CLN["Lowercase, remove punctuation\nstrip extra whitespace"]
        TFIDF_V["TF-IDF Vectorizer\nfor LR and RF"]
        EMB_V["Word Embedding Sequences\npadded to length 50\nfor LSTM"]
    end

    subgraph INTENT["Intent Classification - Fine-Tuned Models"]
        LR_M["Logistic Regression\nML baseline"]
        RF_M["Random Forest\nEnsemble baseline"]
        LSTM_M["LSTM\nDeep Learning\nSelected for routing"]
        PRED["Predicted Intent\ne.g. card_not_working\nConfidence score"]
    end

    subgraph ROUTER["Intent Router - The Novel Layer"]
        MAP["Intent to Node Type Mapping\ncard_not_working -> FAQ + Event\nbalance_enquiry -> FAQ + AccountState\ncomplaint -> Event + FAQ\npreference_update -> Preference"]
        SELECTED["Node Types Selected\nfor this query"]
    end

    subgraph EMG["Editable Memory Graph - Built Once, Traversed Many Times"]
        FAQ_N[("FAQ Nodes\nStatic policy docs\n8 nodes")]
        PREF_N[("Preference Nodes\nUser settings\n3 nodes")]
        EVENT_N[("Event Nodes\nComplaints, bookings\n4 nodes")]
        ACCT_N[("Account State Nodes\nAccount facts\n3 nodes")]
        CRUD_OPS["CRUD Operations\nInsert / Update / Delete\nreal time, no retraining"]
    end

    subgraph TOKEN_TRACK["Token Efficiency Measurement"]
        TOK_C["count_tokens via Gemini API\nper query, per condition"]
        TOK_A["Condition A tokens\ndirect prompt only\n~50-80 tokens"]
        TOK_B["Condition B tokens\nflat RAG docs included\n~400-700 tokens"]
        TOK_C2["Condition C tokens\n2-3 targeted nodes only\n~150-250 tokens"]
    end

    subgraph RAG["Response Generation - Gemini 2.5 Flash"]
        PROMPT["Build Minimal Prompt\nIntent + Selected Nodes + Query"]
        GEM["Gemini 2.5 Flash\nGrounded Response"]
        CONF{{"Confidence\nCheck"}}
        GRND["Grounded Answer"]
        FALL["Fallback Response"]
    end

    RESP["Final Response to User"]

    UQ --> CLN
    CLN --> TFIDF_V
    CLN --> EMB_V
    TFIDF_V --> LR_M
    TFIDF_V --> RF_M
    EMB_V --> LSTM_M
    LR_M --> PRED
    RF_M --> PRED
    LSTM_M --> PRED

    PRED --> MAP
    MAP --> SELECTED

    SELECTED --> FAQ_N
    SELECTED --> PREF_N
    SELECTED --> EVENT_N
    SELECTED --> ACCT_N

    CRUD_OPS -.-> FAQ_N
    CRUD_OPS -.-> PREF_N
    CRUD_OPS -.-> EVENT_N
    CRUD_OPS -.-> ACCT_N

    FAQ_N --> PROMPT
    PREF_N --> PROMPT
    EVENT_N --> PROMPT
    ACCT_N --> PROMPT

    UQ --> PROMPT
    PRED --> PROMPT

    PROMPT --> TOK_C
    TOK_C --> GEM
    GEM --> CONF
    CONF -- "Yes" --> GRND
    CONF -- "No" --> FALL
    GRND --> RESP
    FALL --> RESP

    TOK_A -.->|compare| TOK_B
    TOK_B -.->|compare| TOK_C2
```

---

## Diagram 2 - The Graphify Analogy Applied to Banking AI

```mermaid
flowchart LR
    subgraph PROBLEM["What Happens Without the Graph"]
        P_Q["User Query:\nHow do I reset my PIN?"]
        P_LOAD["Load ALL Documents\ninto prompt context:\n- 8 FAQ docs\n- 3 preference records\n- 4 event records\n- 3 account records\n= 18 documents\n~700 tokens"]
        P_LLM["Gemini tries to find\nrelevant info inside\nlarge noisy context\nLost in the Middle problem"]
        P_COST["High token cost\nHigh hallucination risk\nSlower response"]
    end

    subgraph SOLUTION["What the EMG Approach Does"]
        S_Q["User Query:\nHow do I reset my PIN?"]
        S_INTENT["LSTM classifies intent:\ncard_pin_change\nconfidence 0.91"]
        S_MAP["Router maps to:\nFAQ nodes only"]
        S_SEARCH["Search only FAQ subgraph\n8 nodes searched\ntop 2 returned"]
        S_PROMPT["Build minimal prompt:\n- 2 FAQ nodes\n~180 tokens total"]
        S_LLM["Gemini answers with\nfocused accurate context\nNo noise, no confusion"]
        S_COST["Lower token cost\nHigher accuracy\nFaster response"]
    end

    P_Q --> P_LOAD --> P_LLM --> P_COST
    S_Q --> S_INTENT --> S_MAP --> S_SEARCH --> S_PROMPT --> S_LLM --> S_COST
```

---

## Diagram 3 - Editable Memory Graph Node Structure

```mermaid
graph TD
    ROOT["Banking AI Memory Graph\n18 nodes, 4 types"]

    ROOT --> FAQ_G["FAQ Subgraph\n8 nodes - static policy"]
    ROOT --> USER_G["User Memory Subgraph\n10 nodes - dynamic, editable"]

    FAQ_G --> F1["faq_balance_check\nHow to check balance..."]
    FAQ_G --> F2["faq_transfer_limit\nDaily limit 10000 euros..."]
    FAQ_G --> F3["faq_card_pin_reset\nReset PIN via ATM or app..."]
    FAQ_G --> F4["faq_card_cancel\nFreeze card in app..."]
    FAQ_G --> F5["faq_direct_debit\nManage in Payments section..."]
    FAQ_G --> F6["faq_exchange_rate\nUpdated daily, check in app..."]
    FAQ_G --> F7["faq_top_up\nTop up via app or branch..."]
    FAQ_G --> F8["faq_statement\nMonthly PDF in app history..."]

    USER_G --> PREF_G["Preference Nodes\n3 nodes - editable"]
    USER_G --> EVENT_G["Event Nodes\n4 nodes - editable"]
    USER_G --> ACCT_G["Account State Nodes\n3 nodes - editable"]

    PREF_G --> P1["pref_channel\nValue: SMS\nLast updated: 2026-05-29"]
    PREF_G --> P2["pref_language\nValue: English formal\nLast updated: 2026-01-01"]
    PREF_G --> P3["pref_contact_time\nValue: 9am-12pm morning\nLast updated: 2026-03-15"]

    EVENT_G --> E1["event_complaint_001\nBilling dispute 2026-04-15\nStatus: under review"]
    EVENT_G --> E2["event_callback_001\nCallback 2026-05-30 10am\nAgent: Sarah"]
    EVENT_G --> E3["event_application_001\nJoint account applied\nRef: JA2026001"]
    EVENT_G --> E4["event_dispute_002\nUnauthorised 250 euro\nRef: D2026002 open"]

    ACCT_G --> A1["acct_type\nCurrent Account Gold tier\nOpened 2022"]
    ACCT_G --> A2["acct_limits\nTransfer: 10000/day\nCard: 3000/day"]
    ACCT_G --> A3["acct_kyc\nKYC valid 2027-12-31\nPassport verified"]
```

---

## Diagram 4 - Intent to Node Type Routing Table

```mermaid
flowchart LR
    subgraph INTENTS["Predicted Intents (Banking77)"]
        I1["balance_enquiry"]
        I2["card_pin_change"]
        I3["complaint"]
        I4["contact_customer_support"]
        I5["transfer_charge"]
        I6["lost_or_stolen_card"]
        I7["transaction_not_recognised"]
        I8["verify_my_identity"]
        I9["top_up_by_bank_transfer"]
        I10["exchange_rate"]
    end

    subgraph NODES["Node Types Searched"]
        N_FAQ["FAQ Nodes"]
        N_PREF["Preference Nodes"]
        N_EVENT["Event Nodes"]
        N_ACCT["Account State Nodes"]
    end

    I1 --> N_FAQ
    I1 --> N_ACCT
    I2 --> N_FAQ
    I3 --> N_EVENT
    I3 --> N_FAQ
    I4 --> N_PREF
    I4 --> N_EVENT
    I5 --> N_FAQ
    I5 --> N_ACCT
    I6 --> N_FAQ
    I7 --> N_EVENT
    I7 --> N_FAQ
    I8 --> N_ACCT
    I8 --> N_FAQ
    I9 --> N_FAQ
    I9 --> N_PREF
    I10 --> N_FAQ
```

---

## Diagram 5 - Token Efficiency: How the Graph Reduces Token Cost

```mermaid
flowchart TB
    subgraph COND_A["Condition A - Direct LLM (No Context)"]
        A_PROMPT["Prompt: query only\nTokens: ~60-80\nLow cost but hallucinates\nNo retrieved context"]
        A_OUT["Response: often wrong\nROUGE-L: lowest\nCost: cheapest\nAccuracy: worst"]
    end

    subgraph COND_B["Condition B - Flat RAG (All Similar Documents)"]
        B_SEARCH["Retrieve ALL docs\nwith cosine similarity\nNo type filter\nReturns 3 docs regardless of type"]
        B_PROMPT["Prompt: query + 3 docs\nTokens: ~400-700\nHigh cost, noisy context\nLost in the Middle risk"]
        B_OUT["Response: sometimes wrong\nROUGE-L: medium\nCost: highest\nAccuracy: medium"]
    end

    subgraph COND_C["Condition C - Intent-Guided EMG-RAG (This Thesis)"]
        C_INTENT["Classify intent\nSelect 1-2 node types only"]
        C_SEARCH["Search only relevant\nnode type subgraph\n2-3 targeted nodes returned"]
        C_PROMPT["Prompt: query + 2-3 nodes\nTokens: ~150-250\nTargeted context only\nNo noise"]
        C_OUT["Response: most accurate\nROUGE-L: highest\nCost: 40-60% less than B\nAccuracy: best"]
    end

    A_PROMPT --> A_OUT
    B_SEARCH --> B_PROMPT --> B_OUT
    C_INTENT --> C_SEARCH --> C_PROMPT --> C_OUT

    A_OUT -.->|compare| B_OUT
    B_OUT -.->|compare| C_OUT
```

---

## Diagram 6 - CRUD Operations: Real-Time Memory Update Without Retraining

```mermaid
sequenceDiagram
    actor User
    participant SYS as System
    participant IC as LSTM Classifier
    participant NX as NetworkX Graph
    participant CDB as ChromaDB
    participant GEM as Gemini 2.5 Flash

    User->>SYS: "I want SMS alerts instead of email"
    SYS->>IC: classify intent
    IC-->>SYS: intent=update_preference, conf=0.91

    SYS->>NX: UPDATE node pref_channel\nold: email -> new: SMS
    SYS->>CDB: re-embed and update pref_channel
    CDB-->>SYS: updated in 45ms

    SYS->>CDB: retrieve with filter node_type=Preference
    CDB-->>SYS: pref_channel + pref_contact_time returned

    SYS->>GEM: minimal prompt\n~180 tokens
    GEM-->>User: "Notification preference updated to SMS."

    Note over SYS: No model retraining. One node updated.\nNext query reflects new preference immediately.

    User->>SYS: "Cancel my callback booking"
    SYS->>IC: classify intent
    IC-->>SYS: intent=contact_customer_support, conf=0.88

    SYS->>NX: DELETE node event_callback_001
    SYS->>CDB: delete event_callback_001
    CDB-->>SYS: deleted in 32ms

    SYS->>GEM: minimal prompt with confirmation
    GEM-->>User: "Your scheduled callback has been cancelled."
```

---

## Diagram 7 - Evaluation Pipeline (4 Dimensions)

```mermaid
flowchart TB
    subgraph EVAL_INPUT["20 Banking Test Queries + Ground Truth Answers"]
        Q_SET["Each query run through all 3 conditions\nA: Direct LLM\nB: Flat RAG\nC: Intent EMG-RAG"]
    end

    subgraph QUALITY["Quality Metrics"]
        ROUGE["ROUGE-L Score\nword overlap with ground truth"]
        BLEU["BLEU Score\nn-gram precision"]
        EM["Exact Match Count\nnormalised string comparison"]
    end

    subgraph EFFICIENCY["Token Efficiency Metrics"]
        TOK_COUNT["Gemini count_tokens API\nper query per condition"]
        AVG_TOK["Average input tokens per condition"]
        RATIO["Token ratio: C vs B\nshows efficiency gain"]
    end

    subgraph CLASSIFIER_EVAL["Intent Classifier Evaluation"]
        ACC["Accuracy on Banking77 test set\nLR vs RF vs LSTM"]
        F1["Weighted F1, Precision, Recall"]
        CONF_M["Confusion Matrix"]
        TRAIN_T["Training time comparison"]
    end

    subgraph MEMORY_EVAL["Memory CRUD Evaluation"]
        INSERT_L["Insert latency (ms)"]
        UPDATE_L["Update latency (ms)"]
        DELETE_L["Delete latency (ms)"]
    end

    subgraph RESULTS["Final Results Table"]
        CLAIM["CLAIM VERIFIED IF:\nLSTM accuracy > 90%\nCondition C ROUGE-L > Condition B\nCondition C tokens < Condition B tokens\nAll CRUD < 100ms"]
    end

    Q_SET --> ROUGE
    Q_SET --> BLEU
    Q_SET --> EM
    Q_SET --> TOK_COUNT
    TOK_COUNT --> AVG_TOK --> RATIO
    ROUGE --> RESULTS
    BLEU --> RESULTS
    EM --> RESULTS
    RATIO --> RESULTS
    ACC --> RESULTS
    INSERT_L --> RESULTS
    UPDATE_L --> RESULTS
    DELETE_L --> RESULTS
    RESULTS --> CLAIM
```

---

## Diagram 8 - Technology Stack

```mermaid
mindmap
    root((Tech Stack))
        Python 3.11
        Data
            Banking77 HuggingFace
                13083 samples
                77 intents
                saved to CSV
        Intent Classifiers
            scikit-learn
                Logistic Regression
                Random Forest
                TF-IDF vectorizer
            TensorFlow Keras
                LSTM
                Word Embeddings
        Memory Graph
            networkx
                Graph structure
                CRUD operations
            ChromaDB
                Node embeddings
                Type-filtered search
            sentence-transformers
                all-MiniLM-L6-v2
        Response Generation
            Google Generative AI
                gemini-2.5-flash
                count_tokens API
                Grounded generation
        Evaluation
            rouge-score ROUGE-L
            nltk BLEU
            sklearn metrics
            Token count comparison
        Outputs
            matplotlib plots
            pandas tables
            thesis_results txt
```

---

## What Is Novel in This Architecture

| Component | What Existing Work Does | What This Thesis Adds |
|-----------|------------------------|----------------------|
| RAG retrieval | Flat cosine similarity, all types | Intent-filtered, only relevant node types |
| Memory graph | EMG with flat similarity retrieval (Wang 2024) | EMG with intent-guided typed traversal |
| Token cost | Not measured in EMG-RAG papers | Measured with Gemini count_tokens API |
| Knowledge graph | Built once, static content (GraphRAG, Graphify) | Built once, editable via CRUD, dynamic user data |
| Evaluation | Quality metrics only | Quality + token efficiency + CRUD latency |
| Domain | General assistant (Wang 2024) | Banking domain, Banking77 benchmark |

The table shows that no single existing paper combines all of these elements. The combination is the contribution.

---

*Jai Prabhas Malluri - 24310875 - System Architecture v3*
