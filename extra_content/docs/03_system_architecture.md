# System Architecture
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised AI Assistants

**Student:** Jai Prabhas Malluri (24310875)

---

## Architecture Philosophy

The architecture is designed around three problems. Token exhaustion is solved by graph traversal (only 2 targeted nodes retrieved per query). Dynamic memory is solved by editable graph nodes (CRUD without retraining). Intent-context mismatch is solved by the routing layer (intent classifier controls which node types are searched). Each layer solves one problem.

The system is implemented as a Flask web application (`app.py`) with a browser-based UI. Pre-trained PyTorch models are loaded at startup. The EMG is held in-memory in NetworkX + ChromaDB and survives restarts via a JSON persistence layer.

---

## Diagram 1 - Full System Architecture with Token Cost Awareness

```mermaid
flowchart TB
    subgraph USER["User Input"]
        UQ["User Query"]
    end

    subgraph PREPROCESS["Text Preprocessing"]
        CLN["Lowercase, remove punctuation\nstrip extra whitespace"]
        TFIDF_V["TF-IDF Vectorizer\nfor LR and RF (training only)"]
        EMB_V["Word Sequence Encoder\nvocab 12k · maxlen 30\nfor BiLSTM inference"]
    end

    subgraph INTENT["Intent Classification — PyTorch BiLSTM (CLINC150)"]
        BL_M["Bidirectional LSTM\nvocab 12k · embed 128 · hidden 128\ndropout 0.3 · output 150 classes"]
        PRED["Predicted Intent\ne.g. report_fraud\nConfidence score"]
    end

    subgraph ROUTER["Intent Router — The Novel Layer"]
        MAP["Intent to Node Type Mapping\n150 CLINC150 intents mapped\nreport_fraud → [Event, FAQ]\nbalance → [AccountState, Event]\nmeal_suggestion → [Preference, FAQ]\ncalendar → [Event]"]
        SELECTED["Node Types Selected\nfor this query"]
    end

    subgraph EMG["Editable Memory Graph — NetworkX + ChromaDB + all-MiniLM-L6-v2"]
        FAQ_N[("FAQ Nodes (10)\nDomain policy knowledge\nRarely changes")]
        PREF_N[("Preference Nodes (4)\nUser-specific settings\nEditable anytime")]
        EVENT_N[("Event Nodes (8)\nTimestamped personal events\nHigh-change frequency")]
        ACCT_N[("AccountState Nodes (3)\nCurrent profile facts\nPeriodically updated")]
        DOC_N[("Document Nodes (dynamic)\nUploaded file chunks\nPDF / TXT / DOCX / MD")]
        CRUD_OPS["CRUD Operations\nINSERT · UPDATE · DELETE\nreal time · no retraining\n< 25 ms each"]
    end

    subgraph TOKEN_TRACK["Token Efficiency Measurement"]
        TOK_C["gc.models.count_tokens\nGemini count_tokens API\nper query · per condition"]
        TOK_A["Condition A tokens\ndirect prompt only\n~28 avg tokens"]
        TOK_B["Condition B tokens\nflat RAG 3 docs\n~157 avg tokens"]
        TOK_C2["Condition C tokens\n2 targeted typed nodes\n~145 avg tokens"]
    end

    subgraph RAG["Response Generation — Gemini 2.5 Flash (google.genai)"]
        PROMPT["Build Prompt\nIntent + Selected Nodes + Query"]
        GEM["Gemini 2.5 Flash\nGrounded Response"]
        GRND["Grounded Answer\nPersonalised to Priya's EMG"]
    end

    RESP["Final Response to User (JSON)\nAll 3 conditions · steps · highlights"]

    UQ --> CLN
    CLN --> TFIDF_V
    CLN --> EMB_V
    EMB_V --> BL_M
    BL_M --> PRED

    PRED --> MAP
    MAP --> SELECTED

    SELECTED --> FAQ_N
    SELECTED --> PREF_N
    SELECTED --> EVENT_N
    SELECTED --> ACCT_N
    SELECTED --> DOC_N

    CRUD_OPS -.-> FAQ_N
    CRUD_OPS -.-> PREF_N
    CRUD_OPS -.-> EVENT_N
    CRUD_OPS -.-> ACCT_N
    CRUD_OPS -.-> DOC_N

    FAQ_N --> PROMPT
    PREF_N --> PROMPT
    EVENT_N --> PROMPT
    ACCT_N --> PROMPT

    UQ --> PROMPT
    PRED --> PROMPT

    PROMPT --> TOK_C
    TOK_C --> GEM
    GEM --> GRND
    GRND --> RESP

    TOK_A -.->|compare| TOK_B
    TOK_B -.->|compare| TOK_C2
```

---

## Diagram 2 - The Intent-Guided Retrieval Advantage

```mermaid
flowchart LR
    subgraph PROBLEM["What Happens Without Intent Routing (Flat RAG)"]
        P_Q["User Query:\nWhat is my account balance?"]
        P_LOAD["Flat cosine similarity:\nAll 25 nodes searched\nTop 3 returned regardless of type\nMay return travel/medication nodes\n~157 avg tokens"]
        P_LLM["Gemini receives noisy context\nLost in the Middle problem\nIrrelevant facts in prompt"]
        P_COST["Higher token cost\nLower answer precision\nMixed irrelevant context"]
    end

    subgraph SOLUTION["What EMG-RAG Does"]
        S_Q["User Query:\nWhat is my account balance?"]
        S_INTENT["BiLSTM classifies intent: balance\nRoute to: [AccountState, Event]"]
        S_MAP["ChromaDB filtered query:\nwhere node_type IN [AccountState, Event]"]
        S_SEARCH["Top-2 typed nodes returned:\nacct_financial, event_rent"]
        S_PROMPT["Minimal prompt:\n~145 avg tokens"]
        S_LLM["Gemini answers with\nprecise personalised context"]
        S_COST["Fewer tokens · 7.5% reduction\nHigher ROUGE-L · Better quality"]
    end

    P_Q --> P_LOAD --> P_LLM --> P_COST
    S_Q --> S_INTENT --> S_MAP --> S_SEARCH --> S_PROMPT --> S_LLM --> S_COST
```

---

## Diagram 3 - Editable Memory Graph Node Structure (25 Nodes, 16 Edges)

```mermaid
graph TD
    ROOT["Personal AI Memory Graph\n25 nodes · 5 types · 16 directed edges\nUser persona: Priya · Dublin"]

    ROOT --> FAQ_G["FAQ Subgraph\n10 nodes — domain policy"]
    ROOT --> USER_G["User Memory Subgraph\n15 nodes — dynamic, editable"]

    FAQ_G --> F1["faq_transfers\nMoney transfers: 1-3 days domestic\nDaily limit EUR 10,000"]
    FAQ_G --> F2["faq_credit\nImprove credit: pay on time\nutilisation below 30%"]
    FAQ_G --> F3["faq_fraud\nReport fraud: call bank or freeze\nTemp credit issued"]
    FAQ_G --> F4["faq_travel\nNotify bank before travel\nTravel insurance required"]
    FAQ_G --> F5["faq_hotel\nFree cancel 48h before check-in\nCheck-in 3pm / out 11am"]
    FAQ_G --> F6["faq_pto\nPTO: 5 days notice · 25 days annual\nUnused > 5 days do not roll over"]
    FAQ_G --> F7["faq_medication\nNever skip prescribed meds\nSet daily reminders"]
    FAQ_G --> F8["faq_diet\nVegetarian gluten-free:\nlegumes · leafy greens · quinoa"]
    FAQ_G --> F9["faq_car\nOil change 5000-7500 miles / 6 months\nTyre pressure monthly"]
    FAQ_G --> F10["faq_capabilities\nJarvis assists: banking, travel,\nhealth, calendar, work, car, commute"]

    USER_G --> PREF_G["Preference Nodes (4)"]
    USER_G --> EVENT_G["Event Nodes (8)"]
    USER_G --> ACCT_G["AccountState Nodes (3)"]

    PREF_G --> P1["pref_dietary\nVegetarian · gluten-intolerant\nAllergic to shellfish"]
    PREF_G --> P2["pref_commute\nSandymount → Grand Canal Dock\nDART · leaves 8am weekdays"]
    PREF_G --> P3["pref_notifications\nEmail preferred · quiet 10pm-7am\nUpdated 30 May 2026"]
    PREF_G --> P4["pref_music\nLo-fi hip hop commute\nJazz evenings · no heavy metal"]

    EVENT_G --> E1["event_flight\nRyanair FR2241 Dublin→Paris CDG\n10 Jul 2026 07:45 · Ref RY7821345"]
    EVENT_G --> E2["event_hotel\nHotel Le Marais Paris 10-17 Jul\nFree cancel until 8 Jul"]
    EVENT_G --> E3["event_pto\nPTO approved 10-17 Jul 2026\n5 working days · Ref HR2026PTO041"]
    EVENT_G --> E4["event_rent\nRent EUR 1,200 on 1st of month\nNext due 1 Jun 2026"]
    EVENT_G --> E5["event_fraud\nFraud dispute FD2026001\nEUR 250 · resolution 5 Jun 2026"]
    EVENT_G --> E6["event_medication\nRamipril 5mg daily 8am\nLast dose 30 May 2026"]
    EVENT_G --> E7["event_car\nCar service Dublin Motors 15 Jun\nOil · tyres · brakes"]
    EVENT_G --> E8["event_course\nAdvanced ML Coursera\nStarted 1 Jun · deadline 26 Jul 2026"]

    ACCT_G --> A1["acct_financial\nBalance ~EUR 3,200\nCredit score 710 · Income EUR 4,500"]
    ACCT_G --> A2["acct_health\nAge 29 · Ramipril 5mg daily\nVHI Plan B · 10k steps/day"]
    ACCT_G --> A3["acct_work\nSenior Data Analyst TechCorp Dublin\nManager: Sarah Collins · Remote Fri"]
```

---

## Diagram 4 - Intent to Node Type Routing Table (CLINC150)

```mermaid
flowchart LR
    subgraph INTENTS["Predicted Intents (CLINC150 subset — 34 mapped)"]
        I1["balance"]
        I2["report_fraud"]
        I3["book_flight"]
        I4["restaurant_suggestion"]
        I5["pto_request"]
        I6["reminder_update"]
        I7["traffic"]
        I8["credit_score"]
        I9["book_hotel"]
        I10["play_music"]
        I11["calendar"]
        I12["greeting / oos"]
    end

    subgraph NODES["Node Types Searched"]
        N_FAQ["FAQ Nodes"]
        N_PREF["Preference Nodes"]
        N_EVENT["Event Nodes"]
        N_ACCT["AccountState Nodes"]
    end

    I1  --> N_ACCT
    I1  --> N_EVENT
    I2  --> N_EVENT
    I2  --> N_FAQ
    I3  --> N_EVENT
    I3  --> N_FAQ
    I4  --> N_PREF
    I4  --> N_FAQ
    I5  --> N_ACCT
    I5  --> N_EVENT
    I6  --> N_EVENT
    I6  --> N_PREF
    I7  --> N_PREF
    I7  --> N_ACCT
    I8  --> N_ACCT
    I8  --> N_FAQ
    I9  --> N_EVENT
    I9  --> N_FAQ
    I10 --> N_PREF
    I11 --> N_EVENT
    I12 --> N_FAQ
```

---

## Diagram 5 - Token Efficiency: Three-Condition Comparison

```mermaid
flowchart TB
    subgraph COND_A["Condition A — Direct LLM (No Context)"]
        A_PROMPT["Prompt: query only\nAvg tokens: ~28\nNo retrieved context\nLow cost but limited personalisation"]
        A_OUT["Response: generic\nROUGE-L: baseline\nNo user-specific facts"]
    end

    subgraph COND_B["Condition B — Flat RAG (Unfiltered Cosine)"]
        B_SEARCH["Cosine similarity across ALL node types\nNo type filter\nTop-3 nodes returned"]
        B_PROMPT["Prompt: query + 3 mixed-type nodes\nAvg tokens: ~157\nMay include irrelevant type context"]
        B_OUT["Response: partially personalised\nROUGE-L: medium\nHigher token cost"]
    end

    subgraph COND_C["Condition C — Intent-Guided EMG-RAG (This Thesis)"]
        C_INTENT["BiLSTM classifies intent\nRouter selects 1-2 node types"]
        C_SEARCH["ChromaDB filtered query\nOnly relevant node types searched\nTop-2 typed nodes returned"]
        C_PROMPT["Prompt: query + 2 targeted nodes\nAvg tokens: ~145\n7.5% fewer than Condition B"]
        C_OUT["Response: most personalised\nROUGE-L: highest\nBest quality · lowest cost"]
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
    participant APP as Flask app.py
    participant NX as NetworkX DiGraph
    participant CDB as ChromaDB (cosine)
    participant GEM as Gemini 2.5 Flash

    User->>APP: POST /api/crud {op: UPDATE, node_id: pref_notifications, content: "Prefers email..."}
    APP->>NX: node["content"] = new_content
    APP->>CDB: re-embed with all-MiniLM-L6-v2 · update metadata
    CDB-->>APP: updated in ~20 ms
    APP->>APP: persist to user_data/custom_nodes.json

    User->>APP: POST /api/chat {query: "How do I prefer to be notified?"}
    APP->>APP: classify intent → reminder_update
    APP->>CDB: filtered query: node_type IN [Event, Preference]
    CDB-->>APP: pref_notifications (updated content) returned
    APP->>GEM: prompt with new preference content (~145 tokens)
    GEM-->>User: "You prefer email for all notifications..."

    Note over APP: No model retrained. Node updated once.\nNext query reflects new preference immediately.

    User->>APP: POST /api/upload (PDF file)
    APP->>APP: extract text · chunk 350 words · max 8 chunks
    APP->>NX: add Document nodes with type=Document
    APP->>CDB: embed and store with node_type=Document metadata
    APP->>APP: persist to custom_nodes.json + uploaded_files.json
    APP-->>User: {chunks: N, nodes: [...], total_nodes: 25+N}
```

---

## Diagram 7 - Flask API Routes

```mermaid
flowchart TB
    subgraph CLIENT["Browser Client"]
        UI["index.html\nVis.js graph · Chat UI\n3-panel response view"]
    end

    subgraph FLASK["Flask app.py — Routes"]
        R1["GET /\nServe index.html"]
        R2["GET /api/graph\nGraph nodes + edges for vis.js\n5 node type styles (FAQ/Pref/Event/Acct/Doc)"]
        R3["POST /api/chat\nFull 5-step pipeline\nReturns: intent, steps, 3 conditions, highlights"]
        R4["POST /api/crud\nINSERT · UPDATE · DELETE\nUpdates NetworkX + ChromaDB + JSON"]
        R5["POST /api/upload\nPDF / TXT / DOCX / MD → Document nodes\nmax 8 chunks · 350 words each"]
        R6["POST /api/file/delete\nRemove file + all its Document nodes"]
        R7["GET /api/conversations\nLast 50 from conversations.json"]
        R8["GET /api/files\nUploaded files metadata"]
        R9["GET /api/node/<id>\nRead single node"]
        R10["PUT /api/node/<id>\nInline edit single node"]
    end

    subgraph PERSIST["user_data/ — Persistence"]
        J1["conversations.json\nLast 200 query logs"]
        J2["custom_nodes.json\nInserted + document nodes"]
        J3["uploaded_files.json\nFile metadata + node IDs"]
        J4["uploads/\nSaved file copies"]
    end

    UI --> R1
    UI --> R2
    UI --> R3
    UI --> R4
    UI --> R5
    UI --> R6
    UI --> R7
    UI --> R8
    UI --> R9
    UI --> R10

    R3 --> J1
    R4 --> J2
    R5 --> J2
    R5 --> J3
    R5 --> J4
    R6 --> J2
    R6 --> J3
    R6 --> J4
```

---

## Diagram 8 - Technology Stack

```mermaid
mindmap
    root((Tech Stack))
        Python 3.11
        Data
            CLINC150 HuggingFace
                22500 samples
                150 intents
                10 life domains
            Synthetic EMG
                Priya persona
                Dublin-based
                25 static nodes
        Intent Classifier
            PyTorch
                Bidirectional LSTM
                vocab 12k · embed 128 · hidden 128
                Dropout 0.3
            scikit-learn
                TF-IDF vectorizer
                Label Encoder
                LR + RF (trained, not used in inference)
        Memory Graph
            networkx
                DiGraph structure
                25 nodes · 16 edges
                CRUD with threading.Lock
            ChromaDB
                In-memory cosine collection
                Type-filtered vector search
            sentence-transformers
                all-MiniLM-L6-v2
                Node embeddings
        Response Generation
            google.genai
                gemini-2.5-flash
                count_tokens API
                Grounded generation
        Web Application
            Flask
                REST API (10 routes)
                JSON persistence
            Waitress WSGI
                Production server
                4 threads
            Vis.js
                Graph visualisation
        File Processing
            pdfplumber / PyPDF2
                PDF text extraction
            python-docx
                DOCX extraction
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
| Domain | General assistant (Wang 2024) | Multi-domain personal assistant (CLINC150, 10 life domains) |
| Implementation | Research scripts | Flask web app with file upload + graph UI |

The table shows that no single existing paper combines all of these elements. The combination is the contribution.

---

*Jai Prabhas Malluri - 24310875 - System Architecture v3 (updated to reflect Flask app, CLINC150, PyTorch BiLSTM, 25-node EMG)*
