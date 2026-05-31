# Implementation Plan
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised AI Assistants

**Student:** Jai Prabhas Malluri (24310875)
**Novelty Rating:** 9/10
**Entry point:** app.py (Flask web application)

---

## What the Code Does (Top Level)

The implementation is a Flask web application with a knowledge graph backend. Pre-trained models are loaded on startup from `saved_models/`. The Flask server exposes REST API endpoints consumed by a browser-based chat interface.

```
Startup (app.py top-to-bottom)
  Step 1   Load intent classifier models from saved_models/
             tfidf.pkl, label_encoder.pkl, word_to_idx.pkl, lstm_best.pt
  Step 2   Build Editable Memory Graph (EMG)
             25 static nodes across 4 types, 16 directed edges
             Backed by NetworkX + ChromaDB (in-memory, cosine)
  Step 3   Connect Gemini 2.5 Flash via google.genai Client
  Step 4   Start Flask server (Waitress WSGI in production)

Flask routes
  GET  /                    Serve chat UI (templates/index.html)
  GET  /api/graph           Graph nodes + edges for vis.js rendering
  POST /api/chat            Full EMG-RAG pipeline (5 steps)
  POST /api/crud            INSERT / UPDATE / DELETE a node
  POST /api/upload          Upload file → chunk → Document nodes
  POST /api/file/delete     Remove file and its Document nodes
  GET  /api/conversations   Last 50 conversations from JSON log
  GET  /api/files           Uploaded files metadata
  GET  /api/node/<id>       Read a single node
  PUT  /api/node/<id>       Inline-edit a single node
```

---

## Key Design Decisions vs Previous Versions

### v1 — Banking77 (abandoned)
Banking77 covers only 77 banking intents in a single domain. The EMG node type system requires multi-domain routing; single-domain intent labels cannot demonstrate typed traversal across FAQ, Preference, Event, and AccountState subgraphs.

### v2 — CLINC150 classification only
CLINC150 (150 intents, 10 life domains, 22,500 queries) provides the correct breadth. Upgraded to this benchmark for classification training and benchmarking.

### v3 — Flask app with file upload and persistence (current)
The single-script experiment is replaced by a Flask web app. This supports:
- A browser UI with real-time graph visualization
- File upload (PDF, TXT, DOCX, MD) converted to Document nodes
- Persistent conversation history and custom node storage in `user_data/`
- Waitress WSGI server for stable multi-threaded serving

---

## Intent Classifier — PyTorch BiLSTM

### Architecture

```python
class IntentLSTM(nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim, nc):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 batch_first=True, bidirectional=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim * 2, nc)

# Instantiated as:
lstm_model = IntentLSTM(12000, 128, 128, num_classes)
```

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Vocabulary size | 12,000 |
| Embedding dim | 128 |
| Hidden dim | 128 (bidirectional → 256 to FC) |
| Dropout | 0.3 |
| Output classes | 150 (CLINC150) + 1 OOS |
| Sequence length | 30 tokens (padded / truncated) |
| Saved model | `saved_models/lstm_best.pt` |

### Supporting files loaded at startup

| File | Purpose |
|------|---------|
| `saved_models/tfidf.pkl` | Fitted TF-IDF vectorizer (used for LR/RF during training) |
| `saved_models/label_encoder.pkl` | sklearn LabelEncoder mapping int → intent label |
| `saved_models/word_to_idx.pkl` | Vocabulary dict: token → integer index |
| `saved_models/lstm_best.pt` | PyTorch state dict for BiLSTM |

### Classification pipeline (per query)

```python
def classify_intent(query):
    cleaned = clean_text(query)          # lowercase, strip punctuation
    seq     = text_to_seq(cleaned, word_to_idx, maxlen=30)
    xin     = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logits = lstm_model(xin)
        probs  = torch.softmax(logits, dim=1)[0]
        pid    = probs.argmax().item()
        intent = le.classes_[pid]
        conf   = float(probs[pid].item())
    return intent, conf, latency_ms
```

---

## Editable Memory Graph — 25 Nodes, 16 Edges

### Node types and counts

| Type | Count | Description |
|------|-------|-------------|
| FAQ | 10 | Domain policy knowledge (transfers, credit, fraud, travel, hotel, PTO, medication, diet, car, capabilities) |
| Preference | 4 | User-specific settings (dietary, commute, notifications, music) |
| Event | 8 | Timestamped personal events (flight, hotel, PTO, rent, fraud dispute, medication, car service, ML course) |
| AccountState | 3 | Current profile facts (financial, health, work) |
| Document | dynamic | Chunks from uploaded files (PDF/TXT/DOCX/MD) |

### Static node IDs in code

```
FAQ:          faq_transfers, faq_credit, faq_fraud, faq_travel, faq_hotel,
              faq_pto, faq_medication, faq_diet, faq_car, faq_capabilities

Preference:   pref_dietary, pref_commute, pref_notifications, pref_music

Event:        event_flight, event_hotel, event_pto, event_rent, event_fraud,
              event_medication, event_car, event_course

AccountState: acct_financial, acct_health, acct_work
```

### Directed edges (16 total)

```python
EDGES = [
    ("event_flight",       "faq_travel"),
    ("event_hotel",        "faq_hotel"),
    ("event_fraud",        "faq_fraud"),
    ("event_rent",         "acct_financial"),
    ("event_medication",   "faq_medication"),
    ("event_pto",          "acct_work"),
    ("event_car",          "faq_car"),
    ("pref_dietary",       "faq_diet"),
    ("pref_commute",       "acct_work"),
    ("pref_notifications", "acct_work"),
    ("acct_financial",     "acct_work"),
    ("acct_health",        "faq_medication"),
    ("event_course",       "acct_work"),
    ("event_fraud",        "acct_financial"),
    ("event_rent",         "faq_transfers"),
    ("faq_credit",         "acct_financial"),
]
```

### ChromaDB collection

```python
chroma_client = chromadb.Client()
emg_col = chroma_client.create_collection(
    "emg_demo", metadata={"hnsw:space": "cosine"})
```

Embeddings: `all-MiniLM-L6-v2` via sentence-transformers. Each node stored with `node_type`, `label`, `intents` metadata for filtered retrieval.

---

## Chat Pipeline — _run_pipeline(query)

Five steps, all timed, all returned in the API response for UI rendering.

### Step 1 — Text Preprocessing
```python
cleaned = clean_text(query)   # lowercase, regex strip non-alphanumeric, collapse whitespace
```

### Step 2 — Intent Classification
```python
intent, conf, clf_ms = classify_intent(query)
```
Returns: intent label (CLINC150), confidence (0–1), latency in ms.

### Step 3 — Intent-to-Node-Type Routing
```python
node_types = INTENT_NODE_MAP.get(intent, ["FAQ"])
```
`INTENT_NODE_MAP` maps 34 CLINC150 intents to 1–2 node type lists. OOS and unknown intents default to `["FAQ"]`.

### Step 4a — Intent-Guided EMG Retrieval
```python
c_nodes = intent_retrieve(query, intent, top_k=2)
```
Queries ChromaDB with `where={"node_type": {"$in": node_types}}` filter. Also searches Document nodes if any have been uploaded. Falls back to flat retrieval if filtered result is empty.

### Step 4b — Flat Retrieval (Condition B baseline)
```python
b_nodes = flat_retrieve(query, top_k=3)
```
Cosine similarity across all node types, no filter.

### Step 5 — Three-Condition Generation
```python
# Condition A: direct LLM (no context)
prompt_a = f"You are Jarvis, a personalised AI assistant. Answer briefly.\nQuery: {query}\nAnswer:"

# Condition B: flat RAG
prompt_b = f"You are Jarvis. Use only the context below.\nContext:\n{ctx_b}\nQuery: {query}\nAnswer:"

# Condition C: intent-guided EMG-RAG
prompt_c = (f"You are Jarvis, a personalised AI assistant.\n"
            f"Detected intent: {intent}\n"
            f"Use only the retrieved personalised information.\n\n"
            f"Retrieved Info:\n{ctx_c}\nQuery: {query}\nPersonalised Answer:")
```

Token counts measured with `gc.models.count_tokens(model=MODEL, contents=prompt)` for each condition. All three Gemini calls made sequentially. Token reduction = `(tok_b - tok_c) / tok_b * 100`.

---

## CRUD Operations — /api/crud

Supports `op`: INSERT, UPDATE, DELETE.

```
INSERT: add node to NetworkX + embed in ChromaDB + persist to custom_nodes.json
UPDATE: update node content/label in NetworkX + re-embed in ChromaDB + persist
DELETE: remove from NetworkX + ChromaDB + clean from custom_nodes.json + files metadata
```

All operations protected by `threading.Lock()` (`_graph_lock`). Latency measured per operation.

---

## File Upload — /api/upload

```
Allowed extensions: .pdf, .txt, .docx, .md
Max upload size: 50 MB
Chunk size: 350 words, overlap 30 words, max 8 chunks per file
Node type: Document
Node ID format: doc_{safe_stem}_{timestamp}_{chunk_index}
```

Text extracted with: `pdfplumber` (primary) → `PyPDF2` (fallback) for PDF; `python-docx` for DOCX; direct read for TXT/MD.

Nodes persisted in `user_data/custom_nodes.json` and survive server restarts.

---

## Persistence Layer

All runtime state is stored in `user_data/`:

| File | Content |
|------|---------|
| `conversations.json` | Last 200 conversation records (query, intent, response excerpt, tokens, ms) |
| `custom_nodes.json` | All non-static nodes: manually inserted + document chunks |
| `uploaded_files.json` | File metadata: original name, saved filename, size, upload time, node IDs |
| `uploads/` | Saved copies of uploaded files |

On startup, `custom_nodes.json` is replayed into the graph so uploaded documents and manual inserts survive restarts.

---

## Gemini Integration

```python
from google import genai as _genai
gc = _genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash"

def call_gemini(prompt):
    return gc.models.generate_content(model=MODEL, contents=prompt).text.strip()

def count_tokens(prompt):
    return gc.models.count_tokens(model=MODEL, contents=prompt).total_tokens
```

If `GOOGLE_API_KEY` is not set, `call_gemini` returns a mock string and `count_tokens` falls back to word-count estimation (`len(prompt.split()) * 1.3`).

---

## Pass/Fail Criteria for Thesis Claims

| Claim | Pass Condition |
|-------|---------------|
| BiLSTM is competitive classifier | BiLSTM accuracy on CLINC150 test set > 70% (CLINC150 is harder than Banking77) |
| Best classifier meets quality bar | Best model accuracy > 80% on CLINC150 |
| EMG-RAG quality is best | ROUGE-L Condition C > Condition B > Condition A |
| Graph retrieval is token-efficient | Avg tokens Condition C < Avg tokens Condition B |
| CRUD is fast enough | All CRUD operations < 100 ms |

If all five pass: the thesis claim is fully proved.
If token counts for C and B are close: still a valid finding — discuss graph size as a variable in limitations.

---

## Running the Application

```
python app.py
→ Opens at http://localhost:5000

Production (auto-detected if waitress installed):
  Uses Waitress WSGI server, 4 threads, port 5000

Development fallback:
  Uses Werkzeug dev server, threaded=True, debug=False
```

---

*Jai Prabhas Malluri - 24310875 - Implementation Plan v3 (updated to reflect Flask web app)*
