# app.py — Personal AI Assistant powered by EMG-RAG
# Flask backend: knowledge graph, file uploads, conversation persistence.
# Run: python app.py  →  http://localhost:5000

import os, re, json, time, pickle, warnings, threading
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

for _env in [
    os.path.join(script_dir, ".env"),
    os.path.join(script_dir, "..", ".env"),
    os.path.join(script_dir, "..", "..", ".env"),
]:
    if os.path.exists(_env):
        load_dotenv(_env)
        break

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── user data persistence ──────────────────────────────────────────────────────
USER_DATA_DIR  = os.path.join(script_dir, "user_data")
UPLOADS_DIR    = os.path.join(USER_DATA_DIR, "uploads")
CONV_FILE      = os.path.join(USER_DATA_DIR, "conversations.json")
CUSTOM_NODES_F = os.path.join(USER_DATA_DIR, "custom_nodes.json")
FILES_META_F   = os.path.join(USER_DATA_DIR, "uploaded_files.json")

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR,   exist_ok=True)

def _load_json(path, default=None):
    if default is None:
        default = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] save {path}: {e}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

_graph_lock = threading.Lock()

# ── file parsing ───────────────────────────────────────────────────────────────
ALLOWED_EXT = {".pdf", ".txt", ".docx", ".md"}

def _extract_text(filepath, original_name):
    ext = os.path.splitext(original_name)[1].lower()
    try:
        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    return " ".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                pass
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    r = PyPDF2.PdfReader(f)
                    return " ".join(page.extract_text() or "" for page in r.pages)
            except ImportError:
                pass
            return ""
        elif ext == ".docx":
            from docx import Document as DocxDoc
            doc = DocxDoc(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[WARN] text extraction: {e}")
        return ""
    return ""

def _chunk_text(text, chunk_size=350, overlap=30):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── 1. Load intent classifier ──────────────────────────────────────────────────
print("[1/4] Loading intent classifier models...")

with open(os.path.join(script_dir, "saved_models", "tfidf.pkl"),         "rb") as f: tfidf       = pickle.load(f)
with open(os.path.join(script_dir, "saved_models", "label_encoder.pkl"), "rb") as f: le          = pickle.load(f)
with open(os.path.join(script_dir, "saved_models", "word_to_idx.pkl"),   "rb") as f: word_to_idx = pickle.load(f)

num_classes = len(le.classes_)

class IntentLSTM(nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim, nc):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, nc)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(self.dropout(torch.cat([h[0], h[1]], dim=1)))

lstm_model = IntentLSTM(12000, 128, 128, num_classes)
lstm_model.load_state_dict(torch.load(
    os.path.join(script_dir, "saved_models", "lstm_best.pt"), weights_only=True))
lstm_model.eval()
print(f"    LSTM ready  ({num_classes} CLINC150 intent classes)")

# ── 2. Build Editable Memory Graph ────────────────────────────────────────────
print("[2/4] Building Editable Memory Graph...")

sbert         = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
try:   chroma_client.delete_collection("emg_demo")
except: pass
emg_col   = chroma_client.create_collection("emg_demo", metadata={"hnsw:space": "cosine"})
emg_graph = nx.DiGraph()

EMG_NODES = [
    # ── FAQ ──────────────────────────────────────────────────────────────────
    {"id": "faq_transfers",   "type": "FAQ",
     "label": "Transfers FAQ",
     "content": "Money transfers: 1-3 days domestic, up to 5 days international. Daily limit EUR 10,000.",
     "intents": ["make_transfer", "bill_balance"]},
    {"id": "faq_credit",      "type": "FAQ",
     "label": "Credit FAQ",
     "content": "Improve credit score: pay bills on time, keep utilisation below 30%, avoid multiple applications.",
     "intents": ["credit_score", "improve_credit_score"]},
    {"id": "faq_fraud",       "type": "FAQ",
     "label": "Fraud FAQ",
     "content": "Report fraud: call bank or freeze card in app. Disputes resolved in 5 business days. Temp credit issued.",
     "intents": ["report_fraud", "card_declined"]},
    {"id": "faq_travel",      "type": "FAQ",
     "label": "Travel FAQ",
     "content": "Notify bank before international travel. Travel insurance before departure. Carry-on 10kg limit.",
     "intents": ["book_flight", "travel_alert"]},
    {"id": "faq_hotel",       "type": "FAQ",
     "label": "Hotel FAQ",
     "content": "Free cancellation up to 48 hours before check-in. Check-in 3pm, check-out 11am.",
     "intents": ["book_hotel", "cancel_reservation"]},
    {"id": "faq_pto",         "type": "FAQ",
     "label": "PTO Policy",
     "content": "PTO requests need 5 business days notice. Annual allowance 25 days. Unused days beyond 5 do not roll over.",
     "intents": ["pto_request", "pto_balance"]},
    {"id": "faq_medication",  "type": "FAQ",
     "label": "Medication FAQ",
     "content": "Never skip prescribed medication. Set daily reminders. Keep dose log. Store in cool dry place.",
     "intents": ["reminder_update"]},
    {"id": "faq_diet",        "type": "FAQ",
     "label": "Diet FAQ",
     "content": "Balanced vegetarian gluten-free diet: legumes for protein, leafy greens, rice, quinoa, nuts.",
     "intents": ["restaurant_suggestion", "meal_suggestion"]},
    {"id": "faq_car",         "type": "FAQ",
     "label": "Car Maintenance",
     "content": "Oil change every 5000-7500 miles or 6 months. Tyre pressure monthly. Annual service: brakes, fluids.",
     "intents": ["oil_change_when", "schedule_maintenance"]},
    {"id": "faq_capabilities","type": "FAQ",
     "label": "System Help",
     "content": "Jarvis assists with banking, travel, health, calendar, work PTO, restaurants, car, commute.",
     "intents": ["what_can_i_ask_you", "greeting"]},

    # ── Preference ────────────────────────────────────────────────────────────
    {"id": "pref_dietary",    "type": "Preference",
     "label": "Dietary Prefs",
     "content": "Priya: vegetarian, gluten-intolerant. Allergic to shellfish. Preferred: Italian and Indian. Dinner under 600 cal.",
     "intents": ["restaurant_suggestion", "meal_suggestion"]},
    {"id": "pref_commute",    "type": "Preference",
     "label": "Commute Prefs",
     "content": "Commutes Sandymount D4 to Grand Canal Dock D2. DART from Sandymount. Leaves 8am weekdays.",
     "intents": ["traffic", "directions"]},
    {"id": "pref_notifications","type": "Preference",
     "label": "Notification Prefs",
     "content": "Prefers email for all notifications (updated 30 May 2026). Quiet hours 10pm-7am.",
     "intents": ["reminder_update"]},
    {"id": "pref_music",      "type": "Preference",
     "label": "Music Prefs",
     "content": "Lo-fi hip hop and ambient for commute. Jazz evenings. Morning: low tempo instrumental. Dislikes heavy metal.",
     "intents": ["play_music"]},

    # ── Event ─────────────────────────────────────────────────────────────────
    {"id": "event_flight",    "type": "Event",
     "label": "Paris Flight",
     "content": "Ryanair FR2241 Dublin T1 to Paris CDG, 10 Jul 2026 07:45. Return FR2242 17 Jul 19:30. Ref RY7821345.",
     "intents": ["flight_status", "book_flight"]},
    {"id": "event_hotel",     "type": "Event",
     "label": "Paris Hotel",
     "content": "Hotel Le Marais Paris, 10-17 Jul 2026, Ref HLM2026. Free cancel until 8 Jul. Breakfast included.",
     "intents": ["book_hotel", "cancel_reservation"]},
    {"id": "event_pto",       "type": "Event",
     "label": "PTO Approved",
     "content": "PTO approved: 10-17 Jul 2026, 5 working days, Paris. Approved by Sarah Collins 20 May 2026. Ref HR2026PTO041.",
     "intents": ["pto_balance", "pto_request_status"]},
    {"id": "event_rent",      "type": "Event",
     "label": "Rent Due",
     "content": "Rent EUR 1,200 on 1st of each month to Dublin Lettings. Next due 1 Jun 2026. Direct debit active.",
     "intents": ["bill_due", "bill_balance"]},
    {"id": "event_fraud",     "type": "Event",
     "label": "Fraud Dispute",
     "content": "Fraud dispute FD2026001: EUR 250 unauthorised 29 May 2026. Temp credit issued. Resolution by 5 Jun 2026.",
     "intents": ["report_fraud"]},
    {"id": "event_medication","type": "Event",
     "label": "Medication Log",
     "content": "Daily Ramipril 5mg at 8am. Last dose 30 May 2026. Next dose 31 May 2026 08:00. Do not skip.",
     "intents": ["reminder", "reminder_update"]},
    {"id": "event_car",       "type": "Event",
     "label": "Car Service",
     "content": "Car service at Dublin Motors 15 Jun 2026 10am. Oil change, tyre rotation, brake check. Reg 211D12345.",
     "intents": ["schedule_maintenance"]},
    {"id": "event_course",    "type": "Event",
     "label": "ML Course",
     "content": "Advanced ML on Coursera, started 1 Jun 2026. 8-week duration. Deadline 26 Jul 2026.",
     "intents": ["todo_list", "calendar"]},

    # ── AccountState ──────────────────────────────────────────────────────────
    {"id": "acct_financial",  "type": "AccountState",
     "label": "Financial State",
     "content": "Account balance ~EUR 3,200. Credit score 710 (Good). Monthly income EUR 4,500 net. 17 PTO days remaining.",
     "intents": ["balance", "credit_score", "pto_balance"]},
    {"id": "acct_health",     "type": "AccountState",
     "label": "Health State",
     "content": "Age 29. Daily Ramipril 5mg for blood pressure. Health insurance VHI Plan B. Step goal 10,000/day.",
     "intents": ["vaccines", "insurance"]},
    {"id": "acct_work",       "type": "AccountState",
     "label": "Work State",
     "content": "Senior Data Analyst at TechCorp Dublin. Manager: Sarah Collins. Office: Grand Canal Dock. Remote Fridays.",
     "intents": ["pto_request", "make_call"]},
]

EDGES = [
    ("event_flight",    "faq_travel"),    ("event_hotel",    "faq_hotel"),
    ("event_fraud",     "faq_fraud"),     ("event_rent",     "acct_financial"),
    ("event_medication","faq_medication"),("event_pto",      "acct_work"),
    ("event_car",       "faq_car"),       ("pref_dietary",   "faq_diet"),
    ("pref_commute",    "acct_work"),     ("pref_notifications","acct_work"),
    ("acct_financial",  "acct_work"),     ("acct_health",    "faq_medication"),
    ("event_course",    "acct_work"),     ("event_fraud",    "acct_financial"),
    ("event_rent",      "faq_transfers"), ("faq_credit",     "acct_financial"),
]

def add_node(n):
    emg_graph.add_node(n["id"], type=n["type"], content=n["content"],
                       label=n["label"], intents=n.get("intents", []))
    emb = sbert.encode(n["content"]).tolist()
    emg_col.add(ids=[n["id"]], embeddings=[emb], documents=[n["content"]],
                metadatas=[{"node_type": n["type"], "label": n["label"],
                            "intents": json.dumps(n.get("intents", []))}])

for n in EMG_NODES:
    add_node(n)
for src, tgt in EDGES:
    emg_graph.add_edge(src, tgt)

# Restore persisted custom nodes (uploads + manual inserts)
_restored = 0
for _cn in _load_json(CUSTOM_NODES_F, []):
    if _cn.get("id") and _cn["id"] not in emg_graph.nodes:
        try:
            add_node(_cn)
            _restored += 1
        except Exception as _e:
            print(f"    [WARN] custom node {_cn.get('id')}: {_e}")
if _restored:
    print(f"    Restored {_restored} persisted custom nodes")

print(f"    EMG ready  ({emg_graph.number_of_nodes()} nodes, {emg_graph.number_of_edges()} edges)")

# ── 3. Intent routing table ───────────────────────────────────────────────────
INTENT_NODE_MAP = {
    "balance": ["AccountState","Event"], "transactions": ["Event","AccountState"],
    "transfer": ["FAQ","AccountState"],  "bill_balance": ["Event","AccountState"],
    "bill_due": ["Event","AccountState"],"pay_bill": ["Event","FAQ"],
    "credit_score": ["AccountState","FAQ"], "improve_credit_score": ["FAQ","AccountState"],
    "report_fraud": ["Event","FAQ"],     "card_declined": ["FAQ","AccountState"],
    "spending_history": ["AccountState","Event"], "income": ["AccountState"],
    "book_flight": ["Event","FAQ"],      "flight_status": ["Event"],
    "book_hotel": ["Event","FAQ"],       "cancel_reservation": ["Event","FAQ"],
    "travel_alert": ["FAQ","Event"],     "carry_on": ["FAQ"],
    "calendar": ["Event"],              "reminder": ["Event","Preference"],
    "reminder_update": ["Event","Preference"], "todo_list": ["Event"],
    "restaurant_suggestion": ["Preference","FAQ"], "meal_suggestion": ["Preference","FAQ"],
    "recipe": ["FAQ","Preference"],      "calories": ["FAQ","AccountState"],
    "vaccines": ["FAQ","AccountState"],  "insurance": ["AccountState","FAQ"],
    "pto_request": ["AccountState","Event"], "pto_balance": ["AccountState","Event"],
    "pto_request_status": ["Event","AccountState"], "schedule_maintenance": ["Event","FAQ"],
    "last_maintenance": ["Event","FAQ"], "traffic": ["Preference","AccountState"],
    "directions": ["Preference","AccountState"], "play_music": ["Preference"],
    "what_can_i_ask_you": ["FAQ"],       "greeting": ["FAQ","Preference"],
    "oos": ["FAQ"],
}

# ── 4. Gemini client ──────────────────────────────────────────────────────────
print("[3/4] Connecting Gemini...")
gc = None
if GOOGLE_API_KEY:
    try:
        from google import genai as _genai
        gc = _genai.Client(api_key=GOOGLE_API_KEY)
        print("    Gemini ready")
    except Exception as e:
        print(f"    Gemini unavailable: {e}")
else:
    print("    GOOGLE_API_KEY not set — will return mock responses")

MODEL = "gemini-2.5-flash"

def call_gemini(prompt):
    if gc is None:
        return "[Mock] Gemini not configured. Set GOOGLE_API_KEY in .env"
    try:
        return gc.models.generate_content(model=MODEL, contents=prompt).text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"

def count_tokens(prompt):
    if gc is None:
        return int(len(prompt.split()) * 1.3)
    try:
        return gc.models.count_tokens(model=MODEL, contents=prompt).total_tokens
    except:
        return int(len(prompt.split()) * 1.3)

# ── helpers ───────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def text_to_seq(text, w2i, maxlen=30):
    tokens = str(text).split()
    seq = [w2i.get(t, 1) for t in tokens]
    return (seq + [0] * maxlen)[:maxlen]

def classify_intent(query):
    cleaned = clean_text(query)
    t0 = time.time()
    seq = text_to_seq(cleaned, word_to_idx)
    xin = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logits = lstm_model(xin)
        probs  = torch.softmax(logits, dim=1)[0]
        pid    = probs.argmax().item()
        intent = le.classes_[pid]
        conf   = round(float(probs[pid].item()), 4)
    return intent, conf, round((time.time() - t0) * 1000, 1)

def flat_retrieve(query, top_k=3):
    q_emb = sbert.encode(query).tolist()
    n = min(top_k, emg_col.count())
    res = emg_col.query(query_embeddings=[q_emb], n_results=n)
    out = []
    if res["documents"] and res["documents"][0]:
        for doc, meta, nid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            out.append({"node_id": nid, "node_type": meta["node_type"],
                        "label": meta.get("label", nid), "content": doc})
    return out

def intent_retrieve(query, intent, top_k=2):
    node_types = INTENT_NODE_MAP.get(intent, ["FAQ"])
    q_emb = sbert.encode(query).tolist()
    wf = ({"node_type": node_types[0]} if len(node_types) == 1
          else {"node_type": {"$in": node_types}})
    n = min(top_k, emg_col.count())
    res = emg_col.query(query_embeddings=[q_emb], n_results=n, where=wf)
    out = []
    if res["documents"] and res["documents"][0]:
        for doc, meta, nid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            out.append({"node_id": nid, "node_type": meta["node_type"],
                        "label": meta.get("label", nid), "content": doc})

    # Also search uploaded Document nodes as additional context
    doc_count = sum(1 for _, d in emg_graph.nodes(data=True) if d.get("type") == "Document")
    if doc_count > 0:
        try:
            dr = emg_col.query(query_embeddings=[q_emb], n_results=min(2, doc_count),
                               where={"node_type": "Document"})
            if dr["documents"] and dr["documents"][0]:
                existing = {x["node_id"] for x in out}
                for doc, meta, nid in zip(dr["documents"][0], dr["metadatas"][0], dr["ids"][0]):
                    if nid not in existing:
                        out.append({"node_id": nid, "node_type": meta["node_type"],
                                    "label": meta.get("label", nid), "content": doc})
        except Exception:
            pass

    return out or flat_retrieve(query, top_k=top_k)

def _save_conversation(query, intent, response, tokens, total_ms):
    convs = _load_json(CONV_FILE, [])
    convs.append({
        "id": len(convs) + 1,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
        "query": query,
        "intent": intent,
        "response": response[:600],
        "tokens": tokens,
        "ms": total_ms,
    })
    _save_json(CONV_FILE, convs[-200:])

# ── Flask routes ──────────────────────────────────────────────────────────────
print("[4/4] Starting Flask...")

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"error": str(e), "trace": traceback.format_exc()[-500:]}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    type_cfg = {
        "FAQ":          {"bg": "#1565c0", "border": "#0d47a1", "font": "#ffffff", "shape": "star",     "size": 22},
        "Preference":   {"bg": "#2e7d32", "border": "#1b5e20", "font": "#ffffff", "shape": "diamond",  "size": 20},
        "Event":        {"bg": "#e65100", "border": "#bf360c", "font": "#ffffff", "shape": "triangle", "size": 22},
        "AccountState": {"bg": "#6a1b9a", "border": "#4a148c", "font": "#ffffff", "shape": "hexagon",  "size": 26},
        "Document":     {"bg": "#00838f", "border": "#006064", "font": "#ffffff", "shape": "box",      "size": 20},
    }
    vis_nodes, vis_edges = [], []
    type_counts = {}
    for nid, data in emg_graph.nodes(data=True):
        t = data.get("type", "FAQ")
        c = type_cfg.get(t, {"bg": "#555", "border": "#333", "font": "#fff", "shape": "dot", "size": 18})
        type_counts[t] = type_counts.get(t, 0) + 1
        vis_nodes.append({
            "id": nid,
            "label": data.get("label", nid),
            "group": t,
            "title": data.get("content", ""),
            "color": {
                "background": c["bg"], "border": c["border"],
                "highlight": {"background": "#ffeb3b", "border": "#f9a825"},
                "hover":     {"background": "#fff176", "border": "#f9a825"},
            },
            "font":  {"color": c["font"], "size": 12},
            "shape": c["shape"],
            "size":  c["size"],
            "shadow": {"enabled": True, "size": 8, "color": "rgba(0,0,0,0.6)"},
        })
    for src, tgt in emg_graph.edges():
        vis_edges.append({
            "from": src, "to": tgt, "arrows": "to",
            "color": {"color": "#2a2a50", "highlight": "#ffeb3b", "hover": "#7c4dff"},
            "width": 1.5,
            "smooth": {"type": "dynamic"},
        })
    return jsonify({
        "nodes": vis_nodes, "edges": vis_edges,
        "node_count": emg_graph.number_of_nodes(),
        "edge_count":  emg_graph.number_of_edges(),
        "type_counts": type_counts,
    })


_LOG = os.path.join(script_dir, "chat_debug.txt")

def _log(msg):
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    _log("=== request received ===")
    try:
        body_bytes = request.get_data()
        _log(f"raw body: {body_bytes[:200]}")
        data  = json.loads(body_bytes) if body_bytes else {}
        query = str(data.get("query", "")).strip()
        _log(f"query: {query!r}")
        if not query:
            return jsonify({"error": "Empty query"}), 400
    except Exception as e:
        _log(f"PARSE ERROR: {e}")
        return jsonify({"error": f"Bad request: {e}"}), 400
    try:
        result = _run_pipeline(query)
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _log(f"PIPELINE EXCEPTION:\n{tb}")
        return jsonify({"error": str(e), "trace": tb[-800:]}), 500


def _run_pipeline(query):
    steps = []
    t_total = time.time()

    # Step 1: preprocessing
    t0 = time.time()
    cleaned = clean_text(query)
    steps.append({"step": 1, "icon": "search",
                  "label": "Text Preprocessing",
                  "detail": f'Input: "{query[:60]}"  →  Cleaned: "{cleaned[:60]}"',
                  "ms": round((time.time()-t0)*1000, 1)})

    # Step 2: intent classification
    t0 = time.time()
    intent, conf, clf_ms = classify_intent(query)
    node_types = INTENT_NODE_MAP.get(intent, ["FAQ"])
    steps.append({"step": 2, "icon": "brain",
                  "label": "Intent Classification  (CLINC150 BiLSTM)",
                  "detail": f"Predicted intent: <b>{intent}</b>  |  Confidence: <b>{round(conf*100,1)}%</b>",
                  "ms": clf_ms, "badge": intent, "badge_color": "#e65100"})

    # Step 3: routing
    t0 = time.time()
    steps.append({"step": 3, "icon": "route",
                  "label": "Intent-to-Node-Type Router",
                  "detail": f"Intent <b>{intent}</b> → search node types: <b>{', '.join(node_types)}</b>",
                  "ms": round((time.time()-t0)*1000, 2), "node_types": node_types})

    # Step 4a: EMG retrieval
    t0 = time.time()
    c_nodes = intent_retrieve(query, intent, top_k=2)
    emg_ms  = round((time.time()-t0)*1000, 1)
    steps.append({"step": 4, "icon": "graph",
                  "label": "EMG Retrieval  (Intent-Guided)",
                  "detail": (f"Retrieved {len(c_nodes)} nodes from [{', '.join(node_types)}]: "
                             f"<b>{', '.join(n['node_id'] for n in c_nodes)}</b>"),
                  "ms": emg_ms,
                  "retrieved_ids": [n["node_id"] for n in c_nodes],
                  "retrieved_nodes": c_nodes})

    # Step 4b: flat retrieve
    b_nodes = flat_retrieve(query, top_k=3)

    # Step 5: generate all three conditions
    prompt_a = (f"You are Jarvis, a personalised AI assistant. Answer briefly.\n\n"
                f"Query: {query}\nAnswer:")
    tok_a = count_tokens(prompt_a)

    ctx_b    = "\n".join([f"- {n['content']}" for n in b_nodes]) or "No context."
    prompt_b = (f"You are Jarvis. Use only the context below.\n\n"
                f"Context:\n{ctx_b}\n\nQuery: {query}\nAnswer:")
    tok_b    = count_tokens(prompt_b)

    ctx_c    = "\n".join([f"[{n['node_type']}] {n['content']}" for n in c_nodes]) or "No context."
    prompt_c = (f"You are Jarvis, a personalised AI assistant.\n"
                f"Detected intent: {intent}\n"
                f"Use only the retrieved personalised information.\n\n"
                f"Retrieved Info:\n{ctx_c}\n\nQuery: {query}\nPersonalised Answer:")
    tok_c    = count_tokens(prompt_c)

    t0 = time.time()
    resp_a = call_gemini(prompt_a)
    resp_b = call_gemini(prompt_b)
    resp_c = call_gemini(prompt_c)
    gen_ms = round((time.time()-t0)*1000, 0)

    tok_a = int(tok_a); tok_b = int(tok_b); tok_c = int(tok_c)
    tok_reduction = round(float((tok_b - tok_c) / max(tok_b, 1) * 100), 1)

    steps.append({"step": 5, "icon": "sparkles",
                  "label": "Gemini 2.5 Flash — Grounded Generation",
                  "detail": (f"Tokens — A: <b>{tok_a}</b>  "
                              f"B (Flat RAG): <b>{tok_b}</b>  "
                              f"C (EMG-RAG): <b>{tok_c}</b>  "
                              f"({tok_reduction}% fewer than B)"),
                  "ms": int(gen_ms)})

    total_ms = int(round((time.time()-t_total)*1000, 0))

    # Save conversation
    try:
        _save_conversation(query, intent, resp_c, tok_c, total_ms)
    except Exception:
        pass

    def clean_node(n):
        return {"node_id": str(n["node_id"]), "node_type": str(n["node_type"]),
                "label": str(n.get("label", n["node_id"])), "content": str(n["content"])}

    payload = {
        "query":        str(query),
        "intent":       str(intent),
        "confidence":   float(conf),
        "node_types":   [str(t) for t in node_types],
        "steps": [{k: (str(v) if isinstance(v, list) else v) for k, v in s.items()} for s in steps],
        "conditions": {
            "A": {"response": str(resp_a), "tokens": tok_a,
                  "nodes": [], "label": "Direct LLM (no context)"},
            "B": {"response": str(resp_b), "tokens": tok_b,
                  "nodes": [clean_node(n) for n in b_nodes],
                  "label": "Flat RAG (cosine, 3 docs)"},
            "C": {"response": str(resp_c), "tokens": tok_c,
                  "nodes": [clean_node(n) for n in c_nodes],
                  "label": "Intent EMG-RAG (typed, 2 nodes)"},
        },
        "graph_highlights": {
            "emg_nodes":  [str(n["node_id"]) for n in c_nodes],
            "flat_nodes": [str(n["node_id"]) for n in b_nodes],
        },
        "token_reduction": tok_reduction,
        "total_ms":        total_ms,
    }
    return Response(
        json.dumps(payload, ensure_ascii=False, default=str),
        mimetype="application/json", status=200,
    )


@app.route("/api/crud", methods=["POST"])
def api_crud():
    data = request.get_json(force=True)
    op   = data.get("op", "").upper()
    t0   = time.time()

    if op == "INSERT":
        nid     = data["node_id"]
        ntype   = data["node_type"]
        content = data["content"]
        label   = data.get("label", nid)
        if nid in emg_graph.nodes:
            return jsonify({"error": f"Node {nid} already exists"}), 400
        with _graph_lock:
            emg_graph.add_node(nid, type=ntype, content=content, label=label, intents=[])
            emb = sbert.encode(content).tolist()
            emg_col.add(ids=[nid], embeddings=[emb], documents=[content],
                        metadatas=[{"node_type": ntype, "label": label, "intents": "[]"}])
        cn = _load_json(CUSTOM_NODES_F, [])
        cn.append({"id": nid, "type": ntype, "content": content, "label": label, "intents": []})
        _save_json(CUSTOM_NODES_F, cn)

    elif op == "UPDATE":
        nid   = data["node_id"]
        if nid not in emg_graph.nodes:
            return jsonify({"error": f"Node {nid} not found"}), 404
        node    = emg_graph.nodes[nid]
        content = data.get("content", node.get("content", ""))
        label   = data.get("label", node.get("label", nid)) or node.get("label", nid)
        with _graph_lock:
            node["content"] = content
            node["label"]   = label
            emg_col.update(
                ids=[nid],
                embeddings=[sbert.encode(content).tolist()],
                documents=[content],
                metadatas=[{"node_type": node.get("type","FAQ"), "label": label,
                            "intents": json.dumps(node.get("intents", []))}],
            )
        cn = _load_json(CUSTOM_NODES_F, [])
        for n in cn:
            if n.get("id") == nid:
                n["content"] = content
                n["label"]   = label
                break
        _save_json(CUSTOM_NODES_F, cn)

    elif op == "DELETE":
        nid = data["node_id"]
        if nid not in emg_graph.nodes:
            return jsonify({"error": f"Node {nid} not found"}), 404
        with _graph_lock:
            emg_graph.remove_node(nid)
            emg_col.delete(ids=[nid])
        cn = _load_json(CUSTOM_NODES_F, [])
        _save_json(CUSTOM_NODES_F, [n for n in cn if n.get("id") != nid])
        files = _load_json(FILES_META_F, [])
        for f in files:
            if nid in f.get("node_ids", []):
                f["node_ids"].remove(nid)
        _save_json(FILES_META_F, files)
    else:
        return jsonify({"error": "op must be INSERT|UPDATE|DELETE"}), 400

    ms = round((time.time()-t0)*1000, 2)
    return jsonify({"op": op, "node_id": data.get("node_id"), "ms": ms,
                    "total_nodes": emg_graph.number_of_nodes(),
                    "message": f"{op} completed in {ms} ms"})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    fobj = request.files["file"]
    if not fobj or not fobj.filename:
        return jsonify({"error": "Empty filename"}), 400

    original_name = fobj.filename
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    safe_stem  = re.sub(r"[^\w\-]", "_", os.path.splitext(original_name)[0])[:40]
    ts         = int(time.time())
    saved_name = f"{ts}_{safe_stem}{ext}"
    filepath   = os.path.join(UPLOADS_DIR, saved_name)
    fobj.save(filepath)

    text = _extract_text(filepath, original_name)
    if not text.strip():
        os.remove(filepath)
        return jsonify({"error": "Could not extract text. Check file has readable content."}), 400

    chunks  = _chunk_text(text)[:8]
    base_id = re.sub(r"[^\w]", "_", safe_stem)[:20]

    created = []
    with _graph_lock:
        for i, chunk in enumerate(chunks):
            nid  = f"doc_{base_id}_{ts}_{i}"
            name = original_name
            short_name = (name[:22] + "…") if len(name) > 25 else name
            node = {
                "id":          nid,
                "type":        "Document",
                "label":       f"{short_name} §{i+1}",
                "content":     chunk[:800],
                "intents":     [],
                "source_file": original_name,
            }
            if nid not in emg_graph.nodes:
                add_node(node)
                created.append(node)

    if not created:
        return jsonify({"error": "No nodes could be created (duplicates?)"}), 400

    files_meta = _load_json(FILES_META_F, [])
    files_meta.append({
        "original_name": original_name,
        "saved_as":      saved_name,
        "size_bytes":    os.path.getsize(filepath),
        "uploaded_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "node_ids":      [n["id"] for n in created],
        "chunks":        len(created),
    })
    _save_json(FILES_META_F, files_meta)

    cn = _load_json(CUSTOM_NODES_F, [])
    cn.extend(created)
    _save_json(CUSTOM_NODES_F, cn)

    return jsonify({
        "filename": original_name,
        "chunks":   len(created),
        "nodes":    [n["id"] for n in created],
        "message":  f"Created {len(created)} knowledge nodes from '{original_name}'",
        "total_nodes": emg_graph.number_of_nodes(),
    })


@app.route("/api/file/delete", methods=["POST"])
def api_delete_file():
    data     = request.get_json(force=True)
    original = data.get("filename", "")
    files    = _load_json(FILES_META_F, [])
    target   = next((f for f in files if f["original_name"] == original), None)
    if not target:
        return jsonify({"error": "File not found"}), 404

    removed = 0
    with _graph_lock:
        for nid in target.get("node_ids", []):
            if nid in emg_graph.nodes:
                emg_graph.remove_node(nid)
                try:  emg_col.delete(ids=[nid])
                except: pass
                removed += 1

    _save_json(FILES_META_F, [f for f in files if f["original_name"] != original])
    cn = _load_json(CUSTOM_NODES_F, [])
    _save_json(CUSTOM_NODES_F, [n for n in cn if n.get("source_file") != original])

    saved_path = os.path.join(UPLOADS_DIR, target.get("saved_as", ""))
    if os.path.exists(saved_path):
        os.remove(saved_path)

    return jsonify({"message": f"Deleted '{original}' and {removed} nodes",
                    "total_nodes": emg_graph.number_of_nodes()})


@app.route("/api/conversations")
def api_conversations():
    convs = _load_json(CONV_FILE, [])
    return jsonify({"conversations": convs[-50:][::-1]})


@app.route("/api/files")
def api_files():
    files = _load_json(FILES_META_F, [])
    return jsonify({"files": files[::-1]})


@app.route("/api/node/<node_id>", methods=["GET"])
def api_get_node(node_id):
    if node_id not in emg_graph.nodes:
        return jsonify({"error": "Not found"}), 404
    d = emg_graph.nodes[node_id]
    return jsonify({"id": node_id, "type": d.get("type","FAQ"),
                    "label": d.get("label", node_id), "content": d.get("content","")})


@app.route("/api/node/<node_id>", methods=["PUT"])
def api_edit_node(node_id):
    if node_id not in emg_graph.nodes:
        return jsonify({"error": "Not found"}), 404
    data        = request.get_json(force=True)
    node        = emg_graph.nodes[node_id]
    new_content = data.get("content", node.get("content",""))
    new_label   = data.get("label",   node.get("label", node_id))
    with _graph_lock:
        node["content"] = new_content
        node["label"]   = new_label
        existing_intents = json.dumps(node.get("intents", []))
        emg_col.update(
            ids=[node_id],
            embeddings=[sbert.encode(new_content).tolist()],
            documents=[new_content],
            metadatas=[{"node_type": node.get("type","FAQ"), "label": new_label,
                        "intents": existing_intents}],
        )
    cn = _load_json(CUSTOM_NODES_F, [])
    for n in cn:
        if n.get("id") == node_id:
            n["content"] = new_content
            n["label"]   = new_label
            break
    _save_json(CUSTOM_NODES_F, cn)
    return jsonify({"message": "Updated", "node_id": node_id})


if __name__ == "__main__":
    print("\n  Open browser at  http://localhost:5000\n")
    try:
        from waitress import serve
        print("  Using Waitress WSGI server\n")
        serve(app, host="0.0.0.0", port=5000, threads=4)
    except ImportError:
        print("  Using Werkzeug dev server\n")
        app.run(debug=False, port=5000, threaded=True, use_reloader=False)
