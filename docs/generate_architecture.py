# generate_architecture.py
# Generates a high-quality system architecture PNG for inclusion in the thesis Word document.
# Run this before generate_report.py.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "outputs", "system_architecture.png")
os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)

# ─── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ─── helpers ────────────────────────────────────────────────────────────────────

def region(ax, x, y, w, h, title, bg, border, title_color=None):
    tc = title_color or border
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       linewidth=2.2, edgecolor=border,
                       facecolor=bg, alpha=0.28, zorder=1)
    ax.add_patch(p)
    ax.text(x + 0.22, y + h - 0.22, title, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=tc, zorder=3)


def box(ax, x, y, w, h, text, bg, border, tsize=8.5, bold=False, tcolor="black", zorder=4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       linewidth=1.6, edgecolor=border, facecolor=bg, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=tsize, fontweight="bold" if bold else "normal",
            color=tcolor, multialignment="center", zorder=zorder + 1)


def arr(ax, x1, y1, x2, y2, color="#555", lw=1.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=6)


def darr(ax, x1, y1, x2, y2, color="#555", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"),
                zorder=6)

# ──────────────────────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────────────────────
ax.text(9, 23.65, "Intent-Guided EMG-RAG: System Architecture",
        ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a1a")
ax.text(9, 23.25, "Knowledge Graph-Based Editable Memory for Token-Efficient Personalised AI Assistants",
        ha="center", va="top", fontsize=10, style="italic", color="#444")

# ──────────────────────────────────────────────────────────────────────────────
# USER QUERY
# ──────────────────────────────────────────────────────────────────────────────
box(ax, 5.8, 22.0, 6.4, 0.75,
    "User Natural Language Query",
    "#f3e5f5", "#6a1b9a", tsize=11, bold=True)
arr(ax, 9, 22.0, 9, 21.4)

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 — INTENT CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
region(ax, 0.3, 15.9, 17.4, 5.3,
       "Layer 1  —  Intent Classification   "
       "(CLINC150: 150 intents · 10 life domains · 22,500 queries · peer-reviewed benchmark)",
       "#e3f2fd", "#1565c0")

# preprocessing
box(ax, 3.8, 20.5, 10.4, 0.65,
    "Text Preprocessing:  lowercase  ·  remove punctuation  ·  collapse whitespace",
    "#bbdefb", "#1565c0", tsize=9)
arr(ax, 7, 20.5, 5.8, 20.1)   # → TF-IDF
arr(ax, 11, 20.5, 12.2, 20.1)  # → Word seq

# feature boxes
box(ax, 3.0, 19.3, 5, 0.65,
    "TF-IDF Vectorizer\n8,000 terms  ·  unigram-bigram",
    "#bbdefb", "#1565c0", tsize=8.5)
box(ax, 10.0, 19.3, 5, 0.65,
    "Word Sequence Encoder\nvocab 12k  ·  maxlen 30 tokens",
    "#bbdefb", "#1565c0", tsize=8.5)

# arrow from TF-IDF to LR and RF
arr(ax, 4.2, 19.3, 2.6, 18.75)
arr(ax, 6.4, 19.3, 6.4, 18.75)
# arrow from word seq to LSTM
arr(ax, 12.5, 19.3, 12.5, 18.75)

# classifiers
box(ax, 0.7, 17.35, 3.6, 1.25,
    "Logistic Regression\n\nAccuracy: 82.78%  ✓ Best\nlbfgs · C=1.0",
    "#1565c0", "#0d47a1", tsize=8.5, bold=True, tcolor="white")

box(ax, 4.6, 17.35, 3.6, 1.25,
    "Random Forest\n\nAccuracy: 78.44%\n200 estimators",
    "#1976d2", "#1565c0", tsize=8.5, tcolor="white")

box(ax, 8.7, 17.35, 5.5, 1.25,
    "Bidirectional LSTM\n\nAccuracy: 75.53%\nembed 128d · hidden 128 · dropout 0.3",
    "#2196f3", "#1565c0", tsize=8.5, tcolor="white")

# CLINC150 badge
box(ax, 14.5, 17.35, 3.0, 1.25,
    "CLINC150\n22,500 queries\n150 intents\n10 domains",
    "#e8eaf6", "#3949ab", tsize=8)
darr(ax, 14.5, 18.0, 14.2, 18.0, color="#3949ab")

# intent output
box(ax, 5.3, 16.0, 7.4, 0.65,
    "Predicted Intent Label  +  Confidence Score",
    "#e8f5e9", "#2e7d32", tsize=10, bold=True)

# arrows classifier → intent output
arr(ax, 2.5, 17.35, 7.0, 16.65, color="#2e7d32")
arr(ax, 6.4, 17.35, 7.5, 16.65, color="#2e7d32")
arr(ax, 11.4, 17.35, 10.0, 16.65, color="#2e7d32")

arr(ax, 9, 16.0, 9, 15.4)

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 — INTENT ROUTER
# ──────────────────────────────────────────────────────────────────────────────
region(ax, 0.3, 13.65, 17.4, 1.55,
       "Layer 2  —  Intent-to-Node-Type Router   "
       "(Novel Contribution: classifier output becomes typed retrieval signal)",
       "#fff3e0", "#e65100")

box(ax, 0.7, 13.75, 7.8, 1.1,
    "CLINC150 Intent Mapping Table  (150 intents → 4 node types)\n"
    "balance → [AccountState, Event]   ·   report_fraud → [Event, FAQ]\n"
    "meal_suggestion → [Preference, FAQ]   ·   calendar → [Event]",
    "#ffe0b2", "#e65100", tsize=8.3)

box(ax, 9.0, 13.75, 8.4, 1.1,
    "Selected Target Node Types  (1 – 2 types per query)\n"
    "{ FAQ  ·  Preference  ·  Event  ·  AccountState }\n"
    "→ Restricts ChromaDB search space  before  similarity scoring",
    "#fff8e1", "#e65100", tsize=8.5, bold=True)

arr(ax, 8.5, 14.3, 9.0, 14.3, color="#e65100", lw=2)
arr(ax, 9, 13.65, 9, 13.1)

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — EDITABLE MEMORY GRAPH
# ──────────────────────────────────────────────────────────────────────────────
region(ax, 0.3, 7.5, 17.4, 5.95,
       "Layer 3  —  Editable Memory Graph   "
       "(NetworkX graph structure  +  ChromaDB vector store  +  all-MiniLM-L6-v2 embeddings)",
       "#e8f5e9", "#2e7d32")

# ChromaDB search bar
box(ax, 1.0, 12.2, 12.2, 0.65,
    "ChromaDB Filtered Vector Search  ·  cosine similarity within selected node types only"
    "   ←  Solves Problem 3: Intent-Context Mismatch",
    "#c8e6c9", "#2e7d32", tsize=8.5)
arr(ax, 7.1, 12.2, 7.1, 11.7)   # to node boxes

# 4 node type boxes
NODE_Y = 10.35
for i, (title, count, desc, bg) in enumerate([
    ("FAQ\nNodes",         "(10)", "Domain policy\nknowledge\nRarely changes",   "#4caf50"),
    ("Preference\nNodes",  "(4)",  "User-specific\nsettings\nEditable anytime",  "#66bb6a"),
    ("Event\nNodes",       "(6+)", "Timestamped\npersonal events\nHigh-change",  "#81c784"),
    ("AccountState\nNodes","(2)",  "Current profile\nfacts\nPeriodically updated","#a5d6a7"),
]):
    bx = 1.0 + i * 3.1
    box(ax, bx, NODE_Y, 2.85, 1.6,
        f"{title}\n{count}\n{desc}",
        bg, "#2e7d32", tsize=8, tcolor="white")
    arr(ax, bx + 1.4, 11.7, bx + 1.4, 11.95, color="#2e7d32")
    arr(ax, bx + 1.4, NODE_Y, bx + 1.4, 9.85, color="#2e7d32")

# Top-2 nodes
box(ax, 2.0, 8.6, 10.0, 0.85,
    "Top-2 Typed Nodes  ·  Most Relevant Personalised Context Retrieved",
    "#dcedc8", "#558b2f", tsize=10, bold=True)

# arrow from nodes to top-2
arr(ax, 7, 9.85, 7, 9.45, color="#2e7d32")

# ── CRUD side panel ────────────────────────────────────────────────────────────
region(ax, 13.5, 7.5, 4.15, 5.95,
       "Dynamic\nMemory\nManagement", "#f1f8e9", "#1b5e20")

box(ax, 13.7, 11.8, 3.7, 0.75,
    "Real-World Memory Updates",
    "#c8e6c9", "#1b5e20", tsize=8.5)

box(ax, 13.7, 10.85, 3.7, 0.82,
    "INSERT  new node\n< 21 ms  (embed + graph + chroma)",
    "#a5d6a7", "#1b5e20", tsize=8)

box(ax, 13.7, 9.9, 3.7, 0.82,
    "UPDATE  node content\n< 21 ms  (re-embed + chroma update)",
    "#b9f6ca", "#1b5e20", tsize=8)

box(ax, 13.7, 8.95, 3.7, 0.82,
    "DELETE  node\n< 8 ms  (graph remove + chroma delete)",
    "#ccff90", "#1b5e20", tsize=8)

box(ax, 13.7, 7.75, 3.7, 0.9,
    "[OK]  No Model Retraining\n[OK]  Instant effect on next query",
    "#f1f8e9", "#1b5e20", tsize=8, bold=True)

# dashed arrows from CRUD to node columns
darr(ax, 13.5, 11.25, 12.85, 11.25, color="#1b5e20")
darr(ax, 13.5, 10.3, 12.85, 10.3,  color="#1b5e20")
darr(ax, 13.5, 9.35, 12.85, 9.35,  color="#1b5e20")

ax.text(13.1, 10.5, "live\nupdates", ha="center", va="center",
        fontsize=7, style="italic", color="#1b5e20")

# arrow from Top-2 nodes to Layer 4
arr(ax, 7, 8.6, 7, 8.0, color="#558b2f", lw=1.8)

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4 — RESPONSE GENERATION
# ──────────────────────────────────────────────────────────────────────────────
region(ax, 0.3, 1.4, 17.4, 6.35,
       "Layer 4  —  Response Generation   "
       "(Gemini 2.5 Flash  via  Google AI SDK)",
       "#fce4ec", "#880e4f")

# Prompt builder
box(ax, 0.7, 5.7, 5.3, 1.7,
    "Prompt Builder\n\n"
    "[System: You are Jarvis,\n  intent={detected_intent}]\n"
    "[{NodeType}: {content_node_1}]\n"
    "[{NodeType}: {content_node_2}]\n"
    "[Query: {user_query}]",
    "#f8bbd0", "#880e4f", tsize=8)

# Token counter
box(ax, 6.4, 5.7, 5.3, 1.7,
    "Token Counter\nGemini  count_tokens  API\n"
    "────────────────────────────\n"
    "Cond A  (no context):   28.4 avg\n"
    "Cond B  (flat RAG):   157.1 avg\n"
    "Cond C  (EMG-RAG):   145.4 avg\n"
    "→ 7.5% fewer tokens than B",
    "#fce4ec", "#880e4f", tsize=8)

# Gemini box
box(ax, 12.0, 5.7, 5.4, 1.7,
    "Gemini 2.5 Flash\n\n"
    "Grounded generation\nfrom retrieved evidence\n"
    "(reduces hallucination\nvs parametric recall)",
    "#f48fb1", "#880e4f", tsize=8.5)

arr(ax, 6.0,  6.6, 6.4,  6.6,  color="#880e4f")
arr(ax, 11.7, 6.6, 12.0, 6.6,  color="#880e4f")
arr(ax, 14.7, 5.7, 14.7, 5.15, color="#880e4f")

# Confidence check
box(ax, 12.4, 4.2, 4.5, 0.9,
    "Confidence Check  +  Fallback Mechanism",
    "#fff9c4", "#f57f17", tsize=9)
arr(ax, 14.7, 4.2, 14.7, 3.7,  color="#f57f17")

# Output branches
box(ax, 11.0, 2.8, 3.0, 0.78,
    "[OK] Grounded Response\nROUGE-L 0.587 · BLEU 0.264",
    "#c8e6c9", "#2e7d32", tsize=8, bold=True)

box(ax, 15.2, 2.8, 2.3, 0.78,
    "[!] Template\nFallback",
    "#ffccbc", "#d84315", tsize=8.5)

arr(ax, 13.8, 3.7, 12.5, 3.58, color="#2e7d32")
arr(ax, 15.5, 3.7, 16.0, 3.58, color="#d84315")

# Evaluation summary box (bottom left)
box(ax, 0.7, 1.55, 10.5, 3.9,
    "Evaluation Results  (20 synthetic personalised queries  ·  6 life domains)\n\n"
    "Condition A — Direct LLM (no context):   ROUGE-L 0.107  ·  BLEU 0.009  ·  28.4 avg tokens\n"
    "Condition B — Flat RAG (cosine, 3 docs):  ROUGE-L 0.532  ·  BLEU 0.213  · 157.1 avg tokens\n"
    "Condition C — Intent EMG-RAG (2 nodes):   ROUGE-L 0.587  ·  BLEU 0.264  · 145.4 avg tokens\n\n"
    "Intent Classification on CLINC150  (5,500 test queries · 150 intents):\n"
    "LR 82.78%  ·  RF 78.44%  ·  BiLSTM 75.53%    "
    "|   Published dual-encoder baseline: ~92%\n\n"
    "CRUD Latency:  INSERT 16.9 ms  ·  UPDATE 20.2 ms  ·  DELETE 7.5 ms  (all < 100 ms)",
    "#fce4ec", "#880e4f", tsize=8.2)

# Curved dashed arrow: query → prompt builder (showing query feeds into prompt)
ax.annotate("", xy=(2.0, 5.7), xytext=(7.0, 22.0),
            arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.2,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=0.35"),
            zorder=6)
ax.text(0.7, 14.3, "Query also\nfed to\nPrompt Builder",
        ha="center", va="top", fontsize=6.5, style="italic", color="#6a1b9a")

# Curved dashed arrow: intent label → prompt builder
ax.annotate("", xy=(3.4, 7.4), xytext=(8.5, 16.0),
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=-0.25"),
            zorder=6)
ax.text(2.0, 11.8, "Intent\nalso fed\nto Prompt",
        ha="center", va="top", fontsize=6.5, style="italic", color="#2e7d32")

# ─── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#e3f2fd", edgecolor="#1565c0", label="Layer 1: Intent Classification"),
    mpatches.Patch(facecolor="#fff3e0", edgecolor="#e65100", label="Layer 2: Intent Router (Novel)"),
    mpatches.Patch(facecolor="#e8f5e9", edgecolor="#2e7d32", label="Layer 3: Editable Memory Graph"),
    mpatches.Patch(facecolor="#fce4ec", edgecolor="#880e4f", label="Layer 4: Response Generation"),
]
ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, 0.0),
          fontsize=8.5, framealpha=0.9, edgecolor="#888")

plt.tight_layout(pad=0.5)
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Architecture diagram saved -> " + output_path)
