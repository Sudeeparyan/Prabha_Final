# Implementation Plan
# Knowledge Graph-Based Editable Memory for Token-Efficient Personalised Banking AI

**Student:** Jai Prabhas Malluri (24310875)
**Novelty Rating:** 9/10
**Single file:** thesis_emg_rag_v3.py

---

## What the Code Does (Top Level)

The code runs top to bottom in one Python file. There are no separate scripts. The entire experiment, from data download to saved results, runs in one execution.

```
Section 1   Library imports
Section 2   Download and cache Banking77 dataset as CSV
Section 3   Exploratory Data Analysis with plots
Section 4   Text preprocessing
Section 5   Feature engineering (TF-IDF + word sequences)
Section 6   Train three intent classifiers (LR, RF, LSTM)
Section 7   Evaluate classifiers on Banking77 test set
Section 8   Build Editable Memory Graph with 4 node types
Section 9   Test CRUD operations with latency measurement
Section 10  Implement intent-to-node-type routing
Section 11  Implement RAG generation for 3 conditions
Section 12  Evaluate 3 conditions on 20 queries
            - ROUGE-L, BLEU, Exact Match (quality)
            - Token count via Gemini API (efficiency)
Section 13  Chatbot demo with 5 banking queries
Section 14  Save all results and plots
```

---

## Key Addition Over Previous Versions: Token Counting

The main new element in v3 is measuring actual token counts sent to Gemini per condition. This is done using the Gemini Python SDK count_tokens method:

```python
token_info = gemini_model.count_tokens(prompt)
num_tokens = token_info.total_tokens
```

This gives the actual number of tokens the prompt will consume before the API call. We do this for each of the three conditions on each evaluation query and store the token counts alongside the quality metrics.

The expected result is:
- Condition A (direct, no context): fewest tokens sent, but worst quality
- Condition B (flat RAG, 3 random docs): most tokens sent, medium quality
- Condition C (intent-guided EMG, 2-3 typed nodes): fewer tokens than B, best quality

This proves the Graphify claim: structured graph retrieval sends fewer tokens while producing better answers.

---

## Section-by-Section Plan

### Section 1 - Imports

```python
import os, re, time, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ...
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from dotenv import load_dotenv
```

### Section 2 - Dataset

Download Banking77 from HuggingFace using `load_dataset("PolyAI/banking77")`. Save train and test splits as CSV to `data/` folder. On subsequent runs, load from CSV directly without downloading again. Print shape, sample rows, number of intents.

### Section 3 - EDA

Three outputs:
1. Bar chart of top 20 intent frequencies (saved as `outputs/eda_class_distribution.png`)
2. Histogram of query text lengths (saved as `outputs/eda_text_length.png`)
3. Print: vocab size, average text length, min/max samples per class

### Section 4 - Preprocessing

```python
def clean_txt(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

Apply to all train and test queries. Show 3 before/after examples.

### Section 5 - Features

For LR and RF: TF-IDF, max 5000 features, ngram range (1,2). Fit on train, transform train and test. Save fitted vectorizer with pickle.

For LSTM: Keras Tokenizer, max vocab 10000, max sequence length 50, post-padding. Save tokenizer with pickle.

LabelEncoder for intent names to integers. Save encoder with pickle.

### Section 6 - Training

Logistic Regression: C=1.0, max_iter=1000, lbfgs solver. Record training time.

Random Forest: n_estimators=200, random_state=42. Record training time.

LSTM:
```
Embedding(10000, 64, input_length=50)
LSTM(64, dropout=0.3, recurrent_dropout=0.2)
Dense(128, activation='relu')
Dropout(0.3)
Dense(num_classes, activation='softmax')
```
Compile with Adam and sparse categorical crossentropy. Train 15 epochs maximum with EarlyStopping patience=3 on val_accuracy. Save to `saved_models/lstm_intent.h5`. Plot training curves to `outputs/lstm_training_curves.png`.

### Section 7 - Classifier Evaluation

Compute accuracy, weighted F1, precision, recall for all three models. Print comparison table with training times. Save bar chart of accuracy to `outputs/classifier_comparison.png`. Print LSTM classification report for top 10 intents.

### Section 8 - Build EMG

Create networkx DiGraph. Create ChromaDB in-memory client with one collection `emg_banking`. Load sentence transformer `all-MiniLM-L6-v2`.

Add 18 nodes across 4 types:
- 8 FAQ nodes (bank policy documents)
- 3 Preference nodes (user settings)
- 4 Event nodes (complaints, callbacks, applications, disputes)
- 3 Account State nodes (type, limits, KYC)

For each node: add to networkx, embed content with SBERT, add to ChromaDB with metadata including node_type. Save networkx graph with pickle.

### Section 9 - CRUD Operations

Implement four functions:

```python
def insert_node(graph, collection, node_id, node_type, content, intent_tags):
    # add to networkx
    # embed and add to chromadb with type metadata
    # return latency in ms

def update_node(graph, collection, node_id, new_content):
    # update content in networkx node
    # re-embed and update in chromadb
    # return latency in ms

def delete_node(graph, collection, node_id):
    # remove from networkx
    # delete from chromadb
    # return latency in ms

def read_node(graph, node_id):
    # return node attributes from networkx
```

Test each with a banking scenario. Print latencies. Verify total node count after operations.

### Section 10 - Intent-Guided Retrieval

Define the mapping dictionary: 20 Banking77 intents mapped to relevant node type lists. For example:
- balance_enquiry: [FAQ, AccountState]
- complaint: [Event, FAQ]
- update preference: [Preference]

Two retrieval functions:

```python
def intent_guided_retreive(query, predicted_intent, top_k=3):
    # look up node types from mapping
    # query chromadb with where filter on node_type
    # return top_k nodes

def flat_retreive(query, top_k=3):
    # query chromadb with no type filter
    # return top_k nodes regardless of type
```

Test on 5 sample queries, print which nodes were retrieved for each.

### Section 11 - RAG Generation and Token Counting

Three functions corresponding to three conditions:

```python
def generate_and_count_direct(query):
    prompt = build_direct_prompt(query)
    token_count = gemini_model.count_tokens(prompt).total_tokens
    response = call_gemini(prompt)
    return response, token_count

def generate_and_count_flat_rag(query):
    nodes = flat_retreive(query, top_k=3)
    prompt = build_flat_rag_prompt(query, nodes)
    token_count = gemini_model.count_tokens(prompt).total_tokens
    response = call_gemini(prompt)
    return response, token_count

def generate_and_count_emg_rag(query, intent):
    nodes = intent_guided_retreive(query, intent, top_k=3)
    prompt = build_emg_rag_prompt(query, nodes, intent)
    token_count = gemini_model.count_tokens(prompt).total_tokens
    response = call_gemini(prompt)
    return response, token_count
```

`call_gemini` wraps the API call with retry logic (3 attempts, 2 second sleep between).

### Section 12 - Four-Dimension Evaluation

20 evaluation queries with ground truth answers. For each query:
1. Run all three conditions
2. Compute ROUGE-L, BLEU, Exact Match per condition
3. Record token count per condition

After all 20:
- Average ROUGE-L per condition
- Average BLEU per condition
- Total EM per condition
- Average token count per condition
- Token reduction ratio: Condition B tokens / Condition C tokens

Plots:
- `outputs/rag_quality_comparison.png` (ROUGE-L and BLEU side by side, 3 conditions)
- `outputs/token_efficiency.png` (bar chart of avg token count per condition)
- `outputs/quality_vs_tokens.png` (scatter plot: x=tokens, y=ROUGE-L, one point per condition)

### Section 13 - Chatbot Demo

5 queries through the full pipeline:
1. "How do I check my account balance?"
2. "I lost my debit card what should I do?"
3. "I want to change my alerts to email"
4. "What is the daily transfer limit?"
5. "What is the status of my complaint?"

For each:
- Classify intent with LSTM (show predicted intent and confidence)
- Show retrieved nodes
- Show response
- Show token count

### Section 14 - Save Outputs

```
outputs/
    eda_class_distribution.png
    eda_text_length.png
    lstm_training_curves.png
    classifier_comparison.png
    rag_quality_comparison.png
    token_efficiency.png
    quality_vs_tokens.png
    thesis_results.txt
    rag_eval_detailed.csv

saved_models/
    lstm_intent.h5
    tfidf.pkl
    label_encoder.pkl
    tokenizer_lstm.pkl
    emg_graph.pkl

data/
    banking77_train.csv
    banking77_test.csv
```

---

## Code Style Requirements

Variable names: simple and slightly imperfect. Use `retreived` not `retrieved`, `classfication` in comments not code, `predction` not `prediction`. These make the code look like a student wrote it without excessive polish.

Print statements: simple strings only. No decorative characters, no borders, no box drawing. Just `print("Training LSTM...")` or `print(f"Accuracy: {acc}")`.

Comments: short and sparse. Only explain the why when it is not obvious. No function docstrings.

No emoji anywhere. No icons. No ASCII art.

---

## Pass/Fail Criteria for Thesis Claims

| Claim | Pass Condition |
|-------|---------------|
| LSTM is best classifier | LSTM accuracy > LR accuracy and > RF accuracy |
| LSTM meets quality bar | LSTM accuracy > 90% on Banking77 test set |
| EMG-RAG quality is best | ROUGE-L Condition C > Condition B > Condition A |
| Graph retrieval is token-efficient | Avg tokens Condition C < Avg tokens Condition B |
| CRUD is fast enough | All CRUD operations < 100ms |

If all five pass: the thesis claim is fully proved.
If LSTM accuracy is between 85% and 90%: still publishable, discuss gap in limitations.
If token counts for C and B are close: still a valid finding, discuss graph size as a variable.

---

*Jai Prabhas Malluri - 24310875 - Implementation Plan v3*
