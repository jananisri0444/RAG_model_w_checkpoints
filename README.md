# RAG_model_w_checkpoints
# ConvoRAG — Conversation Intelligence System (100% Local, No API Keys)

A full RAG pipeline that processes conversation data chronologically, detects topic shifts, builds a user persona, and powers an intelligent chatbot — **entirely offline, no paid API needed**.

---

## What Changed vs. the Original

| Component | Before | After |
|---|---|---|
| **Embeddings** | TF-IDF + SVD (LSA, sklearn) | `all-MiniLM-L6-v2` via `sentence-transformers` |
| **Chat responses** | Anthropic Claude API (paid) | Local intent-aware response builder |
| **API key required** | Yes (`ANTHROPIC_API_KEY`) | **No** |
| **Internet at runtime** | Yes | **No** (model cached locally after first download) |

---

## Architecture

```
conversations.csv
       │
       ▼
 rag_processor.py
 ├── parse_conversations()       → flat chronological message list
 ├── build_embeddings()          → sentence-transformers all-MiniLM-L6-v2
 ├── detect_topic_segments()     → cosine-distance rolling window
 ├── build_message_checkpoints() → every-100-message summaries
 └── extract_persona()           → rule-based user profile
       │
       ▼
  app.py (Flask API)
 ├── generate_local_response()   → intent-routing + context synthesis
 ├── POST /api/chat              → retrieval + local answer
 ├── GET  /api/persona           → JSON persona
 ├── GET  /api/topics            → all topic segments
 └── GET  /api/checkpoints       → 100-message checkpoints
       │
       ▼
  static/index.html  (chat UI)
```

---

## How Topic Change Detection Works

1. **Encode every message** with `all-MiniLM-L6-v2` (384-dim dense vectors, ~80 MB model, runs on CPU).
2. **Slide a rolling window** of 5 messages across the chronological stream.
3. At each step, compute **cosine distance** between the current window's mean embedding and the previous window's.
4. If `distance > 0.35` and the current segment already has `≥ 5 messages`, mark a **topic boundary**.
5. Each segment gets a **local keyword summary** — no API call needed.

Result: chronologically ordered topic segments like:
```
Topic 1 → msgs 0–142    → "moving, city, portland, culinary, bookstore"
Topic 2 → msgs 143–310  → "music, band, yoga, stress, classic cars"
```

---

## How Local Response Generation Works

`generate_local_response()` in `app.py` uses **intent detection** + **retrieved context**:

| Detected Intent | Response Strategy |
|---|---|
| `persona` / `general` | Full persona summary (traits, habits, facts, comm style) |
| `habits` | Bullet-list of extracted habits |
| `comm_style` | Communication style metrics |
| `topics` | Retrieved topic segment summaries |
| `personal_facts` | Family, location, job, pet mentions |
| fallback | Keyword-grounded summary of top retrieved messages |

Supporting evidence (actual message quotes) is always appended from the FAISS-style dot-product retrieval.

---

## How Persona Is Built

`extract_persona()` scans User 1's messages with rule-based NLP:

| Category | Method |
|---|---|
| **Habits** | Regex for sleep/food/exercise/reading/music/gaming keywords |
| **Personal facts** | Pattern-match "I am/I'm a …", "live in …", "my mom/dog" etc. |
| **Personality** | Count humour, empathy, enthusiasm signals + question rate |
| **Communication style** | Avg word count, emoji frequency, exclamation/question rates |

No LLM, no hallucination — all evidence directly from the text.

---

## Quick Start (Local)

```bash
# 1. Unzip / clone
cd rag_chatbot

# 2. Install dependencies (~400 MB including the embedding model)
pip install -r requirements.txt

# 3. Place your data
#    Put conversations.csv inside the data/ folder:
mkdir -p data
cp /path/to/conversations.csv data/

# 4. Pre-build the RAG index (downloads model once, ~5–15 min first time)
python src/rag_processor.py

# 5. Start the server
python src/app.py

# 6. Open http://localhost:5000
```

> **No API key needed. No internet required at runtime after first model download.**

---

## Project Structure

```
rag_chatbot/
├── data/
│   ├── conversations.csv        # ← place your input data here
│   └── rag_state.pkl            # auto-generated cache (ignored by git)
├── src/
│   ├── rag_processor.py         # parsing, embeddings, topic detection, persona
│   └── app.py                   # Flask API + local response generation
├── static/
│   └── index.html               # chat UI
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Cloud Deployment (no API key required)

### Render (free tier)
- Connect GitHub repo
- Build command: `pip install -r requirements.txt && python src/rag_processor.py`
- Start command: `gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 180 src.app:app`
- **No environment variables needed**

### Streamlit Community Cloud
Replace `app.py` with a `streamlit_app.py` (straightforward port) and deploy for free.

### Hugging Face Spaces
Use the `gradio` or `streamlit` SDK; the sentence-transformers model is fully supported.

### Docker
```bash
docker build -t convorag .
docker run -p 5000:5000 convorag
# No -e API_KEY needed
```

---

## Sample Chatbot Questions

| Question | What it uses |
|---|---|
| "What kind of person is this user?" | Full persona (traits + habits + facts) |
| "What are their habits?" | Persona habits section |
| "How do they communicate?" | Comm style metrics + sample messages |
| "Do they mention any hobbies?" | Topic retrieval + message chunks |
| "What topics come up most?" | Topic segment summaries |
| "Tell me about their relationships" | Personal facts + retrieved messages |

---

## Design Decisions

- **`sentence-transformers` over TF-IDF+SVD** — dense semantic embeddings detect topic shifts based on meaning, not just word frequency; the 384-dim MiniLM model is fast enough for 200k messages on CPU.
- **No vector DB** — embeddings stored in a NumPy array, serialised to pickle. Loads in <2 s; scales fine up to ~500k messages.
- **Intent-aware response builder** — routing by intent gives structured, readable answers without needing a generative LLM. Each intent maps to a different output template backed by real retrieved evidence.
- **Local keyword summaries** — topic segment summaries use word frequency, zero cost at index time.
- **Single Gunicorn worker** — shares in-memory state; add Redis/Postgres for multi-worker production.
- **Threshold tuning** — `TOPIC_CHANGE_THRESHOLD=0.35`, `TOPIC_WINDOW=5` in `rag_processor.py`.

Demo Images: 
<img width="1911" height="907" alt="Screenshot 2026-06-17 132257" src="https://github.com/user-attachments/assets/61653a47-7476-42be-9df0-0f77ce7682fd" />
<img width="1918" height="897" alt="Screenshot 2026-06-17 132349" src="https://github.com/user-attachments/assets/1d20546e-5df2-4293-91c6-c94f6917ebe5" />
<img width="1908" height="892" alt="image" src="https://github.com/user-attachments/assets/37665a3f-16e9-4440-87f6-5205d004eac2" />
<img width="1911" height="892" alt="image" src="https://github.com/user-attachments/assets/7b162914-f7bc-4b74-a2c5-da2a176e23e4" />
<img width="1918" height="901" alt="image" src="https://github.com/user-attachments/assets/423f5591-6b44-4a25-ba6b-bcf3eda3f008" />



Loom Video Demo Link: https://www.loom.com/share/6c4eae873cf8499ca2525ddbdffcf66e


