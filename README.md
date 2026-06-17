# ConvoRAG — Conversation Intelligence System

A full RAG pipeline that processes conversation data chronologically, detects topic shifts, builds a user persona, and powers an intelligent chatbot

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

Demo Imgaes: 
<img width="1918" height="901" alt="Screenshot 2026-06-17 135526" src="https://github.com/user-attachments/assets/f457142b-84d5-4214-b06d-945b9c91462e" />
<img width="1911" height="892" alt="Screenshot 2026-06-17 135507" src="https://github.com/user-attachments/assets/2a0bd098-77ea-433b-bfc4-a010c000662e" />
<img width="1908" height="892" alt="Screenshot 2026-06-17 133143" src="https://github.com/user-attachments/assets/f4138273-c5a4-41dd-a8de-7fa1c890e4cb" />
<img width="1912" height="903" alt="Screenshot 2026-06-17 132840" src="https://github.com/user-attachments/assets/6d2ea539-e987-41d4-bdd3-2fdd9829e031" />
<img width="1917" height="902" alt="Screenshot 2026-06-17 132813" src="https://github.com/user-attachments/assets/4e79973f-6fb7-4618-8f93-90e7462e3fc9" />

Demo video link:
https://www.loom.com/share/6c4eae873cf8499ca2525ddbdffcf66e




