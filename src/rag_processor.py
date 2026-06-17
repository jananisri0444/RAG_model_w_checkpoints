"""
RAG Processor: Parses conversations, detects topic changes chronologically,
creates checkpoints, and extracts user persona.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — runs 100% locally,
no paid API required.
"""

import csv, json, re, os, pickle
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
TOPIC_CHANGE_THRESHOLD = 0.35   # cosine-distance above this → new topic
TOPIC_WINDOW = 5                # messages in rolling window
MIN_TOPIC_MESSAGES = 5          # min segment size before splitting
CHECKPOINT_INTERVAL = 100       # 100-message checkpoint frequency
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 80 MB, fast, free
BATCH_SIZE = 256                # encode this many messages at once

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'rag_state.pkl')

STOPWORDS = {
    'i','a','an','the','is','it','to','of','and','in','that','you','me','we',
    'they','he','she','my','your','do','have','be','are','was','were','for',
    'not','but','so','if','on','at','as','by','or','from','what','how','when',
    'where','who','this','with','just','up','about','can','know','like','really',
    'think','user','yeah','yes','no','okay','ok','oh','well','too','very','there',
    'also','dont',"don't",'im',"i'm",'thats',"that's",'its',"it's",'would',
    'could','should','get','got','go','going','been','had','has','will','just',
    'did','now','then','see','said','say','much','more','some','than','their',
    'our','they','them','its'
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))


def parse_conversations(csv_path):
    messages = []
    msg_id = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for conv_idx, row in enumerate(reader):
            if not row:
                continue
            for line in row[0].split('\n'):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^(User \d+):\s*(.+)', line)
                if m:
                    messages.append({
                        'id': msg_id,
                        'conv_idx': conv_idx,
                        'speaker': m.group(1),
                        'text': m.group(2).strip(),
                    })
                    msg_id += 1
    return messages


def build_embeddings(messages, model):
    """Encode all messages with sentence-transformers in batches."""
    texts = [m['text'] for m in messages]
    print(f"  Encoding {len(texts):,} messages with '{EMBEDDING_MODEL}' …")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalised → dot-product == cosine similarity
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def detect_topic_segments(messages, embeddings):
    segments = []
    seg_start = 0
    prev_window_emb = None

    i = 0
    while i < len(messages):
        w_end = min(i + TOPIC_WINDOW, len(messages))
        window_emb = embeddings[i:w_end].mean(axis=0)
        window_emb = window_emb / (np.linalg.norm(window_emb) + 1e-10)

        if prev_window_emb is not None and (i - seg_start) >= MIN_TOPIC_MESSAGES:
            dist = cosine_distance(window_emb, prev_window_emb)
            if dist > TOPIC_CHANGE_THRESHOLD:
                segments.append({
                    'topic_id': len(segments) + 1,
                    'start_msg': seg_start,
                    'end_msg': i - 1,
                    'messages': messages[seg_start:i],
                    'centroid': embeddings[seg_start:i].mean(axis=0),
                })
                seg_start = i

        prev_window_emb = window_emb
        i += TOPIC_WINDOW

    # last segment
    if seg_start < len(messages):
        segments.append({
            'topic_id': len(segments) + 1,
            'start_msg': seg_start,
            'end_msg': len(messages) - 1,
            'messages': messages[seg_start:],
            'centroid': embeddings[seg_start:].mean(axis=0),
        })

    return segments


def summarize_segment(messages):
    """Lightweight local keyword summary — no API call needed."""
    all_text = ' '.join(m['text'] for m in messages).lower()
    words = re.findall(r"[a-z']{3,}", all_text)
    freq = Counter(w for w in words if w not in STOPWORDS)
    top = [w for w, _ in freq.most_common(8)]
    speakers = sorted(set(m['speaker'] for m in messages))
    mid = messages[len(messages) // 2]['text'][:100]
    return (f"Topics: {', '.join(top)}. "
            f"Speakers: {', '.join(speakers)}. "
            f"Sample: \"{mid}\"")


def build_message_checkpoints(messages):
    checkpoints = []
    for i in range(0, len(messages), CHECKPOINT_INTERVAL):
        chunk = messages[i:i + CHECKPOINT_INTERVAL]
        checkpoints.append({
            'checkpoint_id': len(checkpoints) + 1,
            'start_msg': i,
            'end_msg': min(i + CHECKPOINT_INTERVAL - 1, len(messages) - 1),
            'messages': chunk,
            'summary': summarize_segment(chunk),
        })
    return checkpoints


def extract_persona(messages):
    user1_msgs = [m['text'] for m in messages if m['speaker'] == 'User 1']
    all_text = ' '.join(user1_msgs).lower()

    habits = []
    sleep_hits = re.findall(r"\b(wake up early|night owl|late night|can't sleep|insomnia|sleep late|stay up|early riser|morning person)\b", all_text)
    if sleep_hits: habits.append(f"Sleep pattern: {Counter(sleep_hits).most_common(1)[0][0]}")
    food_hits = re.findall(r"\b(cook|cooking|recipe|eat|food|restaurant|meal|lunch|dinner|breakfast|coffee|tea|vegan|vegetarian|healthy)\b", all_text)
    if len(food_hits) >= 4: habits.append(f"Food-oriented ({', '.join(set(food_hits[:4]))})")
    exercise_hits = re.findall(r"\b(gym|workout|run|running|yoga|hike|hiking|exercise|cycling|swimming|fitness|sport)\b", all_text)
    if exercise_hits: habits.append(f"Active: {', '.join(set(exercise_hits[:4]))}")
    if len(re.findall(r"\b(read|book|novel|library|kindle|reading)\b", all_text)) >= 3: habits.append("Reader / book lover")
    music_hits = re.findall(r"\b(music|song|playlist|concert|guitar|piano|sing|band|listen|album)\b", all_text)
    if music_hits: habits.append(f"Music fan: {', '.join(set(music_hits[:3]))}")
    if len(re.findall(r"\b(game|gaming|video game|console|pc game|gamer)\b", all_text)) >= 2: habits.append("Gamer")

    personal_facts = []
    jobs = re.findall(r"(?:i(?:'m| am) (?:a |an )?)([^\s][\w\s]{2,24}?)(?:\s*(?:and|who|that|,|\.|$))", all_text)
    if jobs: personal_facts.append(f"Mentions being: {jobs[0].strip()}")
    locs = re.findall(r"(?:live in|from|moved to|moving to|based in)\s+([\w\s]{2,25}?)(?:\s*[,\.\n]|$)", all_text)
    if locs: personal_facts.append(f"Location: {locs[0].strip()}")
    family = re.findall(r"\bmy (mom|dad|mother|father|sister|brother|wife|husband|girlfriend|boyfriend|kids|children|family|son|daughter)\b", all_text)
    if family: personal_facts.append(f"Family mentions: {', '.join(set(family[:5]))}")
    pets = re.findall(r"\bmy (dog|cat|puppy|kitten|pet|bird|hamster|fish)\b", all_text)
    if pets: personal_facts.append(f"Pets: {', '.join(set(pets))}")

    traits = []
    humor = len(re.findall(r"\b(haha|lol|lmao|hehe|funny|hilarious|joke|laugh)\b", all_text))
    if humor >= 5: traits.append("Humorous / light-hearted")
    empathy = len(re.findall(r"\b(sorry|i understand|that must|feel for|hope you|take care|here for you)\b", all_text))
    if empathy >= 3: traits.append("Empathetic")
    positive = len(re.findall(r"\b(awesome|amazing|great|love|wonderful|fantastic|excited|happy|cool)\b", all_text))
    if positive >= 10: traits.append("Enthusiastic / positive")
    q_count = sum(t.count('?') for t in user1_msgs)
    if q_count > len(user1_msgs) * 0.4: traits.append("Inquisitive / asks many questions")
    if not traits: traits.append("Balanced / neutral communicator")

    avg_len = float(np.mean([len(t.split()) for t in user1_msgs])) if user1_msgs else 0
    excl_rate = sum(t.count('!') for t in user1_msgs) / max(len(user1_msgs), 1)
    q_rate = sum(t.count('?') for t in user1_msgs) / max(len(user1_msgs), 1)
    emoji_count = sum(1 for t in user1_msgs for c in t if 0x1F300 <= ord(c) <= 0x1FAFF)

    return {
        'habits': habits or ['No distinctive habits detected from text'],
        'personal_facts': personal_facts or ['No clear personal facts found'],
        'personality_traits': traits,
        'communication_style': {
            'avg_message_length_words': round(avg_len, 1),
            'tone': 'casual/friendly' if excl_rate > 0.3 else 'measured/reserved',
            'uses_emojis': emoji_count > 5,
            'exclamation_heavy': excl_rate > 0.5,
            'question_heavy': q_rate > 0.4,
            'total_messages_as_user1': len(user1_msgs),
        }
    }


def build_rag_state(force=False):
    if not force and os.path.exists(CACHE_PATH):
        print("Loading cached RAG state …")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    print("Building RAG state from scratch …")
    csv_path = os.path.join(DATA_DIR, 'conversations.csv')

    print("1/5  Parsing …")
    messages = parse_conversations(csv_path)
    print(f"     {len(messages):,} messages")

    print("2/5  Loading sentence-transformers model …")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("3/5  Embedding messages (all-MiniLM-L6-v2, local) …")
    embeddings = build_embeddings(messages, model)

    print("4/5  Topic segmentation …")
    topic_segments = detect_topic_segments(messages, embeddings)
    for seg in topic_segments:
        seg['summary'] = summarize_segment(seg['messages'])
    print(f"     {len(topic_segments)} topic segments")

    print("5/5  100-message checkpoints + persona extraction …")
    message_checkpoints = build_message_checkpoints(messages)
    print(f"     {len(message_checkpoints)} checkpoints")
    persona = extract_persona(messages)

    state = {
        'messages': messages,
        'embeddings': embeddings,
        'model': model,           # keep model in state for query-time embedding
        'topic_segments': topic_segments,
        'message_checkpoints': message_checkpoints,
        'persona': persona,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(state, f)
    print("Saved to cache.")
    return state


def embed_query(query, state):
    """Embed a single query using the stored sentence-transformers model."""
    return state['model'].encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def retrieve(query, state, top_k_topics=3, top_k_msgs=5):
    q_emb = embed_query(query, state)

    topic_scores = []
    for seg in state['topic_segments']:
        c = seg['centroid']
        c = c / (np.linalg.norm(c) + 1e-10)
        score = float(np.dot(q_emb, c))
        topic_scores.append((score, seg))
    topic_scores.sort(key=lambda x: x[0], reverse=True)

    msg_scores = []
    for i, emb in enumerate(state['embeddings']):
        score = float(np.dot(q_emb, emb))
        msg_scores.append((score, state['messages'][i]))
    msg_scores.sort(key=lambda x: x[0], reverse=True)

    return {
        'relevant_topics': [s for _, s in topic_scores[:top_k_topics]],
        'relevant_messages': [m for _, m in msg_scores[:top_k_msgs]],
    }


if __name__ == '__main__':
    state = build_rag_state(force=True)
    print("\n=== Persona ===")
    print(json.dumps(state['persona'], indent=2))
    print(f"\n=== Topic Segments (first 5) ===")
    for seg in state['topic_segments'][:5]:
        print(f"  Topic {seg['topic_id']}: msgs {seg['start_msg']}–{seg['end_msg']} ({len(seg['messages'])} msgs)")
        print(f"    {seg['summary'][:100]}")
