"""
Flask API for ConvoRAG chatbot.
100% local — no external API keys required.
Response generation uses retrieved context + rule-based NLP synthesis.
"""

import os, sys, json, re
from collections import Counter
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from rag_processor import build_rag_state, retrieve

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

print("Initialising RAG system …")
STATE = build_rag_state()
print("Ready.")


# ── Local response generation ─────────────────────────────────────────────────

def _keywords(text):
    """Return lowercase word set from text, filtering stopwords."""
    STOP = {'the','a','an','is','it','to','of','and','in','that','you','me',
            'we','they','he','she','do','have','be','are','was','were','for',
            'not','but','so','if','on','at','as','by','or','from','what',
            'how','when','where','who','this','with','just','about','can'}
    return {w for w in re.findall(r"[a-z']{3,}", text.lower()) if w not in STOP}


def _detect_intent(query):
    """Map the user query to a response strategy."""
    q = query.lower()
    if re.search(r'\b(habit|routine|daily|sleep|eat|food|exercise|workout|hobby|hobbies)\b', q):
        return 'habits'
    if re.search(r'\b(person|character|like|who|personality|trait|nature|kind of person)\b', q):
        return 'persona'
    if re.search(r'\b(talk|speak|communicate|style|tone|message|word|language|express)\b', q):
        return 'comm_style'
    if re.search(r'\b(topic|subject|discuss|conversation|talk about|theme)\b', q):
        return 'topics'
    if re.search(r'\b(family|friend|relationship|partner|pet|dog|cat|mom|dad|sibling)\b', q):
        return 'personal_facts'
    if re.search(r'\b(fact|detail|information|tell me|what do you know)\b', q):
        return 'facts'
    return 'general'


def _persona_answer(persona, retrieval, query):
    """Build a structured answer about the user's persona."""
    lines = []

    intent = _detect_intent(query)
    cs = persona['communication_style']

    if intent in ('persona', 'general', 'facts'):
        lines.append("**Based on the conversation data, here is what we know about this user:**\n")
        lines.append(f"**Personality traits:** {', '.join(persona['personality_traits'])}")
        lines.append(f"**Habits & interests:** {'; '.join(persona['habits'])}")
        if persona['personal_facts'] and persona['personal_facts'] != ['No clear personal facts found']:
            lines.append(f"**Personal facts:** {'; '.join(persona['personal_facts'])}")
        lines.append(f"**Communication style:** {cs['tone']} tone, averaging {cs['avg_message_length_words']} words per message.")
        if cs['question_heavy']:
            lines.append("They ask a lot of questions, suggesting curiosity.")
        if cs['exclamation_heavy']:
            lines.append("They use frequent exclamations, indicating an expressive style.")

    elif intent == 'habits':
        lines.append("**User habits and interests detected from conversations:**\n")
        for h in persona['habits']:
            lines.append(f"• {h}")

    elif intent == 'comm_style':
        lines.append("**Communication style analysis:**\n")
        lines.append(f"• Tone: {cs['tone']}")
        lines.append(f"• Average message length: {cs['avg_message_length_words']} words")
        lines.append(f"• Question-heavy: {'Yes' if cs['question_heavy'] else 'No'}")
        lines.append(f"• Exclamation-heavy: {'Yes' if cs['exclamation_heavy'] else 'No'}")
        lines.append(f"• Uses emojis: {'Yes' if cs['uses_emojis'] else 'No'}")
        lines.append(f"• Total messages as User 1: {cs['total_messages_as_user1']:,}")

    elif intent == 'topics':
        lines.append("**Most relevant topic segments from the conversation data:**\n")
        for seg in retrieval['relevant_topics']:
            lines.append(f"• Topic {seg['topic_id']} (msgs {seg['start_msg']}–{seg['end_msg']}): {seg['summary'][:120]}")

    elif intent == 'personal_facts':
        lines.append("**Personal facts mentioned by the user:**\n")
        for fact in persona['personal_facts']:
            lines.append(f"• {fact}")

    if retrieval['relevant_messages']:
        lines.append("\n**Supporting evidence from conversations:**")
        shown = 0
        for msg in retrieval['relevant_messages']:
            text = msg["text"].strip()
            if text.endswith("?"):
                continue
            if shown >= 3:
                break
            lines.append(
                f'  › {msg["speaker"]}: "{text[:120]}"'
            )
            shown += 1
    return '\n'.join(lines)


def generate_local_response(query, retrieval, persona):
    """
    Fully local response generator.
    Routes to specialised answer builders based on query intent.
    """
    intent = _detect_intent(query)

    # --- persona / habits / style / personal facts → structured persona answer
    if intent in ('persona', 'habits', 'comm_style', 'personal_facts', 'general', 'facts'):
        return _persona_answer(persona, retrieval, query)

    # --- topic exploration
    if intent == 'topics':
        lines = ["**Relevant conversation topics retrieved:**\n"]
        for seg in retrieval['relevant_topics']:
            lines.append(f"• **Topic {seg['topic_id']}** (msgs {seg['start_msg']}–{seg['end_msg']})")
            lines.append(f"  {seg['summary']}")
        if retrieval['relevant_messages']:
            lines.append("\n**Sample messages from matching topics:**")
            for msg in retrieval['relevant_messages'][:3]:
                lines.append(f'  › {msg["speaker"]}: "{msg["text"][:120]}"')
        return '\n'.join(lines)

    # --- fallback: keyword-grounded summary from retrieved chunks
    lines = ["**Retrieved context for your query:**\n"]
    q_kw = _keywords(query)

    if retrieval['relevant_topics']:
        lines.append("**Relevant topic segments:**")
        for seg in retrieval['relevant_topics'][:2]:
            lines.append(f"• Topic {seg['topic_id']}: {seg['summary'][:150]}")

    if retrieval['relevant_messages']:
        lines.append("\n**Most semantically similar messages:**")
        for msg in retrieval['relevant_messages'][:4]:
            lines.append(f'  › {msg["speaker"]}: "{msg["text"][:120]}"')

    # Derive a one-line summary from keyword overlap
    all_retrieved_text = ' '.join(m['text'] for m in retrieval['relevant_messages'])
    common_kw = _keywords(all_retrieved_text) & q_kw
    if common_kw:
        lines.append(f"\n**Key themes in retrieved content:** {', '.join(sorted(common_kw))}")

    return '\n'.join(lines) if len(lines) > 1 else "No sufficiently relevant content found for this query. Try rephrasing or asking about the user's habits, personality, or topics discussed."


# ── Flask routes ──────────────────────────────────────────────────────────────

def build_context_dict(retrieval, persona):
    """Build the context payload returned alongside every chat answer."""
    return {
        'persona_snapshot': {
            'traits': persona['personality_traits'],
            'habits': persona['habits'],
            'comm_style': persona['communication_style']['tone'],
        },
        'retrieved_topics': [
            {'topic_id': s['topic_id'], 'start_msg': s['start_msg'],
             'end_msg': s['end_msg'], 'summary': s['summary']}
            for s in retrieval['relevant_topics']
        ],
        'retrieved_messages': [
            {'speaker': m['speaker'], 'text': m['text']}
            for m in retrieval['relevant_messages']
        ],
    }


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = (data or {}).get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    retrieval = retrieve(query, STATE)
    answer = generate_local_response(query, retrieval, STATE['persona'])

    return jsonify({
        'answer': answer,
        **build_context_dict(retrieval, STATE['persona']),
    })


@app.route('/api/persona')
def get_persona():
    return jsonify(STATE['persona'])


@app.route('/api/topics')
def get_topics():
    return jsonify([{
        'topic_id': s['topic_id'],
        'start_msg': s['start_msg'],
        'end_msg': s['end_msg'],
        'message_count': len(s['messages']),
        'summary': s['summary'],
    } for s in STATE['topic_segments']])


@app.route('/api/checkpoints')
def get_checkpoints():
    return jsonify([{
        'checkpoint_id': c['checkpoint_id'],
        'start_msg': c['start_msg'],
        'end_msg': c['end_msg'],
        'message_count': len(c['messages']),
        'summary': c['summary'],
    } for c in STATE['message_checkpoints']])


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
