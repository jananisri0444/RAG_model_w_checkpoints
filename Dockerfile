FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"


WORKDIR /app

# Install dependencies (sentence-transformers downloads model on first run)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project
COPY . .

# Pre-build RAG state at image build time (requires data/conversations.csv)
# Uncomment the line below if you want the index baked into the image:
# RUN python src/rag_processor.py

EXPOSE 7860

ENV FLASK_ENV=production

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "180", "src.app:app"]
