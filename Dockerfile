FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src
ENV DATA_DIR=/app/data/sample_docs
ENV INDEX_DIR=/app/data/index

EXPOSE 8000

CMD ["python", "-m", "agentic_rag_assistant.main"]
