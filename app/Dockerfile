# ── Port-a-Prof FastAPI app ───────────────────────────────────────────
# Serves the frontend and calls llama-server for all model inference.
# llama-server runs in a separate container (see docker-compose.yml).

FROM python:3.11-slim

# ffmpeg — required for audio conversion (webm/mp4 → wav) before transcription
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pillow \
    python-multipart

# Copy app files
COPY port-a-prof.py .
COPY index.html .
COPY logo.png .
COPY static/ ./static/

EXPOSE 8001

CMD ["uvicorn", "port-a-prof:app", "--host", "0.0.0.0", "--port", "8001"]
