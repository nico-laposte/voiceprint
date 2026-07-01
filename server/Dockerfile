# Railway build from Dockerfile. ffmpeg est indispensable pour décoder
# le mp4/aac produit par Safari iOS et tous les autres formats.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU-only : plus léger et suffisant pour un proto (pas de GPU sur Railway).
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchaudio==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway fournit $PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
