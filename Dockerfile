# Image Python slim : suffisant, pas besoin de CUDA (inférence CPU,
# cf. discussion architecture — ECAPA tourne bien sur CPU pour nos fenêtres de 2.5s)
FROM python:3.11-slim

# ffmpeg est requis par torchaudio pour décoder le webm/opus envoyé par le
# navigateur lors de l'enrôlement (cf. decode_audio_bytes dans main.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway fournit le port via $PORT (pas un port fixe) — voir CMD ci-dessous.
# Le shell form (pas exec form) est nécessaire pour que $PORT soit interpolé.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
