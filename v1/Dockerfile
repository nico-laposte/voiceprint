# Image Python slim : suffisant, pas besoin de CUDA (inférence CPU,
# cf. discussion architecture — ECAPA tourne bien sur CPU pour nos fenêtres de 2.5s)
FROM python:3.11-slim

# ffmpeg est requis par torchaudio pour décoder le webm/opus envoyé par le
# navigateur lors de l'enrôlement (cf. decode_audio_bytes dans main.py).
# build-essential (gcc) est requis car webrtcvad compile une extension C
# native à l'installation pip — sans lui, "pip install" échoue sur webrtcvad.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# torch/torchaudio installés séparément depuis l'index CPU-only de PyTorch :
# sans ça, pip installe par défaut la version CUDA qui tire ~2 Go de paquets
# nvidia-* totalement inutiles ici (Railway n'a pas de GPU, inférence CPU
# uniquement comme prévu dans l'architecture).
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway fournit le port via $PORT (pas un port fixe) — voir CMD ci-dessous.
# Le shell form (pas exec form) est nécessaire pour que $PORT soit interpolé.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
