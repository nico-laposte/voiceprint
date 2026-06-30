"""
Serveur principal.

- POST /enroll : enrôle une personne à partir d'un fichier audio (webm/wav, ~10-15s)
- GET  /speakers : liste les personnes enrôlées
- DELETE /speakers/{name} : supprime une personne
- WS   /ws/identify : flux audio live -> identification en continu

Lancer avec : uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import collections
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from speaker_id import SpeakerIdentifier, SAMPLE_RATE
from vad import VoiceActivityDetector

app = FastAPI()

# CORS large pour le dev : à restreindre à ton domaine en prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

identifier = SpeakerIdentifier()
vad = VoiceActivityDetector(aggressiveness=2)


@app.get("/health")
async def health():
    # Utilisé par Railway pour vérifier que le service a bien démarré
    # (notamment que le modèle ECAPA a fini de charger).
    return {"status": "ok", "enrolled_count": len(identifier.enrolled)}

# --- Paramètres de fenêtrage (cf. discussion architecture) ---
WINDOW_SECONDS = 2.5
HOP_SECONDS = 1.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)
SIMILARITY_THRESHOLD = 0.45

# Lissage temporel : on garde les N derniers résultats et on vote,
# pour éviter que le nom affiché change à chaque fenêtre sur du bruit limite.
SMOOTHING_WINDOW = 3


def decode_audio_bytes(raw: bytes, filename_hint: str = "audio.webm") -> np.ndarray:
    """
    Décode un blob audio (webm/opus depuis Chrome, ou mp4/AAC depuis Safari)
    en numpy float32 mono 16kHz.

    On passe par un vrai fichier temporaire (plutôt qu'un BytesIO en mémoire)
    car torchaudio/ffmpeg a besoin de l'extension du fichier pour deviner
    fiablement le conteneur — la détection automatique depuis un flux en
    mémoire sans extension échoue souvent sur MP4 ("Format not recognised"),
    notamment pour l'audio envoyé par Safari (qui encode en audio/mp4, pas
    en webm comme Chrome/Firefox).
    """
    suffix = Path(filename_hint).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        # backend="ffmpeg" est obligatoire ici : le backend "soundfile" (souvent
        # choisi par défaut par torchaudio) ne sait PAS décoder l'AAC/MP4 envoyé
        # par Safari iOS ("Format not recognised" sinon), seul ffmpeg le sait.
        available = torchaudio.list_audio_backends()
        if "ffmpeg" not in available:
            raise RuntimeError(
                f"Backend ffmpeg indisponible dans torchaudio (backends trouvés: {available}). "
                "Vérifie que ffmpeg est bien installé dans l'image Docker."
            )
        waveform, sr = torchaudio.load(tmp.name, backend="ffmpeg")  # (channels, samples)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy().astype(np.float32)


@app.post("/enroll")
async def enroll(name: str, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        audio = decode_audio_bytes(raw, filename_hint=file.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio illisible: {e}")

    duration = len(audio) / SAMPLE_RATE
    if duration < 3.0:
        raise HTTPException(
            status_code=400,
            detail=f"Audio trop court ({duration:.1f}s) — au moins 5-10s recommandées pour un bon embedding.",
        )

    embedding = identifier.enroll(name, audio)
    return {"name": name, "duration_s": round(duration, 2), "embedding_dim": len(embedding)}


@app.get("/speakers")
async def list_speakers():
    return {"speakers": list(identifier.enrolled.keys())}


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    identifier.remove(name)
    return {"removed": name}


@app.websocket("/ws/identify")
async def ws_identify(websocket: WebSocket):
    """
    Le client envoie des chunks audio binaires en continu (PCM float32
    mono 16kHz, cf. AudioWorklet côté front). On accumule dans un buffer
    glissant, et dès qu'on a assez d'échantillons pour une fenêtre,
    on lance VAD -> ECAPA -> comparaison -> on renvoie le résultat en JSON.
    """
    await websocket.accept()

    # Buffer circulant en float32. On utilise une liste simple + slicing,
    # suffisant à cette échelle (quelques secondes de buffer max).
    buffer = np.zeros(0, dtype=np.float32)
    recent_results = collections.deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            message = await websocket.receive_bytes()
            # Le client envoie du PCM float32 brut, little-endian
            chunk = np.frombuffer(message, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            # Tant qu'on a assez de samples pour une fenêtre complète,
            # on traite et on avance de HOP_SAMPLES (chevauchement géré ici)
            while len(buffer) >= WINDOW_SAMPLES:
                window = buffer[:WINDOW_SAMPLES]
                buffer = buffer[HOP_SAMPLES:]  # avance du hop, garde le reste pour chevauchement

                if not vad.has_speech(window):
                    recent_results.append(None)
                    await websocket.send_json({"status": "silence"})
                    continue

                result = identifier.identify(window, threshold=SIMILARITY_THRESHOLD)
                recent_results.append(result["name"])

                # Vote de lissage sur les dernières fenêtres avec voix détectée
                votes = [r for r in recent_results if r is not None]
                if votes:
                    smoothed_name = max(set(votes), key=votes.count)
                else:
                    smoothed_name = None

                await websocket.send_json({
                    "status": "speech",
                    "name": smoothed_name,
                    "raw_name": result["name"],
                    "score": round(result["score"], 3),
                    "all_scores": {k: round(v, 3) for k, v in result["all_scores"].items()},
                })

    except WebSocketDisconnect:
        print("[ws_identify] client déconnecté")
