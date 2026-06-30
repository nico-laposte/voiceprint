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
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
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
SIMILARITY_THRESHOLD = 0.35  # Baissé de 0.45 : l'audio micro iPhone en conditions
                              # réelles produit des scores plus bas qu'en test propre

# Lissage temporel : on garde les N derniers résultats et on vote,
# pour éviter que le nom affiché change à chaque fenêtre sur du bruit limite.
SMOOTHING_WINDOW = 3


def decode_audio_bytes(raw: bytes, filename_hint: str = "audio.webm") -> np.ndarray:
    """
    Décode un blob audio (webm/opus depuis Chrome, ou mp4/AAC depuis Safari)
    en numpy float32 mono 16kHz.

    On appelle ffmpeg directement en sous-processus pour convertir le blob
    en WAV PCM, puis on lit ce WAV avec soundfile. C'est plus robuste que de
    laisser torchaudio détecter et charger dynamiquement ses propres
    bindings ffmpeg : sur certaines images Docker, torchaudio.load(...,
    backend="ffmpeg") échoue silencieusement à charger les bibliothèques
    ffmpeg même quand le binaire ffmpeg est bien installé et fonctionnel
    en ligne de commande (cf. torchaudio.list_audio_backends() qui ne
    retourne alors que ['soundfile']). En passant par le binaire ffmpeg
    système via subprocess, on évite complètement ce problème de binding.
    """
    suffix = Path(filename_hint).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as src_tmp:
        src_tmp.write(raw)
        src_tmp.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_tmp:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", src_tmp.name,
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "wav",
                    wav_tmp.name,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg a échoué à convertir l'audio : {result.stderr[-500:]}")

            audio, sr = sf.read(wav_tmp.name, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # ffmpeg a déjà resamplé à SAMPLE_RATE via -ar, donc sr == SAMPLE_RATE ici
    return audio.astype(np.float32)


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


@app.get("/debug")
async def debug():
    """Diagnostic rapide : état du modèle et des enrôlements."""
    import torchaudio
    return {
        "enrolled": list(identifier.enrolled.keys()),
        "threshold": SIMILARITY_THRESHOLD,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "audio_backends": torchaudio.list_audio_backends(),
    }


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
            chunk = np.frombuffer(message, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= WINDOW_SAMPLES:
                window = buffer[:WINDOW_SAMPLES]
                buffer = buffer[HOP_SAMPLES:]

                try:
                    has_voice = True  # VAD désactivée temporairement pour diagnostic
                    # has_voice = vad.has_speech(window)
                except Exception as e:
                    print(f"[ws_identify] erreur VAD: {e}")
                    has_voice = True

                if not has_voice:
                    recent_results.append(None)
                    await websocket.send_json({"status": "silence"})
                    continue

                try:
                    result = identifier.identify(window, threshold=SIMILARITY_THRESHOLD)
                except Exception as e:
                    print(f"[ws_identify] erreur ECAPA: {e}")
                    await websocket.send_json({"status": "error", "detail": str(e)})
                    continue

                recent_results.append(result["name"])
                votes = [r for r in recent_results if r is not None]
                smoothed_name = max(set(votes), key=votes.count) if votes else None

                await websocket.send_json({
                    "status": "speech",
                    "name": smoothed_name,
                    "raw_name": result["name"],
                    "score": round(result["score"], 3),
                    "all_scores": {k: round(v, 3) for k, v in result["all_scores"].items()},
                })

    except WebSocketDisconnect:
        print("[ws_identify] client déconnecté")
    except Exception as e:
        print(f"[ws_identify] erreur inattendue: {e}")
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
        except Exception:
            pass
