"""
Serveur principal.

- POST /enroll : enrôle une personne à partir d'un fichier audio
- GET  /speakers : liste les personnes enrôlées
- DELETE /speakers/{name} : supprime une personne
- WS   /ws/identify : flux audio live -> identification en continu
- GET  /health : healthcheck Railway
"""

import collections
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from speaker_id import SpeakerIdentifier, SAMPLE_RATE
from vad import VoiceActivityDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

identifier = SpeakerIdentifier()
vad = VoiceActivityDetector(aggressiveness=2)

WINDOW_SECONDS = 2.5
HOP_SECONDS = 1.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)
SIMILARITY_THRESHOLD = 0.35
SMOOTHING_WINDOW = 3


def decode_audio_bytes(raw: bytes, filename_hint: str = "audio.webm") -> np.ndarray:
    suffix = Path(filename_hint).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as src_tmp:
        src_tmp.write(raw)
        src_tmp.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_tmp:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", src_tmp.name,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_tmp.name],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg a échoué : {result.stderr[-500:]}")
            audio, sr = sf.read(wav_tmp.name, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


@app.get("/health")
async def health():
    return {"status": "ok", "enrolled_count": len(identifier.enrolled)}


@app.post("/enroll")
async def enroll(name: str, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        audio = decode_audio_bytes(raw, filename_hint=file.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio illisible: {e}")
    duration = len(audio) / SAMPLE_RATE
    if duration < 3.0:
        raise HTTPException(status_code=400, detail=f"Audio trop court ({duration:.1f}s).")
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
    await websocket.accept()
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
                    has_voice = vad.has_speech(window)
                except Exception:
                    has_voice = True

                if not has_voice:
                    recent_results.append(None)
                    await websocket.send_json({"status": "silence"})
                    continue

                try:
                    result = identifier.identify(window, threshold=SIMILARITY_THRESHOLD)
                except Exception as e:
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
        print(f"[ws_identify] erreur: {e}")
