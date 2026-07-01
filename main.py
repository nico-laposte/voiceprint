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
    print("[echo] connexion acceptée")
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            msg_bytes = message.get("bytes") or b""
            msg_text = message.get("text") or ""
            print(f"[echo] type={msg_type} bytes={len(msg_bytes)} text={len(msg_text)}")
            await websocket.send_json({"status": "echo", "received": True})
    except WebSocketDisconnect:
        print("[echo] déconnecté")
    except Exception as e:
        print(f"[echo] erreur: {e}")
