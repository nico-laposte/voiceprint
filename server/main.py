"""
main.py — Couche TRANSPORT (Phase 1 : HTTP POST).

Volontairement fine : elle ne fait que recevoir des octets, appeler la cervelle
(core.py) et le stockage (storage.py), puis renvoyer du JSON. Toute la logique
"métier" vit ailleurs. En Phase 2, on ajoutera un endpoint WebSocket ICI même,
qui appellera EXACTEMENT le même `processor` et le même `store`.

Endpoints :
    POST /enroll     champ 'speaker' + un ou plusieurs fichiers 'files'
    POST /identify   un fichier 'file'
    GET  /speakers   liste {nom: nb_échantillons}
    DELETE /speakers/{name}
    GET  /health
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core import AudioProcessor, NotEnoughSpeech, match
from storage import SpeakerStore

# --- Configuration (surchargeable par variables d'env sur Railway) ---------- #
DATA_DIR = os.environ.get("DATA_DIR", "/data")            # <- volume Railway monté ici
DB_PATH = os.path.join(DATA_DIR, "speakers.db")
MODEL_DIR = os.path.join(DATA_DIR, "ecapa")               # cache du modèle sur le volume
# Seuil de rejet open-set. Point à CALIBRER (voir README). Départ prudent/bas.
THRESHOLD = float(os.environ.get("THRESHOLD", "0.25"))
# Origines autorisées (ton GitHub Pages). "*" pour un proto ; restreins ensuite.
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="VoiceID — Phase 1 (POST)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargés UNE fois au démarrage (le modèle est lourd : jamais par requête).
processor: AudioProcessor
store: SpeakerStore


@app.on_event("startup")
def _startup() -> None:
    global processor, store
    processor = AudioProcessor(model_dir=MODEL_DIR)
    store = SpeakerStore(DB_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": THRESHOLD, "speakers": store.speakers()}


@app.get("/speakers")
def list_speakers() -> dict:
    return {"speakers": store.speakers(), "threshold": THRESHOLD}


@app.delete("/speakers/{name}")
def delete_speaker(name: str) -> dict:
    if not store.delete(name):
        raise HTTPException(404, f"Locuteur '{name}' introuvable.")
    return {"deleted": name, "speakers": store.speakers()}


@app.post("/enroll")
async def enroll(
    speaker: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict:
    """Enrôle un locuteur à partir d'un ou plusieurs fichiers/enregistrements."""
    added, skipped = 0, []
    for f in files:
        raw = await f.read()
        try:
            emb = processor.embed(raw)
        except (NotEnoughSpeech, ValueError) as e:
            skipped.append({"file": f.filename, "reason": str(e)})
            continue
        count = store.add(speaker, emb.vector)

    total = store.speakers().get(speaker.strip(), 0)
    if total == 0:
        raise HTTPException(
            422,
            {"message": "Aucun échantillon exploitable.", "skipped": skipped},
        )
    return {
        "speaker": speaker.strip(),
        "samples_total": total,
        "skipped": skipped,
        "hint": (
            "Astuce : ajoute aussi un échantillon enregistré au micro du téléphone "
            "pour aligner les conditions d'enrôlement et d'identification."
        ),
    }


@app.post("/identify")
async def identify(file: UploadFile = File(...)) -> dict:
    """Identifie le locuteur d'un clip (open-set : peut répondre 'inconnu')."""
    raw = await file.read()
    try:
        emb = processor.embed(raw)
    except NotEnoughSpeech as e:
        raise HTTPException(422, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = match(emb, store.centroids(), THRESHOLD)
    return {
        "decision": result.decision,
        "is_known": result.is_known,
        "score": round(result.score, 4),
        "threshold": THRESHOLD,
        "low_speech": result.low_speech,
        "speech_seconds": round(emb.speech_seconds, 2),
        "scores": {k: round(v, 4) for k, v in sorted(
            result.scores.items(), key=lambda kv: kv[1], reverse=True
        )},
    }
