"""
core.py — La "cervelle" du système, totalement découplée du transport (HTTP/WebSocket)
et du stockage. On y trouve UNIQUEMENT :

    bytes audio bruts  ->  décodage 16 kHz mono  ->  VAD  ->  embedding ECAPA (L2-normalisé)
    embedding + centroïdes  ->  décision (locuteur / inconnu)

En Phase 2 (WebSocket temps réel), on réutilise CE fichier tel quel : seule la plomberie
autour change. C'est là tout l'intérêt du découpage.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch

SAMPLE_RATE = 16000
EMB_DIM = 192  # dimension des embeddings ECAPA-TDNN


# --------------------------------------------------------------------------- #
#  Résultats (types simples, sérialisables tels quels en JSON)
# --------------------------------------------------------------------------- #
@dataclass
class Embedding:
    """Un embedding L2-normalisé + quelques métadonnées de qualité."""
    vector: np.ndarray                 # shape (192,), norme 1
    speech_seconds: float              # durée de parole détectée par le VAD
    low_speech: bool                   # True si trop peu de voix pour être fiable


@dataclass
class Match:
    decision: str                      # nom du locuteur, ou "inconnu"
    score: float                       # cosinus au meilleur centroïde
    is_known: bool
    scores: dict[str, float] = field(default_factory=dict)  # cosinus par locuteur
    low_speech: bool = False


class NotEnoughSpeech(Exception):
    """Levée quand le clip ne contient pas assez de voix pour un embedding fiable."""


# --------------------------------------------------------------------------- #
#  Le processeur audio
# --------------------------------------------------------------------------- #
class AudioProcessor:
    """
    Charge ECAPA + Silero VAD UNE seule fois (au boot), puis transforme
    n'importe quel conteneur audio (mp4/aac de l'iPhone, wav, mp3, m4a, ogg...)
    en embedding. Le décodage passe par ffmpeg, donc le format d'entrée n'a
    aucune importance : c'est ce qui permet d'enrôler par fichier ET d'identifier
    au micro avec EXACTEMENT le même pipeline en aval.
    """

    def __init__(
        self,
        model_dir: str = "/tmp/ecapa",
        min_speech_seconds: float = 1.0,
        vad_threshold: float = 0.5,
    ) -> None:
        self.min_speech_seconds = min_speech_seconds
        self.vad_threshold = vad_threshold

        # ECAPA-TDNN (SpeechBrain). Téléchargé une fois au premier boot.
        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
            run_opts={"device": "cpu"},
        )

        # Silero VAD (modèle embarqué dans le paquet pip, pas de réseau au runtime).
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._vad_available = True
        except Exception:
            # Dégradation propre : si le VAD n'est pas dispo, on garde tout l'audio.
            self._vad_available = False

    # ----- étapes unitaires (faciles à tester isolément) ------------------- #

    def decode(self, raw: bytes) -> np.ndarray:
        """N'importe quel conteneur -> float32 mono 16 kHz dans [-1, 1] via ffmpeg."""
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "pipe:1",
            ],
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise ValueError(
                "Décodage audio impossible. "
                "Le fichier est vide ou dans un format que ffmpeg n'a pas su lire."
            )
        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if wav.size == 0:
            raise ValueError("Le fichier audio décodé est vide.")
        return wav

    def apply_vad(self, wav: np.ndarray) -> tuple[np.ndarray, float]:
        """Ne conserve que les segments de parole. Retourne (audio_voix, secondes)."""
        if not self._vad_available:
            return wav, len(wav) / SAMPLE_RATE

        tensor = torch.from_numpy(wav)
        ts = self._get_speech_timestamps(
            tensor, self._vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.vad_threshold,
        )
        if not ts:
            return np.zeros(0, dtype=np.float32), 0.0

        segments = [wav[t["start"]: t["end"]] for t in ts]
        speech = np.concatenate(segments)
        return speech, len(speech) / SAMPLE_RATE

    def _encode(self, wav: np.ndarray) -> np.ndarray:
        """ECAPA -> vecteur 192D L2-normalisé."""
        with torch.no_grad():
            emb = self.encoder.encode_batch(torch.from_numpy(wav).unsqueeze(0))
        vec = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # ----- pipeline complet ------------------------------------------------ #

    def embed(self, raw: bytes) -> Embedding:
        """bytes -> Embedding. C'est LE point d'entrée unique (enrôlement ET identif)."""
        wav = self.decode(raw)
        speech, seconds = self.apply_vad(wav)

        if seconds < self.min_speech_seconds:
            # Repli : on tente quand même sur l'audio brut, mais on signale le doute.
            if len(wav) / SAMPLE_RATE < self.min_speech_seconds:
                raise NotEnoughSpeech(
                    f"Trop peu de voix détectée ({seconds:.1f}s). "
                    f"Il en faut au moins {self.min_speech_seconds:.0f}s."
                )
            speech, seconds, low = wav, len(wav) / SAMPLE_RATE, True
        else:
            low = seconds < 2.0  # ECAPA aime ~2-3 s : en dessous, embedding moins stable

        vec = self._encode(speech)
        return Embedding(vector=vec, speech_seconds=seconds, low_speech=low)


# --------------------------------------------------------------------------- #
#  Comparaison (fonctions pures — aucune dépendance au modèle ni au stockage)
# --------------------------------------------------------------------------- #
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """a et b sont supposés L2-normalisés -> le cosinus est un simple produit scalaire."""
    return float(np.dot(a, b))


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """Moyenne des embeddings d'un locuteur, puis re-normalisation L2."""
    m = np.mean(np.stack(vectors, axis=0), axis=0)
    n = np.linalg.norm(m)
    return (m / n).astype(np.float32) if n > 0 else m.astype(np.float32)


def match(
    emb: Embedding,
    centroids: dict[str, np.ndarray],
    threshold: float,
) -> Match:
    """
    Compare un embedding aux centroïdes des locuteurs enrôlés.
    OPEN-SET : si le meilleur cosinus < threshold -> "inconnu".
    """
    if not centroids:
        return Match(decision="inconnu", score=0.0, is_known=False,
                     scores={}, low_speech=emb.low_speech)

    scores = {name: cosine(emb.vector, c) for name, c in centroids.items()}
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    is_known = best_score >= threshold

    return Match(
        decision=best_name if is_known else "inconnu",
        score=best_score,
        is_known=is_known,
        scores=scores,
        low_speech=emb.low_speech,
    )
