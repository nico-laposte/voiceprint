"""
Encapsule le modèle ECAPA-TDNN (SpeechBrain) : extraction d'embedding
à partir d'un buffer audio numpy, et comparaison cosinus à une base
de locuteurs enrôlés.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000

# DATA_DIR pointe par défaut vers le dossier du script (utile en local).
# En prod sur Railway, on le fait pointer vers le volume monté (cf. railway.json
# et la variable d'env DATA_DIR="/data") pour que le modèle téléchargé et les
# embeddings enrôlés survivent aux redéploiements.
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
EMBEDDINGS_PATH = DATA_DIR / "enrolled_speakers.json"
MODEL_CACHE_DIR = DATA_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"


class SpeakerIdentifier:
    def __init__(self, device: str | None = None):
        # device="cuda" si tu as un GPU dispo, sinon CPU (suffisant pour
        # des fenêtres de 2.5s, l'inférence ECAPA est rapide même sur CPU.
        # Sur Railway sans GPU dédié, ce sera toujours CPU.)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[SpeakerIdentifier] chargement du modèle ECAPA-TDNN sur {self.device}...")
        print(f"[SpeakerIdentifier] cache modèle: {MODEL_CACHE_DIR}")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODEL_CACHE_DIR),
            run_opts={"device": self.device},
        )
        self.enrolled: dict[str, list[float]] = self._load_enrolled()
        print(f"[SpeakerIdentifier] {len(self.enrolled)} locuteur(s) enrôlé(s): {list(self.enrolled.keys())}")
        # Préchauffage : un premier appel factice pour que PyTorch compile
        # les kernels JIT — sans ça, la première vraie inférence peut prendre
        # 30-60s, ce qui dépasse le timeout de Safari et cause "Load failed".
        print("[SpeakerIdentifier] préchauffage du modèle...")
        dummy = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)  # 3s de silence
        self.extract_embedding(dummy)
        print("[SpeakerIdentifier] préchauffage terminé, serveur prêt.")

    def _load_enrolled(self) -> dict[str, list[float]]:
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, "r") as f:
                return json.load(f)
        return {}

    def _save_enrolled(self):
        with open(EMBEDDINGS_PATH, "w") as f:
            json.dump(self.enrolled, f)

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        audio: numpy array float32, mono, 16kHz, valeurs dans [-1, 1]
        Retourne un embedding numpy 1D (192 dims pour ce modèle ECAPA).
        """
        wav_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, n_samples)
        with torch.no_grad():
            emb = self.classifier.encode_batch(wav_tensor)  # (1, 1, 192)
        return emb.squeeze().cpu().numpy()

    def enroll(self, name: str, audio: np.ndarray):
        """
        Enrôle une personne à partir d'un segment audio (idéalement 10-15s).
        Si la personne existe déjà, on MOYENNE avec l'embedding existant
        plutôt que d'écraser, pour permettre un ré-enrôlement incrémental.
        """
        new_emb = self.extract_embedding(audio)
        if name in self.enrolled:
            existing = np.array(self.enrolled[name])
            averaged = (existing + new_emb) / 2.0
            self.enrolled[name] = averaged.tolist()
        else:
            self.enrolled[name] = new_emb.tolist()
        self._save_enrolled()
        return self.enrolled[name]

    def remove(self, name: str):
        if name in self.enrolled:
            del self.enrolled[name]
            self._save_enrolled()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def identify(self, audio: np.ndarray, threshold: float = 0.45) -> dict:
        """
        Retourne {"name": str|None, "score": float, "all_scores": dict}
        name est None si aucun score ne dépasse le seuil -> "inconnu" côté client.
        """
        if not self.enrolled:
            return {"name": None, "score": 0.0, "all_scores": {}}

        query_emb = self.extract_embedding(audio)
        scores = {
            name: self.cosine_similarity(query_emb, np.array(ref_emb))
            for name, ref_emb in self.enrolled.items()
        }
        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score >= threshold:
            return {"name": best_name, "score": best_score, "all_scores": scores}
        return {"name": None, "score": best_score, "all_scores": scores}
