"""
storage.py — Persistance des embeddings enrôlés.

Pour 10 locuteurs, pas besoin de Postgres : SQLite sur le VOLUME Railway suffit
largement. On stocke les embeddings individuels (pas seulement les centroïdes),
ce qui permet d'ajouter des échantillons plus tard et de recalculer le centroïde.

Un cache mémoire des centroïdes est maintenu pour l'identification (aucune lecture
disque sur le chemin critique). SQLite ne sert qu'à la persistance entre redéploys.
"""

from __future__ import annotations

import sqlite3
import threading

import numpy as np

from core import centroid


class SpeakerStore:
    def __init__(self, db_path: str) -> None:
        # check_same_thread=False : FastAPI peut appeler depuis plusieurs threads.
        # On sérialise nous-mêmes les écritures avec un verrou (un seul writer).
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker  TEXT NOT NULL,
                vec      BLOB NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._db.commit()

        # Cache mémoire : {speaker: [vecteurs]} et {speaker: centroïde}.
        self._samples: dict[str, list[np.ndarray]] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._load()

    # ----- chargement initial --------------------------------------------- #
    def _load(self) -> None:
        rows = self._db.execute("SELECT speaker, vec FROM embeddings").fetchall()
        self._samples.clear()
        for speaker, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            self._samples.setdefault(speaker, []).append(vec)
        self._recompute_all()

    def _recompute_all(self) -> None:
        self._centroids = {
            name: centroid(vecs) for name, vecs in self._samples.items() if vecs
        }

    # ----- API utilisée par les endpoints --------------------------------- #
    def add(self, speaker: str, vec: np.ndarray) -> int:
        """Ajoute un échantillon et met à jour le centroïde du locuteur."""
        speaker = speaker.strip()
        if not speaker:
            raise ValueError("Le nom du locuteur est vide.")
        with self._lock:
            self._db.execute(
                "INSERT INTO embeddings (speaker, vec) VALUES (?, ?)",
                (speaker, vec.astype(np.float32).tobytes()),
            )
            self._db.commit()
            self._samples.setdefault(speaker, []).append(vec.astype(np.float32))
            self._centroids[speaker] = centroid(self._samples[speaker])
            return len(self._samples[speaker])

    def centroids(self) -> dict[str, np.ndarray]:
        """Snapshot des centroïdes (chemin critique de l'identification)."""
        return dict(self._centroids)

    def speakers(self) -> dict[str, int]:
        """{nom: nombre d'échantillons}."""
        return {name: len(v) for name, v in self._samples.items()}

    def delete(self, speaker: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM embeddings WHERE speaker = ?", (speaker,)
            )
            self._db.commit()
            existed = self._samples.pop(speaker, None) is not None
            self._centroids.pop(speaker, None)
            return existed or cur.rowcount > 0
