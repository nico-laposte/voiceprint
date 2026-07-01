"""
VAD (Voice Activity Detection) basée sur webrtcvad.
Sert de filtre rapide avant d'appeler ECAPA (qui est plus coûteux) :
inutile de calculer un embedding sur du silence/bruit de fond.
"""

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad accepte seulement 10, 20 ou 30 ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class VoiceActivityDetector:
    def __init__(self, aggressiveness: int = 2):
        # aggressiveness: 0 (permissif) à 3 (strict). 2 est un bon défaut
        # pour un micro de smartphone en environnement normal.
        self.vad = webrtcvad.Vad(aggressiveness)

    def has_speech(self, audio: np.ndarray, min_ratio: float = 0.3) -> bool:
        """
        audio: float32 mono 16kHz dans [-1, 1]
        Découpe en frames de 30ms, retourne True si au moins `min_ratio`
        des frames contiennent de la voix selon webrtcvad.
        """
        # webrtcvad attend du PCM 16-bit signé
        pcm16 = (audio * 32767).astype(np.int16).tobytes()
        n_frames = len(audio) // FRAME_SAMPLES
        if n_frames == 0:
            return False

        speech_frames = 0
        frame_bytes = FRAME_SAMPLES * 2  # 2 bytes par sample en int16
        for i in range(n_frames):
            frame = pcm16[i * frame_bytes : (i + 1) * frame_bytes]
            if len(frame) < frame_bytes:
                break
            if self.vad.is_speech(frame, SAMPLE_RATE):
                speech_frames += 1

        return (speech_frames / n_frames) >= min_ratio
