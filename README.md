# VoicePrint — Phase 1 (POST)

Identification de locuteur open-set. Enrôlement par fichier **ou** micro, identification
push-to-talk depuis Safari iOS. La « cervelle » (`core.py`) est découplée du transport :
la Phase 2 (WebSocket temps réel) réutilisera `core.py` et `storage.py` sans y toucher.

```
server/
  core.py          # décodage ffmpeg -> VAD -> ECAPA -> cosinus/seuil  (transport-agnostique)
  storage.py       # SQLite sur le volume Railway + cache mémoire des centroïdes
  main.py          # FastAPI : /enroll /identify /speakers  (couche fine)
  requirements.txt
  Dockerfile       # ffmpeg + torch CPU
client/
  index.html       # à héberger sur GitHub Pages
```

## 1. Serveur sur Railway

1. Nouveau projet → déploie depuis le repo, **dossier `server/`** (build via Dockerfile).
2. **Ajoute un volume** monté sur **`/data`** (embeddings + cache du modèle y survivent aux redeploys).
3. Variables d'environnement (optionnelles) :
   - `DATA_DIR=/data` (défaut)
   - `THRESHOLD=0.25` (seuil de rejet open-set — voir calibration)
   - `ALLOW_ORIGINS=https://TON-USER.github.io` (restreins le CORS une fois en place)
4. Prévois **≥ 2 Go de RAM** : torch + SpeechBrain sont lourds et le **premier boot télécharge
   ECAPA** (quelques dizaines de Mo) → démarrage lent la première fois, rapide ensuite grâce au volume.

Vérifie : `GET https://TON-APP.up.railway.app/health` → `{"status":"ok",...}`.

## 2. Client sur GitHub Pages

1. Ouvre `client/index.html`, remplace `SERVER_URL` par l'URL Railway.
2. Pousse `index.html` sur une branche publiée par Pages. HTTPS est fourni (obligatoire pour le micro).
3. Ouvre la page sur l'iPhone. Le micro exige un appui (geste utilisateur) — c'est déjà le cas ici.

## 3. Calibration du seuil (l'étape à ne pas sauter)

Le `THRESHOLD` sépare « locuteur connu » de « inconnu ». Il **n'est pas transférable** entre
conditions de micro ; il se règle empiriquement :

1. Enrôle 2–3 personnes (idéalement avec **un échantillon micro** chacune, pas seulement des fichiers).
2. Lance des identifications : mêmes personnes **et** des inconnus, en lisant le `cos` affiché.
3. Choisis un seuil **entre** le cosinus des vrais (souvent 0.35–0.7) et celui des inconnus
   (souvent < 0.25). Démarre bas (0.25) : peu de faux « inconnu », quelques faux positifs, puis remonte.

**Le piège n°1 reste le mismatch fichier↔micro.** Si tu n'enrôles que par fichiers propres mais
identifies au micro iPhone, les cosinus s'effondrent. D'où le bouton « enrôler au micro » : ajoute
au moins un échantillon dans les conditions réelles d'usage.

## 4. Notes iOS

- La capture utilise `MediaRecorder` (sort du `audio/mp4`/AAC sur iOS) → décodé par ffmpeg côté serveur.
  Aucun resampling JS nécessaire en Phase 1.
- `echoCancellation`/`noiseSuppression`/`autoGainControl` sont désactivés pour un signal cohérent
  (iOS peut néanmoins les ignorer).
- Garde l'app au premier plan : iOS suspend l'audio écran verrouillé / en arrière-plan.

## 5. Cap vers la Phase 2 (temps réel)

On ajoutera un endpoint **WebSocket** dans `main.py` qui, au lieu d'un blob unique, bufferise
l'audio en **fenêtre glissante** (2–3 s, hop 0,5–1 s), appelle le **même** `processor.embed` +
`match`, puis **lisse** la sortie (vote majoritaire / EMA) pour éviter le papillonnement
connu/inconnu. Côté client, `MediaRecorder` cède la place à un **AudioWorklet** (PCM 16 kHz).
`core.py` et `storage.py` restent inchangés.
