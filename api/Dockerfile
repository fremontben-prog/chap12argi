# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Crop Yield Prediction API
# Build  : docker build -t crop-yield-api .
# Run    : docker run -p 8000:8000 \
#            -v $(pwd)/models_par_culture:/app/models_par_culture:ro \
#            -v $(pwd)/model_metadata.json:/app/model_metadata.json:ro \
#            crop-yield-api
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="votre.email@example.com"
LABEL description="API FastAPI — Prédiction de rendement agricole"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/api/models_par_culture \
    METADATA_PATH=/api/model_metadata.json

WORKDIR /api

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY main.py .

# Copie des artefacts ML (modèles + métadonnées)
# Ces fichiers peuvent aussi être montés en volume (recommandé en production)
COPY models_par_culture/ ./models_par_culture/
COPY model_metadata.json .

# Port exposé par Uvicorn
EXPOSE 8000

# Healthcheck Docker natif
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" \
    || exit 1

# Démarrage de l'API (workers ajustables via la variable WORKERS)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
