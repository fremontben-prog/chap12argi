"""
API FastAPI — Prédiction de Rendement Agricole
Endpoints : POST /predict  |  POST /recommend
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(BASE_DIR / "models_par_culture")))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(BASE_DIR / "model_metadata.json")))

# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🌾 Crop Yield Prediction API",
    description=(
        "API de prédiction de rendement agricole basée sur des modèles "
        "GradientBoosting entraînés par culture. "
        "Features : pluviométrie (mm/an), température moyenne (°C), pesticides (tonnes)."
    ),
    version="1.0.0",
)

models: dict = {}
metadata: dict = {}
historical_means: dict = {}   # t/ha par culture
crop_metrics: dict = {}       # R², RMSE, MAE… par culture


def _load_resources() -> None:
    """Charge les modèles .joblib et les métadonnées JSON."""
    global models, metadata, historical_means, crop_metrics

    # --- Métadonnées ---
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Structure réelle : metadata["crop_metrics"][crop] = {R2, RMSE, MAE_t_ha, ...}
        crop_metrics = metadata.get("crop_metrics", {})

        # Moyennes historiques en hg/ha → converties en t/ha
        data_stats = metadata.get("crop_data_stats", {})
        historical_means = {
            crop: stats["mean_yield"] / 10_000
            for crop, stats in data_stats.items()
        }
    else:
        print(f"[WARN] Métadonnées introuvables : {METADATA_PATH}")
        metadata         = {}
        crop_metrics     = {}
        historical_means = {}

    # --- Modèles ---
    pattern = str(MODELS_DIR / "model_*.joblib")
    for path in sorted(glob.glob(pattern)):
        stem = Path(path).stem                              # "model_cassava"
        crop = stem.removeprefix("model_").replace("_", " ")  # "cassava"
        try:
            models[crop] = joblib.load(path)
        except Exception as exc:
            print(f"[WARN] Impossible de charger {path}: {exc}")

    if not models:
        print(f"[WARN] Aucun modèle trouvé dans '{MODELS_DIR}'.")
    else:
        print(f"[OK] {len(models)} modèles chargés : {sorted(models.keys())}")


@app.on_event("startup")
def startup_event():
    _load_resources()


# ──────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    crop: str = Field(..., example="cassava",
                      description="Nom de la culture (anglais, minuscules)")
    rainfall_mm: float = Field(..., ge=0, le=10000, example=1200.0,
                               description="Pluviométrie annuelle (mm/an)")
    avg_temp: float = Field(..., ge=-10, le=60, example=22.0,
                            description="Température moyenne annuelle (°C)")
    pesticides_tonnes: float = Field(..., ge=0, example=50000.0,
                                     description="Quantité de pesticides (tonnes)")

    @validator("crop")
    def crop_lowercase(cls, v):
        return v.strip().lower()


class YieldPrediction(BaseModel):
    crop: str
    yield_hg_ha: float  = Field(..., description="Rendement prédit (hg/ha)")
    yield_t_ha: float   = Field(..., description="Rendement prédit (t/ha)")
    mae_t_ha: Optional[float]         = Field(None, description="MAE du modèle (t/ha)")
    vs_historique_pct: Optional[float] = Field(None, description="Écart vs moyenne historique (%)")
    model_r2: Optional[float]          = Field(None, description="R² du modèle sur le jeu de test")
    fiabilite: str = Field(..., description="Fiabilité du modèle")


class RecommendRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0, le=10000, example=1200.0,
                               description="Pluviométrie annuelle (mm/an)")
    avg_temp: float = Field(..., ge=-10, le=60, example=22.0,
                            description="Température moyenne annuelle (°C)")
    pesticides_tonnes: float = Field(..., ge=0, example=50000.0,
                                     description="Quantité de pesticides (tonnes)")


class RecommendResponse(BaseModel):
    conditions: dict
    recommendations: list[YieldPrediction]
    best_crop: str
    best_yield_t_ha: float


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _features(rainfall_mm: float, avg_temp: float, pesticides_tonnes: float) -> np.ndarray:
    """Vecteur de features dans l'ordre attendu par les modèles."""
    return np.array([[rainfall_mm, pesticides_tonnes, avg_temp]])


def _build_prediction(crop: str, rainfall_mm: float,
                      avg_temp: float, pesticides_tonnes: float) -> YieldPrediction:
    model       = models[crop]
    X           = _features(rainfall_mm, avg_temp, pesticides_tonnes)
    yield_hg_ha = float(model.predict(X)[0])
    yield_t_ha  = yield_hg_ha / 10_000

    # Métriques depuis crop_metrics (structure réelle du JSON)
    m        = crop_metrics.get(crop, {})
    r2       = m.get("R2")
    mae_t_ha = m.get("MAE_t_ha")

    # Écart vs moyenne historique
    hist_mean = historical_means.get(crop)
    vs_hist   = None
    if hist_mean and hist_mean > 0:
        vs_hist = round((yield_t_ha - hist_mean) / hist_mean * 100, 1)

    if r2 is None:
        fiabilite = "Inconnue"
    elif r2 >= 0.7:
        fiabilite = f"Élevée (R²={r2:.3f})"
    else:
        fiabilite = f"Faible (R²={r2:.3f})"

    return YieldPrediction(
        crop=crop,
        yield_hg_ha=round(yield_hg_ha, 2),
        yield_t_ha=round(yield_t_ha, 4),
        mae_t_ha=round(mae_t_ha, 3) if mae_t_ha is not None else None,
        vs_historique_pct=vs_hist,
        model_r2=round(r2, 4) if r2 is not None else None,
        fiabilite=fiabilite,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Healthcheck — vérifie que l'API est en ligne."""
    return {
        "status": "ok",
        "models_loaded": len(models),
        "available_crops": sorted(models.keys()),
    }


@app.get("/crops", tags=["Info"])
def list_crops():
    """Liste des cultures disponibles."""
    return {"crops": sorted(models.keys()), "count": len(models)}


@app.post("/predict", response_model=YieldPrediction, tags=["Prédiction"])
def predict(body: PredictRequest):
    """
    Prédit le rendement pour **une culture** donnée.

    - **crop** : nom de la culture (`cassava`, `wheat`, `rice`…)
    - **rainfall_mm** : pluviométrie annuelle en mm
    - **avg_temp** : température moyenne en °C
    - **pesticides_tonnes** : quantité de pesticides en tonnes
    """
    if body.crop not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Culture '{body.crop}' inconnue. Disponibles : {sorted(models.keys())}",
        )
    try:
        return _build_prediction(body.crop, body.rainfall_mm,
                                 body.avg_temp, body.pesticides_tonnes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommandation"])
def recommend(body: RecommendRequest):
    """
    Prédit le rendement de **toutes les cultures** et retourne le classement
    par rendement décroissant.

    - **rainfall_mm** : pluviométrie annuelle en mm
    - **avg_temp** : température moyenne en °C
    - **pesticides_tonnes** : quantité de pesticides en tonnes
    """
    if not models:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé.")

    predictions = []
    for crop in sorted(models.keys()):
        try:
            predictions.append(
                _build_prediction(crop, body.rainfall_mm,
                                  body.avg_temp, body.pesticides_tonnes)
            )
        except Exception as exc:
            print(f"[WARN] Prédiction échouée pour '{crop}': {exc}")

    predictions.sort(key=lambda p: p.yield_t_ha, reverse=True)
    best = predictions[0] if predictions else None

    return RecommendResponse(
        conditions={
            "rainfall_mm": body.rainfall_mm,
            "avg_temp": body.avg_temp,
            "pesticides_tonnes": body.pesticides_tonnes,
        },
        recommendations=predictions,
        best_crop=best.crop if best else "",
        best_yield_t_ha=best.yield_t_ha if best else 0.0,
    )