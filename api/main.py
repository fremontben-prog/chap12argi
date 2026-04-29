"""
API FastAPI — Prédiction de Rendement Agricole
Endpoints : POST /predict  |  POST /recommend
"""

import os
import json
import glob
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_config

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(BASE_DIR / "models_par_culture")))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(BASE_DIR / "model_metadata.json")))

# ──────────────────────────────────────────────────────────────────────────────
# État global
# ──────────────────────────────────────────────────────────────────────────────

models: dict           = {}
metadata: dict         = {}
historical_means: dict = {}
crop_metrics: dict     = {}


def _load_resources() -> None:
    """Charge les modèles .joblib et les métadonnées JSON."""
    global models, metadata, historical_means, crop_metrics

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        crop_metrics = metadata.get("crop_metrics", {})
        data_stats   = metadata.get("crop_data_stats", {})
        historical_means = {
            crop: stats["mean_yield"] / 10_000
            for crop, stats in data_stats.items()
        }
    else:
        print(f"[WARN] Métadonnées introuvables : {METADATA_PATH}")
        metadata = {}; crop_metrics = {}; historical_means = {}

    pattern = str(MODELS_DIR / "model_*.joblib")
    for path in sorted(glob.glob(pattern)):
        stem = Path(path).stem
        crop = stem.removeprefix("model_").replace("_", " ")
        try:
            models[crop] = joblib.load(path)
        except Exception as exc:
            print(f"[WARN] Impossible de charger {path}: {exc}")

    if not models:
        print(f"[WARN] Aucun modèle trouvé dans '{MODELS_DIR}'.")
    else:
        print(f"[OK] {len(models)} modèles chargés : {sorted(models.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan (remplace @on_event déprécié)
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield


app = FastAPI(
    title="🌾 Crop Yield Prediction API",
    description=(
        "API de prédiction de rendement agricole basée sur des modèles "
        "GradientBoosting entraînés par culture. "
        "Features : pluviométrie (mm/an), température moyenne (°C), pesticides (tonnes)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic V2
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_config = model_config(protected_namespaces=())

    crop: str = Field(..., description="Nom de la culture (anglais, minuscules)",
                      json_schema_extra={"example": "cassava"})
    rainfall_mm: float = Field(..., ge=0, le=10000,
                               description="Pluviométrie annuelle (mm/an)",
                               json_schema_extra={"example": 1200.0})
    avg_temp: float = Field(..., ge=-10, le=60,
                            description="Température moyenne annuelle (°C)",
                            json_schema_extra={"example": 22.0})
    pesticides_tonnes: float = Field(..., ge=0,
                                     description="Quantité de pesticides (tonnes)",
                                     json_schema_extra={"example": 50000.0})

    @field_validator("crop")
    @classmethod
    def crop_lowercase(cls, v: str) -> str:
        return v.strip().lower()


class YieldPrediction(BaseModel):
    model_config = model_config(protected_namespaces=())

    crop: str
    yield_hg_ha: float  = Field(..., description="Rendement prédit (hg/ha)")
    yield_t_ha: float   = Field(..., description="Rendement prédit (t/ha)")
    mae_t_ha: Optional[float]          = Field(None, description="MAE du modèle (t/ha)")
    vs_historique_pct: Optional[float] = Field(None, description="Écart vs moyenne historique (%)")
    model_r2: Optional[float]          = Field(None, description="R² du modèle sur le jeu de test")
    fiabilite: str = Field(..., description="Fiabilité du modèle")


class RecommendRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0, le=10000,
                               json_schema_extra={"example": 1200.0})
    avg_temp: float    = Field(..., ge=-10, le=60,
                               json_schema_extra={"example": 22.0})
    pesticides_tonnes: float = Field(..., ge=0,
                                     json_schema_extra={"example": 50000.0})


class RecommendResponse(BaseModel):
    conditions: dict
    recommendations: list[YieldPrediction]
    best_crop: str
    best_yield_t_ha: float


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _features(rainfall_mm: float, avg_temp: float,
              pesticides_tonnes: float) -> np.ndarray:
    return np.array([[rainfall_mm, pesticides_tonnes, avg_temp]])


def _build_prediction(crop: str, rainfall_mm: float,
                      avg_temp: float, pesticides_tonnes: float) -> YieldPrediction:
    X           = _features(rainfall_mm, avg_temp, pesticides_tonnes)
    yield_hg_ha = float(models[crop].predict(X)[0])
    yield_t_ha  = yield_hg_ha / 10_000

    m        = crop_metrics.get(crop, {})
    r2       = m.get("R2")
    mae_t_ha = m.get("MAE_t_ha")

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
    return {
        "status": "ok",
        "models_loaded": len(models),
        "available_crops": sorted(models.keys()),
    }


@app.get("/crops", tags=["Info"])
def list_crops():
    return {"crops": sorted(models.keys()), "count": len(models)}


@app.post("/predict", response_model=YieldPrediction, tags=["Prédiction"])
def predict(body: PredictRequest):
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
