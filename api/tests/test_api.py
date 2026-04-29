"""
api/tests/test_api.py
Tests unitaires — Crop Yield Predictor
"""

import json
import os
import numpy as np
import joblib
import pytest
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Fixtures sur disque
# ──────────────────────────────────────────────────────────────────────────────

FIXTURES_DIR  = Path(__file__).parent / "fixtures"
MODELS_DIR    = FIXTURES_DIR / "models_par_culture"
METADATA_PATH = FIXTURES_DIR / "model_metadata.json"

CROPS = ["cassava", "maize", "wheat"]

FAKE_METADATA = {
    "architecture": "per_crop_models",
    "model_type": "GradientBoostingRegressor",
    "features": ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"],
    "target": "hg/ha_yield",
    "crops": CROPS,
    "crop_metrics": {
        "cassava": {"R2": 0.951, "RMSE": 19598.0, "MAE": 11228.0, "MAE_t_ha": 1.123, "overfit": 0.035},
        "maize":   {"R2": 0.918, "RMSE":  7715.0, "MAE":  4291.0, "MAE_t_ha": 0.429, "overfit": 0.072},
        "wheat":   {"R2": 0.949, "RMSE":  4137.0, "MAE":  2674.0, "MAE_t_ha": 0.267, "overfit": 0.038},
    },
    "crop_data_stats": {
        "cassava": {"n_samples": 2045, "mean_yield": 150479.0, "std_yield": 89739.0},
        "maize":   {"n_samples": 4121, "mean_yield":  36310.0, "std_yield": 27456.0},
        "wheat":   {"n_samples": 3857, "mean_yield":  30116.0, "std_yield": 18388.0},
    },
    "global_performance": {"mean_r2": 0.939, "median_r2": 0.949, "n_models": 3},
}


class _FakeModel:
    """Retourne toujours 136700 hg/ha = 13.67 t/ha."""
    def predict(self, X):
        return np.array([136700.0])


def _create_fixtures():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for crop in CROPS:
        joblib.dump(_FakeModel(), MODELS_DIR / f"model_{crop.replace(' ', '_')}.joblib")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(FAKE_METADATA, f)


_create_fixtures()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Variables d'env AVANT import de main
# ──────────────────────────────────────────────────────────────────────────────

os.environ["MODELS_DIR"]    = str(MODELS_DIR)
os.environ["METADATA_PATH"] = str(METADATA_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Import de main puis injection directe des fakes dans les globaux
# ──────────────────────────────────────────────────────────────────────────────

import main as main_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Injection directe — contourne tout problème de cache ou de lifespan
main_module.models = {crop: _FakeModel() for crop in CROPS}
main_module.crop_metrics = FAKE_METADATA["crop_metrics"]
main_module.historical_means = {
    crop: stats["mean_yield"] / 10_000
    for crop, stats in FAKE_METADATA["crop_data_stats"].items()
}

client = TestClient(main_module.app)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_models_loaded(self):
        r = client.get("/")
        data = r.json()
        assert "models_loaded" in data
        assert data["models_loaded"] == len(CROPS)

    def test_crops_endpoint(self):
        r = client.get("/crops")
        assert r.status_code == 200
        data = r.json()
        assert "crops" in data
        assert data["count"] == len(CROPS)
        for crop in CROPS:
            assert crop in data["crops"]


# ──────────────────────────────────────────────────────────────────────────────
# Tests — POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:

    VALID_PAYLOAD = {
        "crop": "cassava",
        "rainfall_mm": 1200.0,
        "avg_temp": 22.0,
        "pesticides_tonnes": 50000.0,
    }

    def test_predict_valid_returns_200(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_predict_returns_expected_fields(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        data = r.json()
        for field in ("crop", "yield_hg_ha", "yield_t_ha", "fiabilite"):
            assert field in data, f"Champ manquant : {field}"

    def test_predict_yield_t_ha_equals_hg_ha_divided_by_10000(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        data = r.json()
        assert abs(data["yield_t_ha"] - data["yield_hg_ha"] / 10_000) < 1e-3

    def test_predict_fake_model_returns_13_67(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        data = r.json()
        assert abs(data["yield_t_ha"] - 13.67) < 0.01

    def test_predict_crop_case_insensitive(self):
        payload = {**self.VALID_PAYLOAD, "crop": "CASSAVA"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_unknown_crop_returns_404(self):
        payload = {**self.VALID_PAYLOAD, "crop": "unknown_crop"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 404

    def test_predict_negative_rainfall_returns_422(self):
        payload = {**self.VALID_PAYLOAD, "rainfall_mm": -100.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_missing_field_returns_422(self):
        payload = {"crop": "cassava", "rainfall_mm": 1200.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_model_r2_in_response(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        data = r.json()
        assert data["model_r2"] == pytest.approx(0.951, abs=1e-3)

    def test_predict_vs_historique_is_computed(self):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        data = r.json()
        assert data["vs_historique_pct"] is not None
        assert isinstance(data["vs_historique_pct"], float)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — POST /recommend
# ──────────────────────────────────────────────────────────────────────────────

class TestRecommendEndpoint:

    VALID_PAYLOAD = {
        "rainfall_mm": 1200.0,
        "avg_temp": 22.0,
        "pesticides_tonnes": 50000.0,
    }

    def test_recommend_valid_returns_200(self):
        r = client.post("/recommend", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_recommend_returns_all_crops(self):
        r = client.post("/recommend", json=self.VALID_PAYLOAD)
        assert len(r.json()["recommendations"]) == len(CROPS)

    def test_recommend_sorted_descending(self):
        r = client.post("/recommend", json=self.VALID_PAYLOAD)
        yields = [rec["yield_t_ha"] for rec in r.json()["recommendations"]]
        assert yields == sorted(yields, reverse=True)

    def test_recommend_best_crop_matches_first_recommendation(self):
        r = client.post("/recommend", json=self.VALID_PAYLOAD)
        data = r.json()
        assert data["best_crop"] == data["recommendations"][0]["crop"]

    def test_recommend_conditions_echoed_in_response(self):
        r = client.post("/recommend", json=self.VALID_PAYLOAD)
        conditions = r.json()["conditions"]
        assert conditions["rainfall_mm"]       == self.VALID_PAYLOAD["rainfall_mm"]
        assert conditions["avg_temp"]          == self.VALID_PAYLOAD["avg_temp"]
        assert conditions["pesticides_tonnes"] == self.VALID_PAYLOAD["pesticides_tonnes"]

    def test_recommend_missing_field_returns_422(self):
        r = client.post("/recommend", json={"rainfall_mm": 1200.0})
        assert r.status_code == 422

    def test_recommend_extreme_temperature_returns_200(self):
        payload = {**self.VALID_PAYLOAD, "avg_temp": 30.0}
        r = client.post("/recommend", json=payload)
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Utilitaires internes
# ──────────────────────────────────────────────────────────────────────────────

class TestInternalHelpers:

    def test_features_shape(self):
        X = main_module._features(1200.0, 22.0, 50000.0)
        assert X.shape == (1, 3)

    def test_features_order(self):
        X = main_module._features(1200.0, 22.0, 50000.0)
        assert X[0][0] == 1200.0    # rainfall
        assert X[0][1] == 50000.0   # pesticides
        assert X[0][2] == 22.0      # temp