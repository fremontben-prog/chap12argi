"""
api/tests/test_integration.py
Tests d'intégration — Crop Yield Predictor

Utilisent les VRAIS modèles .joblib pour valider les prédictions réelles.
Nécessitent que les variables d'environnement MODELS_DIR et METADATA_PATH
pointent vers les vrais fichiers (configuré dans le job CI 'integration').

Lance avec :
    pytest api/tests/test_integration.py -v -m integration
"""

import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────────────────────────────────────
# Skip automatique si les vrais modèles sont absents
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR    = Path(os.getenv("MODELS_DIR", "api/models_par_culture"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "api/model_metadata.json"))

REAL_MODELS_AVAILABLE = (
    MODELS_DIR.exists()
    and len(list(MODELS_DIR.glob("model_*.joblib"))) == 10
    and METADATA_PATH.exists()
)

skip_if_no_models = pytest.mark.skipif(
    not REAL_MODELS_AVAILABLE,
    reason="Vrais modèles .joblib introuvables — tests d'intégration ignorés",
)

if REAL_MODELS_AVAILABLE:
    from main import app
    client = TestClient(app, raise_server_exceptions=True)
    client.__enter__()
else:
    client = None


# ──────────────────────────────────────────────────────────────────────────────
# Valeurs de référence issues du pipeline Phase 4
# (conditions : rainfall=1200, temp=22, pesticides=50000)
# ──────────────────────────────────────────────────────────────────────────────

REFERENCE_CONDITIONS = {
    "rainfall_mm": 1200.0,
    "avg_temp": 22.0,
    "pesticides_tonnes": 50000.0,
}

# Rendements attendus ± tolérance 15% (variabilité due aux splits train/test)
REFERENCE_YIELDS = {
    "manioc":              (13.67, 2.0),   # (t/ha attendu, tolérance t/ha)
    "pommes de terre":             (16.69, 2.5),
    "bananes plantains et autres": ( 9.95, 2.8),
    "patates douces":       ( 6.54, 2.4),
    "ignames":                 (12.04, 1.4),
    "riz":                 ( 3.80, 0.5),
    "sorgho":              ( 2.29, 0.5),
    "blé":                ( 2.25, 0.5),
    "maïs":                ( 2.01, 0.8),
    "soja":              ( 1.67, 0.3),
}


# ──────────────────────────────────────────────────────────────────────────────
# Tests d'intégration — /predict
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegrationPredict:

    @skip_if_no_models
    def test_all_10_models_loaded(self):
        r = client.get("/crops")
        assert r.json()["count"] == 10

    @skip_if_no_models
    @pytest.mark.parametrize("crop,expected,tolerance", [
        (crop, val, tol)
        for crop, (val, tol) in REFERENCE_YIELDS.items()
    ])
    def test_predict_yield_close_to_reference(self, crop, expected, tolerance):
        """Vérifie que chaque modèle prédit dans la plage attendue."""
        payload = {"crop": crop, **REFERENCE_CONDITIONS}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, f"Erreur API pour {crop}: {r.text}"
        yield_val = r.json()["yield_t_ha"]
        assert abs(yield_val - expected) <= tolerance, (
            f"{crop} : prédit {yield_val:.3f} t/ha, "
            f"attendu {expected:.2f} ± {tolerance} t/ha"
        )

    @skip_if_no_models
    def test_predict_r2_above_threshold(self):
        """Tous les modèles doivent avoir R² ≥ 0.78 (seuil minimum du projet)."""
        for crop in REFERENCE_YIELDS:
            r = client.post("/predict", json={"crop": crop, **REFERENCE_CONDITIONS})
            r2 = r.json().get("model_r2")
            assert r2 is not None, f"R² absent pour {crop}"
            assert r2 >= 0.78, f"{crop} : R²={r2:.3f} < seuil 0.78"

    @skip_if_no_models
    def test_predict_fiabilite_elevee_for_good_models(self):
        """Les modèles R²≥0.7 doivent retourner fiabilite='Élevée'."""
        for crop in REFERENCE_YIELDS:
            r = client.post("/predict", json={"crop": crop, **REFERENCE_CONDITIONS})
            data = r.json()
            if data.get("model_r2", 0) >= 0.7:
                assert "levée" in data["fiabilite"], (
                    f"{crop} : fiabilite='{data['fiabilite']}' inattendue"
                )

    @skip_if_no_models
    def test_predict_yield_positive_for_all_crops(self):
        """Le rendement doit toujours être positif."""
        for crop in REFERENCE_YIELDS:
            r = client.post("/predict", json={"crop": crop, **REFERENCE_CONDITIONS})
            assert r.json()["yield_t_ha"] > 0, f"{crop} : rendement négatif"

    @skip_if_no_models
    def test_predict_different_conditions_change_yield(self):
        """Des conditions différentes doivent produire des prédictions différentes."""
        payload_dry  = {"crop": "manioc", "rainfall_mm": 200.0,  "avg_temp": 18.0, "pesticides_tonnes": 1000.0}
        payload_wet  = {"crop": "manioc", "rainfall_mm": 3000.0, "avg_temp": 28.0, "pesticides_tonnes": 200000.0}
        yield_dry = client.post("/predict", json=payload_dry).json()["yield_t_ha"]
        yield_wet = client.post("/predict", json=payload_wet).json()["yield_t_ha"]
        assert yield_dry != yield_wet, "Conditions différentes → prédictions identiques (suspect)"


# ──────────────────────────────────────────────────────────────────────────────
# Tests d'intégration — /recommend
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegrationRecommend:

    @skip_if_no_models
    def test_recommend_returns_10_crops(self):
        r = client.post("/recommend", json=REFERENCE_CONDITIONS)
        assert len(r.json()["recommendations"]) == 10

    @skip_if_no_models
    def test_recommend_best_crop_is_pommes de terre(self):
        """Dans les conditions de référence, pommes de terre doit être recommandé."""
        r = client.post("/recommend", json=REFERENCE_CONDITIONS)
        assert r.json()["best_crop"] == "pommes de terre", (
            f"Meilleure culture attendue : pommes de terre, "
            f"obtenu : {r.json()['best_crop']}"
        )

    @skip_if_no_models
    def test_recommend_sorted_descending(self):
        r = client.post("/recommend", json=REFERENCE_CONDITIONS)
        yields = [rec["yield_t_ha"] for rec in r.json()["recommendations"]]
        assert yields == sorted(yields, reverse=True)

    @skip_if_no_models
    def test_recommend_all_yields_positive(self):
        r = client.post("/recommend", json=REFERENCE_CONDITIONS)
        for rec in r.json()["recommendations"]:
            assert rec["yield_t_ha"] > 0, f"{rec['crop']} : rendement négatif"


    @skip_if_no_models
    def test_recommend_tropical_climate_favors_manioc_or_ignames(self):
        """Climat chaud et humide → manioc ou ignames dans le top 3."""
        tropical = {"rainfall_mm": 2500.0, "avg_temp": 29.0, "pesticides_tonnes": 10000.0}
        r = client.post("/recommend", json=tropical)
        top3 = [rec["crop"] for rec in r.json()["recommendations"][:3]]
        assert any(c in top3 for c in ["manioc", "ignames"]), (
            f"Aucune culture tropicale dans le top 3 : {top3}"
        )
