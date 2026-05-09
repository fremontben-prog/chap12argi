# 🌾 Crop Yield Predictor — Pipeline ML Rendement Agricole

> Prédiction et recommandation de rendements agricoles par culture, basé sur des données FAO réelles.  
> Architecture **1 modèle par culture** — anti-leakage — API FastAPI + UI Streamlit + CI/CD GitHub Actions.

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Architecture](#architecture)
- [Données](#données)
- [Pipeline ML](#pipeline-ml)
- [API FastAPI](#api-fastapi)
- [UI Streamlit](#ui-streamlit)
- [Tests](#tests)
- [CI/CD](#cicd)
- [MLflow](#mlflow)
- [Installation & Lancement](#installation--lancement)
- [Structure du projet](#structure-du-projet)

---

## Aperçu du projet

Ce projet prédit le rendement agricole (en t/ha) pour **10 cultures** à partir de 3 variables climatiques et agronomiques. Il expose deux fonctions métier principales :

| Fonction | Description |
|---|---|
| `predict_yield(crop, rainfall, temp, pesticides)` | Prédit le rendement d'une culture donnée dans des conditions spécifiques |
| `recommend_crop(rainfall, temp, pesticides)` | Classe toutes les cultures par rendement prédit et recommande la meilleure |

**Résultats globaux :**

| Métrique | Valeur |
|---|---|
| R² moyen (10 cultures) | **0.9133** |
| R² médian | **0.9253** |
| Modèles fiables (R² ≥ 0.70) | **10 / 10** |
| Modèles gagnants | XGBoost (6) · GradientBoosting (4) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Design anti-leakage                          │
│                                                                 │
│  Problème : un modèle unique voit toutes les cultures →         │
│             leakage via les features Crop_*                     │
│                                                                 │
│  Solution : 1 modèle entraîné PAR culture                       │
│             → pas de confusion inter-cultures                   │
│             → chaque modèle calibré sur ses vraies données      │
└─────────────────────────────────────────────────────────────────┘

  predict_yield()  →  models[crop].predict(conditions)
  recommend_crop() →  boucle sur tous les models, classement final
```

**Stack technique :**

| Composant | Technologie |
|---|---|
| ML Pipeline | Python · scikit-learn · XGBoost |
| Tracking | MLflow |
| API | FastAPI · Uvicorn |
| UI | Streamlit |
| Conteneurisation | Docker · Docker Compose |
| CI/CD | GitHub Actions |
| Tests | pytest · pytest-cov · httpx |

---

## Données

**Source :** FAO (Food and Agriculture Organization)  
**Dataset :** `data/merged/dataset_consolide.csv`

| Propriété | Valeur |
|---|---|
| Observations | 28 242 |
| Cultures | 10 |
| Features | 3 |
| Cible | `hg/ha_yield` |

**Features :**

| Variable | Description | Unité |
|---|---|---|
| `average_rain_fall_mm_per_year` | Pluviométrie annuelle | mm/an |
| `avg_temp` | Température moyenne | °C |
| `pesticides_tonnes` | Quantité de pesticides | tonnes |

**10 cultures modélisées :**
bananes plantains et autres · blé · ignames · manioc · maïs · patates douces · pommes de terre · riz · soja · sorgho

---

## Pipeline ML

Le pipeline complet se lance avec :

```bash
python src/pipeline.py
```

### Phase 1 — Comparaison des modèles de base

Pour chaque culture, 5 modèles sont évalués en cross-validation :

| Modèle | R² typique | Remarque |
|---|---|---|
| Ridge / Lasso | < 0.35 | Relations non-linéaires → inadapté |
| RandomForest | 0.88 – 0.95 | Bon R² mais overfit plus élevé |
| GradientBoosting | 0.81 – 0.91 | Bon équilibre biais/variance |
| XGBoost | 0.87 – 0.95 | Meilleur score composite sur 6/10 cultures |

### Phase 2 — Optimisation & score composite

Sélection du meilleur modèle par culture via un **score composite** :

```
Score = R²_test  −  0.5 × |overfit|  −  0.3 × std_cv
```

| Culture | Modèle | R² test | MAE (t/ha) |
|---|---|---|---|
| Bananes plantains | XGBoost | 0.8138 | 1.491 |
| Blé | XGBoost | 0.9500 | 0.270 |
| Ignames | XGBoost | 0.9328 | 0.665 |
| Manioc | GradientBoosting | 0.9513 | 1.123 |
| Maïs | XGBoost | 0.9178 | 0.456 |
| Patates douces | XGBoost | 0.8890 | 1.090 |
| Pommes de terre | GradientBoosting | 0.9367 | 1.545 |
| Riz | XGBoost | 0.9340 | 0.301 |
| Soja | GradientBoosting | 0.8940 | 0.145 |
| Sorgho | GradientBoosting | 0.9135 | 0.208 |

### Phase 3 — Importance des variables

| Culture | Pluie (mm/an) | Pesticides (t) | Température (°C) |
|---|---|---|---|
| Bananes plantains | **58.6%** | 31.4% | 9.9% |
| Blé | **60.6%** | 12.3% | 27.1% |
| Ignames | 41.9% | 11.5% | **46.5%** |
| Manioc | 28.3% | **44.7%** | 27.0% |
| Maïs | **48.8%** | 13.2% | 38.0% |
| Soja | 27.6% | **45.7%** | 26.7% |

> **Insights :** La pluviométrie est le driver principal pour 7/10 cultures. Ignames et maïs sont les plus sensibles à la température (impact climatique fort). Manioc et soja requièrent un usage intensif de pesticides.

### Phase 4 — Fonctions métier

**Exemple pour les conditions : 1 200 mm/an · 22°C · 50 000 t pesticides**

```
predict_yield('bananes plantains et autres', 1200, 22, 50000)
→ 9.95 t/ha  [7.15 – 12.76 t/ha]  ✓ Fiabilité élevée (R²=0.814)

recommend_crop(1200, 22, 50000)
→ 1. Pommes de terre  16.69 t/ha  (GradientBoosting)
   2. Manioc           13.67 t/ha  (GradientBoosting)
   3. Ignames          12.04 t/ha  (XGBoost)
   ...
```

---

## API FastAPI

### Lancement local (sans Docker)

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Lancement avec Docker

```bash
cd api
docker build --no-cache -t crop-yield-api .
docker run -p 8000:8000 crop-yield-api
```

### Lancement avec Docker Compose

```bash
docker compose up
```

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Health check + nombre de modèles chargés |
| GET | `/crops` | Liste des cultures disponibles |
| POST | `/predict` | Prédire le rendement d'une culture |
| POST | `/recommend` | Recommander la meilleure culture |

**POST /predict — exemple :**

```json
// Request
{
  "crop": "manioc",
  "rainfall_mm": 1200.0,
  "avg_temp": 22.0,
  "pesticides_tonnes": 50000.0
}

// Response
{
  "crop": "manioc",
  "yield_hg_ha": 136700.0,
  "yield_t_ha": 13.67,
  "model_r2": 0.9513,
  "fiabilite": "Élevée",
  "vs_historique_pct": -9.1,
  "interval_low": 11.71,
  "interval_high": 15.63
}
```

**POST /recommend — exemple :**

```json
// Request
{
  "rainfall_mm": 1200.0,
  "avg_temp": 22.0,
  "pesticides_tonnes": 50000.0
}

// Response
{
  "best_crop": "pommes de terre",
  "conditions": { "rainfall_mm": 1200.0, "avg_temp": 22.0, "pesticides_tonnes": 50000.0 },
  "recommendations": [
    { "rank": 1, "crop": "pommes de terre", "yield_t_ha": 16.69, "model_r2": 0.9367 },
    { "rank": 2, "crop": "manioc",          "yield_t_ha": 13.67, "model_r2": 0.9513 },
    ...
  ]
}
```

**Variables d'environnement :**

| Variable | Défaut | Description |
|---|---|---|
| `MODELS_DIR` | `models_par_culture` | Chemin vers les fichiers `.joblib` |
| `METADATA_PATH` | `model_metadata.json` | Chemin vers les métadonnées |

### Documentation interactive

Une fois l'API lancée : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## UI Streamlit

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

L'interface permet de saisir les conditions climatiques, visualiser les prédictions et explorer les recommandations par culture de manière interactive.

---

## Tests

### Tests unitaires

Utilisent des **modèles factices** (`_FakeModel` → retourne toujours 13.67 t/ha fixe).  
Rapides, sans dépendances externes, toujours lancés en CI.

```bash
pytest api/tests/test_api.py \
  --cov=api \
  --cov-report=term-missing \
  -v
```

Variables d'environnement pour les fixtures :
```bash
MODELS_DIR=api/tests/fixtures/models_par_culture
METADATA_PATH=api/tests/fixtures/model_metadata.json
```

**Couverture (29 tests) :**

| Classe | Tests |
|---|---|
| `TestHealthEndpoints` | GET `/`, `/crops` — status, structure |
| `TestPredictEndpoint` | 200/404/422, yield calcul, R², case-insensitive |
| `TestRecommendEndpoint` | tri, best_crop, echo conditions, validation |
| `TestInternalHelpers` | shape et ordre des features `_features()` |

### Tests d'intégration

Utilisent les **vrais modèles `.joblib`** téléchargés depuis GitHub Releases.  
Skippés automatiquement si les modèles sont absents (CI/CD safe).

```bash
pytest api/tests/test_integration.py -v -m integration
```

Variables d'environnement :
```bash
MODELS_DIR=api/models_par_culture
METADATA_PATH=api/model_metadata.json
```

**11 tests d'intégration :**

| Classe | Tests clés |
|---|---|
| `TestIntegrationPredict` | 10 modèles chargés · yields ±tolérance · R²≥0.78 · rendements positifs · conditions différentes → prédictions différentes |
| `TestIntegrationRecommend` | 10 cultures · pommes de terre top 1 · tri décroissant · climat tropical → manioc/ignames top 3 |

---

## CI/CD

Pipeline **GitHub Actions** (`.github/workflows/cicd.yml`) :

```
push (master/develop)  ──┐
pull_request (master)  ──┴──► [1. test] ──► [2. integration*] ──► [3. build] ──► [4. push-docker**]

* uniquement sur push (pas PR)
** uniquement sur master
```

| Job | Déclencheur | Description |
|---|---|---|
| **test** | Toujours | Tests unitaires + coverage XML |
| **integration** | Push uniquement | Télécharge `models.tar.gz` depuis GitHub Releases, lance les tests d'intégration |
| **build** | Après test (+ integration si applicable) | `docker build` sans push, cache GHA |
| **push-docker** | Master + push uniquement | Build & push `latest` + `sha` vers Docker Hub |

### Secrets requis

| Secret | Usage |
|---|---|
| `DOCKERHUB_USERNAME` | Login Docker Hub |
| `DOCKERHUB_TOKEN` | Token Docker Hub (push) |
| `GITHUB_TOKEN` | Téléchargement des modèles depuis Releases (automatique) |

### Publier les modèles (mise à jour)

Après réentraînement, archiver et publier les modèles comme Release GitHub :

```bash
tar -czf models.tar.gz models_par_culture model_metadata.json
# Puis créer/mettre à jour le tag 'models-latest' dans GitHub Releases
# et uploader models.tar.gz comme asset de la release
```

---

## MLflow

Le tracking MLflow est intégré au pipeline. Pour visualiser toutes les expériences :

```bash
# Depuis le répertoire racine du projet
mlflow ui
```

Puis ouvrir [http://localhost:5000](http://localhost:5000)

Les runs sont organisés par culture, avec les métriques R², RMSE, MAE, overfit et std_cv pour chaque modèle et chaque fold de cross-validation.

---

## Installation & Lancement

### Prérequis

- Python 3.10+
- Docker & Docker Compose
- (optionnel) MLflow

### Lancement complet

```bash
# 1. Cloner le dépôt
git clone <repo-url>
cd chap12agri

# 2. Entraîner les modèles
python src/pipeline.py
# → Les modèles sont sauvegardés dans api/models_par_culture/
# → Les PNG de résultats sont dans png_res/

# 3. Lancer l'API (Docker)
cd api
docker build --no-cache -t crop-yield-api .
docker run -p 8000:8000 crop-yield-api

# 4. Lancer l'UI Streamlit
cd streamlit
pip install -r requirements.txt
streamlit run app.py

# 5. Visualiser les expériences MLflow
mlflow ui   # depuis le répertoire racine
```

---

## Structure du projet

```
chap12agri/
├── src/
│   └── pipeline.py                  # Pipeline ML complet (phases 1–5)
│
├── data/
│   └── merged/
│       └── dataset_consolide.csv    # Dataset FAO (28 242 obs)
│
├── api/
│   ├── main.py                      # Application FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── model_metadata.json          # Métriques et métadonnées des modèles
│   ├── models_par_culture/          # Modèles .joblib (1 par culture)
│   │   ├── model_blé.joblib
│   │   ├── model_manioc.joblib
│   │   └── ...
│   └── tests/
│       ├── test_api.py              # Tests unitaires (modèles factices)
│       ├── test_integration.py      # Tests d'intégration (vrais .joblib)
│       └── fixtures/
│           ├── model_metadata.json
│           └── models_par_culture/
│
├── streamlit/
│   ├── app.py                       # Interface utilisateur Streamlit
│   └── requirements.txt
│
├── png_res/                         # Graphiques générés par le pipeline
│   ├── phase1_comparaison_modeles.png
│   ├── phase2_modeles_optimises.png
│   ├── phase3_importance_et_metier.png
│   └── phase4_recommandation_cultures.png
│
├── .github/
│   └── workflows/
│       └── cicd.yml                 # Pipeline CI/CD GitHub Actions
│
├── docker-compose.yml
└── README.md
```

---

## Résultats & visualisations

| Fichier | Contenu |
|---|---|
| `png_res/phase1_comparaison_modeles.png` | R² comparatif Ridge/RF/XGB/GB par culture |
| `png_res/phase2_modeles_optimises.png` | Résultats post-optimisation avec score composite |
| `png_res/phase3_importance_et_metier.png` | Importance des variables + insights métier |
| `png_res/phase4_recommandation_cultures.png` | Classement des cultures pour les conditions de test |

---

*Données : FAO — Food and Agriculture Organization of the United Nations*
