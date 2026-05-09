# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ML — PRÉDICTION DE RENDEMENT AGRICOLE
# ═══════════════════════════════════════════════════════════════════════════
# Architecture : UN MODÈLE PAR CULTURE (sans leakage Crop_*)
# Fonction 1   : Prédiction du rendement pour une culture donnée
# Fonction 2   : Recommandation — classement de toutes les cultures
#
# PIPELINE :
#   Phase 0 — Environnement & chargement
#   Phase 1 — Comparaison des modèles de base (par culture)
#   Phase 2 — Optimisation multi-modèles (RF + XGBoost + GB) + sélection
#   Phase 3 — Importance des variables & interprétation métier
#   Phase 4 — Fonctions predict_yield / recommend_crop
#   Phase 5 — Sauvegarde meilleur modèle + métadonnées API
# ═══════════════════════════════════════════════════════════════════════════

import os
import json
import time
import warnings
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from dotenv import load_dotenv
from pathlib import Path

from sklearn.model_selection  import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model     import Ridge, Lasso
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import mean_squared_error, r2_score, mean_absolute_error
from xgboost                  import XGBRegressor


# ───────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ───────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

CLIMATE_FEATURES = [
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
]
TARGET      = 'hg/ha_yield'
TARGET_UNIT = 'hg/ha'

R2_GOOD = 0.70
R2_MED  = 0.50

warnings.filterwarnings("ignore", message=".*mlflow.sklearn.*")
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def title_print(title: str):
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  " + title.upper().ljust(66) + "║")
    print("╚" + "═" * 68 + "╝")

def line_print():
    print("─" * 70)

def compute_metrics(y_true, y_pred) -> dict:
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)
    mape     = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    mae_t_ha = mae / 10000
    return {
        'RMSE':     round(rmse,     4),
        'MAE':      round(mae,      4),
        'R2':       round(r2,       4),
        'MAPE':     round(mape,     4),
        'MAE_t_ha': round(mae_t_ha, 3),
    }

def color_r2(v: float) -> str:
    if v >= R2_GOOD: return '#2ecc71'
    if v >= R2_MED:  return '#e67e22'
    return '#e74c3c'


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 — ENVIRONNEMENT & CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 0 — Environnement & chargement")

load_dotenv()

chemin_complet = Path(os.getenv("CSV_MERGED")) / os.getenv("DATA_FILE")
png_dir        = Path(os.getenv("PNG_RES"));  png_dir.mkdir(parents=True, exist_ok=True)
api_dir        = Path(os.getenv("API_REP"));  api_dir.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(chemin_complet, index_col=0)

print(f"  Shape (avant Dropna)          : {df_raw.shape}")
df_raw = df_raw.dropna(subset=CLIMATE_FEATURES + [TARGET, 'Crop'])

# Définition du dictionnaire de traduction
traduction_cultures = {
    'potatoes': 'pommes de terre',
    'cassava': 'manioc',
    'maize': 'maïs',
    'wheat': 'blé',
    'sorghum': 'sorgho',
    'soybean': 'soja',
    'rice': 'riz',
    'yams': 'ignames',
    'plantains and others': 'bananes plantains et autres',
    'sweet potatoes': 'patates douces'
}

# Application de la traduction des cultures
df_raw['Crop'] = df_raw['Crop'].map(traduction_cultures).fillna(df_raw['Crop'])


CROPS = sorted(df_raw['Crop'].unique().tolist())

print(f"  Dataset         : {chemin_complet}")
print(f"  Shape           : {df_raw.shape}")
print(f"  Cultures ({len(CROPS):>2})  : {CROPS}")
print(f"  Features        : {CLIMATE_FEATURES}")
print(f"  Target          : {TARGET}")

missing_feat = [f for f in CLIMATE_FEATURES if f not in df_raw.columns]
if missing_feat:
    raise ValueError(f"[!] Features manquantes dans le dataset : {missing_feat}")

print("\n  Statistiques descriptives :")
print(df_raw[CLIMATE_FEATURES + [TARGET]].describe().round(2).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — COMPARAISON DES MODÈLES DE BASE PAR CULTURE
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 1 — Comparaison des modèles de base par culture")

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_01"))

MODELS_BASE = {
    'Ridge':            Ridge(alpha=1.0),
    'Lasso':            Lasso(alpha=0.1, random_state=SEED, max_iter=5000),
    'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    'XGBoost':          XGBRegressor(
                            n_estimators=100, learning_rate=0.1, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, device='cpu'
                        )
}

phase1_results  = {}
crop_data_stats = {}
crop_splits     = {}

for crop in CROPS:
    df_crop   = df_raw[df_raw['Crop'] == crop].copy()
    n_samples = len(df_crop)

    if n_samples < 50:
        print(f"  [~]  {crop:<20} ignorée ({n_samples} lignes insuffisantes)")
        continue

    X = df_crop[CLIMATE_FEATURES].astype(float)
    y = df_crop[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    crop_splits[crop] = (X_train, X_test, y_train, y_test)
    crop_data_stats[crop] = {
        'n_samples':  n_samples,
        'mean_yield': round(float(y.mean()), 0),
        'std_yield':  round(float(y.std()),  0),
    }

    phase1_results[crop] = {}
    print(f"\n  * {crop} ({n_samples:,} lignes)")
    line_print()

    for modelname, model in MODELS_BASE.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        metrics_train = compute_metrics(y_train, model.predict(X_train))
        metrics_test  = compute_metrics(y_test,  model.predict(X_test))
        overfit       = round(metrics_train['R2'] - metrics_test['R2'], 4)

        with mlflow.start_run(run_name=f"{crop}__{modelname}__base"):
            mlflow.set_tag('phase', 'base_comparison')
            mlflow.set_tag('crop',  crop)
            mlflow.set_tag('model', modelname)
            mlflow.log_param('n_train',  len(X_train))
            mlflow.log_param('n_test',   len(X_test))
            mlflow.log_param('features', str(CLIMATE_FEATURES))
            mlflow.log_metric('train_R2',      metrics_train['R2'])
            mlflow.log_metric('test_R2',       metrics_test['R2'])
            mlflow.log_metric('test_RMSE',     metrics_test['RMSE'])
            mlflow.log_metric('test_MAE',      metrics_test['MAE'])
            mlflow.log_metric('test_MAPE',     metrics_test['MAPE'])
            mlflow.log_metric('test_MAE_t_ha', metrics_test['MAE_t_ha'])
            mlflow.log_metric('overfit_gap',   overfit)
            mlflow.sklearn.log_model(model, name='model')

        phase1_results[crop][modelname] = {**metrics_test, 'overfit': overfit}
        print(f"    {modelname:<20} R²={metrics_test['R2']:.4f} | "
              f"RMSE={metrics_test['RMSE']:>10.0f} | "
              f"MAE={metrics_test['MAE_t_ha']:.2f} t/ha | "
              f"Overfit={overfit:.4f}  ({time.time()-t0:.1f}s)")

# ── Visualisation Phase 1 ─────────────────────────────────────────────────
title_print("Phase 1 — Visualisation comparaison modèles de base")

df_r2_heat = pd.DataFrame(
    {crop: {m: v['R2'] for m, v in models.items()}
     for crop, models in phase1_results.items()}
).T

fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(phase1_results) * 0.6 + 2)))
fig.suptitle("Phase 1 — Comparaison des modèles de base par culture\n"
             "(features : rainfall, température, pesticides — sans leakage Crop_*)",
             fontsize=12)

sns.heatmap(df_r2_heat, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'R²'})
axes[0].set_title("R² par culture × modèle\n(vert=bon, rouge=faible)")
axes[0].set_xlabel("Modèle")
axes[0].set_ylabel("Culture")

best_per_crop = df_r2_heat.idxmax(axis=1)
best_r2       = df_r2_heat.max(axis=1).sort_values()
colors_best   = [color_r2(v) for v in best_r2]
best_r2.plot(kind='barh', ax=axes[1], color=colors_best, edgecolor='white')
axes[1].axvline(R2_GOOD, color='green',  linestyle='--', lw=1.2, label=f'Seuil bon ({R2_GOOD})')
axes[1].axvline(R2_MED,  color='orange', linestyle='--', lw=1.2, label=f'Seuil moyen ({R2_MED})')
for i, (crop, v) in enumerate(best_r2.items()):
    axes[1].text(v + 0.005, i, f"{best_per_crop[crop]} ({v:.3f})", va='center', fontsize=8)
axes[1].set_title("Meilleur modèle par culture\n(R² du meilleur modèle)")
axes[1].set_xlabel("R²")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(png_dir / 'phase1_comparaison_modeles.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
print("  [OK] phase1_comparaison_modeles.png sauvegardé")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — OPTIMISATION MULTI-MODÈLES (RF + XGBoost + GB) + SÉLECTION
# ═══════════════════════════════════════════════════════════════════════════
# Pour chaque culture :
#   → GridSearchCV 5-fold sur les 3 modèles candidats
#   → Sélection par score composite :
#       score = R²_test − λ_overfit × |overfit| − λ_std × std_cv
#   → Log de tous les trials + du meilleur run dans MLflow
#   → Comparaison avant/après optimisation
#
# Pondérations :
#   LAMBDA_OVERFIT : pénalise le gap train/test        (défaut 0.5)
#   LAMBDA_STD     : pénalise l'instabilité CV         (défaut 0.3)
#
# Intuition :
#   - LAMBDA_OVERFIT élevé → favorise les modèles qui généralisent bien
#   - LAMBDA_STD élevé     → favorise les modèles stables entre les folds
#   - Les deux à 0         → revient à choisir sur le R² test seul
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 2 — Optimisation multi-modèles (RF + XGBoost + GB)")

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_02"))

# ── Pondérations du score composite ───────────────────────────────────────
LAMBDA_OVERFIT = 0.5   # pénalité sur |R²_train − R²_test|
LAMBDA_STD     = 0.3   # pénalité sur std du CV R²

def composite_score(test_r2: float, overfit: float, cv_std: float) -> float:
    """
    Score composite de sélection de modèle.
    Plus élevé = meilleur compromis performance / généralisation / stabilité.

    score = R²_test − LAMBDA_OVERFIT × |overfit| − LAMBDA_STD × std_cv
    """
    return test_r2 - LAMBDA_OVERFIT * abs(overfit) - LAMBDA_STD * cv_std

print(f"  Score composite : R²_test "
      f"− {LAMBDA_OVERFIT} × |overfit| "
      f"− {LAMBDA_STD} × std_cv")

# ── Grilles d'hyperparamètres par modèle ──────────────────────────────────
CANDIDATE_MODELS = {
    'RandomForest': {
        'estimator': RandomForestRegressor(random_state=SEED, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
        },
    },
    'XGBoost': {
        'estimator': XGBRegressor(random_state=SEED, n_jobs=-1,device='cpu'),
        'param_grid': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample':     [0.8, 1.0],
        },
    },
    'GradientBoosting': {
        'estimator': GradientBoostingRegressor(random_state=SEED),
        'param_grid': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample':     [0.8, 1.0],
        },
    },
}

# Stockage des résultats Phase 2
crop_models          = {}   # { crop: best_estimator }
crop_metrics         = {}   # { crop: metrics }
crop_best_params     = {}   # { crop: best_params }
crop_best_model_name = {}   # { crop: nom du modèle gagnant }
crop_scores_detail   = {}   # { crop: { model: score_composite } } pour traçabilité
phase2_runs          = {}   # { crop: run_id MLflow du meilleur }

line_print()
print(f"  {'Culture':<22} | {'Modèle gagnant':<18} | {'Score':>7} | "
      f"{'Test R²':>8} | {'Overfit':>8} | {'Std CV':>7} | {'MAE t/ha':>9}")
line_print()

for crop in CROPS:
    if crop not in crop_splits:
        continue

    X_train, X_test, y_train, y_test = crop_splits[crop]
    t0 = time.time()

    # ── Boucle sur les 3 modèles candidats ────────────────────────────────
    best_score     = -np.inf   # score composite
    best_estimator = None
    best_name      = None
    best_gs        = None
    best_cv_std    = None
    scores_detail  = {}

    for model_name, config in CANDIDATE_MODELS.items():
        gs = GridSearchCV(
            config['estimator'],
            config['param_grid'],
            cv=5, scoring='r2',
            n_jobs=-1, verbose=0, refit=True
        )
        gs.fit(X_train, y_train)

        # Métriques du meilleur trial de ce modèle
        m_test  = compute_metrics(y_test,  gs.best_estimator_.predict(X_test))
        m_train = compute_metrics(y_train, gs.best_estimator_.predict(X_train))
        overfit_m = round(m_train['R2'] - m_test['R2'], 4)
        cv_std_m  = round(
            gs.cv_results_['std_test_score'][gs.best_index_], 4
        )
        score_m = composite_score(m_test['R2'], overfit_m, cv_std_m)
        scores_detail[model_name] = {
            'composite': round(score_m,    4),
            'test_r2':   m_test['R2'],
            'overfit':   overfit_m,
            'cv_std':    cv_std_m,
        }

        # Log tous les trials de ce modèle dans MLflow
        cv_results = pd.DataFrame(gs.cv_results_)
        for _, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"{crop}__{model_name}__gs_trial"):
                mlflow.set_tag('phase',      'hyperopt_trial')
                mlflow.set_tag('crop',       crop)
                mlflow.set_tag('model_type', model_name)
                mlflow.log_params({f"{model_name}__{k}": v
                                   for k, v in row['params'].items()})
                mlflow.log_metric('cv_mean_r2', row['mean_test_score'])
                mlflow.log_metric('cv_std_r2',  row['std_test_score'])

        # Garder le meilleur score composite toutes familles confondues
        if score_m > best_score:
            best_score     = score_m
            best_estimator = gs.best_estimator_
            best_name      = model_name
            best_gs        = gs
            best_cv_std    = cv_std_m

    # ── Métriques finales du modèle gagnant ───────────────────────────────
    metrics_test  = compute_metrics(y_test,  best_estimator.predict(X_test))
    metrics_train = compute_metrics(y_train, best_estimator.predict(X_train))
    overfit       = round(metrics_train['R2'] - metrics_test['R2'], 4)

    # ── Log du meilleur modèle ────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{crop}__BEST__{best_name}") as run:
        run_id = run.info.run_id
        mlflow.set_tag('phase',      'best_model')
        mlflow.set_tag('crop',       crop)
        mlflow.set_tag('model',      best_name)
        mlflow.set_tag('dataset',    'dataset_consolide')
        mlflow.log_params(best_gs.best_params_)
        mlflow.log_metric('composite_score',  best_score)
        mlflow.log_metric('cv_best_r2',       best_gs.best_score_)
        mlflow.log_metric('cv_std_r2',        best_cv_std)
        mlflow.log_metric('test_R2',          metrics_test['R2'])
        mlflow.log_metric('test_RMSE',        metrics_test['RMSE'])
        mlflow.log_metric('test_MAE',         metrics_test['MAE'])
        mlflow.log_metric('test_MAPE',        metrics_test['MAPE'])
        mlflow.log_metric('test_MAE_t_ha',    metrics_test['MAE_t_ha'])
        mlflow.log_metric('overfit_gap',      overfit)
        mlflow.log_metric('lambda_overfit',   LAMBDA_OVERFIT)
        mlflow.log_metric('lambda_std',       LAMBDA_STD)
        mlflow.log_metric('n_train',          len(X_train))
        mlflow.sklearn.log_model(best_estimator, name=f'model_{crop}')

    crop_models[crop]          = best_estimator
    crop_best_model_name[crop] = best_name
    crop_scores_detail[crop]   = scores_detail
    crop_metrics[crop]         = {
        **metrics_test,
        'overfit':          overfit,
        'cv_best_r2':       round(best_gs.best_score_, 4),
        'cv_std':           best_cv_std,
        'composite_score':  round(best_score, 4),
        'model_name':       best_name,
    }
    crop_best_params[crop] = best_gs.best_params_
    phase2_runs[crop]      = run_id

    print(f"  {crop:<22} | {best_name:<18} | {best_score:>7.4f} | "
          f"{metrics_test['R2']:>8.4f} | {overfit:>8.4f} | "
          f"{best_cv_std:>7.4f} | {metrics_test['MAE_t_ha']:>9.3f}  "
          f"({time.time()-t0:.1f}s)")

line_print()

# ── Tableau de détail — scores composites par modèle et par culture ───────
print(f"\n  Détail des scores composites "
      f"(R²_test − {LAMBDA_OVERFIT}×|overfit| − {LAMBDA_STD}×std_cv) :")
line_print()
print(f"  {'Culture':<22} | "
      + " | ".join(f"{m[:18]:>18}" for m in CANDIDATE_MODELS)
      + " | Gagnant")
line_print()
for crop_d, detail in crop_scores_detail.items():
    row_str = f"  {crop_d:<22} | "
    for m in CANDIDATE_MODELS:
        s      = detail.get(m, {}).get('composite', float('nan'))
        marker = " ◄" if m == crop_best_model_name[crop_d] else "  "
        row_str += f"{s:>16.4f}{marker} | "
    row_str += crop_best_model_name[crop_d]
    print(row_str)
line_print()

# Décompte par famille
from collections import Counter
winner_counts = Counter(crop_best_model_name.values())
print(f"\n  Répartition des gagnants : {dict(winner_counts)}")

# ── Visualisation Phase 2 ─────────────────────────────────────────────────
title_print("Phase 2 — Visualisation post-optimisation")

df_p2 = pd.DataFrame(crop_metrics).T.sort_values('R2', ascending=False)

# Comparaison avant/après : R² base (meilleur base) vs R² optimisé
base_r2_best = {
    crop: max(phase1_results[crop].get(m, {}).get('R2', -np.inf)
              for m in ['RandomForest', 'XGBoost', 'GradientBoosting'])
    for crop in crop_models
}
df_compare = pd.DataFrame({
    'R² base (meilleur base)': base_r2_best,
    'R² optimisé':             df_p2['R2'],
}).dropna().sort_values('R² optimisé')

fig, axes = plt.subplots(1, 3, figsize=(22, max(5, len(crop_models) * 0.5 + 2)))
fig.suptitle(
    "Phase 2 — Optimisation multi-modèles (RF / XGBoost / GradientBoosting + GridSearchCV)\n"
    "1 modèle gagnant par culture — features : rainfall, température, pesticides",
    fontsize=12
)

# Panel 1 : Avant / Après
df_compare.plot(kind='barh', ax=axes[0],
                color=['#95a5a6', '#2ecc71'], edgecolor='white', width=0.7)
axes[0].axvline(R2_GOOD, color='green', linestyle='--', lw=1, label=f'Seuil bon ({R2_GOOD})')
axes[0].set_title("R² avant vs après optimisation\n(base: meilleur des 3 non-optimisé)")
axes[0].set_xlabel("R²")
axes[0].legend(fontsize=8)
# Annoter le modèle gagnant
for i, crop in enumerate(df_compare.index):
    r2_val = df_compare.loc[crop, 'R² optimisé']
    axes[0].text(r2_val + 0.005, i,
                 crop_best_model_name.get(crop, ''), va='center', fontsize=7,
                 color='#27ae60')

# Panel 2 : RMSE en t/ha
rmse_tha = (df_p2['RMSE'] / 10000).sort_values()
rmse_tha.plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='white')
axes[1].set_title("RMSE par culture (t/ha)\n(erreur type de prédiction)")
axes[1].set_xlabel("RMSE (t/ha)")
for i, v in enumerate(rmse_tha.values):
    axes[1].text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)

# Panel 3 : MAE t/ha
mae_tha = df_p2['MAE_t_ha'].sort_values()
mae_tha.plot(kind='barh', ax=axes[2], color='#9b59b6', edgecolor='white')
axes[2].set_title("Erreur moyenne (MAE) en t/ha\n(lisible par un agriculteur)")
axes[2].set_xlabel("MAE (t/ha)")
for i, v in enumerate(mae_tha.values):
    axes[2].text(v + 0.005, i, f'±{v:.2f} t/ha', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(png_dir / 'phase2_modeles_optimises.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
print("  [OK] phase2_modeles_optimises.png sauvegardé")

# Résumé console Phase 2
mean_r2   = df_p2['R2'].mean()
median_r2 = df_p2['R2'].median()
good_n    = (df_p2['R2'] >= R2_GOOD).sum()

print(f"\n  R² moyen   : {mean_r2:.4f}")
print(f"  R² médian  : {median_r2:.4f}")
print(f"  Modèles R² ≥ {R2_GOOD} : {good_n}/{len(crop_models)}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — IMPORTANCE DES VARIABLES & INTERPRÉTATION MÉTIER
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 3 — Importance des variables & interprétation métier")

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_03"))

feature_labels = {
    'average_rain_fall_mm_per_year': 'Pluie (mm/an)',
    'pesticides_tonnes':             'Pesticides (t)',
    'avg_temp':                      'Température (°C)',
}

imp_data = {}
for crop, model in crop_models.items():
    imp_data[crop] = {
        feature_labels[f]: round(float(v), 4)
        for f, v in zip(CLIMATE_FEATURES, model.feature_importances_)
    }

df_imp = pd.DataFrame(imp_data).T
print("\n  Importance des variables par culture :")
print(df_imp.round(3).to_string())

FEATURE_SAFE_NAMES = {
    'average_rain_fall_mm_per_year': 'rainfall',
    'pesticides_tonnes':             'pesticides',
    'avg_temp':                      'temperature',
}

with mlflow.start_run(run_name="feature_importance_global"):
    mlflow.set_tag('phase', 'business_insights')
    for crop, model in crop_models.items():
        safe_crop = crop.replace(' ', '_').replace('/', '_')
        for feat, val in zip(CLIMATE_FEATURES, model.feature_importances_):
            safe_feat = FEATURE_SAFE_NAMES[feat]
            mlflow.log_metric(f"imp__{safe_crop}__{safe_feat}", float(val))

temp_corr = (
    df_raw.groupby('Crop')
          .corr(numeric_only=True)
          .loc[(slice(None), TARGET), 'avg_temp']
          .droplevel(1)
          .reindex(list(crop_models.keys()))
          .dropna()
          .sort_values()
          .round(3)
)

rain_corr = (
    df_raw.groupby('Crop')
          .corr(numeric_only=True)
          .loc[(slice(None), TARGET), 'average_rain_fall_mm_per_year']
          .droplevel(1)
          .reindex(list(crop_models.keys()))
          .dropna()
          .sort_values()
          .round(3)
)

yield_by_crop = (df_raw.groupby('Crop')[TARGET]
                        .agg(['mean', 'std', 'min', 'max', 'count'])
                        .reindex(list(crop_models.keys()))
                        .dropna()
                        .sort_values('mean', ascending=False)
                        .round(0))
yield_by_crop['mean_t_ha'] = (yield_by_crop['mean'] / 10000).round(2)

# ── Visualisation Phase 3 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.suptitle("Phase 3 — Importance des variables & Recommandations métier\n"
             "Source : FAO Agriculture CropYield Dataset",
             fontsize=14, y=0.98)

gs_layout = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs_layout[0, 0])
sns.heatmap(df_imp, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=ax1, cbar_kws={'label': 'Importance'})
ax1.set_title("Importance des variables par culture\n"
              "(+ foncé = variable qui drive le + le rendement)", fontsize=10)

ax2 = fig.add_subplot(gs_layout[0, 1])
colors_tc = ['#e74c3c' if v < 0 else '#2ecc71' for v in temp_corr]
ax2.barh(temp_corr.index, temp_corr.values, color=colors_tc, edgecolor='white')
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_title("Impact de la température (°C) sur le rendement\n"
              "(rouge = la chaleur nuit | vert = la chaleur aide)", fontsize=10)
ax2.set_xlabel("Corrélation (r)")
for i, v in enumerate(temp_corr.values):
    ax2.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:.2f}',
             va='center', ha='left' if v >= 0 else 'right', fontsize=9)

ax3 = fig.add_subplot(gs_layout[0, 2])
colors_rc = ['#e74c3c' if v < 0 else '#3498db' for v in rain_corr]
ax3.barh(rain_corr.index, rain_corr.values, color=colors_rc, edgecolor='white')
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_title("Impact de la pluviométrie (mm/an) sur le rendement\n"
              "(rouge = la pluie nuit | bleu = la pluie aide)", fontsize=10)
ax3.set_xlabel("Corrélation (r)")
for i, v in enumerate(rain_corr.values):
    ax3.text(v + (0.003 if v >= 0 else -0.003), i, f'{v:.2f}',
             va='center', ha='left' if v >= 0 else 'right', fontsize=9)

ax4 = fig.add_subplot(gs_layout[1, 0])
med_yield = yield_by_crop['mean_t_ha'].median()
colors_yc = ['#2ecc71' if v >= med_yield else '#e74c3c'
             for v in yield_by_crop['mean_t_ha'].sort_values()]
yield_by_crop['mean_t_ha'].sort_values().plot(
    kind='barh', ax=ax4, color=colors_yc, edgecolor='white'
)
ax4.axvline(med_yield, color='orange', linestyle='--', lw=1.5,
            label=f'Médiane : {med_yield:.1f} t/ha')
ax4.set_title("Rendement moyen historique (t/ha)\n"
              "(vert = au-dessus de la médiane FAO)", fontsize=10)
ax4.set_xlabel("Rendement moyen (t/ha)")
ax4.legend(fontsize=8)

ax5 = fig.add_subplot(gs_layout[1, 1])
yield_trend = df_raw.groupby('Year')[TARGET].mean().reset_index()
ax5.plot(yield_trend['Year'], yield_trend[TARGET] / 10000,
         color='steelblue', lw=2, marker='o', markersize=3)
ax5.set_title("Tendance mondiale du rendement (t/ha)\nÉvolution historique FAO", fontsize=10)
ax5.set_xlabel("Année")
ax5.set_ylabel("Rendement moyen mondial (t/ha)")
ax5.grid(axis='y', alpha=0.3)

ax6 = fig.add_subplot(gs_layout[1, 2])
r2_sorted  = df_p2['R2'].sort_values()
colors_m   = [color_r2(v) for v in r2_sorted]
r2_sorted.plot(kind='barh', ax=ax6, color=colors_m, edgecolor='white')
ax6.axvline(R2_GOOD, color='green',  linestyle='--', lw=1.2, label=f'Bon (≥{R2_GOOD})')
ax6.axvline(R2_MED,  color='orange', linestyle='--', lw=1.2, label=f'Moyen (≥{R2_MED})')
for i, (crop_name, v) in enumerate(r2_sorted.items()):
    winner = crop_best_model_name.get(crop_name, '')
    ax6.text(v + 0.005, i, f'{v:.3f} ({winner[:2]})', va='center', fontsize=8)
ax6.set_title("Fiabilité des modèles par culture (R²)\n"
              "(vert=fiable | orange=moyen | rouge=fragile)", fontsize=10)
ax6.set_xlabel("R²")
ax6.legend(fontsize=8)

plt.savefig(png_dir / 'phase3_importance_et_metier.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
print("  [OK] phase3_importance_et_metier.png sauvegardé")

best_crop_yield  = yield_by_crop['mean_t_ha'].idxmax()
worst_crop_yield = yield_by_crop['mean_t_ha'].idxmin()
temp_pos = temp_corr[temp_corr > 0].idxmax() if (temp_corr > 0).any() else "aucune"
temp_neg = temp_corr[temp_corr < 0].idxmin() if (temp_corr < 0).any() else "aucune"

print(f"""
  ┌─ Insights métier ─────────────────────────────────────────────────┐
  │  Culture la plus productive (historique) : {best_crop_yield:<23}│
  │  Culture la moins productive             : {worst_crop_yield:<23}│
  │  Culture qui profite le + de la chaleur  : {temp_pos:<23}│
  │  Culture la + pénalisée par la chaleur   : {temp_neg:<23}│
  └───────────────────────────────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — FONCTIONS MÉTIER
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 4 — Fonctions métier : predict_yield & recommend_crop")


def predict_yield(crop: str,
                  rainfall_mm: float,
                  avg_temp: float,
                  pesticides_tonnes: float) -> dict:
    if crop not in crop_models:
        raise ValueError(
            f"Culture '{crop}' inconnue.\n"
            f"Cultures disponibles : {sorted(crop_models.keys())}"
        )

    model = crop_models[crop]
    X_input = pd.DataFrame([{
        'average_rain_fall_mm_per_year': rainfall_mm,
        'pesticides_tonnes':             pesticides_tonnes,
        'avg_temp':                      avg_temp,
    }])[CLIMATE_FEATURES]

    pred_hg_ha = float(model.predict(X_input)[0])
    pred_t_ha  = pred_hg_ha / 10000
    rmse_t_ha  = crop_metrics[crop]['RMSE'] / 10000
    r2         = crop_metrics[crop]['R2']

    ref_yield  = crop_data_stats[crop]['mean_yield'] / 10000
    delta_pct  = (pred_t_ha - ref_yield) / ref_yield * 100
    if delta_pct >= 10:
        interpretation = f"[+] Conditions favorables (+{delta_pct:.1f}% vs moyenne historique)"
    elif delta_pct <= -10:
        interpretation = f"[-] Conditions défavorables ({delta_pct:.1f}% vs moyenne historique)"
    else:
        interpretation = f"[~] Conditions dans la normale ({delta_pct:+.1f}% vs moyenne historique)"

    if r2 >= R2_GOOD:
        fiabilite = f"[OK] Élevée (R²={r2:.3f})"
    elif r2 >= R2_MED:
        fiabilite = f"[~]  Moyenne (R²={r2:.3f}) — à interpréter avec prudence"
    else:
        fiabilite = f"[!] Faible (R²={r2:.3f}) — résultat indicatif uniquement"

    return {
        'crop':              crop,
        'model_used':        crop_metrics[crop]['model_name'],
        'yield_hg_ha':       round(pred_hg_ha, 0),
        'yield_t_ha':        round(pred_t_ha,  2),
        'interval_min_t_ha': round(pred_t_ha - rmse_t_ha, 2),
        'interval_max_t_ha': round(pred_t_ha + rmse_t_ha, 2),
        'ref_yield_t_ha':    round(ref_yield,  2),
        'delta_vs_mean_pct': round(delta_pct,  1),
        'model_r2':          r2,
        'model_rmse_t_ha':   round(rmse_t_ha,  2),
        'fiabilite':         fiabilite,
        'interpretation':    interpretation,
    }


def recommend_crop(rainfall_mm: float,
                   avg_temp: float,
                   pesticides_tonnes: float,
                   top_n: int = None,
                   min_r2: float = 0.0) -> pd.DataFrame:
    results = []
    for crop, model in crop_models.items():
        X_input = pd.DataFrame([{
            'average_rain_fall_mm_per_year': rainfall_mm,
            'pesticides_tonnes':             pesticides_tonnes,
            'avg_temp':                      avg_temp,
        }])[CLIMATE_FEATURES]

        pred_hg_ha = float(model.predict(X_input)[0])
        pred_t_ha  = pred_hg_ha / 10000
        r2         = crop_metrics[crop]['R2']
        rmse_t_ha  = crop_metrics[crop]['RMSE'] / 10000

        if r2 < min_r2:
            continue

        ref_yield = crop_data_stats[crop]['mean_yield'] / 10000
        delta_pct = (pred_t_ha - ref_yield) / ref_yield * 100

        results.append({
            'crop':            crop,
            'model_used':      crop_metrics[crop]['model_name'],
            'yield_t_ha':      round(pred_t_ha,  2),
            'interval':        f"±{rmse_t_ha:.2f}",
            'vs_historique':   f"{delta_pct:+.1f}%",
            'model_r2':        r2,
            'n_train_samples': crop_data_stats[crop]['n_samples'],
            'fiabilite':       '[OK]' if r2 >= R2_GOOD else '[~]' if r2 >= R2_MED else '[!]',
        })

    df_rec = (pd.DataFrame(results)
                .sort_values('yield_t_ha', ascending=False)
                .reset_index(drop=True))
    df_rec.insert(0, 'rang', df_rec.index + 1)

    return df_rec.head(top_n) if top_n else df_rec


# ── Démonstration ─────────────────────────────────────────────────────────
print("\n  ── Exemple d'utilisation ────────────────────────────────────────")

CONDITIONS = {
    'rainfall_mm':        1200.0,
    'avg_temp':           22.0,
    'pesticides_tonnes':  50000.0,
}
print(f"\n  Conditions parcelle :")
print(f"    Pluviométrie : {CONDITIONS['rainfall_mm']} mm/an")
print(f"    Température  : {CONDITIONS['avg_temp']} °C")
print(f"    Pesticides   : {CONDITIONS['pesticides_tonnes']:,.0f} tonnes")

example_crop = CROPS[0]
res_f1 = predict_yield(example_crop, **CONDITIONS)
print(f"\n  Fonction 1 — Prédiction pour '{example_crop}' :")
print(f"    Modèle utilisé   : {res_f1['model_used']}")
print(f"    Rendement prédit : {res_f1['yield_t_ha']} t/ha "
      f"[{res_f1['interval_min_t_ha']} – {res_f1['interval_max_t_ha']} t/ha]")
print(f"    {res_f1['interpretation']}")
print(f"    Fiabilité : {res_f1['fiabilite']}")

df_rec = recommend_crop(**CONDITIONS)
print(f"\n  Fonction 2 — Recommandation pour ces conditions :")
print(df_rec[['rang', 'crop', 'model_used', 'yield_t_ha', 'interval',
              'vs_historique', 'model_r2', 'fiabilite']].to_string(index=False))
print(f"\n  * Recommandation : '{df_rec.iloc[0]['crop']}' "
      f"→ {df_rec.iloc[0]['yield_t_ha']} t/ha prédit "
      f"[{df_rec.iloc[0]['model_used']}]")

# ── Visualisation Recommandation ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, max(5, len(df_rec) * 0.55 + 1.5)))

colors_rec = []
for _, row in df_rec.iterrows():
    if row['model_r2'] >= R2_GOOD:   colors_rec.append('#2ecc71')
    elif row['model_r2'] >= R2_MED:  colors_rec.append('#e67e22')
    else:                            colors_rec.append('#e74c3c')

ax.barh(df_rec['crop'][::-1], df_rec['yield_t_ha'][::-1],
        color=colors_rec[::-1], edgecolor='white')

for i, (_, row) in enumerate(df_rec[::-1].reset_index(drop=True).iterrows()):
    ax.text(row['yield_t_ha'] + 0.05, i,
            f"{row['yield_t_ha']} t/ha  {row['vs_historique']}  "
            f"[{row['model_used'][:2]} R²={row['model_r2']:.2f}] {row['fiabilite']}",
            va='center', fontsize=9)

ax.set_title(
    f"Fonction 2 — Recommandation des cultures\n"
    f"Pluie={CONDITIONS['rainfall_mm']} mm | "
    f"Temp={CONDITIONS['avg_temp']}°C | "
    f"Pesticides={CONDITIONS['pesticides_tonnes']:,.0f} t\n"
    f"(n°1=or | vert=modèle fiable R²≥{R2_GOOD} | "
    f"orange=modèle moyen | rouge=fragile)",
    fontsize=11
)
ax.set_xlabel("Rendement prédit (t/ha)")
plt.tight_layout()
plt.savefig(png_dir / 'phase4_recommandation_cultures.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
print("\n  [OK] phase4_recommandation_cultures.png sauvegardé")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — SAUVEGARDE MEILLEUR MODÈLE PAR CULTURE & MÉTADONNÉES API
# ═══════════════════════════════════════════════════════════════════════════

title_print("Phase 5 — Sauvegarde des modèles & métadonnées API")

models_dir = api_dir / 'models_par_culture'
models_dir.mkdir(parents=True, exist_ok=True)

saved_models = {}
for crop, model in crop_models.items():
    safe_name  = crop.replace(' ', '_').replace('/', '_')
    model_name = crop_best_model_name[crop]
    model_path = models_dir / f'model_{safe_name}.joblib'
    joblib.dump(model, model_path)
    saved_models[crop] = str(model_path)
    print(f"  [OK] {crop:<22} [{model_name:<18}] → {model_path.name}")

# Métadonnées complètes pour l'API
model_metadata = {
    'architecture':       'per_crop_models',
    'model_selection':    'best_of_RF_XGBoost_GradientBoosting_via_GridSearchCV',
    'features':           CLIMATE_FEATURES,
    'feature_labels':     feature_labels,
    'target':             TARGET,
    'target_unit':        TARGET_UNIT,
    'seed':               SEED,
    'crops':              CROPS,
    'models_dir':         str(models_dir),
    'saved_models':       saved_models,
    'crop_metrics':       crop_metrics,
    'crop_best_params':   crop_best_params,
    'crop_best_model':    crop_best_model_name,
    'crop_data_stats':    crop_data_stats,
    'mlflow_runs':        phase2_runs,
    'functions': {
        'predict_yield':  'predict_yield(crop, rainfall_mm, avg_temp, pesticides_tonnes) -> dict',
        'recommend_crop': 'recommend_crop(rainfall_mm, avg_temp, pesticides_tonnes, top_n, min_r2) -> DataFrame',
    },
    'leakage_note': (
        "Architecture sans leakage : Crop n'est PAS une feature. "
        "Un modèle distinct (meilleur parmi RF / XGBoost / GradientBoosting) "
        "est calibré par culture sur les seules relations climatiques/agronomiques réelles."
    ),
    'global_performance': {
        'mean_r2':       round(float(df_p2['R2'].mean()),   4),
        'median_r2':     round(float(df_p2['R2'].median()), 4),
        'n_models':      len(crop_models),
        'n_good_models': int((df_p2['R2'] >= R2_GOOD).sum()),
        'winner_counts': dict(Counter(crop_best_model_name.values())),
    },
}

metadata_path = api_dir / os.getenv("MODEL_METADATA")
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"\n  [OK] model_metadata.json → {metadata_path}")
print(f"  [OK] {len(saved_models)} modèles joblib → {models_dir}")

# ── Résumé final ──────────────────────────────────────────────────────────
winner_summary = "  |  ".join(
    f"{k}: {v}" for k, v in Counter(crop_best_model_name.values()).items()
)
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║       RÉSUMÉ FINAL — PIPELINE ML RENDEMENT AGRICOLE                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture    : 1 meilleur modèle / culture (RF|XGB|GB)           ║
║  Features        : rainfall | température | pesticides               ║
║  Leakage Crop_*  : [OK] ÉLIMINÉ                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cultures modélisées    : {len(crop_models):<43}║
║  R² moyen (toutes)      : {df_p2['R2'].mean():<43.4f}║
║  R² médian              : {df_p2['R2'].median():<43.4f}║
║  Modèles fiables (≥{R2_GOOD}) : {int((df_p2['R2'] >= R2_GOOD).sum()):<43}║
╠══════════════════════════════════════════════════════════════════════╣
║  Modèles gagnants par culture :                                      ║
║    {winner_summary:<66}║
╠══════════════════════════════════════════════════════════════════════╣
║  SCREENSHOTS MLflow générés :                                        ║
║    phase1_comparaison_modeles.png                                    ║
║    phase2_modeles_optimises.png                                      ║
║    phase3_importance_et_metier.png                                   ║
║    phase4_recommandation_cultures.png                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  FONCTION 1 — Prédiction                                             ║
║    predict_yield(crop, rainfall_mm, avg_temp, pesticides_tonnes)     ║
║  FONCTION 2 — Recommandation                                         ║
║    recommend_crop(rainfall_mm, avg_temp, pesticides_tonnes)          ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("  → Lancez 'mlflow ui' depuis le terminal pour visualiser toutes les expériences.")
print("  → Les PNG sont dans :", png_dir)
print("  → Les modèles sont dans :", models_dir)