"""
app.py — Interface Streamlit pour l'API de prédiction de rendement agricole
Interroge FastAPI et affiche les résultats sans aucune logique ML.
"""

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go



# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# CSS personnalisé
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 50%, #1b4332 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(116,198,157,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #d8f3dc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #95d5b2;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* Carte métrique */
.metric-card {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    border: 1px solid rgba(116,198,157,0.3);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
}
.metric-card .label {
    color: #95d5b2;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #d8f3dc;
    line-height: 1;
}
.metric-card .unit {
    color: #74c69d;
    font-size: 1rem;
    margin-top: 0.3rem;
}
.metric-card .badge {
    display: inline-block;
    background: rgba(116,198,157,0.2);
    color: #74c69d;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    margin-top: 0.6rem;
    border: 1px solid rgba(116,198,157,0.3);
}

/* Badge écart historique */
.delta-positive { color: #52b788; }
.delta-negative { color: #f4a261; }

/* Alerte erreur API */
.api-error {
    background: rgba(220, 53, 69, 0.1);
    border: 1px solid rgba(220, 53, 69, 0.4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #f8d7da;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d2018;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers API
# ──────────────────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_available_crops() -> list[str]:
    try:
        r = requests.get(f"{API_URL}/crops", timeout=3)
        return r.json().get("crops", [])
    except Exception:
        return []


def call_predict(crop: str, rainfall_mm: float,
                 avg_temp: float, pesticides_tonnes: float) -> dict | None:
    try:
        payload = {
            "crop": crop,
            "rainfall_mm": rainfall_mm,
            "avg_temp": avg_temp,
            "pesticides_tonnes": pesticides_tonnes,
        }
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Erreur API ({e.response.status_code}) : {e.response.json().get('detail', str(e))}")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
    return None


def call_recommend(rainfall_mm: float,
                   avg_temp: float, pesticides_tonnes: float) -> dict | None:
    try:
        payload = {
            "rainfall_mm": rainfall_mm,
            "avg_temp": avg_temp,
            "pesticides_tonnes": pesticides_tonnes,
        }
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Erreur API ({e.response.status_code}) : {e.response.json().get('detail', str(e))}")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🌾 Crop Yield Predictor</h1>
    <p>Prédiction de rendement agricole par apprentissage automatique — GradientBoosting par culture</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Paramètres & statut API
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration API")
    api_url_input = st.text_input("URL de l'API", value=API_URL)
    API_URL = api_url_input.rstrip("/")

    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API connectée")
    else:
        st.error("❌ API non joignable")
        st.caption(f"Vérifiez que l'API tourne sur `{API_URL}`")

    st.divider()
    st.markdown("### 🌍 Conditions environnementales")

    rainfall_mm = st.slider(
        "💧 Pluviométrie (mm/an)",
        min_value=50, max_value=3240, value=1200, step=10,
        help="Précipitations annuelles moyennes en millimètres"
    )
    avg_temp = st.slider(
        "🌡️ Température moyenne (°C)",
        min_value=1, max_value=31, value=22, step=1,
        help="Température annuelle moyenne en degrés Celsius"
    )
    pesticides_tonnes = st.number_input(
        "🧪 Pesticides (tonnes)",
        min_value=0.0, max_value=400000.0, value=50000.0, step=1000.0,
        help="Quantité totale de pesticides utilisés (en tonnes)"
    )

    st.divider()
    st.caption("📊 Plages issues du dataset FAO (28 242 observations)")

# ──────────────────────────────────────────────────────────────────────────────
# Onglets principaux
# ──────────────────────────────────────────────────────────────────────────────

tab_predict, tab_recommend = st.tabs(["🎯  Prédiction par culture", "🏆  Recommandation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown("#### Sélectionnez une culture et obtenez son rendement prédit")

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        crops = get_available_crops() if api_ok else []
        if not crops:
            crops = ["cassava", "maize", "plantains and others", "potatoes",
                     "rice", "sorghum", "soybean", "sweet potatoes", "wheat", "yams"]

        crop = st.selectbox("🌱 Culture", options=crops,
                            format_func=lambda x: x.title())

        st.markdown("##### Contexte sélectionné")
        st.info(
            f"💧 **{rainfall_mm} mm/an**  |  "
            f"🌡️ **{avg_temp} °C**  |  "
            f"🧪 **{pesticides_tonnes:,.0f} t**"
        )

        predict_btn = st.button("🚀 Prédire le rendement", type="primary",
                                use_container_width=True, disabled=not api_ok)

    with col_result:
        if predict_btn:
            with st.spinner("Interrogation de l'API…"):
                result = call_predict(crop, rainfall_mm, avg_temp, pesticides_tonnes)

            if result:
                yield_val  = result.get("yield_t_ha", 0)
                r2         = result.get("model_r2")
                mae        = result.get("mae_t_ha")
                vs_hist    = result.get("vs_historique_pct")
                fiabilite  = result.get("fiabilite", "—")

                # Signe et couleur de l'écart historique
                delta_str = ""
                if vs_hist is not None:
                    sign = "+" if vs_hist >= 0 else ""
                    cls  = "delta-positive" if vs_hist >= 0 else "delta-negative"
                    delta_str = f'<span class="{cls}">{sign}{vs_hist:.1f}% vs historique</span>'

                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Rendement prédit — {crop.title()}</div>
                    <div class="value">{yield_val:.2f}</div>
                    <div class="unit">tonnes / hectare</div>
                    {f'<div style="margin-top:0.5rem">{delta_str}</div>' if delta_str else ''}
                    <div class="badge">{fiabilite}</div>
                </div>
                """, unsafe_allow_html=True)

                # Métriques secondaires
                m1, m2 = st.columns(2)
                m1.metric("R² modèle", f"{r2:.3f}" if r2 else "—")
                m2.metric("MAE", f"± {mae:.3f} t/ha" if mae else "—")

                # Jauge visuelle
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=yield_val,
                    number={"suffix": " t/ha", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 60], "tickcolor": "#74c69d"},
                        "bar": {"color": "#2d6a4f"},
                        "bgcolor": "#1b4332",
                        "steps": [
                            {"range": [0, 10],  "color": "#1b4332"},
                            {"range": [10, 25], "color": "#2d6a4f"},
                            {"range": [25, 60], "color": "#40916c"},
                        ],
                        "threshold": {
                            "line": {"color": "#95d5b2", "width": 3},
                            "thickness": 0.75,
                            "value": yield_val,
                        },
                    },
                    title={"text": f"<b>{crop.title()}</b>",
                           "font": {"color": "#d8f3dc", "size": 14}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#d8f3dc",
                    height=260,
                    margin=dict(t=40, b=10, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.markdown("""
            <div style="
                border: 2px dashed rgba(116,198,157,0.3);
                border-radius: 14px;
                padding: 3rem;
                text-align: center;
                color: #52b788;
                margin-top: 1rem;
            ">
                <div style="font-size:3rem">🌱</div>
                <div style="margin-top:0.8rem; font-size:0.95rem">
                    Sélectionnez une culture et cliquez sur<br><b>Prédire le rendement</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMANDATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_recommend:
    st.markdown("#### Pour vos conditions, quelles cultures maximisent le rendement ?")

    col_ctx, col_btn = st.columns([3, 1])
    with col_ctx:
        st.info(
            f"💧 **{rainfall_mm} mm/an**  |  "
            f"🌡️ **{avg_temp} °C**  |  "
            f"🧪 **{pesticides_tonnes:,.0f} t**  "
            f"— modifiables dans la barre latérale"
        )
    with col_btn:
        recommend_btn = st.button("🏆 Recommander", type="primary",
                                  use_container_width=True, disabled=not api_ok)

    if recommend_btn:
        with st.spinner("Calcul en cours pour toutes les cultures…"):
            result = call_recommend(rainfall_mm, avg_temp, pesticides_tonnes)

        if result:
            recs  = result.get("recommendations", [])
            best  = result.get("best_crop", "")
            df    = pd.DataFrame(recs)

            # ── Meilleure culture ──────────────────────────────────────────
            best_row = df.iloc[0]
            vs = best_row.get("vs_historique_pct")
            vs_str = ""
            if vs is not None:
                sign = "+" if vs >= 0 else ""
                cls  = "delta-positive" if vs >= 0 else "delta-negative"
                vs_str = f'<span class="{cls}">{sign}{vs:.1f}% vs historique</span>'

            st.markdown(f"""
            <div class="metric-card">
                <div class="label">🏆 Culture recommandée</div>
                <div class="value">{best_row['crop'].title()}</div>
                <div class="unit">{best_row['yield_t_ha']:.2f} t/ha prédit</div>
                {f'<div style="margin-top:0.5rem">{vs_str}</div>' if vs_str else ''}
                <div class="badge">{best_row.get('fiabilite','—')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ── Graphique à barres ─────────────────────────────────────────
            st.markdown("##### 📊 Rendements prédits — toutes cultures")

            df_sorted = df.sort_values("yield_t_ha", ascending=True)
            colors = [
                "#40916c" if c == best else "#2d6a4f"
                for c in df_sorted["crop"]
            ]

            fig_bar = go.Figure(go.Bar(
                x=df_sorted["yield_t_ha"],
                y=df_sorted["crop"].str.title(),
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f} t/ha" for v in df_sorted["yield_t_ha"]],
                textposition="outside",
                textfont=dict(color="#d8f3dc", size=11),
                hovertemplate="<b>%{y}</b><br>Rendement : %{x:.3f} t/ha<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,32,24,0.6)",
                font_color="#d8f3dc",
                height=420,
                margin=dict(t=10, b=10, l=10, r=80),
                xaxis=dict(
                    title="Rendement prédit (t/ha)",
                    gridcolor="rgba(116,198,157,0.1)",
                    color="#95d5b2",
                ),
                yaxis=dict(color="#95d5b2"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Tableau détaillé ───────────────────────────────────────────
            st.markdown("##### 📋 Tableau détaillé")

            df_display = df[["crop", "yield_t_ha", "mae_t_ha",
                              "vs_historique_pct", "model_r2", "fiabilite"]].copy()
            df_display.columns = [
                "Culture", "Rendement (t/ha)", "MAE (t/ha)",
                "Écart historique (%)", "R²", "Fiabilité"
            ]
            df_display["Culture"] = df_display["Culture"].str.title()
            df_display.insert(0, "Rang", range(1, len(df_display) + 1))

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rendement (t/ha)": st.column_config.NumberColumn(format="%.3f"),
                    "MAE (t/ha)":       st.column_config.NumberColumn(format="%.3f"),
                    "Écart historique (%)": st.column_config.NumberColumn(format="%.1f"),
                    "R²":               st.column_config.ProgressColumn(
                                            min_value=0, max_value=1, format="%.3f"
                                        ),
                },
            )
    else:
        st.markdown("""
        <div style="
            border: 2px dashed rgba(116,198,157,0.3);
            border-radius: 14px;
            padding: 3rem;
            text-align: center;
            color: #52b788;
            margin-top: 1rem;
        ">
            <div style="font-size:3rem">🏆</div>
            <div style="margin-top:0.8rem; font-size:0.95rem">
                Ajustez les conditions dans la barre latérale<br>
                puis cliquez sur <b>Recommander</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "🌾 Crop Yield Predictor — Frontend Streamlit · Backend FastAPI · "
    "Modèles GradientBoosting entraînés sur le dataset FAO (28 242 obs., 10 cultures)"
)
