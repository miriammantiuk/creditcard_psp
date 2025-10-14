import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import __main__ 

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from creditcard_psp.features import add_time_features

# Legacy-Aliases für Unpickling alter Modelle
__main__.add_time_features = add_time_features
__main__.time_feat_transform = add_time_features  # falls das alte Artefakt diesen Namen erwartet

from creditcard_psp.config import DASHBOARD_DIR, RAW_DATA_DIR
 

# -------- Laden (einfach & robust) --------

@st.cache_resource
def load_model(p: Path):
    return joblib.load(p)

@st.cache_data
def load_any(p: Path):
    s = str(p)
    if s.endswith(".parquet"):
        return pd.read_parquet(p)
    if s.endswith(".npy"):
        return np.load(p)
    if s.endswith(".json"):
        import json
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    if s.endswith(".pkl"):
        return pd.read_pickle(p)
    try:
        return joblib.load(p)
    except Exception:
        return None

# hübsche Namen (Fallback, wenn kein meta.json)
def _to_display(n: str) -> str:
    n = str(n)
    if "__" in n:                      # 'num__amount' -> 'amount'
        n = n.split("__", 1)[1]
    if any(n.startswith(p) for p in ("PSP_", "card_", "country_")):
        parts = n.split("_", 1)        # 'PSP_Simplecard' -> 'PSP=Simplecard'
        if len(parts) == 2:
            n = f"{parts[0]}={parts[1]}"
    return n

# Artefakte laden
model_raw = load_model(DASHBOARD_DIR / "model_raw.joblib")
X_te      = load_any(DASHBOARD_DIR / "X_te.parquet")
P         = load_any(DASHBOARD_DIR / "P_matrix.npy")
exp_cost  = load_any(DASHBOARD_DIR / "exp_cost_matrix.npy")
psps      = load_any(DASHBOARD_DIR / "psps.json")
meta      = load_any(DASHBOARD_DIR / "meta.json") or {}

# Nach dem Laden:
if any(x is None for x in [model_raw, X_te, P, exp_cost, psps]):
    st.error("Artefakte unvollständig: erwartet model_raw.joblib, X_te.parquet, P_matrix.npy, exp_cost_matrix.npy, psps.json.")
    st.stop()

# Shape-Check
if len(X_te) != np.asarray(P).shape[0] or np.asarray(P).shape != np.asarray(exp_cost).shape:
    st.error(f"Shape-Mismatch: len(X_te)={len(X_te)}, P={np.asarray(P).shape}, exp_cost={np.asarray(exp_cost).shape}")
    st.stop()

# -------- Preprocessor / Classifier --------
if isinstance(model_raw, Pipeline) and "classifier" in model_raw.named_steps:
    # Alles vor dem Classifier (z. B. feat -> preprocessor -> selector ...)
    preproc_pipe = model_raw[:-1]
    classifier   = model_raw.named_steps["classifier"]
    # Für Feature-Namen (falls vorhanden) nutzen wir den ColumnTransformer in der Pipeline
    preprocessor = model_raw.named_steps.get("preprocessor", None)
else:
    # Fallback: alte Artefakte ohne Pipeline-Schneiden
    steps        = getattr(model_raw, "named_steps", {})
    preproc_pipe = None
    preprocessor = steps.get("preprocessor", None)
    classifier   = steps.get("classifier") or steps.get("clf") or model_raw

def X(df: pd.DataFrame):
    """Transformiere Roh-DF so, wie es der Classifier erwartet (alle Steps außer Classifier)."""
    if preproc_pipe is not None:
        return preproc_pipe.transform(df)
    # Fallback: explizit feat -> preprocessor, wenn vorhanden
    s = df.copy()
    feat = steps.get("feat")
    if feat is not None:
        s = feat.transform(s)
    if preprocessor is not None:
        s = preprocessor.transform(s)
    return s    
    
# # -------- Preprocessor / Classifier --------
# steps = getattr(model_raw, "named_steps", {})
# preprocessor = steps.get("preprocessor", None)
# classifier   = steps.get("classifier") or steps.get("clf") or (model_raw if not steps else list(steps.values())[-1])

# def X(df):
#     return preprocessor.transform(df) if preprocessor is not None else df

# Feature-Namen (aus meta.json, sonst hübsch ableiten)
feature_names_display = meta.get("feature_names_display")
if not feature_names_display:
    try:
        raw_names = preprocessor.get_feature_names_out() if preprocessor is not None else list(X_te.columns)
    except Exception:
        raw_names = list(X_te.columns)
    feature_names_display = [_to_display(n) for n in raw_names] if raw_names is not None else None

# -------- SHAP (ein Explainer für alles) --------
@st.cache_resource
def make_explainer(_clf, _bg):
    try:
        return shap.Explainer(_clf, _bg)
    except Exception:
        return shap.TreeExplainer(_clf)

bg = X(X_te.sample(min(100, len(X_te)), random_state=42))
explainer = make_explainer(classifier, bg)

# -------- UI --------
st.set_page_config(layout="wide", page_title="PSP-Routing-Cockpit")

# Sidebar
st.sidebar.header("Strategie-Simulator")
alpha = st.sidebar.slider(
    "Strategie: von 'Maximale Sicherheit' bis 'Aggressiv Sparen'",
    min_value=0.0, max_value=0.2, value=0.05, step=0.005, format="%.3f"
)
st.sidebar.markdown(
    """
    **Erläuterung:**
    - **Links (`alpha`=0):** Kosten werden ignoriert. Es wird immer der PSP mit der höchsten Erfolgswahrscheinlichkeit gewählt.
    - **Rechts (`alpha`>0):** Kosten werden stärker gewichtet. Das Modell sucht einen Kompromiss aus hoher Erfolgswahrscheinlichkeit und niedrigen Gebühren.
    """
)

# Performance
st.header(f"Performance-Analyse (alpha = {alpha:.3f})")
score = P - alpha * exp_cost
best_idx = np.argmax(score, axis=1)
n = len(X_te)

avg_success_rate = P[np.arange(n), best_idx].mean()
avg_cost = exp_cost[np.arange(n), best_idx].mean()
psp_selected = np.array(psps)[best_idx]
psp_distribution = pd.Series(psp_selected).value_counts(normalize=True)

c1, c2 = st.columns(2)
c1.metric("Erwartete Erfolgsrate", f"{avg_success_rate:.2%}")
c2.metric("Erwartete Kosten/Transaktion", f"{avg_cost:.2f} CHF")

st.subheader("Verteilung der gewählten PSPs")
st.bar_chart(psp_distribution)

# Gebühren-Tabelle aus XLSX
st.subheader("Gebühren je PSP")
fees_df = pd.read_excel(RAW_DATA_DIR / "PSP_Servicegebuehren.xlsx")
fees_show = fees_df.rename(columns={
        "fee_successful":     "✓",
        "fee_not_successful": "✗"
})
st.dataframe(
    fees_show.sort_values("PSP").reset_index(drop=True),
    hide_index=True,
    column_config={
            "✓": st.column_config.NumberColumn(format="%.2f EUR"),
            "✗": st.column_config.NumberColumn(format="%.2f EUR"),
        }
    )

# Globale Modell-Interpretation
st.header("Globale Modell-Interpretation")
# with st.expander("SHAP Summary Plot anzeigen (Übersicht der Feature-Wichtigkeit)"):
#     sample = X_te.sample(min(500, len(X_te)), random_state=42)
#     Xs = X(sample)
#     vals = explainer(Xs)
#     shap.summary_plot(vals.values, Xs, feature_names=feature_names_display, show=False, plot_size=(6, 4))
#     st.pyplot(plt.gcf())

with st.expander("SHAP Summary Plot anzeigen (Übersicht der Feature-Wichtigkeit)"):
    sample = X_te.sample(min(500, len(X_te)), random_state=42)
    Xs = X(sample)
    vals = explainer(Xs)

    # <<< kleinere Schrift über rc_context >>>
    with mpl.rc_context({
        "font.size": 9,        # Basis-Schrift
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,  # Achsenticks X
        "ytick.labelsize": 8,  # Feature-Namen links
        "legend.fontsize": 8,
    }):
        plt.figure(figsize=(6, 4))
        shap.summary_plot(
            vals.values, Xs,
            feature_names=feature_names_display,
            show=False,
            max_display=15,
            plot_size=(6, 4),
        )
        fig = plt.gcf()
        # Sicherheitshalber auch alle Achsenticks kleiner drehen (inkl. Colorbar)
        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=8)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    st.caption("Links = senkt Erfolg, rechts = erhöht Erfolg. Farbe: Feature-Wert (blau niedrig, rot hoch).")

# Drill-down
df_test_original = load_any(DASHBOARD_DIR / "df.parquet")

st.header("Analyse einzelner Transaktionen")
transaction_index = st.selectbox("Wähle eine Transaktion (Index aus X_te):", options=X_te.index)

try:
    row_pos = X_te.index.get_loc(transaction_index)
except Exception:
    row_pos = int(np.where(X_te.index == transaction_index)[0][0])

st.subheader("Originale Transaktionsdaten")
try:
    st.dataframe(df_test_original.loc[[transaction_index]])
except KeyError:
    st.dataframe(df_test_original.iloc[[row_pos]])

row_p    = np.asarray(P)[row_pos, :]
row_cost = np.asarray(exp_cost)[row_pos, :]
row_score = row_p - alpha * row_cost

selected_idx = int(np.argmax(row_score))
selected_psp = psps[selected_idx]

psp_table = pd.DataFrame({
    "PSP": psps,
    "Erfolgswahrscheinlichkeit (%)": row_p * 100,
    "Erwartete Kosten": row_cost,
    "Score (p - alpha*Erwartete Kosten)": row_score,
})
psp_table["Gewählt (bei aktuellem α)"] = np.arange(len(psps)) == selected_idx

st.dataframe(
    psp_table.sort_values("Score (p - alpha*Erwartete Kosten)", ascending=False).reset_index(drop=True),
    hide_index=True,
    column_config={
        "Erfolgswahrscheinlichkeit (%)": st.column_config.NumberColumn(format="%.2f"),
        "Erwartete Kosten": st.column_config.NumberColumn(format="%.2f EUR"),
        "Score (p - alpha*Erwartete Kosten)": st.column_config.NumberColumn(format="%.3f"),
        "Gewählt (bei aktuellem α)": st.column_config.CheckboxColumn()
    }
)

st.info(
    f"**Gewählter PSP bei α = {alpha:.3f}:** {selected_psp}  "
    f"(p = {row_p[selected_idx]:.2%}, Kosten = {row_cost[selected_idx]:.2f} CHF, "
    f"Score = {row_score[selected_idx]:.3f})"
)
