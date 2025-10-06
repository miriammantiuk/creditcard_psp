from __future__ import annotations
import json
import shap
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, accuracy_score
)
from sklearn.model_selection import train_test_split

app = typer.Typer(add_completion=False)
RNG = 42

# --------- Paths (Config, with Fallback) ----------
try:
    from creditcard_psp.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
except Exception:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    MODELS_DIR = Path("artifacts/models")
    FIGURES_DIR = Path("artifacts/figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ Helper ============

def compute_feature_cols(df: pd.DataFrame, target_hint: str | None = None) -> Tuple[str, List[str]]:
    target = target_hint or ("transaction_success" if "transaction_success" in df.columns else "success")
    assert "PSP" in df.columns and target in df.columns, "DF muss 'PSP' und Target enthalten."
    drop = {
        target, "transaction_id", "tmsp", "tmsp_last",
        "fee_successful", "fee_not_successful",
        "success", "attempt_number",
        "weekday", "hour", "minute",  # delete because of cyclic
    }
    feats = [c for c in df.columns
             if (c not in drop)
             and (not c.startswith("fee_successful__"))
             and (not c.startswith("fee_not_successful__"))]
    if "PSP" not in feats:
        feats.append("PSP")
    return target, feats

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    cats = [c for c in ["PSP", "card", "country"] if c in feature_cols]
    nums = [c for c in feature_cols if c not in cats]
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", ohe)]), cats),
    ])
    return pre

def get_models() -> Dict[str, object]:
    return {
        "LOGREG": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
        "RF":     RandomForestClassifier(n_estimators=200, max_depth=8,
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=RNG),
        "XGB":    XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9,
                                colsample_bytree=0.9, tree_method="hist", eval_metric="logloss",
                                n_jobs=-1, random_state=RNG),
    }

def load_fees(fee_path: Path) -> Dict[str, Dict[str, float]]:
    fees = {}
    try:
        df = pd.read_excel(fee_path)
        df.columns = df.columns.str.strip()
        required = {"PSP", "fee_successful", "fee_not_successful"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in fees file: {missing}")
        df["PSP"] = df["PSP"].astype(str).str.strip()
        for _, r in df.iterrows():
            fees[r["PSP"]] = {
                "fee_successful": float(r["fee_successful"]) if pd.notna(r["fee_successful"]) else 0.0,
                "fee_not_successful": float(r["fee_not_successful"]) if pd.notna(r["fee_not_successful"]) else 0.0,
            }
    except Exception as e:
        print(f"[WARN] Gebühren-Datei nicht geladen: {e} – setze alle Gebühren = 0")
    return fees

def metrics(y_true, y_proba, thr=0.5) -> Dict[str, float]:
    y_proba = np.clip(y_proba, 1e-6, 1-1e-6)
    y_pred = (y_proba >= thr).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0,1]),
        "brier": brier_score_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
    }

def plot_calibration(y_true, y_proba, out_path: Path, title: str):
    y_proba = np.clip(y_proba, 1e-6, 1-1e-6)
    pt, pp = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(pp, pt, marker="o"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path); plt.close()

def predict_p_mat(calib_model, X_df: pd.DataFrame, psps: List[str], psp_col="PSP") -> np.ndarray:
    batches = [X_df.assign(**{psp_col: p}) for p in psps]
    X_all = pd.concat(batches, ignore_index=True)
    proba_all = calib_model.predict_proba(X_all)[:, 1]
    n = len(X_df); k = len(psps)
    return proba_all.reshape(k, n).T  # (n, k)

def shap_summary_plot(pipe_fit, X_df: pd.DataFrame, model_name: str, out_path: Path,
                      random_state: int = RNG):
    """Speichert einen SHAP Summary Plot für ein bereits FITTES Pipeline-Modell (ohne Calibrator)."""

    pre = pipe_fit.named_steps["preprocessor"]
    core = pipe_fit.named_steps["classifier"]

    X_trans = pre.transform(X_df)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f_{i}" for i in range(X_trans.shape[1])])

    try:
        if isinstance(core, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(core)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):  # Klassifikation -> Liste je Klasse
                sv = sv[1] if len(sv) > 1 else sv[0]
        elif isinstance(core, LogisticRegression):
            # LinearExplainer, Fallback auf allgemeines Explainer-API falls nötig
            try:
                explainer = shap.LinearExplainer(core, X_trans)
                sv = explainer.shap_values(X_trans)
            except Exception:
                explainer = shap.Explainer(lambda Z: core.predict_proba(Z)[:,1], X_trans)
                sv = explainer(X_trans).values
        else:
            # Selten benötigt in deinem Setup; KernelExplainer ist teuer -> skippen
            print(f"[INFO] Kein Tree/Linear-Modell für {model_name}; SHAP wird übersprungen.")
            return

        shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary – {model_name}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"SHAP gespeichert: {out_path}")
    except Exception as e:
        print(f"[WARN] SHAP für {model_name} übersprungen: {e}")


def generate_force_plot(pipe_fit, X_row_df): 
    """ Erstellt und zeigt einen SHAP Force Plot für eine einzelne Datenzeile (Transaktion). """
    try:
        shap.initjs()
        pre_fit = pipe_fit.named_steps["preprocessor"]
        core_model = pipe_fit.named_steps["classifier"]
        
        # Extrahiere die transformierten Feature-Namen direkt hier
        try:
            feature_names_out = pre_fit.get_feature_names_out()
        except AttributeError:
            # Fallback, falls die Methode in deiner scikit-learn Version nicht existiert
            # In diesem Fall wird der Plot weniger lesbare Namen haben, aber funktionieren
            n_features = pre_fit.transform(X_row_df).shape[1]
            feature_names_out = [f"feature_{i}" for i in range(n_features)]

        # Transformiere die einzelne Datenzeile
        X_transformed_df = pd.DataFrame(
            pre_fit.transform(X_row_df),
            columns=feature_names_out
        )
        
        # Erstelle den passenden SHAP-Explainer
        if hasattr(core_model, 'coef_'):
            X_train_transformed = pre_fit.transform(X_tr)
            explainer = shap.LinearExplainer(core_model, X_train_transformed)
        else:
            explainer = shap.TreeExplainer(core_model)
            
        # Berechne SHAP-Werte
        shap_values_row = explainer.shap_values(X_transformed_df)
        if isinstance(shap_values_row, list):
            shap_values_row = shap_values_row[1] if len(shap_values_row) > 1 else shap_values_row[0]
            
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        display(shap.force_plot(
            base_value,
            shap_values_row,
            X_transformed_df 
        ))
        
    except Exception as e:
        print(f"[WARN] SHAP Force Plot konnte nicht erstellt werden: {e}")

        
def compute_policy_for_alpha(alpha: float) -> pd.DataFrame:
    score = P - alpha * exp_cost
    best_idx = np.argmax(score, axis=1)
    tx_id = X_te.get("transaction_id", pd.Series(range(len(X_te)))).values
    return pd.DataFrame({
        "transaction_id": tx_id,
        "psp_selected": np.array(psps)[best_idx],
        "expected_success": P[np.arange(len(tx_id)), best_idx],
        "expected_cost_selected": exp_cost[np.arange(len(tx_id)), best_idx],
        "business_score": score[np.arange(len(tx_id)), best_idx],
        "alpha": alpha,
    })

def save_dashboard_artifacts(
    calibrated_model,
    raw_model,
    X_te_df: pd.DataFrame,
    psps_list: List[str],
    fees_dict: Dict[str, Dict[str, float]],
    feature_cols: List[str],
    df_full: pd.DataFrame,
    output_dir: Path,
):
    print(f"\n--- Speichere Artefakte für Dashboard im Ordner '{output_dir}' ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Modelle (joblib)
    joblib.dump(calibrated_model, output_dir / "psp_router_model.joblib")
    joblib.dump(raw_model,        output_dir / "model_raw.joblib")

    # 2) Test-Features (Parquet)
    X_te_df.to_parquet(output_dir / "X_te.parquet", index=True)

    # 3) PSP-Liste
    with open(output_dir / "psps.json", "w", encoding="utf-8") as f:
        json.dump(psps_list, f)

    # 4) P-Matrix + erwartete Kosten
    P = predict_p_mat(calibrated_model, X_te[feature_cols], psps, "PSP")
    np.save(output_dir / "P_matrix.npy", P)

    fee_s = np.array([fees_dict.get(p, {}).get("fee_successful", 0.0)     for p in psps_list], dtype=float)
    fee_f = np.array([fees_dict.get(p, {}).get("fee_not_successful", 0.0) for p in psps_list], dtype=float)
    exp_cost = P * fee_s + (1 - P) * fee_f
    np.save(output_dir / "exp_cost_matrix.npy", exp_cost)

    # 5) Original-DF (Parquet)
    df_full.to_parquet(output_dir / "df.parquet", index=True)

    print("Dashboard-Artefakte gespeichert.")

__all__ = [
    "compute_feature_cols", "make_preprocessor", "get_models",
    "load_fees", "metrics", "plot_calibration", "predict_p_mat",
    "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "MODELS_DIR", "FIGURES_DIR", "RNG", "shap_summary_plot"
]



# ===================== Training =====================

@app.command()
def train(
    df_path: Path = typer.Option(PROCESSED_DATA_DIR / "df.pkl", help="Pfad zu df.parquet"),
    fee_xlsx: Path = typer.Option(RAW_DATA_DIR / "PSP_Servicegebuehren.xlsx", help="Gebühren XLSX"),
    calib: str = typer.Option("sigmoid", help="Kalibrierungsmethode: sigmoid|isotonic"),
    cv: int = typer.Option(3, help="CV-Folds für Kalibrierung"),
    test_size: float = typer.Option(0.2, min=0.05, max=0.4),
    selection_metric: str = typer.Option("log_loss", help="log_loss|brier|avg_precision|auc_roc"),
    models: List[str] = typer.Option(["LOGREG","RF","XGB"], help="welche Modelle trainieren"),
    dump_dashboard_artifacts: bool = typer.Option(True, help="Artefakte für Dashboard zum besten Modell speichern"),
    do_shap: bool = typer.Option(True, help="SHAP-Summary je Modell speichern"),
):
    """Trainiert, evaluiert und speichert die PSP-Routing-Modelle."""
    print("--- Starte Trainings-Pipeline ---")
    
    # ---- Load df (Parquet bevorzugt, mit Fallback) ----
    try:
        if str(df_path).lower().endswith(".parquet"):
            df = pd.read_parquet(df_path)
        elif str(df_path).lower().endswith((".pkl", ".pickle", ".joblib")):
            try:
                df = pd.read_pickle(df_path)
            except Exception:
                df = joblib.load(df_path)
        else:
            # heuristische Versuche
            try:
                df = pd.read_parquet(df_path)
            except Exception:
                try:
                    df = pd.read_pickle(df_path)
                except Exception:
                    df = joblib.load(df_path)
    except Exception as e:
        raise RuntimeError(f"DF konnte nicht geladen werden ({df_path}): {e}")

    target, feature_cols = compute_feature_cols(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feature_cols], df[target].astype(int),
        test_size=test_size, random_state=RNG, stratify=df[target].astype(int)
    )
    fees = load_fees(fee_xlsx)
    psps = sorted(list(X_tr["PSP"].dropna().unique()))

    pre = make_preprocessor(feature_cols)
    zoo = get_models()

    # --- Train & Evaluate Loop ---
    results_data = []
    trained_models = {}
    for name in models:
        print(f"\n--- Training Modell: {name} ---")
        base = zoo[name]
        
        # Kalibriertes Modell fitten
        clf = CalibratedClassifierCV(
            estimator=Pipeline([("preprocessor", pre), ("classifier", base)]),
            cv=cv, method=calib
        ).fit(X_tr, y_tr)
        
        # Metriken berechnen
        y_proba = clf.predict_proba(X_te)[:, 1]
        m = metrics(y_te, y_proba)
        results_data.append({"model": name, **m})
        plot_calibration(y_te, y_proba, FIGURES_DIR / f"calibration_{name}.png", f"Calibration - {name}")
        
        # Unkalibriertes Modell für SHAP fitten
        pipe_raw = Pipeline([("preprocessor", pre), ("classifier", base)]).fit(X_tr, y_tr)
        trained_models[name] = {"calibrated": clf, "raw": pipe_raw}
        
        if do_shap:
            shap_summary_plot(pipe_raw, X_te, name, FIGURES_DIR / f"shap_{name}.png")

    # --- Select Best Model & Save ---
# --- Select Best Model & Save ---
    summary_df = pd.DataFrame(results_data).sort_values(by=selection_metric)
    best_name = summary_df.iloc[0]["model"]
    best_model_calibrated = trained_models[best_name]["calibrated"]
    best_model_raw = trained_models[best_name]["raw"]

    print(f"\n--- Bestes Modell (nach {selection_metric}): {best_name} ---")
    print(summary_df)

    # Speichere allgemeine Artefakte
    summary_df.to_csv(MODELS_DIR / "summary_metrics.csv", index=False)
    joblib.dump(best_model_calibrated, MODELS_DIR / f"best_model_{best_name}.joblib")
    with open(MODELS_DIR / "meta.json","w", encoding="utf-8") as f:
        json.dump({"best_model_name": best_name, "feature_cols": feature_cols, "psps": psps}, f, indent=2)

    # --- Speichere Artefakte für Dashboard ---
    if dump_dashboard_artifacts:
        print("\n--- Speichere Artefakte für Dashboard ---")
        DASHBOARD_DATA_DIR = Path("dashboard_data")
        DASHBOARD_DATA_DIR.mkdir(exist_ok=True)

        # Modelle: joblib
        joblib.dump(best_model_calibrated, DASHBOARD_DATA_DIR / "psp_router_model.joblib")
        joblib.dump(best_model_raw,        DASHBOARD_DATA_DIR / "model_raw.joblib")

        # Original-DF: Parquet
        df.to_parquet(DASHBOARD_DATA_DIR / "df.parquet", index=True)

        # Test-Features: Parquet
        X_te.to_parquet(DASHBOARD_DATA_DIR / "X_te.parquet", index=True)

        # PSPs
        with open(DASHBOARD_DATA_DIR / "psps.json", 'w', encoding="utf-8") as f:
            json.dump(psps, f)

        # Matrizen
        P = predict_p_mat(best_model_calibrated, X_te, psps, "PSP")
        np.save(DASHBOARD_DATA_DIR / "P_matrix.npy", P)

        fee_s = np.array([fees.get(p, {}).get("fee_successful", 0.0) for p in psps], dtype=float)
        fee_f = np.array([fees.get(p, {}).get("fee_not_successful", 0.0) for p in psps], dtype=float)
        exp_cost = P * fee_s + (1 - P) * fee_f
        np.save(DASHBOARD_DATA_DIR / "exp_cost_matrix.npy", exp_cost)

        print("Alle Dashboard-Artefakte gespeichert.")

if __name__ == "__main__":
    app()