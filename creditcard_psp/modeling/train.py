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
from sklearn.compose import ColumnTransformer,make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from creditcard_psp.features import add_time_features, compute_feature_cols
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

# Paths (Config, with Fallback)
try:
    from creditcard_psp.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, DASHBOARD_DIR
except Exception:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    MODELS_DIR = Path("models")
    FIGURES_DIR = Path("figures")
    DASHBOARD_DIR = Path("dashboard_data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# ============ Helper ============

def make_preprocessor_selector() -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            # All numeric columns (including sin/cos) -> Imputer + StandardScaler
            ("num",
             Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
             selector(dtype_include=np.number)),
            # All string/category columns (PSP, card, country, etc.) -> Imputer + OHE
            ("cat",
             Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                       ("ohe", ohe)]),
             selector(dtype_include=["object", "category"])),
        ],
        remainder="drop",
    )
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

# Pretty mapping for feature names
def _to_display(n: str) -> str:
    # remove transformer prefixes
    if "__" in n:
        n = n.split("__", 1)[1]
    # OHE nicer: "PSP_Simplecard" -> "PSP=Simplecard"
    if any(n.startswith(p) for p in ("PSP_", "card_", "country_")):
        var, val = n.split("_", 1)
        n = f"{var}={val}"
    return n

def shap_summary_plot(pipe_fit, X_df, model_name, out_path,max_bg=1000, random_state=42):
    """
    Save a SHAP summary plot for a fitted Pipeline (feat + preprocessor + classifier).
    Runs the feature step (time features) before the preprocessor to avoid missing columns.
    Picks a model-specific explainer (Tree/Linear), otherwise falls back to PermutationExplainer.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import shap
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = tuple()

    if not isinstance(pipe_fit, Pipeline):
        raise ValueError("pipe_fit must be a sklearn Pipeline")

    # run feat + preprocessor exactly as in training (everything except the final classifier)
    preproc_pipe = pipe_fit[:-1]
    X_trans = preproc_pipe.transform(X_df)
    if hasattr(X_trans, "toarray"):  # densify sparse matrices for SHAP if needed
        X_trans = X_trans.toarray()

    pre = pipe_fit.named_steps["preprocessor"]
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f_{i}" for i in range(X_trans.shape[1])])

    feature_names_disp = [_to_display(str(n)) for n in feature_names]

    core = pipe_fit.named_steps["classifier"]
    # unwrap very light wrappers if present (optional)
    if isinstance(core, CalibratedClassifierCV):
        core = core.estimator  

    # Background-Sample for SHAP
    n = X_trans.shape[0]
    rs = np.random.RandomState(random_state)
    bg_idx = rs.choice(n, size=min(n, max_bg), replace=False)
    background = X_trans[bg_idx]

    # Calculate SHAP
    def _to_2d(vals):
        vals = np.asarray(vals)
        if vals.ndim == 3:      # (n, d, C)
            vals = vals[:, :, 1]
        assert vals.ndim == 2   # (n, d)
        return vals

    try:
        if isinstance(core, (RandomForestClassifier, GradientBoostingClassifier)) or (XGBClassifier and isinstance(core, XGBClassifier)):
            expl = shap.TreeExplainer(
                core, data=background, model_output="probability",
                feature_perturbation="interventional"
            )
            sv = expl(X_trans, check_additivity=False)
            vals = _to_2d(getattr(sv, "values", sv))
        elif isinstance(core, LogisticRegression):
            # Linear
            try:
                expl = shap.LinearExplainer(core, background)
                vals = expl(X_trans).values
            except Exception:
                expl = shap.Explainer(lambda Z: core.predict_proba(Z)[:, 1], background)
                vals = expl(X_trans).values
            vals = _to_2d(vals)
        else:
            score_fn = (lambda Z: core.predict_proba(Z)[:, 1]) if hasattr(core, "predict_proba") else core.decision_function
            expl = shap.PermutationExplainer(score_fn, background)
            vals = _to_2d(expl(X_trans).values)
    except Exception:
        score_fn = (lambda Z: core.predict_proba(Z)[:, 1]) if hasattr(core, "predict_proba") else core.decision_function
        vals = _to_2d(shap.Explainer(score_fn, background)(X_trans).values)

    # Plot & save
    plt.figure()
    shap.summary_plot(vals, X_trans, feature_names=feature_names_disp, plot_type="dot", show=False)
    plt.title(f"SHAP Summary – {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"SHAP gespeichert: {out_path}")


def save_dashboard_artifacts(
    calibrated_model,
    raw_model,
    X_te_df: pd.DataFrame,
    psps_list: List[str],
    fees_dict: Dict[str, Dict[str, float]],
    feature_cols: List[str],
    df_full: pd.DataFrame,
    output_dir: Path,
    feature_names_out: List[str] | None = None,
    feature_names_display: List[str] | None = None,
    mapping: Dict[str, str] | None = None,
):
    print(f"\n--- Speichere Artefakte für Dashboard im Ordner '{output_dir}' ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Models (joblib)
    joblib.dump(calibrated_model, output_dir / "psp_router_model.joblib")
    joblib.dump(raw_model,        output_dir / "model_raw.joblib")

    # 2) Test-Features (Parquet)
    X_te_df.to_parquet(output_dir / "X_te.parquet", index=True)

    # 3) PSP-List
    with open(output_dir / "psps.json", "w", encoding="utf-8") as f:
        json.dump(psps_list, f)

    # 4) P-Matrix + expexted Kosten
    P = predict_p_mat(calibrated_model, X_te_df[feature_cols], psps_list, "PSP")
    np.save(output_dir / "P_matrix.npy", P)

    fee_s = np.array([fees_dict.get(p, {}).get("fee_successful", 0.0)     for p in psps_list], dtype=float)
    fee_f = np.array([fees_dict.get(p, {}).get("fee_not_successful", 0.0) for p in psps_list], dtype=float)
    exp_cost = P * fee_s + (1 - P) * fee_f
    np.save(output_dir / "exp_cost_matrix.npy", exp_cost)

    # 5) Original-DF (Parquet)
    df_full.to_parquet(output_dir / "df.parquet", index=True)

    # 6) Meta (Mapping) – for Dashboard
    meta = {
        "feature_names_out": list(map(str, feature_names_out or [])),
        "feature_names_display": feature_names_display or [],
        "mapping": mapping or {},
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Dashboard-Artefakte gespeichert.")


# ===================== Training =====================

@app.command()
def train(
    df_path: Path = typer.Option(PROCESSED_DATA_DIR / "df.pkl", help="Pfad zu df.pkl"),
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

    # Load df 
    try:
        if str(df_path).lower().endswith(".parquet"):
            df = pd.read_parquet(df_path)
        elif str(df_path).lower().endswith((".pkl", ".pickle", ".joblib")):
            try:
                df = pd.read_pickle(df_path)
            except Exception:
                df = joblib.load(df_path)
        else:
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

    feat_step = ("feat", FunctionTransformer(add_time_features, validate=False))
    pre = make_preprocessor_selector()
    zoo = get_models()

    # Train & Evaluate Loop
    results_data = []
    trained_models = {}
    for name in models:
        print(f"\n--- Training Modell: {name} ---")
        base = zoo[name]

        # Calibrierted model
        pipe = Pipeline([feat_step, ("preprocessor", pre), ("classifier", base)])
        clf = CalibratedClassifierCV(estimator=pipe, cv=cv, method=calib).fit(X_tr, y_tr)

        # Metrics
        y_proba = clf.predict_proba(X_te)[:, 1]
        m = metrics(y_te, y_proba)
        results_data.append({"model": name, **m})
        plot_calibration(y_te, y_proba, FIGURES_DIR / f"calibration_{name}.png", f"Calibration - {name}")

        # Uncalibrierted model (for SHAP)
        pipe_raw = Pipeline([feat_step, ("preprocessor", pre), ("classifier", base)]).fit(X_tr, y_tr)
        trained_models[name] = {"calibrated": clf, "raw": pipe_raw}

        if do_shap:
            shap_summary_plot(pipe_raw, X_te, name, FIGURES_DIR / f"shap_{name}.png")

    # Choose best model
    summary_df = pd.DataFrame(results_data).sort_values(by=selection_metric)
    best_name = summary_df.iloc[0]["model"]
    best_model_calibrated = trained_models[best_name]["calibrated"]
    best_model_raw = trained_models[best_name]["raw"]

    print(f"\n--- Bestes Modell (nach {selection_metric}): {best_name} ---")
    print(summary_df)

    # Generate Mapping/Display-Names
    pre_best = best_model_raw.named_steps["preprocessor"]
    try:
        feat_out = pre_best.get_feature_names_out()
    except Exception:
        # Fallback if get_feature_names_out is not available
        X_tmp = pre_best.transform(X_tr.iloc[:1])
        feat_out = np.array([f"f_{i}" for i in range(X_tmp.shape[1])])

    feat_out = list(map(str, feat_out))
    display_names = [_to_display(n) for n in feat_out]
    mapping = dict(zip(feat_out, display_names))

    # Save Artefacts
    summary_df.to_csv(MODELS_DIR / "summary_metrics.csv", index=False)
    joblib.dump(best_model_calibrated, MODELS_DIR / f"best_model_{best_name}.joblib")
    with open(MODELS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model_name": best_name,
            "feature_cols": feature_cols,
            "psps": psps,
            "feature_names_out": feat_out,
            "feature_names_display": display_names,
            "mapping": mapping
        }, f, indent=2)

    # Dashboard-Artefacts
    if dump_dashboard_artifacts:
        print("\n--- Speichere Artefakte für Dashboard ---")
#         DASHBOARD_DATA_DIR = Path("dashboard_data")
#         DASHBOARD_DATA_DIR.mkdir(exist_ok=True)

        # Models: joblib
        joblib.dump(best_model_calibrated, DASHBOARD_DIR / "psp_router_model.joblib")
        joblib.dump(best_model_raw,        DASHBOARD_DIR / "model_raw.joblib")

        # Original-DF: Parquet
        df.to_parquet(DASHBOARD_DIR / "df.parquet", index=True)

        # Test-Features: Parquet
        X_te.to_parquet(DASHBOARD_DIR / "X_te.parquet", index=True)

        # PSPs
        with open(DASHBOARD_DIR / "psps.json", 'w', encoding="utf-8") as f:
            json.dump(psps, f)

        # Matrices
        P = predict_p_mat(best_model_calibrated, X_te[feature_cols], psps, "PSP")
        np.save(DASHBOARD_DIR / "P_matrix.npy", P)

        fee_s = np.array([fees.get(p, {}).get("fee_successful", 0.0) for p in psps], dtype=float)
        fee_f = np.array([fees.get(p, {}).get("fee_not_successful", 0.0) for p in psps], dtype=float)
        exp_cost = P * fee_s + (1 - P) * fee_f
        np.save(DASHBOARD_DIR / "exp_cost_matrix.npy", exp_cost)

        # Meta (mapping) for the dashboard
        with open(DASHBOARD_DIR / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "feature_names_out": feat_out,
                "feature_names_display": display_names,
                "mapping": mapping
            }, f, indent=2)

        print("Alle Dashboard-Artefakte gespeichert.")

if __name__ == "__main__":
    app()
