# creditcard_psp/modeling/predict.py
from __future__ import annotations
from pathlib import Path
import json
import typer
import pandas as pd
import numpy as np
import joblib

# IMPORTANT: importing this ensures the saved pipeline step "feat"
# (FunctionTransformer(time_feat_transform)) can be resolved when loading.
from creditcard_psp.features import add_time_features  # noqa: F401

from creditcard_psp.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer(add_completion=False)
RNG = 42


# ---------- Small helpers ----------
def load_any(p: Path):
    """
    Load a dataframe or object from path with pragmatic format detection.
    Supports parquet, pickle/joblib, xlsx, csv. Falls back to joblib.load.
    """
    s = str(p).lower()
    if s.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    if s.endswith((".pkl", ".pickle", ".joblib")):
        try:
            return pd.read_pickle(p)
        except Exception:
            return joblib.load(p)
    if s.endswith(".xlsx"):
        return pd.read_excel(p)
    if s.endswith(".csv"):
        return pd.read_csv(p)
    # fallback
    try:
        return joblib.load(p)
    except Exception:
        return None


def load_best_model():
    """
    Load the calibrated best model:
    1) Try MODELS_DIR/meta.json -> best_model_{name}.joblib
    2) Fallback to dashboard_data/psp_router_model.joblib
    """
    meta_path = MODELS_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best_name = meta.get("best_model_name")
        if best_name:
            p = MODELS_DIR / f"best_model_{best_name}.joblib"
            if p.exists():
                return joblib.load(p), meta
    # fallback to dashboard model
    dash_model = Path("dashboard_data") / "psp_router_model.joblib"
    if dash_model.exists():
        return joblib.load(dash_model), {}
    raise FileNotFoundError(
        "No model found: neither MODELS_DIR/best_model_*.joblib nor dashboard_data/psp_router_model.joblib"
    )

def time_feat_transform(df):
    # erzeugt sin/cos aus 'tmsp' und entfernt Rohspalten
    return add_time_features(df, time_col="tmsp", drop_raw=True)

def load_fees(fee_path: Path) -> dict[str, dict[str, float]]:
    """
    Load PSP fees from an Excel file with columns:
    - PSP
    - fee_successful
    - fee_not_successful
    Returns dict[psp] -> {"fee_successful": float, "fee_not_successful": float}
    """
    fees = {}
    df = pd.read_excel(fee_path)
    df.columns = df.columns.str.strip()
    required = {"PSP", "fee_successful", "fee_not_successful"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fee file is missing columns: {missing}")
    df["PSP"] = df["PSP"].astype(str).str.strip()
    for _, r in df.iterrows():
        fees[r["PSP"]] = {
            "fee_successful": float(r["fee_successful"]) if pd.notna(r["fee_successful"]) else 0.0,
            "fee_not_successful": float(r["fee_not_successful"]) if pd.notna(r["fee_not_successful"]) else 0.0,
        }
    return fees


def predict_p_mat(calib_model, X_df: pd.DataFrame, psps: list[str], psp_col: str = "PSP") -> np.ndarray:
    """
    For each row in X_df and each PSP in `psps`, compute success probability.
    Returns an (n, k) matrix where n = len(X_df), k = len(psps).
    Implementation: duplicate the data per PSP by overwriting the PSP column,
    concatenate, score once, then reshape back to (n, k).
    """
    batches = [X_df.assign(**{psp_col: p}) for p in psps]
    X_all = pd.concat(batches, ignore_index=True)
    proba_all = calib_model.predict_proba(X_all)[:, 1]
    n = len(X_df)
    k = len(psps)
    return proba_all.reshape(k, n).T  # (n, k)


def coerce_dt(df: pd.DataFrame, col: str = "tmsp") -> pd.DataFrame:
    """
    Ensure `df[col]` is datetime (if present). Coerce on failure.
    """
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


# ---------- CLI ----------
@app.command()
def predict(
    in_path: Path = typer.Option(PROCESSED_DATA_DIR / "synthetic.parquet", help="New data (xlsx/csv/parquet/pkl)"),
    # out_path: Path = typer.Option(PROCESSED_DATA_DIR / "predictions.parquet", help="Output (csv/parquet/xlsx)"),
    out_path: Path = typer.Option(PROCESSED_DATA_DIR / "predictions.csv", help="Output (csv/parquet/xlsx)"),
    all_psps: bool = typer.Option(True, help="Score all PSPs and choose best per row"),
    alpha: float = typer.Option(0.05, min=0.0, help="Cost weight: score = p - alpha * expected_cost"),
    fee_xlsx: Path = typer.Option(RAW_DATA_DIR / "PSP_Servicegebuehren.xlsx", help="Fee table (Excel)"),
):
    """
    Score new transactions with the saved pipeline model.

    - With --all-psps (default):
        * Builds a P-matrix (n × |PSP|)
        * Loads fees from Excel
        * Computes expected costs and the score p - alpha * cost
        * Selects the best PSP per row
        * Writes per-PSP probabilities/costs as wide columns

    - Without --all-psps:
        * Expects a 'PSP' column and only scores that PSP per row
    """
    # 1) Load input
    df = load_any(in_path)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Could not load input or input is empty: {in_path}")
    df = coerce_dt(df, "tmsp")

    # 2) Load model & meta
    model, meta = load_best_model()
    meta_psps = meta.get("psps") if isinstance(meta, dict) else None

    # 3) Score
    if all_psps:
        # PSP universe: prefer meta.json; fallback to PSPs seen in the input
        psps = list(meta_psps) if meta_psps else sorted(
            map(str, df.get("PSP", pd.Series(dtype=str)).dropna().unique())
        )
        if not psps:
            raise RuntimeError("No PSP list found (neither in meta.json nor in the input data).")

        # P matrix
        P = predict_p_mat(model, df, psps, psp_col="PSP")

        # Load fees and compute expected costs & score
        fees = load_fees(fee_xlsx)
        fee_s = np.array([fees.get(p, {}).get("fee_successful", 0.0) for p in psps], dtype=float)
        fee_f = np.array([fees.get(p, {}).get("fee_not_successful", 0.0) for p in psps], dtype=float)

        exp_cost = P * fee_s + (1 - P) * fee_f
        score = P - alpha * exp_cost

        best_idx = np.argmax(score, axis=1)
        best_psp = np.array(psps)[best_idx]
        rows = np.arange(len(df))
        
        # Output (compact): only transaction_id + best_psp
        tx_ids = (
            df["transaction_id"].values
            if "transaction_id" in df.columns
            else np.arange(len(df))
        )
        out = pd.DataFrame({"transaction_id": tx_ids, "best_psp": best_psp})

    else:
        # Simple mode: keep only provided PSP per row
        if "PSP" not in df.columns:
            raise RuntimeError(
                "Column 'PSP' is missing. For single-PSP scoring keep 'PSP' in the input "
                "or use --all-psps."
            )

        tx_ids = (
            df["transaction_id"].values
            if "transaction_id" in df.columns
            else np.arange(len(df))
        )
        out = pd.DataFrame({
            "transaction_id": tx_ids,
            "best_psp": df["PSP"].astype(str).values,
        })

#         # Compact output: per-row selection + (optional) per-PSP details
#         out = pd.DataFrame({
#             "transaction_id": df.get("transaction_id", pd.Series(rows)).values,
#             "psp_selected": best_psp,
#             "p_selected": P[rows, best_idx],
#             "exp_cost_selected": exp_cost[rows, best_idx],
#             "score_selected": score[rows, best_idx],
#         })
#         # Add wide columns for diagnostics/BI
#         for j, p in enumerate(psps):
#             out[f"p__{p}"] = P[:, j]
#             out[f"cost__{p}"] = exp_cost[:, j]

#     else:
#         # Simple mode: only score the PSP provided in the 'PSP' column
#         if "PSP" not in df.columns:
#             raise RuntimeError(
#                 "Column 'PSP' is missing. For single-PSP scoring keep 'PSP' in the input "
#                 "or use --all-psps."
#             )
#         proba = model.predict_proba(df)[:, 1]
#         out = pd.DataFrame({
#             "transaction_id": df.get("transaction_id", pd.Series(range(len(df)))).values,
#             "PSP": df["PSP"].astype(str).values,
#             "p": proba,
#         })

    # 4) Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = str(out_path).lower()
    if s.endswith(".csv"):
        out.to_csv(out_path, index=False)
    elif s.endswith(".xlsx"):
        out.to_excel(out_path, index=False)
    else:
        out.to_parquet(out_path, index=False)

    print(f"OK — wrote {len(out)} rows to: {out_path}")


if __name__ == "__main__":
    app()
