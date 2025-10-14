from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)

from creditcard_psp.config import PROCESSED_DATA_DIR

# app = typer.Typer()

def add_time_features(
    df: pd.DataFrame,
    time_col: str = "tmsp",
    drop_raw: bool = True,
) -> pd.DataFrame:
    """
    Extract weekday, hour, minute incl. cyclic coding and adds columns results as columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transaction attempts.
    time_col : str, default 'tmsp'
        Name of the timestamp column to to extract weekday, hour, minute.

    Returns
    -------
    pd.DataFrame
        The original DataFrame enriched with:
        - 'weekday'
        - 'hour'
        - 'minute'
        - 'weekday_sin'
        - 'weekday_cos'
        - 'hour_sin'
        - 'hour_cos'
        - 'minute_sin'
        - 'minute_cos'
    """  

    out = df.copy()

    if time_col in out.columns:
        dt = pd.to_datetime(out[time_col], errors="coerce")
        hour    = dt.dt.hour.fillna(0).astype(int) % 24
        minute  = dt.dt.minute.fillna(0).astype(int) % 60
        weekday = dt.dt.weekday.fillna(0).astype(int) % 7
    else:
        need = {"hour", "minute", "weekday"}
        if not need.issubset(out.columns):
            raise ValueError(f"Zeitspalten fehlen: liefere '{time_col}' oder hour/minute/weekday.")
        hour    = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int) % 24
        minute  = pd.to_numeric(out["minute"], errors="coerce").fillna(0).astype(int) % 60
        weekday = pd.to_numeric(out["weekday"], errors="coerce").fillna(0).astype(int) % 7

    two_pi = 2.0 * np.pi
    out["weekday_sin"] = np.sin(two_pi * weekday / 7.0)
    out["weekday_cos"] = np.cos(two_pi * weekday / 7.0)
    out["hour_sin"]    = np.sin(two_pi * hour    / 24.0)
    out["hour_cos"]    = np.cos(two_pi * hour    / 24.0)
    out["minute_sin"]  = np.sin(two_pi * minute  / 60.0)
    out["minute_cos"]  = np.cos(two_pi * minute  / 60.0)

    out["weekday"] = weekday
    out["hour"]    = hour
    out["minute"]  = minute

    if drop_raw:
        out.drop(columns=[time_col, "weekday", "hour", "minute"], errors="ignore", inplace=True)

    return out

def compute_feature_cols(
    df: pd.DataFrame,
    target_hint: Optional[str] = None,
    keep_time_cols: bool = True,
) -> tuple[str, List[str]]:
    """
    Ermittelt Target- und Feature-Spalten.
    Hinweis: Wenn `keep_time_cols=True`, bleiben 'tmsp'/'hour'/'minute'/'weekday' im Feature-Set,
    damit ein Pipeline-Step (FunctionTransformer) daraus die zyklischen Features erzeugen kann.
    """
    target = target_hint
    target = target_hint or ("transaction_success" if "transaction_success" in df.columns else "success")
    assert "PSP" in df.columns and target in df.columns, "DF muss 'PSP' und Target enthalten."

    drop = {
        target,
        "transaction_id",
        "fee_successful",
        "fee_not_successful"
    }
    if not keep_time_cols:
        drop |= {"tmsp", "weekday", "hour", "minute"}

    feats = [
        c for c in df.columns
        if (c not in drop)
        and (not c.startswith("fee_successful__"))
        and (not c.startswith("fee_not_successful__"))
    ]
    if "PSP" not in feats:
        feats.append("PSP")
    return target, feats


# @app.command()
# def main(
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     # -----------------------------------------
# ):
#     # Load dataset
#     df = pd.read_csv(input_path)

#     # Build and apply pipeline
#     preproc = make_preprocessor(
#         categorical_cols=['PSP','card','country'],
#         amount_col='amount',
#         time_col='tmsp'
#     )
#     df_transformed = preproc.fit_transform(df)

#     # Convert back to DataFrame with feature names
#     columns = preproc.named_steps['preprocessor'].get_feature_names_out()
#     df_out = pd.DataFrame(df_transformed, columns=columns, index=df.index)

#     # Save
#     df_out.to_csv(output_path, index=False)
#     logger.success(f"Features saved to {output_path}")

# if __name__ == "__main__":
#     app()
