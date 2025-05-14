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

app = typer.Typer()

def add_time_features(
    df: pd.DataFrame,
    time_col: str = 'tmsp'
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

    # Extract weekday, hour, minute
    df['weekday'] = df['tmsp'].dt.weekday
    df['hour']    = df['tmsp'].dt.hour
    df['minute']  = df['tmsp'].dt.minute

    # Cyclic Coding
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour']    / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour']    / 24)
    df['minute_sin']  = np.sin(2 * np.pi * df['minute']    / 60)
    df['minute_cos']  = np.cos(2 * np.pi * df['minute']    / 60)

    return df

def encode_and_scale(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    amount_col: str = 'amount',
    drop_first: bool = True,
    handle_unknown: str = 'ignore'
) -> pd.DataFrame:
    """
    1) One‑Hot‑Encode the given categorical columns with sklearn.OneHotEncoder.
    2) Add a log‑transformed version of `amount_col` as 'amount_log'.
    3) Standard‑scale 'amount_log' into 'amount_scaled'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_cols : list of str
        Columns to be one‑hot encoded.
    amount_col : str, default 'amount'
        Name of the numeric column to log‑transform and scale.
    drop_first : bool, default True
        If True, drop the first dummy level per categorical to avoid multicollinearity.
    handle_unknown : str, default 'ignore'
        How to handle unknown categories during transform.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with:
        - one‑hot dummies for each category (drop_first if requested)
        - 'amount_log'
        - 'amount_scaled'
    """
    # Fit OneHotEncoder
    ohe = OneHotEncoder(
        sparse_output=False,
        drop='first' if drop_first else None,
        handle_unknown=handle_unknown,
        dtype=float
    )
    arr = ohe.fit_transform(df[categorical_cols])

    # Build DataFrame of encoded features
    feature_names = ohe.get_feature_names_out(categorical_cols)
    df_ohe = pd.DataFrame(arr, columns=feature_names, index=df.index)

    # Drop original categorical cols & concat the new dummies
    df = pd.concat([df.drop(columns=categorical_cols), df_ohe], axis=1)

    # Log‑transform the amount
    df['amount_log'] = np.log1p(df[amount_col])

    # Standard‑scale the log amount
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount_log']])

    return df

def make_preprocessor(
    categorical_cols: List[str],
    amount_col: str = 'amount',
    time_col: Optional[str] = None,
    drop_first: bool = True
) -> Pipeline:
    """
    Build a sklearn Pipeline to preprocess features:
      - Numeric: median impute, log1p, StandardScaler
      - Categorical: constant impute, OneHotEncoder
      - Optional time features via add_time_features
    """
    # Numeric pipeline
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=True)),
        ('scale', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(
            sparse_output=False,
            drop='first' if drop_first else None,
            handle_unknown='ignore'
        ))
    ])

    # Assemble transformers
    transformers = [
        ('num', num_pipeline, [amount_col]),
        ('cat', cat_pipeline, categorical_cols)
    ]
    if time_col:
        time_pipeline = Pipeline([
            ('time_feats', FunctionTransformer(lambda df: add_time_features(df, time_col), validate=False))
        ])
        transformers.append(('time', time_pipeline, [time_col]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return Pipeline([('preprocessor', preprocessor)])

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # Load dataset
    df = pd.read_csv(input_path)

    # Build and apply pipeline
    preproc = make_preprocessor(
        categorical_cols=['PSP','card','country'],
        amount_col='amount',
        time_col='tmsp'
    )
    df_transformed = preproc.fit_transform(df)

    # Convert back to DataFrame with feature names
    columns = preproc.named_steps['preprocessor'].get_feature_names_out()
    df_out = pd.DataFrame(df_transformed, columns=columns, index=df.index)

    # Save
    df_out.to_csv(output_path, index=False)
    logger.success(f"Features saved to {output_path}")

if __name__ == "__main__":
    app()
