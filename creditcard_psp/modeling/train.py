from pathlib import Path
import joblib

import pandas as pd
import numpy as np
import typer
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from creditcard_psp.config import PROCESSED_DATA_DIR, MODELS_DIR
from creditcard_psp.dataset import split_train_test
from creditcard_psp.features import make_preprocessor

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path:    Path = MODELS_DIR        / "model_pipeline.joblib",
    target_col:    str  = "transaction_success",
    test_size:     float = 0.2,
    random_seed:   int   = 42
):
    """
    Train a classifier on preprocessed features, evaluate on a hold-out set,
    and save the full sklearn Pipeline to disk.
    """
    # 1) Load features
    logger.info(f"Loading feature data from {features_path}")
    df = pd.read_csv(features_path)

    # 2) Split into train/test
    logger.info(f"Splitting data: test_size={test_size}, random_seed={random_seed}")
    X_train, X_test, y_train, y_test = split_train_test(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_seed,
        stratify=True
    )

    # 3) Build preprocessing + model pipeline
    logger.info("Building preprocessing pipeline + classifier")
    # Define which original columns we need to reconstruct the preprocessor
    # Here we assume your pipeline knows its own columns via the make_preprocessor call.
    preprocessor = make_preprocessor(
        categorical_cols=['PSP', 'card', 'country'],
        amount_col='amount',
        time_col='tmsp'
    )
    clf = LogisticRegression(max_iter=1000, random_state=random_seed)

    pipeline = typer.Context().obj = None  # dummy to satisfy type checkers
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # 4) Fit
    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    # 5) Evaluate
    logger.info("Evaluating on test data")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=4)

    logger.success(f"Test accuracy: {acc:.4f}")
    logger.success(f"Test ROC-AUC   : {auc:.4f}")
    logger.info("Classification Report:\n" + report)

    # 6) Save model pipeline
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.success(f"Saved trained pipeline to {model_path}")

if __name__ == "__main__":
    app()