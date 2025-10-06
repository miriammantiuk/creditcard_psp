from pathlib import Path

import pandas as pd
import joblib

from loguru import logger
from tqdm import tqdm
import typer

from creditcard_psp.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def recommend_best_psp(transaction_row, models_dict, alpha=0.05):
    """
    Recommends the best PSP for a new transaction based on:
    score = p(success) - alpha * expected_cost

    Parameters
    ----------
    transaction_row : pd.Series
        A single row representing a new transaction (same features as training set).
    models_dict : dict
        A dictionary {psp_name: trained_model} for all PSPs.
    alpha : float
        Cost penalty factor (default = 0.05).

    Returns
    -------
    best_psp : str
        PSP with the highest score.
    psp_scores : pd.DataFrame
        DataFrame with p(success), expected cost, and final score per PSP.
    """
    results = []

    for psp, model in models_dict.items():
        # Inject PSP into the row
        row = transaction_row.copy()
        row['PSP'] = psp

        # Model expects a DataFrame
        X = pd.DataFrame([row])

        # Predict success probability
        p_success = model.predict_proba(X)[0][1]

        # Lookup fees (you must add them into the row or merge beforehand)
        fee_s = row['fee_successful']
        fee_f = row['fee_not_successful']
        expected_cost = p_success * fee_s + (1 - p_success) * fee_f
        score = p_success - alpha * expected_cost

        results.append({
            'PSP': psp,
            'p_success': round(p_success, 4),
            'expected_cost': round(expected_cost, 4),
            'score': round(score, 4)
        })

    scores_df = pd.DataFrame(results).sort_values(by='score', ascending=False)
    best_psp = scores_df.iloc[0]['PSP']
    return best_psp, scores_df


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    models_dir: Path = MODELS_DIR,
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    alpha: float = 0.05
):
    logger.info(f"Loading test data from: {features_path}")
    test_df = pd.read_csv(features_path)

    logger.info(f"Loading PSP models from: {models_dir}")
    models_dict = {}
    for model_file in models_dir.glob("model_*.pkl"):
        psp = model_file.stem.replace("model_", "")
        models_dict[psp] = joblib.load(model_file)

    logger.info("Recommending best PSPs for test transactions...")
    best_psps = []
    all_scores = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        best_psp, scores_df = recommend_best_psp(row, models_dict, alpha)
        best_psps.append(best_psp)
        all_scores.append(scores_df)

    test_df['recommended_psp'] = best_psps
    test_df.to_csv(predictions_path, index=False)
    logger.success(f"Saved recommendations to: {predictions_path}")


if __name__ == "__main__":
    app()