from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from creditcard_psp.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_transactions(file_path: Path) -> pd.DataFrame:
    """
    Load data from an Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the loaded transaction data.
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_excel(file_path, index_col=0)
    return df

def assign_transaction_ids(
    df: pd.DataFrame,
    time_col: str = "tmsp",
    country_col: str = "country",
    amount_col: str = "amount",
    keep_only_first: bool = True,
    drop_attempt_number: bool = True,
) -> pd.DataFrame:
    """
    Assign transaction ids (same country, amount, and same minute) and
    keep only the first attempt per transaction if keep_only_first=True.

    Returns the original columns plus:
      - 'transaction_id'
      - 'attempt_number'  (1 for the kept rows if keep_only_first=True)
      - 'transaction_success' (max success in the transaction if 'success' exists)
    """
    df = df.copy()
    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # "Same minute" bucket
    df["minute_bucket"] = df[time_col].dt.floor("T")

    # Sort within group by time (then by original order to break ties deterministically)
    gkey = [country_col, amount_col, "minute_bucket"]
    df = df.sort_values(gkey + [time_col], kind="mergesort")

    df["transaction_id"] = df.groupby(gkey, sort=False).ngroup()

    # Attempt number within transaction (after time-based sorting)
    df["attempt_number"] = df.groupby("transaction_id").cumcount() + 1

    # Transaction-level success flag if available
#     if "success" in df.columns:
#         df["transaction_success"] = df.groupby("transaction_id")["success"].transform("max")

    if keep_only_first:
        df = df[df["attempt_number"] == 1].copy()
        if drop_attempt_number:
            df.drop(columns=["attempt_number"], inplace=True)
            
        # Safety: genau eine Zeile je transaction_id
        assert df["transaction_id"].is_unique, "Mehrere Zeilen pro transaction_id nach dem Filtern!"


    # Clean up helper columns
    df.drop(columns=["minute_bucket"], inplace=True, errors="ignore")
    return df


def merge_service_fees(
    df: pd.DataFrame,
    fee_filepath: Path,
    psp_col: str = 'PSP'
) -> pd.DataFrame:
    """
    Load service fees from an Excel file, normalize decimal separators,
    merge them into the transactions DataFrame on the PSP column,
    and report any missing entries.

    Parameters
    ----------
    df : pd.DataFrame
        Transactions DataFrame containing a column with PSP identifiers.
    fee_filepath : Path
        Path to the Excel file with columns ['PSP', 'fee_successful', 'fee_not_successful'].
    psp_col : str, default 'PSP'
        Name of the column in both df and the fees file to merge on.

    Returns
    -------
    pd.DataFrame
        The original df plus the two new columns:
        'fee_successful' and 'fee_not_successful'.
    """
    logger.info(f"Merging service fees from {fee_filepath}")
    fees = pd.read_excel(fee_filepath)
    for col in ('fee_successful', 'fee_not_successful'):
        fees[col] = (
            fees[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
    merged = df.merge(
        fees,
        on=psp_col,
        how='left',
        validate='many_to_one'
    )
    missing = merged[['fee_successful', 'fee_not_successful']].isna().sum()
    logger.warning(f"Missing fee entries after merge: {missing.to_dict()}")
    return merged

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / 'PSP_Jan_Feb_2019.xlsx',
    fee_path:   Path = RAW_DATA_DIR / 'PSP_Servicegebuehren.xlsx',
    output_path: Path = PROCESSED_DATA_DIR / 'df.pkl',
):
    # Load transactions
    df = load_transactions(input_path)

    # Assign transaction IDs
    df = assign_transaction_ids(df)

    # Merge service fees
    df = merge_service_fees(df, fee_path)
    
    # Save DataFrame
    df.to_pickle(output_path)
    
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
