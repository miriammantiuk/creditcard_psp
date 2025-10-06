from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.model_selection import train_test_split

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
) -> pd.DataFrame:
    """
    Assign a transaction ID to each row by grouping rows with the same values in group_cols
    whose timestamps are within `gap` of each other. Also adds columns for transaction_success
    and attempt_number.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transaction attempts.
    time_col : str, default 'tmsp'
        Name of the timestamp column to use for ordering.
    group_cols : List[str], default ['country', 'amount']
        Columns defining the grouping key for transactions.
    gap : pd.Timedelta, default 1 minute
        Maximum allowed time difference between attempts in the same transaction.

    Returns
    -------
    pd.DataFrame
        The original DataFrame enriched with:
        - 'transaction_id'
        - 'transaction_success'
        - 'attempt_number'
    """
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # exact "same minute": floor timestamp to minute
    df["minute_bucket"] = df[time_col].dt.floor("T")

    # global transaction id per (country, amount, minute)
    df["transaction_id"] = (
        df.groupby([country_col, amount_col, "minute_bucket"], sort=False)
          .ngroup()
    )

    # attempt count within transaction
    df["attempt_number"] = df.groupby("transaction_id").cumcount() + 1

    # transaction-level success (if available)
    if "success" in df.columns:
        df["transaction_success"] = df.groupby("transaction_id")["success"].transform("max")

    # cleanup
    df.drop(columns=["minute_bucket"], inplace=True)
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

def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    time_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train and test sets.

    If `time_col` is given:
      - With 'transaction_id': group-aware chronological split (no leakage across attempts).
      - Without: row-wise chronological split.
    Else: random split (optionally stratified).

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame including feature and target columns.
    target_col : str
        Name of the column to predict.
    test_size : float, default 0.2
        Fraction of data to reserve for the test set.
    random_state : int, default 42
        Seed for shuffling (only used if time_col is None).
    stratify : bool, default True
        Whether to stratify by `target_col` (only used if time_col is None).
    time_col : str or None, default None
        If given, perform a chronological split on this datetime column.

    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    if time_col:
        if time_col not in df.columns:
            raise KeyError(f"'{time_col}' not in columns for chronological split.")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        if "transaction_id" in df.columns:
            tx = df.groupby("transaction_id")[time_col].max().sort_values()
            k = int((1 - test_size) * len(tx))
            train_ids = set(tx.index[:k])
            train = df[df["transaction_id"].isin(train_ids)]
            test  = df[~df["transaction_id"].isin(train_ids)]
        else:
            df = df.sort_values(time_col)
            k = int((1 - test_size) * len(df))
            train, test = df.iloc[:k], df.iloc[k:]

        return (
            train.drop(columns=[target_col]),
            test.drop(columns=[target_col]),
            train[target_col],
            test[target_col],
        )

    # Random split
    X, y = df.drop(columns=[target_col]), df[target_col]
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / 'PSP_Jan_Feb_2019.xlsx',
    fee_path:   Path = RAW_DATA_DIR / 'PSP_Servicegebuehren.xlsx',
    output_path: Path = PROCESSED_DATA_DIR / 'PSP_Jan_Feb_2019_processed.csv',
):
    # Load transactions
    df = load_transactions(input_path)

    # Assign transaction IDs, flags, and attempt numbers
    df = assign_transaction_ids(df)

    # Merge service fees
    df = merge_service_fees(df, fee_path)
    
    # Save processed Dataset
    df.to_csv(output_path, index=False)
    logger.success(f"Speichere das verarbeitete Dataset {output_path}")

    # Chronologic split (e.g.  Timestamp)
    X_tr, X_te, y_tr, y_te = split_train_test(
        df=df,
        target_col=target_col,
        test_size=test_size,   
        time_col='tmsp',       
        stratify=False
    )


    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
