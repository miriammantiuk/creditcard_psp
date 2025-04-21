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
    Load transactions from an Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the Excel file containing transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with the loaded transaction data.
    """
    logger.info(f"Loading transactions from {file_path}")
    df = pd.read_excel(file_path)
    return df

def assign_transaction_ids(
    df: pd.DataFrame,
    time_col: str = 'tmsp',
    group_cols: List[str] = ['country', 'amount'],
    gap: pd.Timedelta = pd.Timedelta(minutes=1)
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
    # Parse timestamp and sort
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(group_cols + [time_col])

    # Compute time difference within each group
    df['time_diff'] = df.groupby(group_cols)[time_col].diff().abs()

    # Flag new transaction when gap is exceeded or at the start
    df['new_transaction'] = df['time_diff'].isna() | (df['time_diff'] > gap)

    # Cumulative sum yields the transaction ID
    df['transaction_id'] = df.groupby(group_cols)['new_transaction'].cumsum()

    # Transaction-level success flag
    tx_key = group_cols + ['transaction_id']
    df['transaction_success'] = df.groupby(tx_key)['success'].transform('max')

    # Attempt number within each transaction
    df['attempt_number'] = df.groupby(tx_key).cumcount() + 1

    # Clean up temporary columns
    df.drop(columns=['time_diff', 'new_transaction'], inplace=True)

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

    If `time_col` is provided, does a chronologic split:
      - Sorts by `time_col`, then takes the last `test_size` fraction as test set.

    Otherwise does a randomized split (optionally stratified by target).

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
        # ensure datetime
        df = df.sort_values(time_col)
        n = len(df)
        split_at = int((1 - test_size) * n)
        train = df.iloc[:split_at]
        test  = df.iloc[split_at:]
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test  = test.drop(columns=[target_col])
        y_test  = test[target_col]
    else:
        strat = df[target_col] if stratify else None
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
    return X_train, X_test, y_train, y_test

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

    # Chronologic split (e.g.  Timestamp)
    X_tr, X_te, y_tr, y_te = split_train_test(
        df=agg_df,
        target_col='transaction_success',
        test_size=0.25,
        time_col='tmsp_last'    
    )

    # random, stratify split
    X_tr2, X_te2, y_tr2, y_te2 = split_train_test(
        df=agg_df,
        target_col='transaction_success',
        test_size=0.2,
        stratify=True
)

    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
