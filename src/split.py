"""
Rolling-origin backtesting splits.

Implements time-series cross-validation where validation windows walk backward
from the most recent data, and training uses all data strictly before each window.
"""

import pandas as pd
from typing import List, Tuple
from datetime import timedelta


def get_rolling_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    val_window_days: int = 28
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Create rolling-origin backtesting splits.

    Each fold has:
    - Validation window: Fixed size (val_window_days)
    - Training window: All data strictly before validation start

    Folds are created by walking backward from the most recent date.

    Args:
        df: DataFrame with Date column
        n_folds: Number of folds to create (default: 3)
        val_window_days: Size of each validation window in days (default: 28)

    Returns:
        List of tuples: [(train_start, train_end, val_start, val_end), ...]
        Each tuple represents one fold's date ranges.

    Raises:
        ValueError: If insufficient data for requested folds
    """
    # Get date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    total_days = (max_date - min_date).days + 1

    print("\n" + "="*80)
    print("Creating rolling-origin splits")
    print("="*80)
    print(f"Data range: {min_date.date()} to {max_date.date()} ({total_days} days)")
    print(f"Requested: {n_folds} folds with {val_window_days}-day validation windows")

    # Check if we have enough data
    min_required_days = n_folds * val_window_days + 28  # +28 for warm-up
    if total_days < min_required_days:
        raise ValueError(
            f"Insufficient data for {n_folds} folds with {val_window_days}-day windows. "
            f"Need at least {min_required_days} days, have {total_days} days."
        )

    splits = []

    for fold_idx in range(n_folds):
        # Calculate validation window (walking backward from max_date)
        val_end = max_date - timedelta(days=fold_idx * val_window_days)
        val_start = val_end - timedelta(days=val_window_days - 1)

        # Training uses all data strictly before validation start
        train_end = val_start - timedelta(days=1)
        train_start = min_date

        splits.append((train_start, train_end, val_start, val_end))

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Training:   {train_start.date()} to {train_end.date()} ({(train_end - train_start).days + 1} days)")
        print(f"  Validation: {val_start.date()} to {val_end.date()} ({(val_end - val_start).days + 1} days)")

    print("="*80 + "\n")

    return splits


def split_data_by_dates(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into training and validation sets based on date ranges.

    Args:
        df: DataFrame with Date column
        train_start: Training start date (inclusive)
        train_end: Training end date (inclusive)
        val_start: Validation start date (inclusive)
        val_end: Validation end date (inclusive)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    # Filter training data
    train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    train_df = df[train_mask].copy()

    # Filter validation data
    val_mask = (df['Date'] >= val_start) & (df['Date'] <= val_end)
    val_df = df[val_mask].copy()

    return train_df, val_df
