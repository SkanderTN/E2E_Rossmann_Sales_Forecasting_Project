"""
Simple baseline models for comparison.

Implements seasonal naive (lag-7) and moving average (7-day) baselines
to benchmark model performance.
"""

import numpy as np
import pandas as pd


def seasonal_naive_forecast(
    df: pd.DataFrame,
    store_id: int,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> np.ndarray:
    """
    Seasonal naive forecast: predict Sales = value from 7 days ago (lag_7).

    Args:
        df: Full dataframe with lag_7 feature already computed
        store_id: Store ID to forecast for
        val_start: Start of validation period
        val_end: End of validation period

    Returns:
        np.ndarray: Predictions (lag_7 values) for the validation period
    """
    # Filter to store and validation window
    mask = (
        (df['Store'] == store_id) &
        (df['Date'] >= val_start) &
        (df['Date'] <= val_end)
    )
    store_val = df[mask].copy()

    # Use lag_7 as prediction
    predictions = store_val['lag_7'].values

    # Fill NaN with median of available lag_7 for this store
    if np.any(np.isnan(predictions)):
        store_data = df[df['Store'] == store_id]
        median_lag7 = store_data['lag_7'].median()
        predictions = np.nan_to_num(predictions, nan=median_lag7)

    return predictions


def moving_average_forecast(
    df: pd.DataFrame,
    store_id: int,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> np.ndarray:
    """
    Moving average forecast: predict Sales = 7-day moving average (roll_mean_7).

    Args:
        df: Full dataframe with roll_mean_7 feature already computed
        store_id: Store ID to forecast for
        val_start: Start of validation period
        val_end: End of validation period

    Returns:
        np.ndarray: Predictions (roll_mean_7 values) for the validation period
    """
    # Filter to store and validation window
    mask = (
        (df['Store'] == store_id) &
        (df['Date'] >= val_start) &
        (df['Date'] <= val_end)
    )
    store_val = df[mask].copy()

    # Use roll_mean_7 as prediction
    predictions = store_val['roll_mean_7'].values

    # Fill NaN with median of available roll_mean_7 for this store
    if np.any(np.isnan(predictions)):
        store_data = df[df['Store'] == store_id]
        median_roll_mean = store_data['roll_mean_7'].median()
        predictions = np.nan_to_num(predictions, nan=median_roll_mean)

    return predictions
