"""
Evaluation utilities for computing metrics across stores and folds.

Functions for evaluating models and baselines, and displaying results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable

from src.metrics import mape, smape, rmse, safe_metric


def evaluate_model(model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a fitted model on validation data.

    Args:
        model: Fitted model with predict() method
        X: Validation features
        y: Validation target (ground truth)

    Returns:
        Dict[str, float]: Dictionary with 'mape', 'smape', 'rmse' keys
    """
    # Get predictions
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        'mape': safe_metric(mape, y, y_pred),
        'smape': safe_metric(smape, y, y_pred),
        'rmse': safe_metric(rmse, y, y_pred)
    }

    return metrics


def evaluate_baseline(
    df: pd.DataFrame,
    baseline_func: Callable,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> Dict[str, float]:
    """
    Evaluate a baseline model across all stores in validation window.

    Args:
        df: Full dataframe with features
        baseline_func: Baseline function (e.g., seasonal_naive_forecast)
        val_start: Validation start date
        val_end: Validation end date

    Returns:
        Dict[str, float]: Dictionary with 'mape', 'smape', 'rmse' keys
    """
    # Get all stores in validation window
    val_mask = (df['Date'] >= val_start) & (df['Date'] <= val_end)
    val_stores = df[val_mask]['Store'].unique()

    all_predictions = []
    all_actuals = []

    # Aggregate predictions and actuals across all stores
    for store_id in val_stores:
        try:
            # Get baseline predictions for this store
            predictions = baseline_func(df, store_id, val_start, val_end)

            # Get actual values for this store
            store_mask = (
                (df['Store'] == store_id) &
                (df['Date'] >= val_start) &
                (df['Date'] <= val_end)
            )
            actuals = df[store_mask]['Sales'].values

            # Only include if lengths match (safeguard)
            if len(predictions) == len(actuals):
                all_predictions.extend(predictions)
                all_actuals.extend(actuals)

        except Exception as e:
            # Skip stores that fail (e.g., insufficient data)
            continue

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Calculate metrics
    if len(all_predictions) > 0:
        metrics = {
            'mape': safe_metric(mape, all_actuals, all_predictions),
            'smape': safe_metric(smape, all_actuals, all_predictions),
            'rmse': safe_metric(rmse, all_actuals, all_predictions)
        }
    else:
        # No valid predictions
        metrics = {
            'mape': np.nan,
            'smape': np.nan,
            'rmse': np.nan
        }

    return metrics


def print_leaderboard(metrics_df: pd.DataFrame):
    """
    Print formatted leaderboard of model performance.

    Shows mean and std of metrics across folds, sorted by MAPE.

    Args:
        metrics_df: DataFrame with columns: fold, model_name, mape, smape, rmse
    """
    print("\n" + "="*80)
    print("MODEL LEADERBOARD")
    print("="*80)

    # Group by model name
    grouped = metrics_df.groupby('model_name').agg({
        'mape': ['mean', 'std'],
        'smape': ['mean', 'std'],
        'rmse': ['mean', 'std']
    }).round(2)

    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={
        'mape_mean': 'MAPE_mean',
        'mape_std': 'MAPE_std',
        'smape_mean': 'sMAPE_mean',
        'smape_std': 'sMAPE_std',
        'rmse_mean': 'RMSE_mean',
        'rmse_std': 'RMSE_std'
    })

    # Sort by MAPE (lower is better)
    grouped = grouped.sort_values('MAPE_mean')

    # Format output
    print(f"{'Model':<20} {'MAPE':<15} {'sMAPE':<15} {'RMSE':<15}")
    print("-"*80)

    for model_name, row in grouped.iterrows():
        mape_str = f"{row['MAPE_mean']:.2f} ± {row['MAPE_std']:.2f}%"
        smape_str = f"{row['sMAPE_mean']:.2f} ± {row['sMAPE_std']:.2f}%"
        rmse_str = f"{row['RMSE_mean']:.0f} ± {row['RMSE_std']:.0f}"

        print(f"{model_name:<20} {mape_str:<15} {smape_str:<15} {rmse_str:<15}")

    print("="*80)

    # Identify best model
    best_model = grouped.index[0]
    best_mape = grouped.iloc[0]['MAPE_mean']
    print(f"\n✓ Best model: {best_model} (MAPE: {best_mape:.2f}%)")
    print()
