"""
Metric calculations with safeguards against invalid values.

Implements MAPE, sMAPE, and RMSE with proper handling of edge cases
like division by zero, NaN, and inf values.
"""

import numpy as np
from typing import Callable


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / y_true) * 100

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: MAPE as percentage, or np.nan if cannot be computed
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out rows where y_true is zero (avoid division by zero)
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # If all true values are zero, cannot compute MAPE
    if len(y_true_filtered) == 0:
        return np.nan

    # Calculate MAPE
    abs_pct_errors = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
    return np.mean(abs_pct_errors) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    sMAPE = mean(|y_true - y_pred| / (|y_true| + |y_pred|)) * 200

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: sMAPE as percentage, or np.nan if cannot be computed
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate denominator
    denominator = np.abs(y_true) + np.abs(y_pred)

    # Filter out rows where both are zero
    mask = denominator != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    denominator_filtered = denominator[mask]

    # If all denominators are zero, cannot compute sMAPE
    if len(denominator_filtered) == 0:
        return np.nan

    # Calculate sMAPE
    abs_errors = np.abs(y_true_filtered - y_pred_filtered)
    smape_values = abs_errors / denominator_filtered
    return np.mean(smape_values) * 200


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    RMSE = sqrt(mean((y_true - y_pred)^2))

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    squared_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(squared_errors))


def safe_metric(metric_func: Callable, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wrapper for metric functions that catches exceptions and validates inputs.

    Args:
        metric_func: Metric function to call (e.g., mape, smape, rmse)
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: Metric value, or np.nan if computation fails or inputs are invalid
    """
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Check for inf or NaN in inputs
        if np.any(np.isinf(y_true)) or np.any(np.isnan(y_true)):
            return np.nan
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return np.nan

        # Check shapes match
        if y_true.shape != y_pred.shape:
            return np.nan

        # Call metric function
        result = metric_func(y_true, y_pred)

        # Check result is valid
        if np.isinf(result) or np.isnan(result):
            return np.nan

        return result

    except Exception as e:
        # Any exception during metric calculation returns NaN
        return np.nan
