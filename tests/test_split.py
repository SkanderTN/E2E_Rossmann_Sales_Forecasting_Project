"""
Tests for rolling-origin splits correctness.

Validates that validation windows are correct size, non-overlapping,
and training data is strictly before validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.split import get_rolling_splits, split_data_by_dates


def test_rolling_splits_count():
    """Test that correct number of splits are returned."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    splits = get_rolling_splits(df, n_folds=3, val_window_days=28)

    assert len(splits) == 3


def test_rolling_splits_no_overlap():
    """Test that validation windows don't overlap."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    splits = get_rolling_splits(df, n_folds=3, val_window_days=28)

    # Check each pair of consecutive folds
    for i in range(len(splits) - 1):
        _, _, val_start_i, val_end_i = splits[i]
        _, _, val_start_j, val_end_j = splits[i + 1]

        # Fold i should be more recent than fold i+1
        assert val_end_i > val_end_j
        # No overlap: fold i+1 should end before fold i starts
        assert val_end_j < val_start_i


def test_rolling_splits_validation_window_size():
    """Test that each validation window is exactly the requested size."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    val_window = 28
    splits = get_rolling_splits(df, n_folds=3, val_window_days=val_window)

    for train_start, train_end, val_start, val_end in splits:
        # Calculate window size (inclusive of both start and end dates)
        window_size = (val_end - val_start).days + 1
        assert window_size == val_window


def test_rolling_splits_train_before_val():
    """Test that training data is strictly before validation."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    splits = get_rolling_splits(df, n_folds=3, val_window_days=28)

    for train_start, train_end, val_start, val_end in splits:
        # Training end should be before validation start
        assert train_end < val_start


def test_split_data_by_dates():
    """Test that split_data_by_dates correctly filters data."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': range(100)})

    train_start = pd.Timestamp('2023-01-01')
    # Use timedelta arithmetic to create valid timestamps instead of
    # non-standard strings like '2023-01-50' which newer pandas rejects.
    train_end = train_start + pd.Timedelta(days=49)  # 50 days total
    val_start = train_end + pd.Timedelta(days=1)
    val_end = val_start + pd.Timedelta(days=19)  # 20-day validation window

    train_df, val_df = split_data_by_dates(df, train_start, train_end, val_start, val_end)

    # Check training data
    assert len(train_df) == 50
    assert train_df['Date'].min() == train_start
    assert train_df['Date'].max() == train_end

    # Check validation data
    assert len(val_df) == 20
    assert val_df['Date'].min() == val_start
    assert val_df['Date'].max() == val_end

    # Check no overlap
    assert train_df['Date'].max() < val_df['Date'].min()


def test_rolling_splits_insufficient_data():
    """Test that error is raised when insufficient data."""
    # Only 50 days of data, not enough for 3 folds of 28 days + warm-up
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    with pytest.raises(ValueError):
        get_rolling_splits(df, n_folds=3, val_window_days=28)


def test_rolling_splits_output_types():
    """Test that splits return correct types."""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    splits = get_rolling_splits(df, n_folds=2, val_window_days=28)

    # Should return list of tuples
    assert isinstance(splits, list)
    assert len(splits) == 2

    for split in splits:
        assert isinstance(split, tuple)
        assert len(split) == 4  # (train_start, train_end, val_start, val_end)

        train_start, train_end, val_start, val_end = split
        # All should be timestamps
        assert isinstance(train_start, pd.Timestamp)
        assert isinstance(train_end, pd.Timestamp)
        assert isinstance(val_start, pd.Timestamp)
        assert isinstance(val_end, pd.Timestamp)
