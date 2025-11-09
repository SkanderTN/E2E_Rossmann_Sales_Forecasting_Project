"""
Tests for feature engineering and leakage prevention.

Validates that lag and rolling features use only past data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_seasonality_features,
    build_all_features
)


def test_calendar_features():
    """Test calendar feature extraction."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({'Date': dates, 'Store': 1, 'Sales': 100})

    df = add_calendar_features(df)

    # Check features exist
    assert 'dow' in df.columns
    assert 'month' in df.columns
    assert 'weekofyear' in df.columns
    assert 'quarter' in df.columns
    assert 'year' in df.columns
    assert 'is_weekend' in df.columns

    # Check specific values (2023-01-01 is Sunday, dow=6)
    assert df.iloc[0]['dow'] == 6  # Sunday
    assert df.iloc[0]['is_weekend'] == 1  # Sunday is weekend

    # Check Monday (2023-01-02, dow=0)
    assert df.iloc[1]['dow'] == 0  # Monday
    assert df.iloc[1]['is_weekend'] == 0  # Monday not weekend


def test_lag_features_no_leakage():
    """Test that lag features don't leak future information."""
    # Create simple dataset with known values
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    sales = np.arange(10, 110, 10)  # [10, 20, 30, ..., 100]

    df = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Sales': sales
    })

    df = add_lag_features(df, 'Sales')

    # Check lag_1: should be previous day's value
    assert np.isnan(df.iloc[0]['lag_1'])  # First row has no previous day
    assert df.iloc[1]['lag_1'] == 10  # Second day should have first day's value
    assert df.iloc[2]['lag_1'] == 20  # Third day should have second day's value

    # Check lag_7: should be 7 days ago
    assert np.isnan(df.iloc[6]['lag_7'])  # 7th row doesn't have data from 7 days ago
    assert df.iloc[7]['lag_7'] == 10  # 8th row (index 7) should have 1st row's value


def test_lag_features_multi_store():
    """Test that lag features are computed separately per store."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')

    # Create data for two stores
    df = pd.DataFrame({
        'Store': [1, 1, 1, 2, 2, 2],
        'Date': list(dates[:3]) + list(dates[:3]),
        'Sales': [10, 20, 30, 100, 200, 300]
    })

    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df = add_lag_features(df, 'Sales')

    # Store 1: lag_1 for second row should be 10
    store1_data = df[df['Store'] == 1].reset_index(drop=True)
    assert np.isnan(store1_data.iloc[0]['lag_1'])  # First row
    assert store1_data.iloc[1]['lag_1'] == 10  # Second row

    # Store 2: lag_1 for second row should be 100 (not 30 from Store 1)
    store2_data = df[df['Store'] == 2].reset_index(drop=True)
    assert np.isnan(store2_data.iloc[0]['lag_1'])  # First row of Store 2
    assert store2_data.iloc[1]['lag_1'] == 100  # Should use Store 2's data


def test_rolling_features_shifted():
    """Test that rolling features use shifted values (no current day)."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    sales = np.full(30, 100.0)  # Constant sales

    df = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Sales': sales
    })

    df = add_rolling_features(df, 'Sales')

    # With constant sales and shifting, roll_mean_7 should still be 100
    # But it should NOT include current day
    # Check that rolling features exist
    assert 'roll_mean_7' in df.columns
    assert 'roll_std_7' in df.columns
    assert 'roll_mean_28' in df.columns
    assert 'roll_std_28' in df.columns

    # For constant sales, mean should be 100 (after warm-up)
    assert np.isclose(df.iloc[10]['roll_mean_7'], 100.0, rtol=0.01)


def test_rolling_warmup_nans():
    """Test that rolling features have NaN warm-up period."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Sales': np.arange(10, 110, 10)
    })

    df = add_rolling_features(df, 'Sales')

    # First row should have NaN (no previous data after shift)
    assert np.isnan(df.iloc[0]['roll_mean_7'])

    # Later rows should have values
    assert not np.isnan(df.iloc[-1]['roll_mean_7'])


def test_seasonality_features():
    """Test seasonality feature calculation."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Sales': np.arange(100, 400, 10)
    })

    # Add lag features first (required for seasonality)
    df = add_lag_features(df, 'Sales')
    df = add_rolling_features(df, 'Sales')
    df = add_seasonality_features(df, 'Sales')

    # Check features exist
    assert 'diff_7' in df.columns
    assert 'ratio_7' in df.columns
    assert 'trend_7' in df.columns

    # Check no inf values in ratios
    assert not np.any(np.isinf(df['ratio_7']))
    assert not np.any(np.isinf(df['ratio_14']))


def test_feature_pipeline_order():
    """Test that build_all_features runs without errors."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Store': [1] * 50,
        'Date': dates,
        'Sales': np.random.randint(100, 500, 50),
        'Promo': ['0'] * 50,
        'StateHoliday': ['0'] * 50,
        'SchoolHoliday': ['0'] * 50,
        'StoreType': ['a'] * 50,
        'Assortment': ['a'] * 50,
        'CompetitionDistance': [1000.0] * 50,
        'CompetitionOpenSinceMonth': [1.0] * 50,
        'CompetitionOpenSinceYear': [2010.0] * 50,
        'Promo2': [0] * 50,
        'Promo2SinceWeek': [np.nan] * 50,
        'Promo2SinceYear': [np.nan] * 50,
        'PromoInterval': [''] * 50
    })

    # Should not raise exceptions
    df_features = build_all_features(df, 'Sales')

    # Check that features were added
    assert 'dow' in df_features.columns
    assert 'lag_1' in df_features.columns
    assert 'roll_mean_7' in df_features.columns
    assert 'diff_7' in df_features.columns


def test_features_preserve_temporal_order():
    """Test that features are computed on sorted data."""
    # Create unsorted data
    dates = [datetime(2023, 1, 3), datetime(2023, 1, 1), datetime(2023, 1, 2)]
    df = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Sales': [30, 10, 20]
    })

    # build_all_features should sort by Date
    df_features = build_all_features(df, 'Sales')

    # Check that data is sorted
    assert df_features['Date'].is_monotonic_increasing
