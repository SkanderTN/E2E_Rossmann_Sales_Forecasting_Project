"""
Leakage-safe feature engineering with strict temporal ordering.

All lag and rolling features are shifted BEFORE aggregation to prevent data leakage.
Features use only past information and are computed per store.
"""

import numpy as np
import pandas as pd
import holidays
from typing import List


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features derived from Date column.

    Features:
    - dow: Day of week (0=Monday, 6=Sunday)
    - month: Month (1-12)
    - weekofyear: ISO week number
    - quarter: Quarter (1-4)
    - year: Year
    - is_weekend: Boolean (1 if Saturday/Sunday, else 0)

    Args:
        df: DataFrame with 'Date' column (datetime type)

    Returns:
        pd.DataFrame: DataFrame with calendar features added
    """
    df = df.copy()

    df['dow'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday features using Germany holidays library.

    Feature:
    - is_holiday: 1 if Date is a German holiday OR StateHoliday != '0', else 0

    Args:
        df: DataFrame with 'Date' and 'StateHoliday' columns

    Returns:
        pd.DataFrame: DataFrame with holiday feature added
    """
    df = df.copy()

    # Get German holidays
    de_holidays = holidays.Germany()

    # Check if date is a German holiday OR has StateHoliday != '0'
    # Ensure StateHoliday column exists; if missing, assume '0' (no holiday)
    if 'StateHoliday' not in df.columns:
        df['StateHoliday'] = '0'

    df['is_holiday'] = df.apply(
        lambda row: 1 if (row['Date'] in de_holidays or row['StateHoliday'] != '0') else 0,
        axis=1
    )

    return df


def add_lag_features(df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
    """
    Add lag features (shifted sales from previous periods).

    CRITICAL: Lags are created by shifting within each store group.
    This prevents leakage by ensuring we only use past values.

    Features:
    - lag_1: Sales from 1 day ago
    - lag_7: Sales from 7 days ago
    - lag_14: Sales from 14 days ago
    - lag_28: Sales from 28 days ago

    Args:
        df: DataFrame sorted by Store, Date with target_col
        target_col: Name of target column (default: 'Sales')

    Returns:
        pd.DataFrame: DataFrame with lag features added
    """
    df = df.copy()

    # Group by Store and shift (within each store)
    df['lag_1'] = df.groupby('Store')[target_col].shift(1)
    df['lag_7'] = df.groupby('Store')[target_col].shift(7)
    df['lag_14'] = df.groupby('Store')[target_col].shift(14)
    df['lag_28'] = df.groupby('Store')[target_col].shift(28)

    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
    """
    Add rolling window features (moving averages and standard deviations).

    CRITICAL: shift(1) BEFORE .rolling() to prevent leakage.
    This ensures rolling windows only use past data, excluding the current day.

    Features:
    - roll_mean_7: 7-day rolling mean (excluding current day)
    - roll_std_7: 7-day rolling std (excluding current day)
    - roll_mean_28: 28-day rolling mean (excluding current day)
    - roll_std_28: 28-day rolling std (excluding current day)

    Args:
        df: DataFrame sorted by Store, Date with target_col
        target_col: Name of target column (default: 'Sales')

    Returns:
        pd.DataFrame: DataFrame with rolling features added
    """
    df = df.copy()

    # CRITICAL: shift BEFORE rolling to prevent leakage
    df['roll_mean_7'] = (
        df.groupby('Store')[target_col]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['roll_std_7'] = (
        df.groupby('Store')[target_col]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['roll_mean_28'] = (
        df.groupby('Store')[target_col]
        .shift(1)
        .rolling(window=28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['roll_std_28'] = (
        df.groupby('Store')[target_col]
        .shift(1)
        .rolling(window=28, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def add_seasonality_features(df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
    """
    Add seasonality and trend features using lag values.

    Features use previously computed lag features, ensuring no leakage.

    Features:
    - diff_7: Sales difference from 7 days ago
    - ratio_7: Sales ratio compared to 7 days ago
    - trend_7: Daily trend over 7 days
    - diff_14: Sales difference from 14 days ago
    - ratio_14: Sales ratio compared to 14 days ago
    - trend_14: Daily trend over 14 days
    - roll_mean_ratio_7_28: Ratio of 7-day to 28-day rolling means
    - roll_mean_diff_7_28: Difference between 7-day and 28-day rolling means

    Args:
        df: DataFrame with lag_7, lag_14, roll_mean_7, roll_mean_28 already computed
        target_col: Name of target column (default: 'Sales')

    Returns:
        pd.DataFrame: DataFrame with seasonality features added
    """
    df = df.copy()

    # 7-day seasonality features
    df['diff_7'] = df[target_col] - df['lag_7']
    df['ratio_7'] = df[target_col] / df['lag_7']
    df['ratio_7'] = df['ratio_7'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df['trend_7'] = df['diff_7'] / 7.0

    # 14-day seasonality features
    df['diff_14'] = df[target_col] - df['lag_14']
    df['ratio_14'] = df[target_col] / df['lag_14']
    df['ratio_14'] = df['ratio_14'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df['trend_14'] = df['diff_14'] / 14.0

    # Rolling mean comparisons
    df['roll_mean_ratio_7_28'] = df['roll_mean_7'] / df['roll_mean_28']
    df['roll_mean_ratio_7_28'] = df['roll_mean_ratio_7_28'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df['roll_mean_diff_7_28'] = df['roll_mean_7'] - df['roll_mean_28']

    return df


def add_store_dow_mean_past(df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
    """
    Add historical average sales per store-weekday using only past data.

    CRITICAL: Uses expanding mean with shift(1) to ensure only past data is used.

    Feature:
    - store_dow_mean_past: Historical mean sales for this store on this weekday

    Args:
        df: DataFrame sorted by Store, Date with 'dow' and target_col
        target_col: Name of target column (default: 'Sales')

    Returns:
        pd.DataFrame: DataFrame with store_dow_mean_past feature added
    """
    df = df.copy()

    # Group by Store and dow, compute expanding mean of shifted values.
    # Use `apply` to run the expanding mean per-group, then align back to the
    # original DataFrame index by dropping the group keys from the resulting
    # MultiIndex. This avoids calling `.reset_index(level=[0,1])` on a Series
    # that doesn't have those levels (which caused the IndexError).
    store_dow_mean = (
        df.groupby(['Store', 'dow'])[target_col]
        .apply(lambda s: s.shift(1).expanding().mean())
    )

    # When using groupby.apply the result is indexed by a MultiIndex
    # (Store, dow, original_index). Reset the first two levels to align to
    # the original row index, then assign back to the DataFrame.
    df['store_dow_mean_past'] = store_dow_mean.reset_index(level=[0, 1], drop=True)

    return df


def build_all_features(df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
    """
    Build all features in correct order, ensuring leakage-safe computation.

    Steps:
    1. Sort by Store, Date ascending (CRITICAL for temporal correctness)
    2. Add calendar features
    3. Add holiday features
    4. Add lag features
    5. Add rolling features
    6. Add seasonality features
    7. Add store-dow historical mean

    Args:
        df: Raw merged dataframe with Store, Date, Sales, and store metadata
        target_col: Name of target column (default: 'Sales')

    Returns:
        pd.DataFrame: DataFrame with all features added
    """
    print("Building features...")
    print(f"Input shape: {df.shape}")

    # CRITICAL: Sort by Store and Date to ensure temporal ordering
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Add features in order (some depend on previous features)
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_seasonality_features(df, target_col)
    df = add_store_dow_mean_past(df, target_col)

    # Count NaNs created by warm-up period
    nan_count = df.isnull().any(axis=1).sum()
    print(f"Features added. {nan_count:,} rows contain NaN (warm-up period)")
    print(f"Output shape: {df.shape}")

    return df


def get_feature_columns() -> List[str]:
    """
    Get list of all feature columns (excludes target and metadata columns).

    These columns will be used for model training and saved with the model bundle.

    Returns:
        List[str]: List of feature column names
    """
    # Calendar features
    calendar_features = ['dow', 'month', 'weekofyear', 'quarter', 'year', 'is_weekend']

    # Holiday features
    holiday_features = ['is_holiday']

    # Lag features
    lag_features = ['lag_1', 'lag_7', 'lag_14', 'lag_28']

    # Rolling features
    rolling_features = ['roll_mean_7', 'roll_std_7', 'roll_mean_28', 'roll_std_28']

    # Seasonality features
    seasonality_features = [
        'diff_7', 'ratio_7', 'trend_7',
        'diff_14', 'ratio_14', 'trend_14',
        'roll_mean_ratio_7_28', 'roll_mean_diff_7_28'
    ]

    # Historical patterns
    historical_features = ['store_dow_mean_past']

    # Exogenous signals (from raw data)
    exogenous_features = [
        'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment',
        'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'
    ]

    # Combine all feature columns
    all_features = (
        calendar_features +
        holiday_features +
        lag_features +
        rolling_features +
        seasonality_features +
        historical_features +
        exogenous_features
    )

    return all_features
