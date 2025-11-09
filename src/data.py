"""
Data loading, validation, and merging utilities.

Functions for downloading data from Kaggle, loading train/store CSV files,
merging them, and applying business rules.
"""

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import DATA_RAW_DIR, TRAIN_FILE, STORE_FILE


def download_data() -> bool:
    """
    Attempt to download Rossmann data from Kaggle using Kaggle API.

    Returns:
        bool: True if data files exist (either already present or successfully downloaded),
              False if download failed.
    """
    train_path = DATA_RAW_DIR / TRAIN_FILE
    store_path = DATA_RAW_DIR / STORE_FILE

    # Check if files already exist
    if train_path.exists() and store_path.exists():
        print(f"Data files already exist in {DATA_RAW_DIR}")
        return True

    print("Data files not found. Attempting to download from Kaggle...")

    # Check for Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n" + "="*80)
        print("ERROR: Kaggle credentials not found!")
        print("="*80)
        print("\nPlease download data manually:")
        print("1. Go to: https://www.kaggle.com/c/rossmann-store-sales/data")
        print("2. Download train.csv and store.csv")
        print(f"3. Place them in: {DATA_RAW_DIR.absolute()}")
        print("\nAlternatively, set up Kaggle API:")
        print("1. Create a Kaggle account and generate API token")
        print("2. Download kaggle.json from your Kaggle account settings")
        print("3. Place it in: ~/.kaggle/kaggle.json")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("="*80 + "\n")
        return False

    try:
        # Download competition files
        print("Downloading from Kaggle (this may take a few minutes)...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", "rossmann-store-sales", "-p", str(DATA_RAW_DIR)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"Kaggle download failed: {result.stderr}")
            print("Please download data manually (see instructions above)")
            return False

        # Extract zip files
        print("Extracting files...")
        zip_path = DATA_RAW_DIR / "rossmann-store-sales.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW_DIR)
            zip_path.unlink()  # Delete zip file after extraction

        # Verify files exist
        if train_path.exists() and store_path.exists():
            print(f"✓ Data downloaded successfully to {DATA_RAW_DIR}")
            return True
        else:
            print("Download completed but expected files not found")
            return False

    except subprocess.TimeoutExpired:
        print("Kaggle download timed out. Please try again or download manually.")
        return False
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install: pip install kaggle")
        print("Or download data manually (see instructions above)")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        print("Please download data manually (see instructions above)")
        return False


def load_train_data() -> pd.DataFrame:
    """
    Load training data from train.csv.

    Returns:
        pd.DataFrame: Training data with parsed dates and correct dtypes.

    Raises:
        FileNotFoundError: If train.csv doesn't exist.
    """
    train_path = DATA_RAW_DIR / TRAIN_FILE

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please run download_data() first or download manually."
        )

    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)

    # Parse date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Convert types
    df['Store'] = df['Store'].astype(int)
    df['Sales'] = df['Sales'].astype(float)
    df['Customers'] = df['Customers'].astype(float)
    df['Open'] = df['Open'].astype(int)
    df['Promo'] = df['Promo'].astype(str)
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['SchoolHoliday'] = df['SchoolHoliday'].astype(str)

    print(f"Loaded {len(df):,} rows, date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique stores: {df['Store'].nunique()}")

    return df


def load_store_data() -> pd.DataFrame:
    """
    Load store metadata from store.csv.

    Returns:
        pd.DataFrame: Store metadata with correct dtypes.

    Raises:
        FileNotFoundError: If store.csv doesn't exist.
    """
    store_path = DATA_RAW_DIR / STORE_FILE

    if not store_path.exists():
        raise FileNotFoundError(
            f"Store data not found at {store_path}. "
            f"Please run download_data() first or download manually."
        )

    print(f"Loading store data from {store_path}...")
    df = pd.read_csv(store_path)

    # Convert types
    df['Store'] = df['Store'].astype(int)
    df['StoreType'] = df['StoreType'].astype(str)
    df['Assortment'] = df['Assortment'].astype(str)
    df['CompetitionDistance'] = df['CompetitionDistance'].astype(float)  # Allow NaN
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].astype(float)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].astype(float)
    df['Promo2'] = df['Promo2'].astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].astype(float)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(float)

    print(f"Loaded metadata for {len(df)} stores")

    return df


def merge_data(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge training data with store metadata.

    Args:
        train_df: Training data from load_train_data()
        store_df: Store metadata from load_store_data()

    Returns:
        pd.DataFrame: Merged dataframe with both sales history and store attributes.

    Raises:
        ValueError: If any stores in train data are missing from store metadata.
    """
    print("Merging train and store data...")

    # Check for missing stores
    train_stores = set(train_df['Store'].unique())
    store_stores = set(store_df['Store'].unique())
    missing_stores = train_stores - store_stores

    if missing_stores:
        raise ValueError(
            f"Found {len(missing_stores)} stores in training data not present in store metadata: "
            f"{sorted(missing_stores)[:10]}{'...' if len(missing_stores) > 10 else ''}"
        )

    # Left merge (all training records should find a store match)
    merged_df = train_df.merge(store_df, on='Store', how='left')

    # Verify no data loss
    assert len(merged_df) == len(train_df), "Row count changed during merge!"

    print(f"Merged data: {len(merged_df):,} rows with {len(merged_df.columns)} columns")

    return merged_df


def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business rules to filter data.

    Business rules:
    - Keep only Open == 1 (stores that are open)
    - Keep only Sales > 0 (for metric stability)

    Args:
        df: Merged dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    print("Applying business rules...")
    print(f"Before filtering: {len(df):,} rows")

    # Filter to open stores
    df_filtered = df[df['Open'] == 1].copy()
    print(f"After Open == 1 filter: {len(df_filtered):,} rows")

    # Filter to positive sales
    df_filtered = df_filtered[df_filtered['Sales'] > 0].copy()
    print(f"After Sales > 0 filter: {len(df_filtered):,} rows")

    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)

    rows_removed = len(df) - len(df_filtered)
    pct_removed = (rows_removed / len(df)) * 100
    print(f"Removed {rows_removed:,} rows ({pct_removed:.1f}%)")

    return df_filtered
