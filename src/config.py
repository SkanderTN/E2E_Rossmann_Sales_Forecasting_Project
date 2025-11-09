"""
Central configuration for Rossmann Sales Forecasting Project.

Contains all paths, hyperparameters, and constants used throughout the project.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# File names
TRAIN_FILE = "train.csv"
STORE_FILE = "store.csv"
FEATURES_FILE = "rossmann_features.parquet"
MODEL_FILE = "model.pkl"
METRICS_FILE = "metrics.csv"

# Validation parameters
N_FOLDS = 3
VALIDATION_WINDOW_DAYS = 28
FORECAST_HORIZON = 28

# Reproducibility
RANDOM_SEED = 42

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
}

# Model configuration
MODEL_TYPE = "lightgbm"  # default, can be overridden via CLI
USE_LOG1P = True  # transform target to log1p scale


def ensure_dirs():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Ensured all required directories exist under {BASE_DIR}")
