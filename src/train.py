"""
Main training script with rolling-origin validation and model comparison.

Orchestrates the complete training pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Rolling-origin validation with baselines
4. Model training and evaluation
5. Final model retraining on all data
6. Saving model and metrics

Usage:
    python -m src.train [--model_type lightgbm|xgboost]
"""

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import src.config as config
from src.data import download_data, load_train_data, load_store_data, merge_data, apply_business_rules
from src.features import build_all_features, get_feature_columns
from src.split import get_rolling_splits, split_data_by_dates
from src.models import SalesForecastModel
from src.baselines import seasonal_naive_forecast, moving_average_forecast
from src.evaluate import evaluate_model, evaluate_baseline, print_leaderboard


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Main training workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Rossmann sales forecasting model')
    parser.add_argument(
        '--model_type',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'xgboost'],
        help='Model type to use (default: lightgbm)'
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROSSMANN SALES FORECASTING - TRAINING PIPELINE")
    print("="*80)
    print(f"Model type: {args.model_type.upper()}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print("="*80 + "\n")

    # Step 1: Setup
    print("Step 1: Setup")
    set_seed(config.RANDOM_SEED)
    config.ensure_dirs()

    # Step 2: Data acquisition
    print("\nStep 2: Data Acquisition")
    if not download_data():
        print("ERROR: Data download failed. Exiting.")
        return

    # Step 3: Data loading
    print("\nStep 3: Data Loading")
    try:
        train_df = load_train_data()
        store_df = load_store_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Step 4: Data merging and business rules
    print("\nStep 4: Data Merging and Business Rules")
    df = merge_data(train_df, store_df)
    df = apply_business_rules(df)

    # Step 5: Feature engineering
    print("\nStep 5: Feature Engineering")
    df = build_all_features(df, target_col='Sales')
    feature_cols = get_feature_columns()

    # Filter to columns that exist in dataframe
    feature_cols_available = [col for col in feature_cols if col in df.columns]
    if len(feature_cols_available) < len(feature_cols):
        missing = set(feature_cols) - set(feature_cols_available)
        print(f"Warning: {len(missing)} features not found in data: {missing}")

    feature_cols = feature_cols_available
    print(f"Using {len(feature_cols)} features for modeling")

    # Drop rows with NaN in feature columns (warm-up period)
    print(f"\nShape before dropping NaN: {df.shape}")
    df_clean = df.dropna(subset=feature_cols).copy()
    print(f"Shape after dropping NaN: {df_clean.shape}")
    rows_dropped = len(df) - len(df_clean)
    print(f"Dropped {rows_dropped:,} rows with NaN ({rows_dropped/len(df)*100:.1f}%)")

    # Step 6: Rolling-origin validation
    print("\nStep 6: Rolling-Origin Validation")

    # Get fold splits
    splits = get_rolling_splits(
        df_clean,
        n_folds=config.N_FOLDS,
        val_window_days=config.VALIDATION_WINDOW_DAYS
    )

    # Prepare results storage
    fold_results = []

    # Select model parameters
    if args.model_type == 'lightgbm':
        model_params = config.LIGHTGBM_PARAMS
    else:
        model_params = config.XGBOOST_PARAMS

    # Iterate over folds
    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{config.N_FOLDS}")
        print(f"{'='*80}")

        # Split data
        train_df_fold, val_df_fold = split_data_by_dates(
            df_clean, train_start, train_end, val_start, val_end
        )

        print(f"Training samples: {len(train_df_fold):,}")
        print(f"Validation samples: {len(val_df_fold):,}")

        # Prepare X and y
        X_train = train_df_fold[feature_cols]
        y_train = train_df_fold['Sales']
        X_val = val_df_fold[feature_cols]
        y_val = val_df_fold['Sales']

        # Train model
        model = SalesForecastModel(
            model_type=args.model_type,
            model_params=model_params,
            use_log1p=config.USE_LOG1P
        )
        model.fit(X_train, y_train, feature_cols)

        # Evaluate model
        print("\nEvaluating model...")
        model_metrics = evaluate_model(model, X_val, y_val)
        print(f"  MAPE:  {model_metrics['mape']:.2f}%")
        print(f"  sMAPE: {model_metrics['smape']:.2f}%")
        print(f"  RMSE:  {model_metrics['rmse']:.0f}")

        # Store result
        fold_results.append({
            'fold': fold_idx + 1,
            'model_name': args.model_type,
            'mape': model_metrics['mape'],
            'smape': model_metrics['smape'],
            'rmse': model_metrics['rmse']
        })

        # Evaluate baselines
        print("\nEvaluating baselines...")

        # Seasonal naive baseline
        print("  Seasonal naive (lag-7)...")
        naive_metrics = evaluate_baseline(
            df_clean,
            seasonal_naive_forecast,
            val_start,
            val_end
        )
        print(f"    MAPE:  {naive_metrics['mape']:.2f}%")
        print(f"    sMAPE: {naive_metrics['smape']:.2f}%")
        print(f"    RMSE:  {naive_metrics['rmse']:.0f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'model_name': 'seasonal_naive',
            'mape': naive_metrics['mape'],
            'smape': naive_metrics['smape'],
            'rmse': naive_metrics['rmse']
        })

        # Moving average baseline
        print("  Moving average (7-day)...")
        ma_metrics = evaluate_baseline(
            df_clean,
            moving_average_forecast,
            val_start,
            val_end
        )
        print(f"    MAPE:  {ma_metrics['mape']:.2f}%")
        print(f"    sMAPE: {ma_metrics['smape']:.2f}%")
        print(f"    RMSE:  {ma_metrics['rmse']:.0f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'model_name': 'moving_average',
            'mape': ma_metrics['mape'],
            'smape': ma_metrics['smape'],
            'rmse': ma_metrics['rmse']
        })

    # Step 7: Save metrics
    print("\nStep 7: Saving Metrics")
    metrics_df = pd.DataFrame(fold_results)
    metrics_path = config.REPORTS_DIR / config.METRICS_FILE
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to {metrics_path}")

    # Print leaderboard
    print_leaderboard(metrics_df)

    # Step 8: Final model training
    print("\nStep 8: Final Model Training")
    print("Retraining model on all available data...")

    # Select best fold by MAPE
    model_metrics_df = metrics_df[metrics_df['model_name'] == args.model_type]
    best_fold_idx = model_metrics_df['mape'].idxmin()
    best_fold = model_metrics_df.loc[best_fold_idx, 'fold']
    best_mape = model_metrics_df.loc[best_fold_idx, 'mape']
    print(f"Best fold: {best_fold} (MAPE: {best_mape:.2f}%)")

    # Train on all data
    X_all = df_clean[feature_cols]
    y_all = df_clean['Sales']

    final_model = SalesForecastModel(
        model_type=args.model_type,
        model_params=model_params,
        use_log1p=config.USE_LOG1P
    )
    final_model.fit(X_all, y_all, feature_cols)

    # Save model
    model_path = config.MODELS_DIR / config.MODEL_FILE
    final_model.save(model_path)

    # Step 9: Save processed features
    print("\nStep 9: Saving Processed Features")
    features_path = config.DATA_PROCESSED_DIR / config.FEATURES_FILE
    df_clean.to_parquet(features_path, index=False)
    print(f"✓ Processed features saved to {features_path}")

    # Step 10: Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Features saved to: {features_path}")
    print(f"Best validation MAPE: {best_mape:.2f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
