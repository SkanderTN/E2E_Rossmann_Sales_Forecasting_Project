"""
Model wrapper for LightGBM/XGBoost with preprocessing pipeline.

Handles numeric imputation, categorical encoding, and model training/prediction
with optional log1p target transformation.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


class SalesForecastModel:
    """
    Sales forecasting model with preprocessing pipeline.

    Wraps LightGBM or XGBoost with scikit-learn pipeline for preprocessing.
    Supports log1p target transformation and column alignment for inference.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_params: Dict[str, Any] = None,
        use_log1p: bool = True
    ):
        """
        Initialize model wrapper.

        Args:
            model_type: "lightgbm" or "xgboost"
            model_params: Hyperparameters for the model
            use_log1p: Whether to apply log1p transformation to target
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_log1p = use_log1p
        self.pipeline = None
        self.feature_cols = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]):
        """
        Fit the model on training data.

        Args:
            X: Training features (full dataframe)
            y: Training target (Sales)
            feature_cols: List of feature column names to use
        """
        print(f"\nTraining {self.model_type.upper()} model...")

        # Store feature columns
        self.feature_cols = feature_cols

        # Select feature columns from X
        X_features = X[feature_cols].copy()

        # Identify numeric and categorical columns
        self.numeric_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Features: {len(feature_cols)} ({len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical)")

        # Transform target if needed
        y_transformed = np.log1p(y) if self.use_log1p else y
        if self.use_log1p:
            print(f"Applied log1p transformation to target")

        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        # Add model to pipeline
        if self.model_type == "lightgbm":
            model = LGBMRegressor(**self.model_params)
        elif self.model_type == "xgboost":
            model = XGBRegressor(**self.model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit pipeline
        self.pipeline.fit(X_features, y_transformed)
        self.fitted = True

        print(f"✓ Model trained on {len(X_features):,} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features (full dataframe or just feature columns)

        Returns:
            np.ndarray: Predictions (on original scale if log1p was used)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Align columns to training features
        X_aligned = self._align_columns(X)

        # Predict
        predictions = self.pipeline.predict(X_aligned)

        # Inverse transform if log1p was used
        if self.use_log1p:
            predictions = np.expm1(predictions)

        return predictions

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align input columns to match training feature columns.

        Adds missing columns as NaN and drops extra columns.

        Args:
            X: Input dataframe

        Returns:
            pd.DataFrame: Aligned dataframe with exactly self.feature_cols
        """
        X_aligned = pd.DataFrame(index=X.index)

        for col in self.feature_cols:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                # Add missing column as NaN
                X_aligned[col] = np.nan

        return X_aligned

    def save(self, filepath: str):
        """
        Save model bundle to disk.

        Bundle includes:
        - pipeline: Fitted scikit-learn pipeline
        - feature_cols: List of feature columns
        - use_log1p: Target transformation flag
        - model_type: Model type used
        - categorical_cols: Categorical column names
        - numeric_cols: Numeric column names
        - train_date: Timestamp of training

        Args:
            filepath: Path to save model (e.g., "models/model.pkl")
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")

        bundle = {
            'pipeline': self.pipeline,
            'feature_cols': self.feature_cols,
            'use_log1p': self.use_log1p,
            'model_type': self.model_type,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'train_date': datetime.now().isoformat(),
            'n_features': len(self.feature_cols)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)

        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SalesForecastModel':
        """
        Load model bundle from disk.

        Args:
            filepath: Path to saved model

        Returns:
            SalesForecastModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            bundle = pickle.load(f)

        # Reconstruct model instance
        model = cls(
            model_type=bundle['model_type'],
            model_params={},  # Already fitted, params not needed
            use_log1p=bundle['use_log1p']
        )

        model.pipeline = bundle['pipeline']
        model.feature_cols = bundle['feature_cols']
        model.categorical_cols = bundle['categorical_cols']
        model.numeric_cols = bundle['numeric_cols']
        model.fitted = True

        print(f"✓ Model loaded from {filepath}")
        print(f"  Model type: {bundle['model_type']}")
        print(f"  Features: {bundle['n_features']}")
        print(f"  Log1p: {bundle['use_log1p']}")
        print(f"  Trained: {bundle.get('train_date', 'unknown')}")

        return model
