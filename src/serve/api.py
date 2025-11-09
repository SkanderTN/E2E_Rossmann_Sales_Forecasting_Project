"""
FastAPI inference service with true recursive forecasting.

Implements recursive multi-step-ahead forecasting where predictions
feed into lag and rolling features for subsequent predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta
from typing import Dict, Any
from pathlib import Path

from src.models import SalesForecastModel
from src.serve.schemas import PredictRequest, PredictResponse, ForecastPoint, HealthResponse
import src.config as config


# Global state
app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Recursive forecasting API for daily store-level sales predictions",
    version="1.0.0"
)

model: SalesForecastModel = None
history_df: pd.DataFrame = None
store_metadata: Dict[int, Dict[str, Any]] = {}
de_holidays = holidays.Germany()


@app.on_event("startup")
async def load_model_and_data():
    """Load model and historical data on API startup."""
    global model, history_df, store_metadata

    print("Loading model and data...")

    # Load model
    model_path = config.MODELS_DIR / config.MODEL_FILE
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please run training first: python -m src.train")
        model = None
        return

    try:
        model = SalesForecastModel.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None
        return

    # Load historical data
    features_path = config.DATA_PROCESSED_DIR / config.FEATURES_FILE
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        print("Please run training first: python -m src.train")
        history_df = None
        return

    try:
        history_df = pd.read_parquet(features_path)
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        print(f"✓ Historical data loaded: {len(history_df):,} rows")
    except Exception as e:
        print(f"ERROR loading historical data: {e}")
        history_df = None
        return

    # Cache store metadata
    store_cols = ['StoreType', 'Assortment', 'CompetitionDistance',
                  'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                  'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    existing_cols = [col for col in store_cols if col in history_df.columns]

    for store_id in history_df['Store'].unique():
        store_data = history_df[history_df['Store'] == store_id].iloc[0]
        store_metadata[store_id] = {col: store_data[col] for col in existing_cols}

    print(f"✓ Cached metadata for {len(store_metadata)} stores")
    print("API ready!\n")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model is not None and history_df is not None:
        return HealthResponse(status="ok", model_loaded=True)
    else:
        return HealthResponse(status="error", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate sales forecasts using recursive forecasting.

    For each forecast day:
    1. Compute calendar and holiday features
    2. Compute lag features from rolling buffer (including previous predictions)
    3. Compute rolling features from buffer
    4. Compute seasonality features
    5. Predict sales
    6. Update rolling buffer with prediction
    """
    # Check model is loaded
    if model is None or history_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure training has been completed."
        )

    # Validate store exists
    store_id = request.store_id
    if store_id not in store_metadata:
        raise HTTPException(
            status_code=400,
            detail=f"Store ID {store_id} not found in dataset"
        )

    # Parse start date
    try:
        requested_start = pd.to_datetime(request.start_date)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {request.start_date}. Use YYYY-MM-DD"
        )

    # Get last history date for this store
    store_history = history_df[history_df['Store'] == store_id].copy()
    if len(store_history) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No historical data for store {store_id}"
        )

    last_history_date = store_history['Date'].max()

    # Determine actual forecast start (must be after last history)
    actual_start = max(last_history_date + timedelta(days=1), requested_start)
    if actual_start > requested_start:
        print(f"Warning: Requested start {requested_start.date()} is before/at last history date. "
              f"Using {actual_start.date()} instead.")

    # Get recent history for initialization (last 56 days for lag_28 + buffer)
    recent_cutoff = last_history_date - timedelta(days=56)
    recent_history = store_history[store_history['Date'] >= recent_cutoff].copy()
    recent_history = recent_history.sort_values('Date').reset_index(drop=True)

    # Initialize rolling buffer with recent sales
    rolling_buffer = recent_history[['Date', 'Sales']].copy()

    # Get store metadata
    store_meta = store_metadata[store_id]

    # Generate forecasts
    forecasts = []

    for day_offset in range(request.horizon):
        current_date = actual_start + timedelta(days=day_offset)

        # Create row for current date
        row = {'Store': store_id, 'Date': current_date}

        # Add store metadata
        row.update(store_meta)

        # Step B: Calendar features
        row['dow'] = current_date.dayofweek
        row['month'] = current_date.month
        row['weekofyear'] = current_date.isocalendar()[1]
        row['quarter'] = current_date.quarter
        row['year'] = current_date.year
        row['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0

        # Step C: Holiday features
        # Check if StateHoliday exists in row, default to '0'
        state_holiday = row.get('StateHoliday', '0')
        row['is_holiday'] = 1 if (current_date in de_holidays or state_holiday != '0') else 0

        # Step D: Apply extra overrides
        if request.extra:
            for key, value in request.extra.items():
                row[key] = value

        # Step E: Compute lag features from buffer
        buffer_sales = rolling_buffer['Sales'].values
        buffer_len = len(buffer_sales)

        row['lag_1'] = buffer_sales[-1] if buffer_len >= 1 else np.nan
        row['lag_7'] = buffer_sales[-7] if buffer_len >= 7 else np.nan
        row['lag_14'] = buffer_sales[-14] if buffer_len >= 14 else np.nan
        row['lag_28'] = buffer_sales[-28] if buffer_len >= 28 else np.nan

        # Step F: Compute rolling features from buffer
        if buffer_len >= 7:
            row['roll_mean_7'] = np.mean(buffer_sales[-7:])
            row['roll_std_7'] = np.std(buffer_sales[-7:])
        else:
            row['roll_mean_7'] = np.mean(buffer_sales) if buffer_len > 0 else np.nan
            row['roll_std_7'] = np.std(buffer_sales) if buffer_len > 0 else np.nan

        if buffer_len >= 28:
            row['roll_mean_28'] = np.mean(buffer_sales[-28:])
            row['roll_std_28'] = np.std(buffer_sales[-28:])
        else:
            row['roll_mean_28'] = np.mean(buffer_sales) if buffer_len > 0 else np.nan
            row['roll_std_28'] = np.std(buffer_sales) if buffer_len > 0 else np.nan

        # Step G: Compute seasonality features
        lag_7 = row['lag_7']
        lag_14 = row['lag_14']
        roll_mean_7 = row['roll_mean_7']
        roll_mean_28 = row['roll_mean_28']

        if not np.isnan(lag_7):
            row['diff_7'] = row['lag_1'] - lag_7
            row['ratio_7'] = row['lag_1'] / lag_7 if lag_7 != 0 else 1.0
            row['trend_7'] = row['diff_7'] / 7.0
        else:
            row['diff_7'] = np.nan
            row['ratio_7'] = 1.0
            row['trend_7'] = np.nan

        if not np.isnan(lag_14):
            row['diff_14'] = row['lag_1'] - lag_14
            row['ratio_14'] = row['lag_1'] / lag_14 if lag_14 != 0 else 1.0
            row['trend_14'] = row['diff_14'] / 14.0
        else:
            row['diff_14'] = np.nan
            row['ratio_14'] = 1.0
            row['trend_14'] = np.nan

        if not np.isnan(roll_mean_7) and not np.isnan(roll_mean_28) and roll_mean_28 != 0:
            row['roll_mean_ratio_7_28'] = roll_mean_7 / roll_mean_28
            row['roll_mean_diff_7_28'] = roll_mean_7 - roll_mean_28
        else:
            row['roll_mean_ratio_7_28'] = 1.0
            row['roll_mean_diff_7_28'] = 0.0

        # Step H: Compute store_dow_mean_past
        # Get historical mean for this store and dow from history
        dow = row['dow']
        store_dow_history = store_history[store_history['dow'] == dow]['Sales']
        if len(store_dow_history) > 0:
            row['store_dow_mean_past'] = store_dow_history.mean()
        else:
            row['store_dow_mean_past'] = store_history['Sales'].mean()

        # Step I: Create DataFrame and align columns
        row_df = pd.DataFrame([row])

        # Step J: Predict
        try:
            y_pred = model.predict(row_df)[0]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error at date {current_date.date()}: {str(e)}"
            )

        # Ensure prediction is non-negative
        y_pred = max(y_pred, 0.0)

        # Step K: Update rolling buffer
        new_entry = pd.DataFrame({
            'Date': [current_date],
            'Sales': [y_pred]
        })
        rolling_buffer = pd.concat([rolling_buffer, new_entry], ignore_index=True)

        # Keep buffer manageable (last 56 days)
        if len(rolling_buffer) > 56:
            rolling_buffer = rolling_buffer.iloc[-56:].reset_index(drop=True)

        # Step L: Store forecast result
        forecasts.append(ForecastPoint(
            date=current_date.strftime('%Y-%m-%d'),
            yhat=round(y_pred, 2)
        ))

    return PredictResponse(store_id=store_id, forecasts=forecasts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
