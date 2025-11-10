# E2E Rossmann Sales Forecasting Project

Production-style daily store-level sales forecasting system for Rossmann stores using LightGBM with true recursive forecasting.

## Problem Statement

Forecast daily sales for each Rossmann store with a 28-day horizon. The system includes:
- Training pipeline with leakage-safe features and rolling-origin validation
- FastAPI inference service with recursive forecasting
- Streamlit dashboard for interactive visualization
- Comprehensive test suite

## Dataset

**Source:** Kaggle Rossmann Store Sales competition
- `train.csv`: Historical daily sales per store
- `store.csv`: Store metadata (type, assortment, competition, etc.)

**Business rules:**
- Filter to Open == 1 (stores that are open)
- Filter to Sales > 0 (for metric stability)
- Granularity: Daily per store

## Features

All features are strictly leakage-safe (use only past data):

**Calendar features:**
- dow, month, weekofyear, quarter, year, is_weekend

**Holiday features:**
- is_holiday (Germany holidays + StateHoliday)

**Lag features (shifted):**
- lag_1, lag_7, lag_14, lag_28

**Rolling features (shifted before rolling):**
- roll_mean_7, roll_std_7, roll_mean_28, roll_std_28

**Seasonality features:**
- diff_7, ratio_7, trend_7
- diff_14, ratio_14, trend_14
- roll_mean_ratio_7_28, roll_mean_diff_7_28

**Historical patterns:**
- store_dow_mean_past (expanding mean per store-weekday)

**Exogenous signals:**
- Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, etc.

## Model

**Primary:** LightGBM (default)
**Alternative:** XGBoost (via CLI flag)

**Preprocessing:**
- Numeric: Median imputation
- Categorical: Most-frequent imputation + OneHotEncoding (handle_unknown='ignore')

**Target transformation:** log1p (inverted with expm1 for metrics and inference)

**Hyperparameters:**
- n_estimators: 500
- learning_rate: 0.05
- max_depth: 7
- Other params in src/config.py

## Validation Strategy

**Rolling-origin backtesting:**
- 3 folds
- Each validation window: 28 days
- Training: All data strictly before validation window
- No overlap between folds

**Baselines for comparison:**
- Seasonal naive (t-7 lag)
- 7-day moving average

**Metrics:**
- Primary: MAPE (Mean Absolute Percentage Error)
- Secondary: sMAPE, RMSE

## Installation

### Prerequisites
- Python 3.9+
- Kaggle account (for data download)

### Setup

1. Clone repository:
```bash
git clone <repo-url>
cd E2E_Rossmann_Sales_Forecasting_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials (optional for auto-download):
```bash
mkdir ~/.kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### 1. Training

Run training pipeline:
```bash
python -m src.train
```

With XGBoost:
```bash
python -m src.train --model_type xgboost
```

**Outputs:**
- `models/model.pkl`: Trained model bundle
- `reports/metrics.csv`: Fold-wise metrics for all models
- `data/processed/rossmann_features.parquet`: Feature-engineered dataset

**Expected results:**
- LightGBM MAPE: ~10-15% (should beat seasonal naive)
- Training time: ~5-15 minutes (depending on hardware)

### 2. Inference API

Start FastAPI service:
```bash
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

`GET /health`
```bash
curl http://localhost:8000/health
```

`POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "start_date": "2015-08-01",
    "horizon": 28
  }'
```

With overrides:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "start_date": "2015-08-01",
    "horizon": 28,
    "extra": {"Promo": 1}
  }'
```

### 3. Dashboard

Start Streamlit dashboard:
```bash
streamlit run src/app/dashboard.py
```

Access at: http://localhost:8501

**Features:**
- Select store and forecast horizon
- View 180-day historical sales
- Generate and visualize 28-day forecasts
- Export forecasts as CSV
- Saved plots in `reports/` directory

### 4. Docker Deployment

Build and run with docker-compose:
```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

Stop services:
```bash
docker-compose down
```

## Project Structure

```
E2E_Rossmann_Sales_Forecasting_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/              - train.csv, store.csv
│   ├── interim/          - intermediate processing artifacts
│   └── processed/        - rossmann_features.parquet
│
├── models/               - model.pkl bundle
│
├── reports/              - metrics.csv, plots (PNG files)
│
├── notebooks/            - optional exploration notebooks
│
├── src/
│   ├── __init__.py
│   ├── config.py         - central configuration
│   ├── data.py           - data loading & merging
│   ├── features.py       - feature engineering
│   ├── baselines.py      - seasonal naive, moving average
│   ├── metrics.py        - MAPE, sMAPE, RMSE
│   ├── split.py          - rolling-origin splits
│   ├── models.py         - LightGBM/XGBoost wrapper
│   ├── train.py          - main training script
│   ├── evaluate.py       - evaluation utilities
│   │
│   ├── serve/
│   │   ├── __init__.py
│   │   ├── api.py        - FastAPI endpoints
│   │   └── schemas.py    - Pydantic request/response models
│   │
│   └── app/
│       ├── __init__.py
│       └── dashboard.py  - Streamlit application
│
└── tests/
    ├── __init__.py
    ├── test_metrics.py
    ├── test_features.py
    ├── test_split.py
    └── test_api.py
```

## License

MIT.
