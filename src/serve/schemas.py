"""
Pydantic models for FastAPI request/response validation.

Defines schemas for predict endpoint and health check.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""

    store_id: int = Field(..., description="Store ID to forecast for", ge=1)
    start_date: str = Field(..., description="Forecast start date in YYYY-MM-DD format")
    horizon: int = Field(28, description="Number of days to forecast", ge=1, le=56)
    extra: Optional[Dict[str, Any]] = Field(None, description="Optional overrides for features (e.g., {'Promo': 1})")

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": 1,
                "start_date": "2015-08-01",
                "horizon": 28,
                "extra": {"Promo": 1}
            }
        }


class ForecastPoint(BaseModel):
    """Single forecast point (date + prediction)."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    yhat: float = Field(..., description="Predicted sales value")


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    store_id: int = Field(..., description="Store ID")
    forecasts: List[ForecastPoint] = Field(..., description="List of forecast points")

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": 1,
                "forecasts": [
                    {"date": "2015-08-01", "yhat": 5263.45},
                    {"date": "2015-08-02", "yhat": 5124.32}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str = Field(..., description="API status ('ok' or 'error')")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True
            }
        }
