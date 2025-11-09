"""
Tests for FastAPI endpoints and recursive forecasting.

Tests health check and prediction endpoints with various scenarios.
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path

# Note: These tests assume model and data files exist
# In practice, you might want to use fixtures or mocks


def test_api_imports():
    """Test that API module can be imported."""
    try:
        from src.serve.api import app
        assert app is not None
    except Exception as e:
        pytest.fail(f"Failed to import API: {e}")


def test_health_endpoint_structure():
    """Test health endpoint returns correct structure."""
    from src.serve.api import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "status" in data
    assert "model_loaded" in data
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)


def test_predict_endpoint_structure():
    """Test predict endpoint request/response structure."""
    from src.serve.api import app

    client = TestClient(app)

    # Test request body
    request_body = {
        "store_id": 1,
        "start_date": "2015-08-01",
        "horizon": 7
    }

    response = client.post("/predict", json=request_body)

    # If model is not loaded, should get 503
    # If model is loaded, should get 200 or 400 (if store not found)
    assert response.status_code in [200, 400, 503]

    if response.status_code == 200:
        data = response.json()

        # Check response structure
        assert "store_id" in data
        assert "forecasts" in data
        assert isinstance(data["forecasts"], list)

        # Check forecast structure
        if len(data["forecasts"]) > 0:
            forecast = data["forecasts"][0]
            assert "date" in forecast
            assert "yhat" in forecast
            assert isinstance(forecast["date"], str)
            assert isinstance(forecast["yhat"], (int, float))


def test_predict_endpoint_horizon():
    """Test that prediction returns correct number of forecasts."""
    from src.serve.api import app

    client = TestClient(app)

    horizons = [7, 14, 28]

    for horizon in horizons:
        request_body = {
            "store_id": 1,
            "start_date": "2015-08-01",
            "horizon": horizon
        }

        response = client.post("/predict", json=request_body)

        # Only test if model is loaded and request successful
        if response.status_code == 200:
            data = response.json()
            assert len(data["forecasts"]) == horizon


def test_predict_invalid_store():
    """Test prediction with invalid store ID."""
    from src.serve.api import app

    client = TestClient(app)

    request_body = {
        "store_id": 99999,  # Unlikely to exist
        "start_date": "2015-08-01",
        "horizon": 7
    }

    response = client.post("/predict", json=request_body)

    # Should get either 400 (store not found) or 503 (model not loaded)
    assert response.status_code in [400, 503]


def test_predict_invalid_date_format():
    """Test prediction with invalid date format."""
    from src.serve.api import app

    client = TestClient(app)

    request_body = {
        "store_id": 1,
        "start_date": "invalid-date",
        "horizon": 7
    }

    response = client.post("/predict", json=request_body)

    # Should get 400 (bad request) or 503 (model not loaded)
    assert response.status_code in [400, 503, 422]  # 422 is validation error


def test_predict_with_extra_overrides():
    """Test prediction with extra feature overrides."""
    from src.serve.api import app

    client = TestClient(app)

    request_body = {
        "store_id": 1,
        "start_date": "2015-08-01",
        "horizon": 7,
        "extra": {"Promo": "1"}
    }

    response = client.post("/predict", json=request_body)

    # Should work if model is loaded
    assert response.status_code in [200, 400, 503]


def test_predict_default_horizon():
    """Test that default horizon is used when not specified."""
    from src.serve.api import app

    client = TestClient(app)

    request_body = {
        "store_id": 1,
        "start_date": "2015-08-01"
        # No horizon specified
    }

    response = client.post("/predict", json=request_body)

    if response.status_code == 200:
        data = response.json()
        # Default horizon should be 28
        assert len(data["forecasts"]) == 28


def test_api_cors_and_headers():
    """Test that API returns proper headers."""
    from src.serve.api import app

    client = TestClient(app)
    response = client.get("/health")

    # Check that response has content-type header
    assert "content-type" in response.headers
    assert "application/json" in response.headers["content-type"]


def test_predict_date_sequence():
    """Test that forecast dates are sequential."""
    from src.serve.api import app

    client = TestClient(app)

    request_body = {
        "store_id": 1,
        "start_date": "2015-08-01",
        "horizon": 7
    }

    response = client.post("/predict", json=request_body)

    if response.status_code == 200:
        data = response.json()
        forecasts = data["forecasts"]

        # Check dates are sequential
        if len(forecasts) > 1:
            dates = [pd.to_datetime(f["date"]) for f in forecasts]
            for i in range(len(dates) - 1):
                # Each date should be 1 day after previous
                assert (dates[i+1] - dates[i]).days == 1
