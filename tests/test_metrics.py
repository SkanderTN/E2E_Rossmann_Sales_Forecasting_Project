"""
Tests for metric calculations and safeguards.

Validates MAPE, sMAPE, RMSE implementations with edge cases.
"""

import pytest
import numpy as np
from src.metrics import mape, smape, rmse, safe_metric


def test_mape_basic():
    """Test basic MAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    result = mape(y_true, y_pred)

    # Expected: (|100-110|/100 + |200-190|/200 + |300-310|/300) / 3 * 100
    # = (10/100 + 10/200 + 10/300) / 3 * 100
    # = (0.1 + 0.05 + 0.0333) / 3 * 100
    # ≈ 6.11%
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert not np.isinf(result)
    assert result > 0
    assert result < 10  # Should be around 6%


def test_mape_with_zeros():
    """Test MAPE with zero values in y_true (should be filtered out)."""
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 190])

    result = mape(y_true, y_pred)

    # Should compute only on [100, 200] entries
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert not np.isinf(result)


def test_mape_all_zeros():
    """Test MAPE when all y_true are zero (should return NaN)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([10, 20, 30])

    result = mape(y_true, y_pred)

    assert np.isnan(result)


def test_smape_basic():
    """Test basic sMAPE calculation."""
    y_true = np.array([100, 200])
    y_pred = np.array([110, 190])

    result = smape(y_true, y_pred)

    # Expected: (|100-110|/(100+110) + |200-190|/(200+190)) / 2 * 200
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert not np.isinf(result)
    assert 0 <= result <= 200


def test_rmse_basic():
    """Test basic RMSE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    result = rmse(y_true, y_pred)

    # Expected: sqrt(mean([10^2, 10^2, 10^2])) = sqrt(100) = 10
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert not np.isinf(result)
    assert np.isclose(result, 10.0, rtol=0.01)


def test_safe_metric_wrapper():
    """Test safe_metric wrapper with invalid inputs."""
    # Test with NaN in inputs
    y_true = np.array([100, np.nan, 200])
    y_pred = np.array([110, 190, 210])

    result = safe_metric(mape, y_true, y_pred)
    assert np.isnan(result)

    # Test with inf in inputs
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, np.inf, 210])

    result = safe_metric(mape, y_true, y_pred)
    assert np.isnan(result)

    # Test with valid inputs
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    result = safe_metric(mape, y_true, y_pred)
    assert not np.isnan(result)
    assert not np.isinf(result)


def test_metric_shapes():
    """Test that metrics handle correct shapes."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    mape_result = mape(y_true, y_pred)
    smape_result = smape(y_true, y_pred)
    rmse_result = rmse(y_true, y_pred)

    # All should return scalars
    assert isinstance(mape_result, (int, float))
    assert isinstance(smape_result, (int, float))
    assert isinstance(rmse_result, (int, float))


def test_metrics_no_exceptions():
    """Test that metrics don't raise exceptions with valid inputs."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([95, 205, 310, 390, 510])

    # Should not raise any exceptions
    try:
        mape(y_true, y_pred)
        smape(y_true, y_pred)
        rmse(y_true, y_pred)
        success = True
    except Exception:
        success = False

    assert success
