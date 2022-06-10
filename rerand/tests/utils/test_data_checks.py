import pytest
import numpy as np
from rerand.utils.data_checks import (
    check_data,
    check_distance_metric,
    check_tol,
    check_max_reps,
    check_variants,
)


def test_check_data():
    """Test nans result in error"""
    x = np.random.normal(0, 1, 100)
    x[0] = np.NaN
    with pytest.raises(ValueError):
        check_data(x)


def test_check_distance_metric():
    """Test non-supported distance metrics result in error"""
    with pytest.raises(ValueError):
        check_distance_metric("Geometric")


def test_check_max_reps():
    """Test invalid max reps cause error"""
    invalid_max_reps = [100.1, 0, -1]
    for i in invalid_max_reps:
        with pytest.raises(ValueError):
            check_max_reps(i)


def test_check_tol():
    """Test invalid tolerance causes error"""
    invalid_tols = [0, -1]
    for i in invalid_tols:
        with pytest.raises(ValueError):
            check_tol(i)


def test_check_variants():
    """Test invalid variant entry causes error"""
    invalid_variants = [{"a": 1}, {"a": 0.5, "b": 0.2}]
    for i in invalid_variants:
        with pytest.raises(ValueError):
            check_variants(i)
