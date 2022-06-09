import pytest
import numpy as np
from rerand.Randomisation import Randomisation


@pytest.fixture
def standard_rand():
    """Returns a plain vanilla randomisation"""
    x = np.random.normal(0, 1, 100)
    return Randomisation(
        covariates=x,
        distance_metric="Euclidean",
        tol=0.1,
        max_reps=100,
    )


@pytest.fixture
def impossible_rand():
    """Returns a randomisation with an
    unachievable tolerance"""
    x = np.random.normal(0, 100, 100)
    return Randomisation(
        covariates=x,
        distance_metric="Euclidean",
        tol=0.01,
        max_reps=10,
    )


def test_Randomisation_init(standard_rand):
    """Test initiation of Randomisation class"""
    assert len(standard_rand.covariates) == 100
    assert standard_rand.tol == 0.1
    assert standard_rand.distance_metric == "Euclidean"
    assert standard_rand.max_reps == 100


def test_randomise_impossible(impossible_rand):
    """Test does not randomise when tolerance too low"""
    assert impossible_rand.randomise() is None
