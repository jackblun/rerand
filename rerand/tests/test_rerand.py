import pytest
import numpy as np
from rerand.Randomisation import Randomisation


@pytest.fixture
def standard_rand():
    """Returns a plain vanilla randomisation"""
    x = np.random.normal(0, 1, 10)
    variants = {"control": 0.5, "treatment": 0.5}
    seeds = range(100)
    return Randomisation(
        covariates=x,
        distance_metric="Euclidean",
        tol=1,
        max_reps=100,
        variants=variants,
        seeds=seeds,
    )


@pytest.fixture
def impossible_rand():
    """Returns a randomisation with an
    unachievable tolerance"""
    x = np.random.normal(0, 100, 100)
    variants = {"control": 0.5, "treatment": 0.5}
    return Randomisation(
        covariates=x,
        distance_metric="Euclidean",
        tol=0.01,
        max_reps=10,
        variants=variants,
    )


@pytest.fixture
def multivar_rand():
    """Returns a randomisation with multiple variants"""
    x = np.random.normal(0, 1, 1000)
    variants = {"a": 0.5, "b": 0.3, "c": 0.2}
    return Randomisation(
        covariates=x,
        distance_metric="Euclidean",
        tol=0.5,
        max_reps=100,
        variants=variants,
    )


def test_Randomisation_init(standard_rand):
    """Test initiation of Randomisation class"""
    assert len(standard_rand.covariates) == 10
    assert standard_rand.tol == 1
    assert standard_rand.distance_metric == "Euclidean"
    assert standard_rand.max_reps == 100


def test_Randomisation_seeds(standard_rand):
    """Test returns expected randomisation"""
    expected = [
        "treatment",
        "treatment",
        "treatment",
        "treatment",
        "control",
        "treatment",
        "control",
        "treatment",
        "treatment",
        "control",
    ]
    result = standard_rand.randomise()
    assert np.array_equal(result, expected)


def test_randomise_impossible(impossible_rand):
    """Test does not randomise when tolerance too low"""
    assert impossible_rand.randomise() is None


def test_randomise_multivar(multivar_rand):
    """Test produces randomisation for multiple variants"""
    assert multivar_rand.randomise() is not None
