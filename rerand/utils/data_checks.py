import numpy as np
import rerand.utils.constants as cons
from numpy.typing import ArrayLike


def check_data(x: ArrayLike) -> None:
    """
    Check covariates data is valid.

        Parameters
        ----------
        x : array-like
            array of covariates

        Returns
        -------
            None (error if checks fail)
    """
    if sum(np.isnan(x)) > 0:
        raise ValueError("NaNs present in data.")


def check_distance_metric(distance_metric: str) -> None:
    """
    Check distance metric is valid.

        Parameters
        ----------
        distance_metric : str
            chosen distance metric

        Returns
        -------
            None (error if checks fail)
    """
    if distance_metric not in cons.DISTANCE_METRICS:
        raise ValueError(
            "Unsupported distance metric. Currently supported metrics are "
            + "".join(cons.DISTANCE_METRICS)
        )
    return None


def check_max_reps(max_reps: float or int) -> None:
    """
    Check max repititions is valid.

        Parameters
        ----------
        max_reps : int or float
            chosen max repetitions

        Returns
        -------
            None (error if checks fail)
    """
    if not isinstance(max_reps, int):
        raise ValueError("Max reps must be an integer")
    if max_reps < 1:
        raise ValueError("Max reps must not be less than 1")


def check_tol(tol: float) -> None:
    """
    Check chosen tolerance is valid.

        Parameters
        ----------
        tol : float
            tolerance

        Returns
        -------
            None (error if checks fail)
    """
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0")


def check_variants(variants: dict) -> None:
    """
    Check variants are valid.

        Parameters
        ----------
        variants : dict
            dict of variant names and randomisation probabilities

        Returns
        -------
            None (error if checks fail)
    """
    if type(variants) is not dict:
        raise TypeError("Variants must be dictionary type.")
    if len(variants) == 1:
        raise ValueError("Must provide more than one variant.")
    if sum(variants.values()) != 1:
        raise ValueError("Randomisation probabilities must sum to 1.")
