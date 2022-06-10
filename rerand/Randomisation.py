from multiprocessing.dummy import Array
import numpy as np
from rerand.utils.data_checks import (
    check_data,
    check_distance_metric,
    check_max_reps,
    check_tol,
)
import logging
from numpy.typing import ArrayLike
from itertools import combinations

logging.basicConfig(level=logging.NOTSET)


class Randomisation:
    """
    A class to represent a randomisation.

    Attributes
    ----------
    covariates : array-like
        Array of covariates
    distance_metric : str
        Chosen distance metric on which to assess balance
    tol : float
        Acceptable distance between groups
    max_reps : int or float
        Maximum number of repeated randomisations
    variants : dict of str: float
        Variant names and randomisation probabilities

    Methods
    -------
    randomise:
        Perform full (re)randomisation process.
    distance:
        Calculate distance between groups.
    check_inputs:
        Verify inputs are valid
    """

    def __init__(
        self,
        covariates: ArrayLike,
        distance_metric: str,
        tol: float,
        max_reps: float or int,
        variants: dict,
    ):
        logging.info("Initialising Randomisation class")
        self.covariates = covariates
        self.tol = tol
        self.max_reps = max_reps
        self.distance_metric = distance_metric
        self.n = len(self.covariates)
        self.variants = variants

        self.check_inputs()

    def randomise(self) -> np.ndarray:
        """
        Generate balanced assignment vector.

        The core method of this module. The following steps are taken:
        1. Randomise assignment.
        2. Check for balance according to distance metric and tolerance.
        3. Repeat steps 1 and 2 until balance achieved or maximum reps reached.
        4. If balance achieved, return assignment vector.

        Returns
        -------
        numpy.ndarray
            Vector of treatment assignments
        """
        for i in range(self.max_reps):

            variant_names = list(self.variants.keys())
            probabilities = list(self.variants.values())
            t = np.random.choice(variant_names, self.n, p=probabilities)

            covariates_by_t = []
            for vname in variant_names:
                covariates_by_t.append(self.covariates[t == vname])

            combos = list(combinations(covariates_by_t, 2))

            dists = []
            for combo in combos:
                dists.append(self.distance(combo[0], combo[1]))

            dist = max(dists)
            logging.info(
                "Randomisation: "
                + str(i + 1)
                + ", Distance = "
                + str(np.round(dist, 2))
            )

            if dist < self.tol:
                logging.info(
                    str(i + 1) + " randomisations needed to achieve balance"
                    " with tolerance " + str(self.tol)
                )
                return t
        logging.warning("Did not achieve balance, tolerance " + str(self.tol))

    def distance(self, x0: ArrayLike, x1: Array, metric: str = "Euclidean") -> float:
        """
        Calculate distance between two NxK numpy arrays.

        According to a specified distance metric, calculate the
        difference between two numpy arrays.
        where N is the number of observations and K is the
        number of variables

        Parameters
        ----------
        x0 : array-like
            NxK array of covariates from control group
        x1 : array-like
            NXK array of covariates from treatment group
        metric : str
            distance metric used

        Returns
        -------
        float
            Distance between two matrices
        """

        x0_bar = np.mean(x0, axis=0)
        x1_bar = np.mean(x1, axis=0)

        if metric == "Euclidean":
            sq_devs = np.power(x0_bar - x1_bar, 2)
            distance = np.sqrt(np.sum(sq_devs))

        return distance

    def check_inputs(self) -> None:
        check_tol(self.tol)
        check_max_reps(self.max_reps)
        check_distance_metric(self.distance_metric)
        check_data(self.covariates)
