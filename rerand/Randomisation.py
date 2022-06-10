import numpy as np
from rerand.utils.data_checks import (
    check_data,
    check_distance_metric,
    check_max_reps,
    check_tol,
)
import logging

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

    Methods
    -------
    randomise:
        Perform full (re)randomisation process.
    distance:
        Calculate distance between groups.
    check_inputs:
        Verify inputs are valid
    """

    def __init__(self, covariates, distance_metric, tol, max_reps):
        logging.info("Initialising Randomisation class")
        self.covariates = covariates
        self.tol = tol
        self.max_reps = max_reps
        self.distance_metric = distance_metric
        self.n = len(self.covariates)

        self.check_inputs()

    def randomise(self):
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
            t_p = np.random.uniform(low=0, high=1, size=self.n)
            t = t_p > 0.5

            x0 = self.covariates[~t]
            x1 = self.covariates[t]

            dist = self.distance(x0, x1)
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

    def distance(self, x0, x1, metric="Euclidean"):
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

    def check_inputs(self):
        check_tol(self.tol)
        check_max_reps(self.max_reps)
        check_distance_metric(self.distance_metric)
        check_data(self.covariates)
