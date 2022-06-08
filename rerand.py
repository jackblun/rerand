import numpy as np
from utils.data_checks import (
    check_data,
    check_distance_metric,
    check_max_reps,
    check_tol,
)


class Randomisation:
    def __init__(self, covariates, distance_metric, tol, max_reps):
        self.covariates = covariates
        self.tol = tol
        self.max_reps = max_reps
        self.distance_metric = distance_metric
        self.n = len(self.covariates)

        self.check_inputs()

    def randomise(self):

        for i in range(self.max_reps):
            t_p = np.random.uniform(low=0, high=1, size=self.n)
            t = t_p > 0.5

            x0 = self.covariates[~t]
            x1 = self.covariates[t]

            dist = self.distance(x0, x1)
            if dist < self.tol:
                print(
                    str(i) + " randomisations needed to achieve balance"
                    " with tolerance " + str(self.tol)
                )
                return t
        print("Did not achieve balance for tolerance " + str(self.tol))

    def distance(self, x0, x1, metric="Euclidean"):
        """
        Method that calculates distance between two NxK matrices,
        where N is the number of observations and K is the
        number of variables

        Params:
        - x0: NxK matrix of covariates from control group
        - x1: NXK matrix of covaraites from treatment group
        - metric: Distance metric used. Supported options are Euclidean.

        Returns:
        - distance: Distance between two matrices
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
