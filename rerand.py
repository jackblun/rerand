import numpy as np


def distance(x0, x1, metric="Euclidean"):
    """
    Function that calculates distance between two NxK matrices,
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
