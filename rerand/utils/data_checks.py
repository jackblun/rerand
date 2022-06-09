import numpy as np
import utils.constants as cons


def check_data(x):
    if sum(np.isnan(x)) > 0:
        raise ValueError("NaNs present in data.")


def check_distance_metric(distance_metric):
    if distance_metric not in cons.DISTANCE_METRICS:
        raise ValueError(
            "Unsupported distance metric. Currently supported metrics are "
            + "".join(cons.DISTANCE_METRICS)
        )


def check_max_reps(max_reps):
    if not isinstance(max_reps, int):
        raise ValueError("Max reps must be an integer")
    if max_reps < 1:
        raise ValueError("Max reps must not be less than 1")


def check_tol(tol):
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0")
