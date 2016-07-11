"""A module for aiding in evaluating the results of models."""

import numpy as np

def intellegently_guess(ratios): 
    """Simulate the error from intellegently guessing (e.g. guessing the mean).

    Assume mean squared error.

    Args: 
    ----
        ratios: 1d np.ndarray of floats

    Return: 
    ------
        error: float
    """

    mean_ratio = ratios.mean()
    nobs = ratios.shape[0]
    error = np.sum((ratios - mean_ratio) ** 2) / nobs

    return error
