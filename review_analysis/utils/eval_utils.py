"""A module for aiding in evaluating the results of models."""

import numpy as np

def intellegently_guess(ratios, pred=None): 
    """Simulate the error from intellegently guessing (e.g. guessing the mean).

    Assume mean squared error.

    Args: 
    ----
        ratios: 1d np.ndarray of floats
        pred (optional): float
            mean to use as the prediction

    Return: 
    ------
        error: float
    """
    
    if not pred: 
        pred = ratios.mean()
    nobs = ratios.shape[0]
    error = np.sum((ratios - pred) ** 2) / nobs

    return error
