"""A module for aiding in evaluating the results of models."""

import numpy as np

def intellegently_guess(ratios, classification=False, pred=None): 
    """Simulate the error from intellegently guessing (e.g. guessing the mean).

    Assume mean squared error.

    Args: 
    ----
        ratios: 1d np.ndarray of floats
        classification (optional): bool
        pred (optional): float
            mean to use as the prediction

    Return: 
    ------
        error: float
    """
    
    if not pred: 
        pred = ratios.mean()
    nobs = ratios.shape[0]
    if classification:
        # Case of predicting on a 2d array meant for Keras model
        if len(ratios.shape) == 2:
            ratios = ratios[:, 1]
        error = ratios.mean() # this is really accuracy
    else: 
        error = np.sum((ratios - pred) ** 2) / nobs

    return error
