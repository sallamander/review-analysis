"""A module for aiding in evaluating the results of models."""

import numpy as np
from sklearn.metrics import confusion_matrix

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

def get_confusion_mat(y_true, y_pred):
    """Use sklearns `confusion_matrix` to return a user friendly confusion mat.

    Args:
    ----
        y_true: 1d np.ndarray
        y_pred: 1d np.ndarray

    Return:
    ------
        confusion_dct: dct
            cell_label (str) : n_obs (int) key-value pairs 
    """ 

    confusion_mat = confusion_matrix(y_true, y_pred)
    
    tn, tp = confusion_mat[0][0], confusion_mat[1][1]
    fn, fp = confusion_mat[1][0], confusion_mat[0][1] 

    confusion_dct = {'true_negatives': tn, 'true_positives': tp, 
                     'false_negatives': fn, 'false_positives': fp}
    
    return confusion_dct
