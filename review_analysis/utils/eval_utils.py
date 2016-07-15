"""A module for aiding in evaluating the results of models."""

import numpy as np
from sklearn.metrics import confusion_matrix

def intellegently_guess(ratios, classification=False, pred=None): 
    """Simulate error as if intellegently guessing (e.g. the mean/majority class).

    Assume mean squared error if not the classification case. 

    Args: 
    ----
        ratios: 1d np.ndarray of floats
        classification (optional): bool
        pred (optional): float
            mean to use as the prediction in a regression setting 

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

class ConfusionMatrix(object): 
    """A utility class for working with a confusion matrix and associated stats."""

    def fit(self, y_true, y_pred): 
        """Store the inputted obs. and generate a confusion matrix.

        Args:
        ----
            y_true: 1d np.ndarray
            y_pred: 1d np.ndarray
        """

        self.y_true, self.y_pred = y_true, y_pred
        self.confusion_mat = confusion_matrix(y_true, y_pred)

        tn, tp = self.confusion_mat[0][0], self.confusion_mat[1][1]
        fn, fp = self.confusion_mat[1][0], self.confusion_mat[0][1] 
        self.confusion_dct = {'true_negatives': tn, 'true_positives': tp, 
                              'false_negatives': fn, 'false_positives': fp}

    def get_cell_counts(self, subsets=None):
        """Return back a user friendly version/subset of the confusion matrix.

        Args:
        ----
            subsets (optional): list or other iterable
                iterable of one or more of <true/false>_<positives/negatives>

        Return:
        ------
            dictionary of cell_label (str) : n_obs (int) pairs
        """

        if not hasattr(self, 'confusion_mat'): 
            raise RuntimeError("Must call `fit` method of ConfusionMatrix first!")
        
        if subsets: 
            confusion_counts = {}
            for subset in subsets:
                confusion_counts[subset] = self.confusion_dct[subset]
            return confusion_counts
        else: 
            return self.confusion_dct

    def get_cell_obs(self, subsets=None):
        """Generate a mask for obs. from given cells of the confusion matrix.

        Args:
        ----
            subsets (optional): list or other iteratble
                iterable of one or more of <true/false>_<positives/negatives>

        Return:
        ------
            confusion_obs: dct
                cell_label (str): obs (1d np.ndarray) pairs
        """
            
        if not hasattr(self, 'confusion_mat'): 
            raise RuntimeError("Must call `fit` method of ConfusionMatrix first!")

        correct_mask = (self.y_true == self.y_pred)
        incorrect_mask = (self.y_true != self.y_pred)

        tp_mask = np.logical_and(correct_mask, self.y_true == 1)
        tn_mask = np.logical_and(correct_mask, self.y_true == 0)
        fn_mask = np.logical_and(incorrect_mask, self.y_true == 1)
        fp_mask = np.logical_and(incorrect_mask, self.y_true == 0)

        confusion_mask_dct = {'true_negatives': tn_mask, 
                              'true_positives': tp_mask, 
                              'false_negatives': fn_mask, 
                              'false_positives': fp_mask}
        
        confusion_obs = {} if subsets else confusion_mask_dct
        if subsets: 
            for subset in subsets:
                confusion_obs[subset] = confusion_mask_dct[subset]
            return confusion_obs
        else: 
            return confusion_obs 
