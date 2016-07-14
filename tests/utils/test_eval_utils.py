import pytest
import numpy as np
from review_analysis.utils.eval_utils import intellegently_guess, \
        ConfusionMatrix
        

class TestEvalUtils: 

    def setup_class(cls):

        y_true = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0])

        cls.confucius = ConfusionMatrix()
        cls.confucius.fit(y_true, y_pred)

    def teardown_class(cls):

        del cls.confucius

    def test_intellegently_guess(self): 
        eps = 1e-3

        ratios = np.array([[0.10], [0.20], [0.30]])
        error = intellegently_guess(ratios)
        assert abs(error - 0.02 / 3) < eps 

        ratios = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
        # this technically returns accuracy
        error = intellegently_guess(ratios, classification=True)
        assert abs(error -  0.75) < eps

        ratios = np.array([[0, 1], [1, 0], [0, 1]])
        error = intellegently_guess(ratios, classification=True)
        assert abs(error -  0.666) < eps

        ratios = np.array([[0.01], [0.05], [0.12]])
        error = intellegently_guess(ratios, pred=0.03)
        assert abs(error - 0.0089 / 3) < eps

    def test_fit(self):

        assert (self.confucius.y_true is not None)
        assert (self.confucius.y_pred is not None)
        assert (self.confucius.confusion_dct is not None)
        assert (len(self.confucius.confusion_dct) == 4)

    def test_get_cell_counts(self): 

        confusion_dct = self.confucius.get_cell_counts()

        assert (confusion_dct['true_negatives'] == 1)
        assert (confusion_dct['true_positives'] == 2)
        assert (confusion_dct['false_positives'] == 3)
        assert (confusion_dct['false_negatives'] == 4)

        subsets = ['true_negatives', 'true_positives']
        confusion_dct = self.confucius.get_cell_counts(subsets)

        assert (len(confusion_dct) == 2)
        assert (confusion_dct['true_negatives'] == 1)
        assert (confusion_dct['true_positives'] == 2)

    def test_get_confusion_obs(self): 

        confusion_obs = self.confucius.get_cell_obs()
            
        assert (len(confusion_obs['true_negatives']) == 10)
        assert (len(confusion_obs['true_positives']) == 10)
        assert (len(confusion_obs['false_positives']) == 10)
        assert (len(confusion_obs['false_negatives']) == 10)
        assert (sum(confusion_obs['true_negatives']) == 1)
        assert (sum(confusion_obs['true_positives']) == 2)
        assert (sum(confusion_obs['false_positives']) == 3)
        assert (sum(confusion_obs['false_negatives']) == 4)

        subsets = ['true_negatives', 'true_positives']
        confusion_obs = self.confucius.get_cell_obs(subsets)

        assert (len(confusion_obs) == 2)
        assert (len(confusion_obs['true_negatives']) == 10)
        assert (len(confusion_obs['true_positives']) == 10)
        assert (sum(confusion_obs['true_negatives']) == 1)
        assert (sum(confusion_obs['true_positives']) == 2)
