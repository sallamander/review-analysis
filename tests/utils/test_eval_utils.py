import pytest
import numpy as np
from review_analysis.utils.eval_utils import intellegently_guess

class TestEvalUtils: 

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
