import pytest
import numpy as np
from review_analysis.utils.eval_utils import intellegently_guess

class TestEvalUtils: 

    def test_intellegently_guess(self): 
        eps = 1e-3

        ratios = np.array([[0.10], [0.20], [0.30]])
        error = intellegently_guess(ratios)
        assert abs(error - 0.02 / 3) < eps 

        ratios = np.array([[0.02], [0.04], [0.06]])
        error = intellegently_guess(ratios)
        assert abs(error - 0.0008 / 3) < eps

        ratios = np.array([[0.01], [0.05], [0.12]])
        error = intellegently_guess(ratios)
        assert abs(error - 0.0062 / 3) < eps
