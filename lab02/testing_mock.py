"""
EE-311
======

Lab 2: NumPy, Matplotlib and Scikit-learn (linear regression example)
---------------------------------------------------------------------

created by François Marelli on 03.03.2021
"""

import importlib
import unittest

import mock
import numpy as np
from sklearn import linear_model

importlib.reload(mock)


class Test(unittest.TestCase):
    def test_predict(self):
        model = linear_model.LinearRegression()
        model.coef_ = np.atleast_1d(2)
        model.intercept_ = 3

        data_X = np.array([1, 2, 3])[:, None]

        prediction = mock.predict(model, data_X)

        np.testing.assert_array_equal(prediction, [5, 7, 9])

    def test_sum(self):
        array = np.arange(10)
        self.assertEqual(mock.sum_bigger(array, 4), 35)


if __name__ == "__main__":
    unittest.main()
