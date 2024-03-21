"""
EE-311
======

Lab 3: logistic regression and gradient descent
----------------------------------------

created by Francois Marelli on 05.03.2020
"""

import unittest
import numpy as np
import homework

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import importlib

importlib.reload(homework)


class Test(unittest.TestCase):
    def test_model_prediction(self):
        data_X, data_y = load_iris(return_X_y=True)

        mask = (data_y == 0) | (data_y == 1)

        data_X = data_X[mask]
        data_y = data_y[mask]

        classifier = LogisticRegression().fit(data_X, data_y)
        beta = np.concatenate((classifier.intercept_, classifier.coef_[0, :])).squeeze()
        p = homework.model_prediction(data_X.copy(), beta.copy())
        p2 = classifier.predict(data_X)

        np.testing.assert_array_equal(p, p2)

        sample_X = np.zeros_like(data_X[0])
        sample_X[0] = (0.25 - beta[0]) / beta[1]
        np.testing.assert_array_equal(
            homework.model_prediction(sample_X[None, :], beta), 1
        )

    def test_gradient(self):
        data_X, data_y = load_iris(return_X_y=True)

        mask = (data_y == 0) | (data_y == 1)

        data_X = data_X[mask]
        data_y = data_y[mask]

        data_X = data_X[:-5]
        data_y = data_y[:-5]

        grad_ref = np.load("test_data.npy")

        beta_0 = np.zeros_like(grad_ref)
        grad_out = homework.logistic_loss_derivative(
            data_X.copy(), data_y.copy(), beta_0
        )

        np.testing.assert_allclose(grad_out, grad_ref, atol=1e-9, rtol=1e-5)

    def test_gradient_descent(self):
        data_X, data_y = load_iris(return_X_y=True)

        mask = (data_y == 0) | (data_y == 1)

        data_X = data_X[mask]
        data_y = data_y[mask]

        classifier = LogisticRegression().fit(data_X, data_y)
        beta = np.concatenate((classifier.intercept_, classifier.coef_[0, :]))

        beta_0 = np.zeros_like(beta)
        beta_out = homework.gradient_descent(
            data_X.copy(), data_y.copy(), beta_0, 1e-1, 5
        )

        p = homework.model_prediction(data_X.copy(), beta_out)
        p2 = classifier.predict(data_X)
        np.testing.assert_array_equal(p, p2)


if __name__ == "__main__":
    unittest.main()
