"""
EE-311
======

Lab 4: support vector machines
------------------------------

created by Adrian Shajkofci and François marelli on 12.03.2020
"""

import unittest
import numpy as np
import homework

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import importlib

importlib.reload(homework)


class TestFunctions(unittest.TestCase):
    def test_accuracy(self):
        np.random.seed(0)
        a = np.ones(10)
        b = np.zeros(10)
        c = np.concatenate([a, b], axis=0)
        d = np.concatenate([a, a], axis=0)
        f = homework.compute_accuracy
        np.testing.assert_array_equal(
            [
                f(a.copy(), a.copy()),
                f(a.copy(), b.copy()),
                f(c.copy(), d.copy()),
                f(d.copy(), c.copy()),
            ],
            [1, 0, 0.5, 0.5],
        )

    def test_decision(self):
        data_X, data_y = load_iris(return_X_y=True)
        mask = (data_y == 0) | (data_y == 1)
        data_X = data_X[mask]
        data_y = data_y[mask]
        data_X = data_X[:, 0:2]

        model = LinearSVC().fit(data_X, data_y)
        predict = model.predict(data_X)

        b = model.intercept_
        w = model.coef_.squeeze()

        np.testing.assert_array_equal(homework.smv_decision(w, b, data_X), predict)


class TestTransform(unittest.TestCase):
    def setUp(self):
        data = np.load("data.npz")
        self.X = data["X"]
        self.Y = data["y"]

    def test_transform(self):
        X = homework.transform_space(self.X.copy())
        self.assertEqual(X.ndim, 2, "Transformed space should be 2D")
        score = LogisticRegression().fit(X, self.Y).score(X, self.Y)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
