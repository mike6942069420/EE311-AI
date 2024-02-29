"""
EE-311
======

Lab 2: NumPy, Matplotlib and Scikit-learn (linear regression example)
---------------------------------------------------------------------

created by François Marelli on 03.03.2021
"""

import unittest
import numpy as np
from sklearn import datasets
import homework

import importlib

importlib.reload(homework)


class Test(unittest.TestCase):
    def test_train(self):
        expected = np.load("test_data.npy")

        wine_full, _ = datasets.load_wine(return_X_y=True)
        wine_X = wine_full[:, np.newaxis, 9]
        wine_y = wine_full[:, 0]

        data_X = wine_X[:20]
        data_y = wine_y[:20]

        predict_X = np.arange(0, 101, 10)[..., None]

        model = homework.train(data_X, data_y)
        prediction = model.predict(predict_X)

        np.testing.assert_array_almost_equal(prediction, expected)

    def test_mean_odd(self):
        array1 = np.arange(10)
        array2 = np.ones(10)
        array3 = np.arange(10) - 1
        array4 = np.ones((3, 3, 3))

        self.assertEqual(homework.mean_odd(array1), 5)
        self.assertEqual(homework.mean_odd(array2), 1)
        self.assertEqual(homework.mean_odd(array3), 3)
        self.assertEqual(homework.mean_odd(array4), 1)


if __name__ == "__main__":
    unittest.main()
