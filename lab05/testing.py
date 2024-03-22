"""
EE-311
======

Lab 5: Dimensionality Reduction
----------------------------------------

created by Zahra Farsijani and François Marelli on 25.03.2020
"""

import unittest
import numpy as np
import homework

import importlib

importlib.reload(homework)


class TestPCACorrelation(unittest.TestCase):
    def setUp(self):
        arrays = np.load("data.npz")
        data_X = arrays["X"]
        data_y = arrays["y"]
        self.w = arrays["w"]
        self.P = self.w.shape[1]
        self.submits = []
        for p in range(self.P):
            p += 1
            self.submits.append(homework.pca_correlation(data_X, data_y, p))

    def test_W_valid(self):
        for p in range(self.P):
            submit = self.submits[p]
            p += 1
            w = self.w[:, :p]

            np.testing.assert_array_equal(
                submit.shape,
                w.shape,
                "Incorrect shape for W: {} instead of {}".format(submit.shape, w.shape),
            )

            for j in range(p):
                for k in range(p):
                    dotproduct = submit[:, j].dot(submit[:, k])
                    if j == k:
                        self.assertAlmostEqual(
                            dotproduct, 1, msg="W is not orthonormal"
                        )
                    else:
                        self.assertAlmostEqual(
                            dotproduct, 0, msg="W is not orthonormal"
                        )

    def test_reduction(self):
        for p in range(self.P):
            submit = self.submits[p]
            p += 1
            w_equal = [False] * p
            w = self.w[:, :p]

            for j in range(p):
                for k in range(p):
                    w_equal[j] = (
                        w_equal[j]
                        or np.allclose(submit[:, k], w[:, j])
                        or np.allclose(-submit[:, k], w[:, j])
                    )

            self.assertTrue(
                np.all(w_equal),
                msg="W does not provide the required principal components",
            )


if __name__ == "__main__":
    unittest.main()
