"""
EE-311
======

Lab 1: Introduction to Python
-----------------------------

created by François Marelli on 03.03.2021
"""

import importlib
import math
import unittest

import mock

importlib.reload(mock)


class Test(unittest.TestCase):
    def test_equation(self):
        for val in range(10):
            result = mock.equation(val)

            solution = -(1 + val) / (3 * val**2 + math.exp(val))
            self.assertAlmostEqual(result, solution)

    def test_condition(self):
        for val in range(10):
            result = mock.condition(val)
            self.assertEqual(result, val > 0 and val <= 5)


if __name__ == "__main__":
    unittest.main()
