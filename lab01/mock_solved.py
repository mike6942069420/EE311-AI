"""
EE-311
======

Lab 1: Introduction to Python
-------------------------------------------------------

created by Francois Marelli on 22.02.2021
"""

import math


def equation(input_var):
    """
    Compute the result of the following equation:
    y = -(1 + x) / (3x^2 + exp(x))

    Parameters:
        input_var: float, value of x

    Returns:
        result: float, result of the equation
    """

    result = -(1 + input_var) / (3 * input_var**2 + math.exp(input_var))

    return result


def condition(input_var):
    """
    Return True if the input argument is strictly bigger than 0 AND smaller
    or equal to 5, and False otherwise

    Parameters:
        input_var: float, the input argument

    Returns:
        result: bool, True if the argument satisfies the condition
    """

    if input_var > 0 and input_var <= 5:
        return True

    return False
