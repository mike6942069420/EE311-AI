"""
EE-311
======

Lab 2: NumPy, Matplotlib and Scikit-learn (linear regression example)
---------------------------------------------------------------------

created by François Marelli on 03.03.2021
"""

from sklearn import linear_model


def mean_odd(array):
    """
    Compute the mean of all the odd elements in an array

    Parameters:
        array: ndarray (int), the array we want to process (it contains at least 1 odd number)

    Returns:
        result: float, the mean of all the odd elements in the input array
    """
            
    return array[array % 2 == 1].mean()


def train(data_X, data_y):
    """
    Create and train a linear regression model given the input data

    Parameters:
        data_X : ndarray of shape (N, 1) with N the number of samples, containing the input points
        data_y: ndarray of shape (N) containing the labels of the dataset

    Returns:
        model: sklearn.linear_model.LinearRegression, the linear regression model object trained on the input data

    """

    return  linear_model.LinearRegression().fit(data_X, data_y)


