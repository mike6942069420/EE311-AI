"""
EE-311
======

Lab 2: NumPy, Matplotlib and Scikit-learn (linear regression example)
---------------------------------------------------------------------

created by François Marelli on 03.03.2021
"""


def sum_bigger(array, thresh):
    """
    Sum all elements of an array that are strictly bigger than a given threshold

    Parameters:
        array: ndarray (float), the array we want to process
        thresh: float, the threshold

    Returns:
        result: float, the sum of all elements in the input array that are strictly bigger than the threshold
    """
    mask = array > thresh
    matching = array[mask]
    return matching.sum()


def predict(model, data_X):
    """
    Use a trained model to predict label values at specific given points

    Parameters:
        model: sklearn.linear_model.LinearRegression, the linear regression model object (already trained)
        data_X: ndarray of shape (N, 1) with N the number of samples, containing the points at which we want to predict labels

    Returns:
        data_out: ndarray of shape (N, ) containing the predicted values
    """
    return model.predict(data_X)
