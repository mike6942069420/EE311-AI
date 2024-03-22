"""
EE-311
======

Lab 5: Dimensionality Reduction
----------------------------------------

created by Zahra Farsijani and François Marelli on 25.03.2020
"""

import numpy as np


def generate_data(n_points):
    """
    Generate a 2D dataset for binary classification.
    The data generated are linearly separable (basic linear SVM accuracy of 100%),
    but if we apply PCA and reduce it to 1 dimension, the data become non linearly
    separable (SVM accuracy under 75%).

    Parameters:
        n_points: int, the number of points wanted in the dataset. As much as
        possible, the data are be balanced (n_points / 2 data in each class).

    Returns:
        data_X: ndarray (n_points, 2), the dataset generated
        data_y: ndarray (n_points, ), the binary labels of the generated dataset [0/1]
    """

    rng = np.random.default_rng()

    means_0 = [0, -10]

    means_1 = [0, 10]

    covariance = [[1000, 0], [0, 1]]

    n0 = n_points // 2

    n1 = n_points - n0

    x0 = rng.multivariate_normal(means_0, covariance, n0)

    x1 = rng.multivariate_normal(means_1, covariance, n1)

    x = np.concatenate((x0, x1))

    y = np.zeros(n_points)

    y[:n0] = 1

    return x, y


def pca_correlation(data_X, data_y, P):
    """
    Extract the most relevant P features from a dataset using the following algorithm:

    1. compute the PCA of the data (USE THE DATA AS IS, DO NOT NORMALIZE IT)
    2. filter the P most informative principal components by ranking them using Pearson's correlation coefficient

    Parameters:
        data_X: ndarray (N, M) the dataset to reduce, with M the number of initial features
        data_y: ndarray (N, ) the binary labels of the dataset [0/1]
        P: int, the number of principal components to keep

    Returns:
        W: ndarray (M, P), the projection matrix such that (data_X @ W) is the reduced dataset
    """

    return
