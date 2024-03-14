"""
EE-311
======

Lab 3: logistic regression and gradient descent
-----------------------------------------------

created by Francois Marelli on 05.03.2020
"""

import numpy as np


def model_prediction(data_X, beta):
    """
    Compute the predicted labels for the points in data_X using a logistic regression model

    Parameters:
        data_X : ndarray of shape (N, M) with N the number of points and M the number of features
        beta: ndarray of shape (M+1) containing the model parameters, beta[0] being the independent term

    Returns:
        labels: an array of shape (N) containing the model class predictions (0/1)
    """
    
    N,_=data_X.shape

    # compute the prediction
    y_pred=(1 / (1 + np.exp(-np.hstack((np.ones((data_X.shape[0], 1)), data_X))@beta)))

    # return the hard decision
    return y_pred > 0.5

    #one liner
    #return  (1 / (1 + np.exp(-np.hstack((np.ones((data_X.shape[0], 1)), data_X))@beta))) > 0.5


def logistic_loss_derivative(data_X, data_y, beta):
    """
    Compute the gradient of the averaged logistic loss function for a logistic regression model

    Parameters:
        data_X : ndarray of shape (N, M) with N the number of points and M the number of features
        data_y: ndarray of shape (N) containing the labels (0/1)
        beta: ndarray of shape (M+1) containing the model parameters, beta[0] being the independent term

    Returns:
        grad: an array of shape (M+1) containing the gradient of the logistic loss
              function with respect to beta averaged over data_X
    """

    N,_=data_X.shape

    # compute the prediction
    y_pred=(1 / (1 + np.exp(-np.hstack((np.ones((data_X.shape[0], 1)), data_X))@beta)))

    # compute the gradient (element-wise multiplication with (error)*(1 x^i))
    grad=(y_pred-data_y).reshape(-1,1)*np.hstack((np.ones((data_X.shape[0], 1)), data_X))

    # average over the columns
    return grad.mean(axis=0)
    
    # one liner 
    #return ((1 / (1 + np.exp(-np.hstack((np.ones((data_X.shape[0], 1)), data_X))@beta)) - data_y).reshape(-1,1) * np.hstack((np.ones((data_X.shape[0], 1)), data_X))).mean(axis=0)
   

def gradient_descent(data_X, data_y, beta_0, stepsize, iterates):
    """
    A gradient descent algorithm, returns the trained model

    DO NOT EDIT

    Parameters:
        data_X : ndarray of shape (N, M) with N the number of points and M the number of features
        data_y: ndarray of shape (N) containing the labels (0/1)
        beta: ndarray of shape (M+1) containing the initial model parameters, beta[0] being the independent term
        stepsize: float, the step size of the gradient descent
        iterates: number of iterations to perform

    Returns:
        beta: ndarray of shape (M+1) containing the trained model parameters, beta[0] being the independent term

    """

    beta = beta_0

    for _ in range(iterates):
        dloss = logistic_loss_derivative(data_X, data_y, beta)
        beta -= dloss * stepsize

    return beta
