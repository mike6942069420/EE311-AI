"""
EE-311
======

Lab 4: support vector machines
------------------------------

created by François Marelli on 12.03.2020
"""


def compute_accuracy(y_predicted, y_ground_truth):
    """
    Compute the accuracy of a binary classifier predictions.

    Parameters:
        y_predicted: array (N, ) containing the binary model predictions, with N the number of points
        y_ground_truth: array (N, ) containing the ground truth binary labels

    Returns:
        accuracy: float between 0.0 and 1.0, the proportion of correct predictions
    """

    return (y_predicted == y_ground_truth).mean()

def smv_decision(w, b, X):
    """
    Given the parameters of a trained linear SVM, compute its predicted class for the input points.

    Parameters:
        w: array (M, ), the normal vector to the SVM separating hyperplane
        b: array (1, ), the intercept of the SVM plane
        X: array (N, M), the inputs for which to compute the predictions

    Return:
        labels: array (N, ) containing the binary class predictions (0/1)
    """

    return (X@w + b) > 0


def transform_space(X):
    """
    Transform the input space so that the two classes are linearly separable (see scatter plot in homework notebook).
    Return a 2D transformed space on which a linear SVM can be trained using function `train_svm_with_transform`.

    Parameters:
        X: array (N, 2), a dataset of N points with 2 features

    Returns:
        X_transformed: array (N, 2), the transformed dataset, with X_transformed[i, :] the transform of X[i, :]
    """

    return abs(X)
