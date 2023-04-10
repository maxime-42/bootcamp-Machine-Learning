import numpy as np

from  day02.ex03.gradient import gradient as grad


def fit_(x:np.ndarray, y:np.ndarray, theta:np.ndarray, alpha:float, max_iter:int=10000):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    # print("derive = ", grad(x, y, theta))
    for _ in range(max_iter):
        derive = alpha *  grad(x, y, theta)
        theta = theta - derive
    return theta
