
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y:np.ndarray, y_hat:np.ndarray):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y_hat.shape != y.shape:
        return None
    square = (y_hat - y)**2
    divizer = len(y)
    return (1/divizer) * np.sum(square) 

def rmse_(y:np.ndarray, y_hat:np.ndarray):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    return sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
    """
        Description:
        Calculate the MAE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
    """
    abs_value = abs(y_hat - y)
    divizer = len(y)
    return (1/divizer) * np.sum(abs_value)

def r2score_(y:np.ndarray, y_hat:np.ndarray):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    numerator = (y_hat - y)**2
    y_bar = np.sum(y) / len(y)
    denomitor = (y - y_bar)**2
    return 1 - (np.sum(numerator)/np.sum(denomitor))


x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])
# Mean squared error
## your implementation
print("exemple 1")
print(mse_(x,y))
print(mean_squared_error(x,y))
## example2:
print("exemple 2")
print(rmse_(x,y))
sqrt(mean_squared_error(x,y))

print("exemple 3")
print(mae_(x,y))
print(mean_absolute_error(x,y))

print("exemple 4")
print(r2score_(x,y))
print(r2_score(x,y))
