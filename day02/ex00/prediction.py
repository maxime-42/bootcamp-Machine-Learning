import numpy as np

from day02.utils.utils_function import check_x_and_theta

def simple_predict(x:np.ndarray, theta:np.ndarray):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if check_x_and_theta(x, theta) is False:
        return None
    y_hat = np.zeros(len(x)).reshape(-1, 1)

    m, n = x.shape
    y_hat = np.full(shape=(m, 1), fill_value=theta[0][0])
    for i in range(m):
        for j in range(n):
            y_hat[i][0] += x[i][j] * theta[j + 1][0]
    print(y_hat)
    return y_hat
