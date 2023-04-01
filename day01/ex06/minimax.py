"""
Normalization involves scaling the values of a feature to a range between 0 and 1.
This is typically done by subtracting the minimum value of the feature and dividing by 
the range of the feature (i.e., the difference between the maximum and minimum values).
"""
import numpy as np

def minmax(x:np.ndarray):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not all([isinstance(x, np.ndarray),  x.shape in [(x.size, 1)]]):
        return None

    xp = np.zeros(len(x))
    max = np.amax(x)
    min = np.amin(x)
    for i in range(len(x)):
        xp[i] = (x[i] - min)/(max - min)
    return xp

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
print(minmax(X))
