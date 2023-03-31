"""test"""
import numpy as np

class TinyStatistician():
    """
        Create a class named TinyStatistician with the following methods.
        All methods take a list or a numpy.array as first parameter. 
        You have to protect your
        functions against input errors
    """
    def __init__(self) -> None:
        pass

    def mean(self, x:np.ndarray = np.array([])):
        """dddd"""
        if len(x) == 0:
            return None
        sum = 0
        for value in x:
            sum += value
        return sum / len(x)

    def median(self, x:np.ndarray):
        """"""
        if len(x) == 0:
            return None
        return np.median(x)

    def quartile(self, x:np.ndarray = np.array([])):
        """Compute the 1st and 3rd quartiles of a 1D array using NumPy"""
        if len(x) == 0:
            return None
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        return q1, q3

    def  percentile(self, x:np.ndarray = np.array([]), p:int = 50):
        """
            Computes the expected percentile of a given non-empty list or array x.
            The method returns the percentile as a float, otherwise None if x is an
            empty list or array or a non expected type object.

            Args:
                x (list or np.ndarray): The input list or array.
                p (float): The desired percentile value between 0 and 100.

            Returns:
             float or None: The computed percentile or None if the input is invalid.
         """
        if len(x) == 0:
            return None
        return np.percentile(x, p)

    def var(self, x:np.ndarray = np.array([])):
        """test"""
        if len(x) == 0:
            return None
        return np.var(x)

    def std(self, x:np.ndarray = np.array([])):
        np.std(x, ddof=1)


a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))
print(TinyStatistician().median(a))
print(TinyStatistician().quartile(a))
print(TinyStatistician().percentile(a, 10))
print(TinyStatistician().percentile(a, 15))
print(TinyStatistician().percentile(a, 20))
print(TinyStatistician().var(a))
print(TinyStatistician().std(a))