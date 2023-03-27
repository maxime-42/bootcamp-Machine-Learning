import numpy as np
import statistics

class TinyStatistician():
    def  __init__(self):
        """init"""
        pass
    def mean(self, x = np.array([])):
        """dddd"""
        if isinstance(x, list):
           return x.mean(x)
    def median(self, x):
        """"""
        return np.median(x)
    
    def quartile(self, x):
        """Compute the 1st and 3rd quartiles of a 1D array using NumPy"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        return q1, q3

    def  percentile(self, x, p):
        """"""
        pass

    def var(self, x):
        """"""
        return np.variance(x)

    def std(self, x):
        np.std(x, ddof=1)