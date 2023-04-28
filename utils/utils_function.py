import numpy as np

def check_x_and_theta(x:np.ndarray, theta:np.ndarray = np.array([[]])):
    """
    Check requirement of argument y and x 
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    return 
        True if the requirement is fine
        False it mean there are some error in args
    """
    return isinstance(x, np.ndarray) and isinstance(theta, np.ndarray) and x.ndim == 2 and theta.ndim == 2 and x.shape[1] == theta.shape[0] - 1
    # return isinstan	ce(x, np.ndarray) and isinstance(theta, np.ndarray) and x.ndim == 2 and theta.ndim == 2 and x.shape[1] == theta.shape[0] - 1

def add_intercept(x=np.ndarray):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return np.column_stack((np.ones(len(x)), x ))

class Standardization():
    def __init__(self):
        """
            Standardization is commonly used when working with features that have different scales or units. 
            This is because many machine learning algorithms assume that all features are on a similar scale, 
            and may not perform well if this assumption is not met.
            Standardization involves subtracting the mean value of each feature and then dividing by its standard deviation. 
            This process transforms the features to have a mean of 0 and a standard deviation of 1.
            Standardization can be useful in situations where some features have values that are much larger or smaller than others, 
            or when the range of values for different features varies widely. 
            It can also help to make the model more interpretable, 
            since the coefficients obtained from the model will be directly comparable across features.
        """
        pass
    
    def fit(self, feature):
        self.mean_ = np.mean(feature, axis=0)
        self.std_ = np.std(feature, axis=0)
        
    def transform(self, feature):
   
        std_feature = np.copy(feature)
        std_feature -= self.mean_
        std_feature /= self.std_
        return std_feature



class Minmax():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def apply(self, X):
        e = 1e-20
        mnmx = (X - self.min) / (self.max - self.min + e)
        return mnmx

    def unapply(self, X):
        e = 1e-20
        return (X * (self.max - self.min + e)) + self.min