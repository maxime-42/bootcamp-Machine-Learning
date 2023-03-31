import numpy as np
from  day01.ex02.fit  import fit_ as i_fit_
from day01.ex01.vec_gradient import simple_predict
from  day01.ex00.gradient  import  check_requirement_args

class MyLinearRegression():
    """
        Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas:np.ndarray, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        #... other methods ...
        # gradient()
    def fit_(self, x, y):
        """call fit function"""
        i_fit_(x, y)

    def predict_(self, x, y):
        simple_predict(x, y)
    

    def loss_elem_(self, y:np.ndarray, y_hat:np.ndarray):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """

        if  all(isinstance(obj, np.ndarray) and obj.shape in [(obj.size, 1)] for  obj in [x, y]) is False : 
            return None
        if y_hat.shape != y.shape:
            return None
        return (y_hat - y)**2


    def loss_(self, y:np.ndarray, y_hat:np.ndarray):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """

        if  all(isinstance(obj, np.ndarray) and obj.shape in [(obj.size, 1)] for  obj in [x, y]) is False : 
            return None
        squared_distances  = loss_elem_(y, y_hat)
        divizer = 2*len(y)
        return (1/divizer) * np.sum(squared_distances)
    
    def mse_(self, y:np.ndarray, y_hat:np.ndarray):
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
