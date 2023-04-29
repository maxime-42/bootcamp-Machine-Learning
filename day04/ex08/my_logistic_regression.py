from day03.ex06.my_logistic_regression import MyLogisticRegression as MyLogr

from day03.ex05.vec_log_gradient import vec_log_gradient 
from day03.ex01.log_pred import logistic_predict_
from day03.ex02.log_loss import log_loss_ 
from day03.ex03.vec_log_loss import vec_log_loss_ 
from day03.ex02.log_loss import log_loss_ 
from utils.utils_function import add_intercept , check_x_and_theta
import numpy as np


class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0):
        # Check on type, data type, value ... if necessary
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penality = penality
        self.lambda_ = lambda_ if self.penality == 12 in self.supported_penalities else 0
    
    def predict_(self, x):
        return logistic_predict_(x, self.theta)
    
    def loss_elem_(self, y, yhat):
        return log_loss_(y, yhat) 


    def loss_(self, x, y, lambda_=0.5):
        y_hat = self.predict_(x)
        eps = 1e-15
        if self.penality != 'l2':
            lambda_ = 0
        # log_loss = y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

        reg = lambda_ * (np.dot(self.theta[1:].T, self.theta[1:])) / (2 * len(y))
        return vec_log_loss_(y, y_hat) + reg


    def vec_reg_linear_grad(self, y, x, theta, lambda_):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not check_x_and_theta(x, theta):
            return None
        if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, theta)]) is False:
            return None
        x = add_intercept(x)
        
        pred = x.dot(theta)
        # is used to set the bias term of the theta vector to zero.
        # This is done because the regularization term only applies 
        # to the non-bias terms of theta,
        # and setting the bias term to zero ensures that it is not affected by the regularization term.
        theta[0] = 0
        if self.penality != 'l2':
            lambda_ = 0
        reg = lambda_ * theta
        new_theta = x.T.dot((pred - y)) + reg
        return new_theta * 1/len(y)


    def fit_(self, x, y):
        for _ in range(self.max_iter):
            grad = vec_log_gradient(x, y, self.theta)
            if grad is None :
                return None
            self.theta = self.theta -  self.alpha *  grad
        return self.theta        
        # return vec_log_gradient(x, y, self.theta)

