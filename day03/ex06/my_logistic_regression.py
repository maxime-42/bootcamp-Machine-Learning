from day03.ex01.log_pred import logistic_predict_
from day03.ex02.log_loss import log_loss_ 
from day03.ex03.vec_log_loss import vec_log_loss_ 
from day03.ex02.log_loss import log_loss_ 
import numpy as np


from day03.ex05.vec_log_gradient import vec_log_gradient 


class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def predict_(self, x):
        return logistic_predict_(x, self.theta)
    
    def loss_elem_(self, y, yhat):
        return log_loss_(y, yhat)


    def loss_(self, x, y):
        y_hat = self.predict_(x)
        eps = 1e-15
        log_loss = y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
        return vec_log_loss_(y, y_hat)


    def fit_(self, x, y):

        for _ in range(self.max_iter):
            grad = vec_log_gradient(x, y, self.theta)
            if grad is None :
                return None
            self.theta = self.theta -  self.alpha *  grad
        return self.theta        
        # return vec_log_gradient(x, y, self.theta)