
import numpy as np

from day02.ex05.mylinearregression import MyLinearRegression
from day04.ex05.reg_logistic_grad import vec_reg_logistic_grad
from day04.ex01.l2_reg import  l2

class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(self, self.theta, self.alpha, self.max_iter)
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_

        def get_params_(self, x):
            return self.predict_()
        
        def gradient_(self, y, x):
            return vec_reg_logistic_grad(y, x, self.theta, self.lambda_)
        
        def fit_(self, x, y):
            for _ in range(max_iter):
                grad = self.gradient_(x, y)
                if grad is None :
                    return None
                theta = theta -  alpha *  grad
            return theta
        
        def loss_elem_(self, y, y_hat):
            if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, y_hat, self.theta)]) is False or y.shape != y_hat.shape:
                return None
            if not isinstance(lambda_, float):
                return None
            return  (y - y_hat).T @ (y - y_hat)
        
        def loss_(self, y, y_hat):
           loss = loss_elem_(y, y_hat)
           reg = lambda_ * l2(self.theta)
           return float(0.5 * (loss + reg) / y.shape[0])
        
        def get_params_(self):
            return self.__dict__

        def set_params_(self, new_val):
            """ 
            Set new attributes to class
            Args:
                new_val [dict]: A dictionary containing the new values to be set.
            Return:
                None
            """
            attr_dict = {'thetas': np.ndarray,
                        'alpha': float,
                        'max_iter': int,
                        'lambda_': float}

            for key, val in new_val.items():
                if key not in attr_dict:
                    return None
                if not isinstance(val, attr_dict[key]):
                    return None
                setattr(self, key, val)