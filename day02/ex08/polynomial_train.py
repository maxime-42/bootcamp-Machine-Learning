import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from day02.ex05.mylinearregression import MyLinearRegression as MyLR

from day02.ex07.polynomial_model import add_polynomial_features

from pydantic import BaseModel, conlist, Config

class LinearRegression(BaseModel):
    """ structure """
    theta: np.ndarray
    max_iter: int
    alpha: int
    
    class Config:
        arbitrary_types_allowed = True

def score(theta:np.ndarray):
    pass

def main():
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = data["Micrograms"]
    y = data["Score"]
    
    # training_array:Trains = Trains[6]
    # training_array= Trains(theta=np.array([[1], [1]]).reshape(-1,1), 
    #                            max_iter=1e-2, alpha=1e2)
    
    # thetat2 = np.array([[1], [1], [1]]).reshape(-1,1),
    # thetat3 = np.array([[1], [1], [1], [1]]).reshape(-1,1),
    # theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
    # theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
    # theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
    #  = [theta1, thetat2, thetat3, theta4, theta5, theta6]
    # alpha = [1e-2, 1e-3, 1e-5, 1e-6, 1e-8,  1e-9]
    # max_iter = [1e2, 1e3, 1e5, 1e6, 1e8,  1e9]
    # for theta in [theta1, thetat2, thetat3, theta4, theta5, theta6]:
    # print(training_array[0])  
if __name__ == "__main__":
    main()