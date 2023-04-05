import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from day02.ex05.mylinearregression import MyLinearRegression as MyLR

def  univariate_lr():
    data = pd.read_csv("spacecraft_data.csv")
    x = np.array(data["Age"]).reshape(-1,1)
    y = np.array(data["Sell_price"]).reshape(-1,1)
    # theta = np.array([18,5])
    theta = np.array([[3], [2]])

    # print(arr.shape)

    total_price = MyLR(theta)
    total_price.fit_(x, y)
    
    y_hat = total_price.predict_(x)
    # print(y_hat)
    plt.plot(x, y_hat, color='r')
    new_theta = total_price.fit_(x, y)
    plt.scatter(x, y)
    plt.show()



univariate_lr()