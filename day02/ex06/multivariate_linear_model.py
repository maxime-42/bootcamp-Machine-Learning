import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from day02.ex05.mylinearregression import MyLinearRegression as MyLR
from  day02.utils.utils_function import check_x_and_theta



def plot(x,y, y_hat, color, theta, mse):
    """
        about : this function try to plotting the data points
        args:
            x : feature
            y : target
            y_hat : predict
            color; color of y_hat points
            theta : it is new parameter
            mse: ccc
        return 
            nothing 
    """
    plt.plot(x, y_hat, "o", color=color[0], label="Sell price")
    plt.scatter(x, y, color=color[1])
    # plt.title(f"Theta : {theta} ; MSE: {mse}")
    plt.show()

def univariate_lr(feature:str, target:str, color:list = [],alpha=0.001, max_iter=100_000) -> None:
    
    data = pd.read_csv("spacecraft_data.csv")
    x = np.array(data[feature]).reshape(-1,1)
    y = np.array(data[target]).reshape(-1,1)
    theta = np.array([[3], [2]])

    linear_re = MyLR(theta, alpha, max_iter)
    linear_re.fit_(x, y)
    y_hat = linear_re.predict_(x)
    # print("y_hat = ", y)
    mse = linear_re.mse_(y, y_hat)
    # print("x = ", y_hat)
    plot(x,y, y_hat, color, linear_re.thetas, mse)

def Multivariate(feature:str, target:str,   color:list = [], alpha=0.001, max_iter=100_000) -> None:
    data = pd.read_csv("spacecraft_data.csv")
    features = ['Age', 'Thrust_power', 'Terameters']
    x = np.array(data[features])
    y = np.array(data["Sell_price"]).reshape(-1, 1)
    theta = np.array([[3], [2], [4], [5]])
    linear_re = MyLR(theta, alpha, max_iter)
    print(check_x_and_theta(x, theta))

    linear_re.fit_(x, y)
    y_hat = linear_re.predict_(x)
    mse = linear_re.mse_(y, y_hat)
    for i in range(x.shape[1]):
        plot(x[:, i],y, y_hat, color, linear_re.thetas, mse)

    # plt.scatter(x[:,1], y, color=color[0])
    # plt.plot(x[:,1], y_hat, ".", color=color[1], label="Sell price")
    # plt.title(f"Theta : {theta} ; MSE: {mse}")
    # plt.show()

if __name__ == "__main__":
    try:

        # univariate_lr("Age", "Sell_price", alpha=1e-2, max_iter= 100_000)
        # univariate_lr("Thrust_power", "Sell_price", alpha=1e-4, max_iter= 100_000)
        # univariate_lr("Terameters", "Sell_price", alpha=1e-4, max_iter= 100_000)
        colore = ["darkblue", "dodgerblue"]
        Multivariate("Age", "Sell_price", colore, alpha=1e-5, max_iter= 100_000)

    except Exception as error_msg:
        print(error_msg)