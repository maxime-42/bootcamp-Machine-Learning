import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from day02.ex05.mylinearregression import MyLinearRegression as MyLR



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
    plt.plot(x, y_hat, "o", color=color, label="Sell price")
    plt.scatter(x, y)
    # plt.title(f"Theta : {theta} ; MSE: {mse}")
    plt.show()

def univariate_lr(feature:str, target:str, alpha=0.001, max_iter=100_000) -> None:
    
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
    plot(x,y, y_hat, 'r', linear_re.thetas, mse)

def Multivariate(feature:str, target:str,  alpha=0.001, max_iter=100_000) -> None:
    data = pd.read_csv("spacecraft_data.csv")
    features = ['Age', 'Thrust_power', 'Terameters']
    # x = data[features]
    # x = x.reshape(-1,1)
    x = np.array(data[features])
    y = np.array(data["Sell_price"]).reshape(-1, 1)
    theta = np.array([[3], [2], [4], [5]])

    linear_re = MyLR(theta, alpha, max_iter)
    linear_re.fit_(x, y)
    y_hat = linear_re.predict_(x)
    mse = linear_re.mse_(y, y_hat)

    # plot(x,y, y_hat, 'r', linear_re.thetas, mse)

    print(y.shape)
    print(x.shape)



if __name__ == "__main__":
    try:

        # univariate_lr("Age", "Sell_price", alpha=1e-2, max_iter= 100_000)
        # univariate_lr("Thrust_power", "Sell_price", alpha=1e-4, max_iter= 100_000)
        # univariate_lr("Terameters", "Sell_price", alpha=1e-4, max_iter= 100_000)
        Multivariate("Age", "Sell_price", alpha=1e-6, max_iter= 100_000)

    except Exception as error_msg:
        print(error_msg)