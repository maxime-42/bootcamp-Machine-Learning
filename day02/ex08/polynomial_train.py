import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from day02.ex05.mylinearregression import MyLinearRegression as MyLR

from day02.ex07.polynomial_model import add_polynomial_features

    
def main():
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = np.array(data["Micrograms"])
    y = np.array(data["Score"]).reshape(-1, 1)

    theta1 = np.array([-20., 160.]).reshape(-1, 1)
    theta2 = np.array([-20., 160., -80.]).reshape(-1, 1)
    theta3 = np.array([-20., 160., -80., 10.]).reshape(-1, 1)
    theta4 = np.array([-20., 160., -80., 10., -1.]).reshape(-1, 1)
    theta5 = np.array([1140., -1850., 1110., -305., 40., -2.]).reshape(-1, 1)
    theta6 = np.array([9110., -18015., 13400., -4935., 966., -96.4, 3.86]).reshape(-1, 1)

    #six separate Linear Regression model
    model1 = MyLR(theta1, alpha=1e-3, max_iter=1000000)
    model2 = MyLR(theta2, alpha=1e-3, max_iter=1000000)
    model3 = MyLR(theta3, alpha=1e-5, max_iter=5000000)
    model4 = MyLR(theta4, alpha=1e-6, max_iter=1000000)
    model5 = MyLR(theta5, alpha=1e-8, max_iter=1000000)
    model6 = MyLR(theta6, alpha=1e-9, max_iter=5000000)
    
    #Trains six separate Linear Regression models with polynomial hypothesis
    plt.figure(figsize=(13, 8))
    plt.scatter(x, y, label='raw', c='black')
    plt.title("Train Polynomial Models")
    plt.xlabel("Micrograms")
    plt.ylabel("Score")
    plt.grid()

    x_fit = np.linspace(1, 7, 100)
    mse_y = np.zeros(6)
    
    #Plot all separate Linear Regression models
    for i, model in enumerate([model1, model2, model3, model4, model5, model6] , start=1):
        xp = add_polynomial_features(x, i)
        model.fit_(xp, y)
        print(f"Training model {i}...")
        y_hat = model.predict_(add_polynomial_features(x_fit, i))
        plt.plot(x_fit,  y_hat, label=f"Pred model {i}" )
        mse_y[i - 1] = model.loss_(y, model.predict_(xp))
        print(f"MSE : {mse_y[i - 1]}\n")
    print("MSE :\n", mse_y.reshape(-1, 1))  
    xx = np.array(["pred 1", "pred 2", "pred 3", "pred 4", "pred 5", "pred 6"])
    plt.legend()
    plt.figure()
    plt.bar(xx, mse_y)
    plt.grid()
    plt.show()
if __name__ == "__main__":
    main()