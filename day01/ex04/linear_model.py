import numpy as np

import pandas as pd
from sklearn.metrics import mean_squared_error
from day01.ex03.my_linear_regression import MyLinearRegression as MyLR



data = pd.read_csv("abc.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)


linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))


Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print(MyLR.mse_(Yscore, Y_model1))
print(mean_squared_error(Yscore, Y_model1))

print(MyLR.mse_(Yscore, Y_model2))
print(mean_squared_error(Yscore, Y_model2))
