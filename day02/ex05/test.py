import numpy as np
from mylinearregression import MyLinearRegression as MyLR
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
theta = np.array([[1.], [1.], [1.], [1.], [1]])
mylr = MyLR(theta)
# Example 0:
y_hat = mylr.predict_(X)
print(y_hat)
# Output:
# array([[8.], [48.], [323.]])

print("# Example 1:")
print(mylr.loss_elem_(Y, y_hat))

print("Example 3:")
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)
# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]]

print(" Example 4:")
y_hat = mylr.predict_(X)
print(y_hat)
# Output:
# array([[23.417..], [47.489..], [218.065...]]

print(" Example 5:")
print(mylr.loss_elem_(Y, y_hat))
# Output:
# array([[0.174..], [0.260..], [0.004..]])

print(" Example 5:")
print(mylr.loss_(Y, y_hat))
# Output