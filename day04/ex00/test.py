import numpy as np

from polynomial_model_extended import add_polynomial_features

print("exemple 1:")
x = np.arange(1,11).reshape(5, 2)
print(add_polynomial_features(x, 3))
# print("exemple 2:")
# print(add_polynomial_features(x, 4))
