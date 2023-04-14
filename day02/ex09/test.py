from data_spliter import data_spliter
import numpy as np

x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
# Example 1:
print("# Example 0:")
print(data_spliter(x1, y, 0.8))