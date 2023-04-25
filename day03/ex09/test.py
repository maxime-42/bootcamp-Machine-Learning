import numpy as np

from confusion_matrix import confusion_matrix_

from sklearn.metrics import confusion_matrix
y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])


print("Example 1:")
## your implementation

print(confusion_matrix_(y, y_hat))
## Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])
## sklearn implementation
print("sklearn implementation:")
print(confusion_matrix(y, y_hat))
## Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])
# Example 2:
## your implementation
print("Example 2:")
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
print("sklearn implementation:")
print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
## Output:
# array([[2 1]
# [0 2]])
print("Example 3:")

print(confusion_matrix_(y, y_hat, df_option=True))
# print("sklearn implementation:")
# confusion_matrix(y, y_hat, labels=['bird', 'dog'], df_option=True)
# print(confusion_matrix(y, y_hat, labels=['bird', 'dog'], df_option=True))

print("exemple 4:")
print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))