"""ex00"""

class Matrix():
    """
        mplement all the following built-in functions (called magic/special methods) 
        for your Matrix class
    """
    def __init__(self, data = list[list[int]]):
        try:
            if isinstance(data[0], int):
                self.shape = (1, len(data[0]))
            else:
                self.shape = (len(data), len(data[0]) )
            self.data = data

            assert self.check_matrix(data), "Error: invalid matrix"
        except AssertionError  as error_msg:
            print(error_msg)

    def check_int_or_float(self, lst)->bool:
        """check if all values in a list are integers or floats."""
        return all([isinstance(i, (int, float)) for i in lst])

    def check_matrix(self, matrix):
        """check if matrix is valide
            check if each row has same size
            check if each line it composede by integer or fload
        """
        size = len(matrix[0])
        for vect in matrix[1:]:
            if size != len(vect) or self.check_int_or_float(vect) is not True:
                return False
        return True


    def __add__(self, other):
        """sum  matrix"""
        try:
            assert self.shape == other.shape, "Error:row and cols must be same "
        except AssertionError as error_msg:
            print(error_msg)
        else:
            result = [ [0 for i in range(self.shape[1])] for j in range(self.shape[0]) ]
            print(f"shape == {self.shape}")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i][j] = self.data[i][j] + other.data[i][j]
            return result


    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        """matrix substract"""
        try:
            assert self.shape == other.shape, "Error:row and cols must be same "
        except AssertionError as error_msg:
            print(error_msg)
        else:
            result = [ [0 for i in range(self.shape[1])] for j in range(self.shape[0]) ]
            print(f"shape == {self.shape}")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i][j] = self.data[i][j] - other.data[i][j]
            return result

    def __truediv__(self, scalar:int|float):
        """divize by a scalare"""
        try:
            assert scalar != 0, "Error: Divize by zero"
        except AssertionError as error_msg:
            print(error_msg)
        else:
            result = [ [item / scalar  for item in vector] for vector in self.data]
            return result

    def __rtruediv__(self, scalar:int|float):
        """divize by a scalare"""
        self.__truediv__(scalar)


    def calcul(self, other):
        """matrix mutiply"""
        result = [ [ 0 for j in range(other.shape[1])] for i in range(self.shape[0])]
        for i in range(len(self.data)):
            for j in range(len(other.data[0])):
                for k in range(len(other.data)):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return result


    def __mul__(self, other):
        """mutiply matrix"""
        try:
            if isinstance(other.data, (int, float)):
                return [[ other * item for item in vector] for vector in self.data]
            # print(type(other.data))
            if isinstance(other.data , list):
                assert self.shape[1] == other.shape[0], "Error: muliplication matrix"
                return self.calcul(other)
        except AssertionError as error_msg:
            print(error_msg)

    def __rmul__(self, other):
        return other.__mul__(self)


    def __rsub__(self, other):
        """substract"""
        return other.__sub__(self)

    def T(self):
        """matrix transposition """
        new_data = []
        for i in range(len(self.data[0])):
            new_list = []
            for j in range(len(self.data)):
                new_list.append(self.data[j][i])
            new_data.append(new_list)
        # print(f"self = {self.data}")
        self.data = new_data
        # print(f"new_ = {new_data}")
        self.shape = (len(new_data), len(new_data[0]))
        return self


class  Vector(Matrix):
    """create vector"""
    def __init__(self, lst):
        """init vector"""
        try:
            # if isinstance(lst[0], list):
                # lst = [lst]
                # raise TypeError("Error: type")
            # print("vector size = ", len(lst[0]))
            super().__init__(lst)
        except TypeError as error_msg:
            print(error_msg)

m1 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0]])
print(m1.shape)

# m1.T()
print("=======create test=======")
v1 = Vector([[1, 2, 3]]) # create a row vector
print(v1.data)
v2 = Vector([[1], [2], [3]]) # create a column vector
print(v2.shape)
v3 = Vector([[1, 2], [3, 4]]) # return an error
print(v3.data)

print("=======transposition =======")
print("Exemple 1:")
Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape)


print("Exemple 2:")
m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
# Output:
# (2, 3)
print(m1.T().data)
# Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape)
# Output:
# (3, 2)

print("exemple multiply : 1")
m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([
    [0.0, 1.0],
    [2.0, 3.0],
    [4.0, 5.0],
    [6.0, 7.0]])
print(m1 * m2)
# Output:
# Matrix([[28., 34.], [56., 68.]]

print("exemple multiply : 2")
m1 = Matrix([[0.0, 1.0, 2.0],
[0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1)
# Output:
# Matrix([[8], [16]])
# Or: Vector([[8], [16]
print("exemple multiply : 3")
v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
# Output:
# Vector([[3],[6],[11]])
#
