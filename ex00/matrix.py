"""ex00"""

class Matrix():
    """this class define operations on  matrix"""
    def __init__(self, data = list[list[int]]):
        try:
            if type(data[0]) == int:
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
        result = [ [ 0 for j in range(other.shape[1])] for i in range(self.shape[0])]
        for i in range(self.data):
            for j in range(other.data[0]):
                for k in range(self.data):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return result
            
           
    def __mul__(self, other):
        """mutiply matrix"""
        try:
            if isinstance(other.data, (int, float)):
                return [[ other * item for item in vector] for vector in self.data]
            print(type(other.data))
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
    

class  Vector(Matrix):
    """create vector"""
    def __init__(self, lst):
        """init vector"""
        try:
            if isinstance(lst[0], list):
                raise TypeError("Error: type")
            super().__init__(lst)
        except TypeError as error_msg:
            print(error_msg)

v1 = Matrix( [[ 1, 1, 1], [1, 1, 1], [1, 1, 1]])

v2 = Matrix([[10, 40, 70], [20, 50, 80], [30, 60, 90]])

# resul = v1 / 2
print(v1 * v2 )
