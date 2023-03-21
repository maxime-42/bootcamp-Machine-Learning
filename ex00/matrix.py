"""ex00"""

class Matrix():
    """operation class matrix"""
    def __init__(self, data = list[list[int]]):
        self.shape = (len(data), len(data[0]) )
        self.data = data
        try:
            result = self.check_matrix(data)
            assert result, "Error: invalid matrix"
        except AssertionError  as error_msg:
            print(error_msg)

    def get_nb_rows(self):
        """get number of line"""
        return len(self.data)

    def get_nb_cols(self):
        """get number of column"""
        return len(self.data[0])

    def check_int_or_float(self, lst)->bool:
        """check if all values in a list are integers or floats."""
        return all([isinstance(i, (int, float)) for i in lst])

    def check_matrix(self, matrix):
        """check matrix"""
        size = len(matrix[0])
        for vect in matrix[1:]:
            if size != len(vect) or self.check_int_or_float(vect) is not True:
                return False
        return True

    def __add__(self, data2):
        """sum  matrix"""
        try:
            print(f"{data2.shape}")
            print(f"{self.shape}")

            assert self.shape == data2.shape, "Error:row and cols must be same "
        except AssertionError as error_msg:
            print(error_msg)
        else:
            result = [ [0 for i in range(data2.get_nb_cols())] for j in range(data2.get_nb_rows()) ]
            print(f"shape == {self.shape}")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i][j] = self.data[i][j] + data2.data[i][j]
            return result



v1 = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
v2 = Matrix( [[ 1, 1, 1], [1, 1, 1], [1, 1, 1]])

resul = v1 + v2
# print(resul)
