
## class to depict a matrix, n * m matrix
import numpy

class Matrix(object):

    def __init__(self, number_of_rows, number_of_columns, data = None):

        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns

        if data:
            self.checkForSanityOfData(data)
            self.data = data
        else:
            self.data = []
            for _ in range(self.number_of_rows):
                self.data.append( [None for j in range(self.number_of_columns)])
            
        # print(self.data)

    def checkForSanityOfData(self, data):
        """
            Method will check if the data provided adheres to the
            norm of self.number_of_rows * self.number_of_columns
            matrix format
            
        """
        pass

    def setRowValues(self, rowNumber, rowValues):
        
        if rowNumber <= 0:
            raise ValueError("Row number input needs to be numeric and greater than 0 ")
        
        if not isinstance(rowValues, list):
            raise ValueError("Row values input needs to be a list of real numbers  ")

        if len(rowValues) != self.number_of_columns:
            raise ValueError("ROw values should be list of real numbers of size %d" %self.number_of_columns)
        
        self.data[rowNumber - 1] = rowValues


    def transpose(self):
        """
            flips a matrix A[i][j] such that it becomes
            A[j][i]

        """

        new_data = [list(i) for i in zip(*self.data)]
        assert new_data == numpy.transpose(self.data).tolist()
        return new_data

    def multiply(self, matrix):
        """

            Multiplies this matrix with the given matrix
            matrix multiplication A[n,m] and B[m,p] is possible 
            only if columns of A matrix is equal to rows of
            B matrix.

            C = A[n,m] * B[m,p] = summation(A[ij]B[jk])
            where num of rows(C) = n
            num of col(C) = p

        """
        if self.number_of_columns != matrix.number_of_rows:
            raise AssertionError("Matrices cannot be multiplied")
    
        computed_data = [[sum(aa * bb for aa, bb in zip(A_row, B_col)) \
                                for B_col in zip(*matrix.data)] \
                                    for A_row in self.data]
    
        expected_response = numpy.dot(self.data, matrix.data).tolist()
        assert computed_data == expected_response



    def add(self, matrix):
        pass

matrix = Matrix(2,4,[[1,2,4,5],[11,22,33,44]])
matrix.transpose()



matrix = Matrix(1,4,[[1,2,4,5]])
matrix.transpose()


matrix = Matrix(4,1,[[1],[2],[4],[5]])
matrix.transpose()

matrix = Matrix(2,2,[[1,2],[3,4]])
matrix.multiply(Matrix(2,2,[[5,6],[7,8]]))

matrix = Matrix(3,2,[[1,2],[3,4],[5,6]])
matrix.multiply(Matrix(2,2,[[5,6],[7,8]]))
