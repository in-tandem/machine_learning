
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
            
        print(self.data)

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

        expected_number_of_columns = self.number_of_rows
        expected_number_of_rows  = self.number_of_columns

        new_data = []

        for i in range(expected_number_of_rows):

            new_data.append([self.data[j][i] for j in range(expected_number_of_columns) ])
            
        print("computed response %s"%new_data)
        print("expected response %s"%numpy.transpose(self.data))
        assert new_data == numpy.transpose(self.data).tolist()


    
    def multiply(self, matrix):
        pass
    
    def add(self, matrix):
        pass

matrix = Matrix(2,4,[[1,2,4,5],[11,22,33,44]])
matrix.transpose()



matrix = Matrix(1,4,[[1,2,4,5]])
matrix.transpose()


matrix = Matrix(4,1,[[1],[2],[4],[5]])
matrix.transpose()