from numpy.random import RandomState

import pandas as panda
import matplotlib.pyplot as plot 
import random
from math import sqrt
remote_location = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def standard_deviation(values):
    average = sum(values) / len(values)

    variance = sum([(average - i)**2/len(values) for i in values])

    return sqrt(variance)


class Perceptron(object):
    
    def __init__(self, epochs, learning_rate, _x_training_set, _y_training_set, standardize = False, random_state = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.standardize = standardize        
        self._x_training_set = _x_training_set
        self._y_training_set = _y_training_set
        self.number_of_training_set = len(self._y_training_set)
        self.weights = []      
        self.random_state = RandomState(random_state if random_state else 1)

    def standardizeInputData(self):
        """

        Standardizing of feature set means substracting the mean of
        each training sample from the feature value and dividing it by
        the standard deviation

        1. take average of j features from i th training sample . say avg
        2. calculate the variance of each j feature
        3. variance(j) = (avg - x(j))**2/len(features)
        4. standard deviation of x(j) = sq rt(variance(j))

        so standardized(x(j)) = x(j) - avg / standard deviation(x(j))

        """
        temp = []

        for i in range(len(self._x_training_set)):
            
            mean = sum(self._x_training_set[i])/ len(self._x_training_set[i])
            std_deviation = standard_deviation(self._x_training_set[i])
            temp.append([ (j - mean)/std_deviation for j in self._x_training_set[i]])            

        return temp
        
    def setup(self):
        
        if self.standardize:
            self._x_training_set = self.standardizeInputData()

        self.initialize_weights(len(self._x_training_set[0]) + 1)
    
    def initialize_weights(self, number_of_weights):

        self.weights = list(self.random_state.normal(loc = 0.0, scale = 0.01, size = len(self._x_training_set[0]) + 1))
    
    def learn(self):
        
        self.setup() 
        epoch_data = {}
        error = 0

        for epoch in range(self.epochs):

            errors =0 

            for i in range(self.number_of_training_set):
                _x = self._x_training_set[i]
                _desired = self._y_training_set[i]
                _weight = self.weights
                
                guess = _weight[0] + sum([_weight[j+1] * _x[j] for j in range(len(_x))])

                ## possible desired values are 0 or 1. our guess function also need
                ## to reflect the same
               
                error = _desired - guess

                ## i am going to reset all the weights
                if error!= 0 :

                    ## resetting the bias unit
                    self.weights[0] = error * self.learning_rate
                    self.weights[1:] =[self.weights[j+1] + error * self.learning_rate * _x[j] \
                                            for j in range(len(_x))]
                    errors+=error
            #saving error at the end of the training set        
            epoch_data[epoch] = errors**2 # sum of least squares, we dont this to be as small as possible
        
        print(epoch_data)

    def predict(self, _x_test_data):
        """

            Given algorithm has been trained using the #learn method
            this method will predict the y values based on the last
            values calculated for weights. This is because
            by the end of the learn method, algorithm has already
            converged as close to 0 error as it can
        """
        prediction = []

        for i in range(len(_x_test_data)):
            prediction.append(self.weights[0] +  \
                    sum([self.weights[j+1] * _x_test_data[i][j] \
                        for j in range(len(_x_test_data[i]))]))

        print(prediction)
        return prediction


