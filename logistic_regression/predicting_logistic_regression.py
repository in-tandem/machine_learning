from numpy.random import RandomState
from numpy import clip
import pandas as panda
import matplotlib.pyplot as plot 
import random
from math import sqrt, exp, log
remote_location = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def standard_deviation(values):
    average = sum(values) / len(values)

    variance = sum([(average - i)**2/len(values) for i in values])

    return sqrt(variance)


class LogisticRegression(object):
    
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

            cost =0 

            for i in range(self.number_of_training_set):
                _x = self._x_training_set[i]
                _desired = self._y_training_set[i]
                _weight = self.weights
                
                weighted_sum = [_weight[0]] + [_weight[j+1] * _x[j] for j in range(len(_x))]
                
                normalized_weighted_sum =  (sum(weighted_sum) - min(weighted_sum))/ (max(weighted_sum) - min(weighted_sum))

                if normalized_weighted_sum > 710:

                    print("sum of weighted sum %d" %sum(weighted_sum))
                    print("min of weighted sum %d" %min(weighted_sum))
                    print("max of weighted sum %d" %max(weighted_sum))
                    print("diff of weighted sum %d" %(max(weighted_sum) - min(weighted_sum)))
                    

                try:
                    guess = 1 / ( 1 + exp(normalized_weighted_sum))
                
                except Exception as e:
                    
                    print("stil getting overflow ... %d" %normalized_weighted_sum)


                error = _desired - guess 

                ## i am going to reset all the weights
                if error!= 0 :

                    ## resetting the bias unit
                    self.weights[0] = error * self.learning_rate
                    self.weights[1:] =[self.weights[j+1] + error * self.learning_rate * _x[j] \
                                            for j in range(len(_x))]

                    ## cost entropy loss function
                    
                    cost+= - ( _desired * log(guess) + (1 - _desired) *log(1-guess))
                    
            #saving error at the end of the training set        
            epoch_data[epoch] = cost ##summation of all such y predictions for a training set
        
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

            weighted_sum = [self.weights[0]] +  \
                    [self.weights[j+1] * _x_test_data[i][j] \
                        for j in range(len(_x_test_data[i]))]

            normalized_weighted_sum = (sum(weighted_sum) - min(weighted_sum))/(max(weighted_sum) - min(weighted_sum))
 
            guess = 1 / ( 1 + exp(normalized_weighted_sum))

            prediction.append( 1 if guess >= 0.5 else 0)

        print(prediction)
        return prediction


