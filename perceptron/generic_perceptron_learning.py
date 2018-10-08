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
    
    def __init__(self, epochs, learning_rate, standardize = False, weight_range = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.standardize = standardize
        self.weight_range = weight_range if weight_range else [-1, 1]
        self.weights = []
        self._x_training_set = None
        self._y_training_set = None
        self.number_of_training_set = 0        

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
            standard_deviation = standard_deviation(self._x_training_set[i])
            temp.append([ (j - mean)/standard_deviation for j in self._x_training_set[i]])            

        return temp
        
    def setup(self):

        self.number_of_training_set = self.setup_training_set()
        self.initialize_weights(len(self._x_training_set[0]) + 1)

    def setup_training_set(self):
        """

        Downloading training set data from UCI ML Repository - Iris DataSet
        
        """

        data = panda.read_csv(remote_location)       

        self._x_training_set = list(data.iloc[0:, [0,2]].values)
        self._y_training_set = [0 if i.lower()!='iris-setosa' else 1 
                                    for i in data.iloc[0:, 4].values]

        if self.standardize:
            self._x_training_set = self.standardizeInputData()

        return len(self._x_training_set)

    def initialize_weights(self, number_of_weights):
        random_weights = [random.uniform(self.weight_range[0], self.weight_range[1]) 
                                for i in range(number_of_weights + 1)]
        # self.weights.append(-1) # setting up bias unit 
        self.weights.extend(random_weights)

    def draw_initial_plot(self, _x_data, _y_data, _x_label, _y_label):
        
        plot.xlabel(_x_label)
        plot.ylabel(_y_label)
        plot.scatter(_x_data,_y_data)
        plot.show()

    def shuffle(self, x_values, y_values):

        temp= list(zip(x_values,y_values))
        random.shuffle(temp)
        unpacked = list(zip(*temp))
        return (unpacked[0],unpacked[1])

    def learn(self):
        
        self.setup() 
        epoch_data = {}
        error = 0

        for epoch in range(self.epochs):

            # we shuffle the training set at the start of each
            # iteration so that we do not end up with the same
            # calculations on the same training sets every time

            # X, Y = self.shuffle(self._x_training_set, self._y_training_set)
            
            _iterable = list(range(self.number_of_training_set))
            random.shuffle(_iterable)
            errors =0 
            for i in _iterable:
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

        self.draw_initial_plot(list(epoch_data.keys()), list(epoch_data.values()),'Epochs', 'Error')

def runMyCode():
    learning_rate = 0.01
    epochs = 40
    random_generator_start = -1
    random_generator_end = 1

    perceptron = Perceptron( \
                    epochs = epochs, \
                    learning_rate = learning_rate, \
                    weight_range = [random_generator_start, random_generator_end])

    perceptron.learn()

runMyCode()        