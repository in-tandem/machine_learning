import pandas as panda
import matplotlib.pyplot as plot 
import random
import numpy as np

remote_location = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

class Perceptron(object):
    
    def __init__(self, epochs, learning_rate, weight_range = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_range = weight_range if weight_range else [-1, 1]
        self.weights = []
        self._x_training_set = None
        self._y_training_set = None
        self.number_of_training_set = 0        

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

        return len(self._x_training_set)

    def initialize_weights(self, number_of_weights):
        random_weights = [random.uniform(self.weight_range[0], self.weight_range[1]) 
                                for i in range(1, number_of_weights)]
        self.weights.append(-1) # setting up bias unit 
        self.weights.extend(random_weights)

    def draw_initial_plot(self, _x_data, _y_data, _x_label, _y_label):
        
        plot.xlabel(_x_label)
        plot.ylabel(_y_label)
        plot.scatter(_x_data,_y_data)
        plot.show()

    def learn(self):
        
        self.setup() 
        epoch_data = {}
        error = 0

        for epoch in range(self.epochs):

            for i in range(self.number_of_training_set):
                _x = self._x_training_set[i]
                _desired = self._y_training_set[i]
                _weight = self.weights
                guess = _weight[0] ## setting up the bias unit

                #below iteration is no longer required 
                # as we would be using numpy matrix multiplication
                # for j in range(len(_x)):
                #     guess += _weight[j+1] * _x[j]
                
                guess = guess + np.dot(_x, _weight[1:])

                error = _desired - guess

                ## i am going to reset all the weights
                if error!= 0 :

                    ## resetting the bias unit
                    self.weights[0] = error * self.learning_rate

                    ## we would no longer iterate since 
                    ## we would be using numpy matrix multiplication
                    ## _x is an instance of numpy.array
                    # for j in range(len(_x)):
                    #     self.weights[j+1] = self.weights[j+1] + error * self.learning_rate * _x[j]

                    self.weights[1:] = self.weights[1:] + error * self.learning_rate * _x

            #saving error at the end of the training set        
            epoch_data[epoch] = error
        
        print(epoch_data)

        self.draw_initial_plot(list(epoch_data.keys()), list(epoch_data.values()),'Epochs', 'Error')

def runMyCode():
    learning_rate = 0.01
    epochs = 10
    random_generator_start = -1
    random_generator_end = 1
    perceptron = Perceptron(epochs, learning_rate, [random_generator_start, random_generator_end])
    perceptron.learn()

runMyCode()        