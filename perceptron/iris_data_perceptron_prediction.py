import pandas as panda

from sklearn.model_selection import train_test_split
from predicting_perceptron import Perceptron
from sklearn.metrics import accuracy_score, mean_absolute_error

remote_location = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


data = panda.read_csv(remote_location)       
_x_training_set = list(data.iloc[0:, [0,2]].values)
_y_training_set = [0 if i.lower()!='iris-setosa' else 1 for i in data.iloc[0:, 4].values]


_x_train, _x_test, _y_train, _y_test = train_test_split( \
                                        _x_training_set,\
                                        _y_training_set, \
                                        test_size = 0.3, \
                                        random_state = 1, \
                                        stratify = _y_training_set)


random_generator_start = -1
random_generator_end = 1

perceptron = Perceptron( \
                learning_rate = 0.01, \
                epochs = 40, \
                _x_training_set = _x_train, \
                _y_training_set = _y_train,
                standardize= False
                )

perceptron.learn()
_y_predicted = perceptron.predict(_x_test)
print(_y_test)
print(accuracy_score(_y_predicted, _y_test))
print(mean_absolute_error(_y_predicted, _y_test))
