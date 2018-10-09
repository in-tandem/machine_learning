'''
implement iris data set perceptron learning using scikit learning
modules

'''

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

iris = datasets.load_iris()
_x_training_set = iris.data[: , [2,3]]
_y_training_set = iris.target 
print(_x_training_set)
print(_y_training_set)

_x_train, _x_test, _y_train, _y_test = train_test_split( \
                                        _x_training_set,\
                                        _y_training_set, \
                                        test_size = 0.3, \
                                        random_state = 1, \
                                        stratify = _y_training_set)

standardize = StandardScaler()
standardize.fit(_x_training_set)
_x_train_standardized = standardize.transform(_x_train)
_x_pred_standardized = standardize.transform(_x_test)
print(_x_train_standardized)
print(_y_train)

perceptron = Perceptron(max_iter=10,eta0 = 0.1, random_state=1)
perceptron.fit(_x_train_standardized,_y_train)

_y_pred = perceptron.predict(_x_pred_standardized)
print("Accuracy of training set %s" %accuracy_score(_y_test,_y_pred))
print(mean_absolute_error(_y_test, _y_pred))
