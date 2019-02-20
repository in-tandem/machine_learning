
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def getRegressionData(n_samples = 100, n_features = 6):

    X, y, coeff = make_regression(n_samples = n_samples, n_features = n_features, shuffle = True, random_state = 12, coef = True )

    return (X, y)

def create_regressor(x_data,y_data):

    shape_of_input = x_data.shape
    shape_of_target = y_data.shape

    model  = Sequential()

    ## number of neurons = 30
    ## kernel_initializer determines how the weights are initialized
    ## activation is the activation function at this particular hidden layer
    ## input_shape is the number of features in a single row.. in this case it is shape_of_input[1]
    ## shape_of_input[0] is the total number of such rows
    model.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'normal', input_dim = shape_of_input[1]))

    # model.add(Dense(units = 30, activation = 'relu', kernel_initializer = 'uniform'))
    # model.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'normal'))
    ## we are predicting single value for the target
    # model.add(Dense(1, activation = 'softmax'))

    model.add(Dense(1))
    ## mean squared error  is the loss function for regression
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model


def fit(model, x_train, y_train):


    pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('keras_regressor', model)
        ])

    param_grid = {

        'keras_regressor__batch_size' : [20,30,50],
        'keras_regressor__epochs' : [100, 200, 300],
        'keras_regressor__x_data' : [x_train],
        'keras_regressor__y_data' : [y_train],
        
    }

    cross_validator = KFold(n_splits = 10, random_state = 12)
    grid = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = cross_validator, n_jobs = -1)
    # grid = GridSearchCV(estimator = pipeline, param_grid = param_grid, n_jobs = -1)
    grid.fit(x_train, y_train)


    print("Best parameters are : ", grid.best_params_, '\n grid best score :', grid.best_score_)


    return grid.best_estimator_


def predict(model, x_test, y_test):

    y_predicted = model.predict(x_test)

    return y_predicted

def execute():

    X,y = getRegressionData(n_samples = 1000, n_features = 4)

    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

    regressor =  KerasRegressor(build_fn = create_regressor)

    best_regressor = fit(regressor,x_train,y_train)

    y_target = best_regressor.predict(x_test)

    actual_error  = mean_squared_error(y_test, y_target)

    print('actual error: ', actual_error)


execute()