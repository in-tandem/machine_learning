from matplotlib import pyplot as plot 
import pandas as panda 
from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression, RANSACRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

# import seaborn as sns
import numpy as np 
import datetime, time


classifiers = [
    LinearRegression(),
    RANSACRegressor(),
    DecisionTreeRegressor(random_state = 1, criterion = 'mse'),
    RandomForestRegressor(random_state = 1, criterion = 'mse'),
    SGDRegressor(),
    SVR( kernel = 'rbf'),    
]


classifier_names = [
            'linear_regression',
            'ransac_regression',
            'decisiontree_regression',
            'randomforest_regression',
            'gradient_descent_regression',
            'svr',               
    
]

classifier_param_grid = [
            
            {},
            {'ransac_regression__min_samples':[50, 75, 125, 200], 'ransac_regression__max_trials':[50, 125, 200], 'ransac_regression__residual_threshold':[5, 10, 14]},
            {'decisiontree_regression__max_depth':[6,7,8,9,10,11]},
            {'randomforest_regression__n_estimators':[1,2,3,5,6]} ,
            {'gradient_descent_regression__max_iter' : [100, 200, 300]},
            {'svr__C':[1, 10, 20]},
    
]


    

class CodeTimer:
    
    """
        Utility custom contextual class for calculating the time 
        taken for a certain code block to execute
    
    """
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        time_taken = datetime.timedelta(milliseconds = self.took)
        print('Code block' + self.name + ' took(HH:MM:SS): ' + str(time_taken))



def runGridSearchAndPredict(pipeline, x_train, y_train, x_test, y_test, param_grid, n_jobs = 1, cv = 10, score = 'neg_mean_squared_error'):
    
    response = {}
    training_timer       = CodeTimer('training')
    testing_timer        = CodeTimer('testing')

    with training_timer:

        gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = cv, n_jobs = n_jobs, scoring = score)

        search = gridsearch.fit(x_train,y_train)

        print("Grid Search Best parameters ", search.best_params_)
        print("Grid Search Best score ", search.best_score_)
            
    with testing_timer:
        y_prediction = gridsearch.predict(x_test)
            
    print("Mean squared error score %s" %mean_squared_error(y_test,y_prediction))
    
    response['testing_time'] = testing_timer.took
    response['_y_prediction'] = y_prediction
    response['training_time'] = training_timer.took    
    response['mean_squared_error'] = mean_squared_error(y_test,y_prediction)
    response['r2_score'] = r2_score(y_test,y_prediction)
    
    
    return response
    
def getRegressionData(n_samples = 10, n_features = 2):

    X, y, coeff = make_regression(n_samples = n_samples, n_features = n_features, shuffle = True, random_state = 12, coef = True )

    return (X, y)

def analyzeRegressionModel():

    X, y = getRegressionData(n_samples = 400, n_features = 4)
    
    _x_train, _x_test, _y_train, _y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

    model_metrics = {}

    for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

            pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    (model_name, model)
            ])

            result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, model_param_grid)

            _y_prediction = result['_y_prediction']

            model_metrics[model_name] = {}
            model_metrics[model_name]['training_time'] = result['training_time']
            model_metrics[model_name]['testing_time'] = result['testing_time']
            model_metrics[model_name]['r2_score'] = result['r2_score']
            model_metrics[model_name]['mean_squared_error'] = result['mean_squared_error']
            
    
    print('Model metrics are \n :', model_metrics)


analyzeRegressionModel()    