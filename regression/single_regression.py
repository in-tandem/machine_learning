from matplotlib import pyplot as plot 
import pandas as panda 
from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np 


def singleLinearRegression(n_samples = 10, n_features = 2):

    X, y, coeff = make_regression(n_samples = 10, n_features = 2, shuffle = True, random_state = 12, coef = True )

    return (X, y)

def createPairPlot(data_frame):

    sns.pairplot(data_frame)
    plot.show()

def plotCorrelationCoefficients(data_frame):
    '''
    you would have to transpose to get an even square 
    so fr eg..dataframe is 10 * 3 matrix. if we dnt 
    transpose cor coef will be a 10 * 10 matrix . which we 
    dont want . we want a 3*3 matrix right..since there are 3 features
    and we want correlation between thiose 3 features. hence we transpose
    our data frame to take it to a 3*10 matrix
    '''
    correlation_coeff = np.corrcoef(data_frame[data_frame.columns.tolist()].values.T)
    sns.heatmap(

            correlation_coeff,
            cbar = True,
            annot = True,
            fmt = '.2f',
            yticklabels = data_frame.columns.tolist(),
            xticklabels = data_frame.columns.tolist(),
    )

    plot.show()

def drawLinearPlot(X,y, model):

    print(X,y)
    plot.scatter(X, y, marker = 'o', s = 45, color = 'blue') ## s had to be a number, if in string it was giving a very weird error
    plot.plot(X, model.predict(X[:, np.newaxis]), color = 'red')
    plot.show()

def regressAgainstOneVariable(X,y):

    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    x_train_scaled = StandardScaler().fit_transform(x_train)

    x_test_scaled = StandardScaler().fit_transform(x_test)

    regressor = LinearRegression()

    training_set = x_train_scaled[: , 0][:, np.newaxis] ## converting 1 d array to 2d, else linearregression was giving errors

    test_set = x_test_scaled[:, 0 ][:, np.newaxis]

    regressor.fit(training_set, y_train) ## we will regress against only one

    y_predict = regressor.predict(test_set)

    print('Mean squared error is ', mean_squared_error(y_test, y_predict), '\n r2 score is ', r2_score(y_test, y_predict))
    drawLinearPlot(X[:, 0],y, regressor)

def regress(X,y):

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    regressor = LinearRegression()

    regressor.fit(x_train, y_train) ## we will regress against only one

    y_predict = regressor.predict(x_test)

    print('Mean squared error is ', mean_squared_error(y_test, y_predict), '\n r2 score is ', r2_score(y_test, y_predict))
    

def analyzeRegressionModel():

    X, y = singleLinearRegression(n_samples = 50)
    
    data_frame = panda.DataFrame(X, columns = ['c1', 'c2'])

    data_frame['y'] =  y 

    createPairPlot(data_frame)
    plotCorrelationCoefficients(data_frame)

    '''
    even though we are taking random regreswsion values, right now based on the view at
    the seaborne pairplot and correlation coeff we can see that d1 vlaues are linearly
    impacting y trarget vcalues. and there is close to no relation with c2 and y
     our regression tools are hence going to regress c1 vs y

    furthermore, we can do another regression of c1,c2 vs y 
    and check for metrics comparison
    '''


    regressAgainstOneVariable(X,y)
    regress(X,y)

analyzeRegressionModel()


