### PROBLEM OVERVIEW
"""
Problem:  use the provided baseball dataset to create a model 
    for predicting player salaries in the 2012 season that uses
    2012 statistics as input.  Use 2011 season as the training set.
"""

### IMPORT MODULES
import pandas as pd
import numpy as np
from sklearn import svm, feature_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error

### LOAD DATA AND CREATE NEW FEATURES 
data = pd.read_csv('baseball_pitching.csv')
# functions for new feature creation
def has_nick(nickName):
    if nickName:
        return 1
    else:
        return 0

def is_yankee(teamID):
    if teamID == 'NYA':
        return 1
    else:
        return 0

data = data[ data['HR'].notnull() ]
data['has nickname'] = data['nameNick'].notnull()
data['has nickname'] = data['has nickname'].apply(has_nick)
data['is yankee'] = data['teamID'].apply(is_yankee)
data['Age'] = data['yearID'] - data['birthYear']
data['Age sqr'] = data['Age'] ** 2
data['logAge'] = np.log(data['Age'])

### PREPARE TRAINING DATA
data_2011 = data[ data['yearID'] == 2011 ]
salary_2011 = data_2011['salary'].values
input_2011 = data_2011[ ["HR", "RBI", 'R', 'X2B', 'X3B', "G", "SB", 'AB', 'CS', 'BB', 'SO', 
    'IBB', 'HBP', 'SF', 'height', 'weight', 'has nickname', 'is yankee', 'SO pitcher', 
    'Age', 'Age sqr'] ]
feature_selection.univariate_selection.f_regression(input_2011, salary_2011)

### PREPARE TESTING DATA
data_2012 = data[ data['yearID'] == 2012 ]
salary_2012 = data_2012['salary'].values
input_2012 = data_2012[ ["HR", "RBI", 'R', 'X2B', 'X3B', "G", "SB", 'AB', 'CS', 'BB', 'SO', 
    'IBB', 'HBP', 'SF', 'height', 'weight', 'has nickname', 'is yankee', 'SO pitcher',
    'Age', 'Age sqr'] ]

### CREATE MODEL
regr = svm.SVR(kernel='poly', degree=2, C=3)
regr.fit(input_2011, salary_2011)
data_2011['salary_predict'] = regr.predict(input_2011)
data_2012['salary_predict'] = regr.predict(input_2012)

def fix_neg_salary(salary):
    if salary < 480000:
        return 480000
    else:
        return salary

data_2011['salary_predict'] = data_2011['salary_predict'].apply(fix_neg_salary)
data_2012['salary_predict'] = data_2012['salary_predict'].apply(fix_neg_salary)
predict_2011 = data_2011['salary_predict'].values
predict_2012 = data_2012['salary_predict'].values

### STATISTICS
print 'R squared train:  ', regr.score(input_2011, salary_2011)
print 'R squared test:  ', regr.score(input_2012, salary_2012)
print 'MSE train:  ', mean_squared_error(predict_2011, salary_2011)
print 'MSE_test:  ', mean_squared_error(predict_2012, salary_2012)
print 'MAE train:  ', mean_absolute_error(predict_2011, salary_2011)
print 'MAE test:  ', mean_absolute_error(predict_2012, salary_2012), '\n'

### OUTPUT DATA
# output_df = data_2012[['playerID', 'yearID', 'salary', 'salary_predict']]
# output_df.to_csv('output_svr.csv')


