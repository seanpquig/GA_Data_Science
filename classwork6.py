import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection, neighbors, linear_model

# stop_dist = pd.read_csv('stop_distance.csv')
# car_mpg = pd.read_csv('car_MPG.csv')



# Do Logistic regression on beer data (class 6 work)
import re
beer = pd.read_csv('http://www-958.ibm.com/software/analytics/'
    'manyeyes/datasets/af-er-beer-dataset/versions/1.txt', delimiter="\t")
beer = beer.dropna()

def good(x):
  if x > 4.3:
    return 1
  else:
    return 0

# regex matching functions for beer type categorization
def ale_match(s):
    if re.search('Ale', s):
        return 1
    else:
        return 0

def stout_match(s):
    if re.search('Stout', s):
        return 1
    else:
        return 0

def ipa_match(s):
    if re.search('IPA', s):
        return 1
    else:
        return 0

beer['Good'] = beer['WR'].apply(good)
beer['Ale'] = beer['Type'].apply(ale_match)
beer['Stout'] = beer['Type'].apply(stout_match)
beer['IPA'] = beer['Type'].apply(ipa_match)

good = beer['Good'].values
input = beer[ ['Reviews', 'ABV'] ].values
# input = beer[ ['Ale', 'Stout', 'IPA'] ].values
# input = beer[ ['ABV', 'Stout']].values

logm = linear_model.LogisticRegression()
feature_selection.univariate_selection.f_regression(input, good)
logm.fit(input, good)
logm.predict(input)
logm.score(input, good)
beer['Good predict Logistic'] = logm.predict(input)


# KNN prediction
clf = neighbors.KNeighborsClassifier(10, weights='uniform')
clf.fit(input, good)
beer['Good predict KNN'] = clf.predict(input)

beer.to_csv('beer.csv')


