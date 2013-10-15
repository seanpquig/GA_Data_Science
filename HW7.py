import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, neighbors, datasets, feature_selection
import re


# Do Logistic regression on beer data (class 6 work)
beer = pd.read_csv('http://www-958.ibm.com/software/analytics/'
    'manyeyes/datasets/af-er-beer-dataset/versions/1.txt', delimiter="\t")
beer = beer.dropna()

def good(x):
  if x > 4.3:
    return 1
  else:
    return 0

beer['Good'] = beer['WR'].apply(good)

input = beer[ ['Reviews', 'ABV'] ].values
good = beer['Good'].values
logm = linear_model.LogisticRegression()
logm.fit(input, good)
logm.predict(input)
logm.score(input, good)

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

def lager_match(s):
    if re.search('Lager', s):
        return 1
    else:
        return 0

beer['Ale'] = beer['Type'].apply(ale_match)
beer['Stout'] = beer['Type'].apply(stout_match)
beer['IPA'] = beer['Type'].apply(ipa_match)
beer['Lager'] = beer['Type'].apply(lager_match)


