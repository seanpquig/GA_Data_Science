"""
This script builds a mode to determine if
a Car is a good buy (0) or bad buy (1)
"""

### Import Modules
import pandas as pd
from sklearn import metrics, tree, cross_validation


### Import data sets
l_train = pd.read_csv('lemon_training.csv')
l_test = pd.read_csv('lemon_test.csv')


### Clean/prepare data sets
l_train = l_train.dropna(axis=1)
l_test = l_test.dropna(axis=1)

features = list(l_train.describe().columns)
features.remove('RefId')
features.remove('IsBadBuy')


### Create test and training sets
train_features = l_train[features].values
train_class = l_train.IsBadBuy.values
OSS_features = l_test[features].values

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(train_features, train_class, test_size=.3)


### Build model
clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
clf.score(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


print 'mislabeled points ', (train_y!=y_pred_train).sum()
print metrics.confusion_matrix(y_train, y_pred_train)
print metrics.confusion_matrix(y_test, y_pred_test)
print metrics.classification_report(y_train, y_pred_train)
print metrics.classification_report(y_test, y_pred_test)


# y_pred = clf.predict(test_X)
# submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred })
# submission.to_csv('submission.csv')


### 