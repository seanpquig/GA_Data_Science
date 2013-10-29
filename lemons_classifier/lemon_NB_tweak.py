"""
This script builds a model to determine if
a Car is a good buy (0) or bad buy (1)
"""

### Import Modules
import numpy as np
import pandas as pd
from sklearn import metrics, naive_bayes, cross_validation, feature_selection
import lemons_functions as lfun


# def main():
### Import data sets
l_train = pd.read_csv('lemon_training.csv')
l_test = pd.read_csv('lemon_test.csv')


### Clean/prepare data sets
# vectorize nationality
l_train['American'] = l_train['Nationality'].apply(lfun.isAmerican)
l_test['American'] = l_test['Nationality'].apply(lfun.isAmerican)

# vectorize big 3 brands
l_train['Ford'] = l_train['TopThreeAmericanName'].apply(lfun.isFord)
l_train['Chrysler'] = l_train['TopThreeAmericanName'].apply(lfun.isChrysler)
l_train['GM'] = l_train['TopThreeAmericanName'].apply(lfun.isGM)
l_test['Ford'] = l_test['TopThreeAmericanName'].apply(lfun.isFord)
l_test['Chrysler'] = l_test['TopThreeAmericanName'].apply(lfun.isChrysler)
l_test['GM'] = l_test['TopThreeAmericanName'].apply(lfun.isGM)

# vectorize auctions
l_train['Adesa'] = l_train['Auction'].apply(lfun.isAdesa)
l_train['Manheim'] = l_train['Auction'].apply(lfun.isMan)
l_test['Adesa'] = l_test['Auction'].apply(lfun.isAdesa)
l_test['Manheim'] = l_test['Auction'].apply(lfun.isMan)

# create auciton price vs retail measure
l_train['cost vs retail'] = l_train['MMRAcquisitionRetailAveragePrice'] - l_train['VehBCost']
l_train['cost vs retail'] = l_train['cost vs retail'].fillna(0)
l_test['cost vs retail'] = l_test['MMRAcquisitionRetailAveragePrice'] - l_test['VehBCost']
l_test['cost vs retail'] = l_test['cost vs retail'].fillna(0)

# Get log and square of odometer reading to account for non-linear effects
l_train['VehOdo sqr'] = l_train['VehOdo'] **2
l_train['VehOdo log'] = np.log(l_train['VehOdo'])
l_test['VehOdo sqr'] = l_test['VehOdo'] **2
l_test['VehOdo log'] = np.log(l_test['VehOdo'])

wheel_dummies = pd.get_dummies(l_train['WheelType'])
l_train = l_train.join(wheel_dummies)

l_train['MMRAcquisitionAuctionAveragePrice'] = l_train['MMRAcquisitionAuctionAveragePrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionAveragePrice']))
l_train['MMRAcquisitionAuctionCleanPrice'] = l_train['MMRAcquisitionAuctionCleanPrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionCleanPrice']))
l_train['MMRAcquisitionRetailAveragePrice'] = l_train['MMRAcquisitionRetailAveragePrice'].fillna(np.mean(l_train['MMRAcquisitionRetailAveragePrice']))
l_train['MMRAcquisitonRetailCleanPrice'] = l_train['MMRAcquisitonRetailCleanPrice'].fillna(np.mean(l_train['MMRAcquisitonRetailCleanPrice']))
l_train['MMRCurrentAuctionAveragePrice'] = l_train['MMRCurrentAuctionAveragePrice'].fillna(np.mean(l_train['MMRCurrentAuctionAveragePrice']))
l_train['MMRCurrentAuctionCleanPrice'] = l_train['MMRCurrentAuctionCleanPrice'].fillna(np.mean(l_train['MMRCurrentAuctionCleanPrice']))
l_train['MMRCurrentRetailAveragePrice'] = l_train['MMRCurrentRetailAveragePrice'].fillna(np.mean(l_train['MMRCurrentRetailAveragePrice']))
l_train['MMRCurrentRetailCleanPrice'] = l_train['MMRCurrentRetailCleanPrice'].fillna(np.mean(l_train['MMRCurrentRetailCleanPrice']))



l_train = l_train.dropna(axis=1)
l_test = l_test.dropna(axis=1)

features = list(l_train.describe().columns)
features.remove('RefId')
features.remove('IsBadBuy')


### Create test and training sets
train_features = l_train[features].values
train_class = l_train.IsBadBuy.values
# OSS_features = l_test[features].values
# Seed PRNG
np.random.seed(1234)
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(train_features, train_class, test_size=.3)

# feature selection
# print features
# print feature_selection.f_classif(train_features, train_class)


### Build model
model = naive_bayes.GaussianNB().fit(X_train, y_train)
model.score(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


### Stats
print 'training:\n', metrics.confusion_matrix(y_train, y_pred_train)
print metrics.classification_report(y_train, y_pred_train)
print 'test:\n', metrics.confusion_matrix(y_test, y_pred_test)
print metrics.classification_report(y_test, y_pred_test)
fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train, pos_label=1)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
print 'train MA: ', model.score(X_train, y_train)
print 'test MA: ', model.score(X_test, y_test)
print 'train AUC: ', metrics.auc(fpr_train, tpr_train)
print 'test AUC: ', metrics.auc(fpr_test, tpr_test)



# Cross Validation
AUCs = []
for i in xrange(10):
    X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(train_features, train_class, test_size=.3)
    y_pred_test = model.fit(X_train, y_train).predict(X_test)
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
    AUCs.append(metrics.auc(fpr_test, tpr_test))
    
print 'avgerage AUC from cross val: ', np.mean(AUCs)


### Do output predicitons for OSS data
# OSS_features = l_test[features].values
# y_pred_OSS = model.predict(OSS_features)
# submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred_OSS })
# submission.to_csv('submission_NB.csv')


# if __name__ == '__main__':
#     main()
