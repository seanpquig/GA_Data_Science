"""
This script builds a mode to determine if
a Car is a good buy (0) or bad buy (1)
"""

### IMPORT MODULES
import numpy as np
import pandas as pd
from sklearn import metrics, naive_bayes, cross_validation, feature_selection


### IMPORT DATA SETS
l_train = pd.read_csv('lemon_training.csv')


### SET UP FEATURES
# auction dummies
# auct_dummies = pd.get_dummies(l_train['Auction'])
# auct_dummies = auct_dummies.rename(columns={'OTHER': 'OTHER_auction'})
# l_train = l_train.join(auct_dummies[['MANHEIM', 'ADESA']])

# make dummies
make_dummies = pd.get_dummies(l_train['Make'])
l_train = l_train.join(make_dummies[['CHEVROLET', 'DODGE']])

# color dummies
# color_dummies = pd.get_dummies(l_train['Color'])
# color_dummies = color_dummies.rename(columns={'OTHER': 'OTHER_color'})
# l_train = l_train.join(color_dummies[['BLUE']])

# transmission dummies
# trans_dummies = pd.get_dummies(l_train['Transmission'])
# del trans_dummies['Manual']
# l_train = l_train.join(trans_dummies)

# wheel type dummies
# l_train['WheelType']=l_train['WheelType'].fillna('Wheel_NAN')
wheel_dummies = pd.get_dummies(l_train['WheelType'])
l_train = l_train.join(wheel_dummies)



# nationality dummies
# nation_dummies = pd.get_dummies(l_train['Nationality'])
# nation_dummies = nation_dummies.rename(columns={'OTHER': 'OTHER_nation'})
# l_train = l_train.join(nation_dummies)

# size dummies
size_dummies = pd.get_dummies(l_train['Size'])
l_train = l_train.join(size_dummies)

# big 3 dummies
big3_dummies = pd.get_dummies(l_train['TopThreeAmericanName'])
big3_dummies = big3_dummies.rename(columns={'OTHER': 'OTHER_big3', 'CHRYSLER': 'CHRYSLER_big3', 'FORD': 'FORD_big3'})
l_train = l_train.join(big3_dummies)

# fill in empty benchmark prices
l_train['MMRAcquisitionAuctionAveragePrice'] = l_train['MMRAcquisitionAuctionAveragePrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionAveragePrice']))
l_train['MMRAcquisitionAuctionCleanPrice'] = l_train['MMRAcquisitionAuctionCleanPrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionCleanPrice']))
# l_train['MMRAcquisitionRetailAveragePrice'] = l_train['MMRAcquisitionRetailAveragePrice'].fillna(np.mean(l_train['MMRAcquisitionRetailAveragePrice']))
# l_train['MMRAcquisitonRetailCleanPrice'] = l_train['MMRAcquisitonRetailCleanPrice'].fillna(np.mean(l_train['MMRAcquisitonRetailCleanPrice']))
l_train['MMRCurrentAuctionAveragePrice'] = l_train['MMRCurrentAuctionAveragePrice'].fillna(np.mean(l_train['MMRCurrentAuctionAveragePrice']))
l_train['MMRCurrentAuctionCleanPrice'] = l_train['MMRCurrentAuctionCleanPrice'].fillna(np.mean(l_train['MMRCurrentAuctionCleanPrice']))
l_train['MMRCurrentRetailAveragePrice'] = l_train['MMRCurrentRetailAveragePrice'].fillna(np.mean(l_train['MMRCurrentRetailAveragePrice']))
l_train['MMRCurrentRetailCleanPrice'] = l_train['MMRCurrentRetailCleanPrice'].fillna(np.mean(l_train['MMRCurrentRetailCleanPrice']))

# state dummies
st_dummies = pd.get_dummies(l_train['VNST'])
l_train = l_train.join(st_dummies[['FL', 'KY', 'MD', 'TX']])

# create new variables from others
l_train['VehicleAge sqr'] = l_train['VehicleAge'] **2
l_train['VehYear log'] = np.log(l_train['VehYear'])
l_train['VehOdo sqr'] = l_train['VehOdo'] **2
l_train['VehOdo log'] = np.log(l_train['VehOdo'])



l_train = l_train.dropna(axis=1)
features = list(l_train.describe().columns)
features.remove('RefId')
features.remove('IsBadBuy')
##### p-value pruning
# features.remove('VNZIP1')
features.remove('IsOnlineSale')
# features.remove('OTHER_auction')
# features.remove('Special')
features.remove('SMALL SUV')
features.remove('SMALL TRUCK')
features.remove('VAN')
# features.remove('OTHER_big3')




### Create test and training sets
train_features = l_train[features].values
train_class = l_train.IsBadBuy.values
# Seed PRNG
np.random.seed(1234)
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(train_features, train_class, test_size=.35)

### FEATURE SELECTION
# p_vals = feature_selection.f_classif(train_features, train_class)[1]
# for i, feat in enumerate(features):
#     print i, '  ', feat, '  ', p_vals[i]



### Build model
model = naive_bayes.GaussianNB().fit(X_train, y_train)
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
for i in xrange(20):
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(train_features, train_class, test_size=.35)
    y_pred_test = model.fit(X_train, y_train).predict(X_test)
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
    AUCs.append(metrics.auc(fpr_test, tpr_test))
    
print 'avgerage AUC cross val: ', np.mean(AUCs)
print 'min AUC cross val: ', min(AUCs)
# print AUCs

### Do output predicitons for OSS data
# OSS_features = l_test[features].values
# y_pred_OSS = model.predict(OSS_features)
# submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred_OSS })
# submission.to_csv('submission_NB.csv')


# if __name__ == '__main__':
#     main()
