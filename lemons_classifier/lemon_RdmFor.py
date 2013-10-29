"""
This script builds a mode to determine if
a Car is a good buy (0) or bad buy (1)
"""

### Import Modules
import numpy as np
import pandas as pd
from sklearn import metrics, naive_bayes, cross_validation
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
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

    # Seed PRNG
    np.random.seed(1234)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(train_features, train_class, test_size=.3)


    ### Build model
    # model = naive_bayes.GaussianNB().fit(X_train, y_train)
    model = ExtraTreesClassifier(max_depth=8).fit(X_train, y_train)
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
    # for i in xrange(10):
    #     X_train, X_test, y_train, y_test = \
    #     cross_validation.train_test_split(train_features, train_class, test_size=.3)
    #     y_pred_test = model.fit(X_train, y_train).predict(X_test)
    #     fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
    #     AUCs.append(metrics.auc(fpr_test, tpr_test))
        
    # print 'AUC cross val: ', AUCs


    ### Do output predicitons for OSS data
    OSS_features = l_test[features].values
    y_pred_OSS = model.predict(OSS_features)
    submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred_OSS })
    # submission.to_csv('submission_NB.csv')


if __name__ == '__main__':
    main()
