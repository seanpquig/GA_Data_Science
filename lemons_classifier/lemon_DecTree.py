"""
This script builds a mode to determine if
a Car is a good buy (0) or bad buy (1)
"""

### Import Modules
import numpy as np
import pandas as pd
from sklearn import metrics, tree, cross_validation


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
    clf = tree.DecisionTreeClassifier(max_depth=99999, min_samples_leaf=1).fit(X_train, y_train)
    clf.score(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)


    ### Stats
    print '\ntraining:\n', metrics.confusion_matrix(y_train, y_pred_train)
    print metrics.classification_report(y_train, y_pred_train)
    print 'test:\n', metrics.confusion_matrix(y_test, y_pred_test)
    print metrics.classification_report(y_test, y_pred_test)
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train, pos_label=1)
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
    print 'train MA: ', clf.score(X_train, y_train)
    print 'test MA: ', clf.score(X_test, y_test)
    print 'train AUC: ', metrics.auc(fpr_train, tpr_train)
    print 'test AUC: ', metrics.auc(fpr_test, tpr_test)

    #Create visual output of graph
    # import StringIO
    # import pydot
    # graph_out = StringIO.StringIO()
    # tree.export_graphviz(clf, out_file=graph_out)
    # graph = pydot.graph_from_dot_data(graph_out.getvalue())
    # graph.write_pdf('graph.pdf')
    # graph_out.close()

    # y_pred = clf.predict(test_X)
    # submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred })
    # submission.to_csv('submission.csv')



if __name__ == '__main__':
    main()


