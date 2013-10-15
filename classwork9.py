from sklearn import datasets, metrics, tree, cross_validation
from matplotlib import pyplot as plt
iris = datasets.load_iris()
y_pred = tree.DecisionTreeClassifier().fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points : %d" % (iris.target != y_pred).sum())
print("Absolutely ridiculously overfit score: %d" % (tree.DecisionTreeClassifier().fit(iris.data, iris.target).score(iris.data, iris.target)))


metrics.confusion_matrix(iris.target, y_pred)

print metrics.classification_report(iris.target, y_pred)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=.3)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))
metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))

clf.set_params(min_samples_leaf=5)
clf.set_params(max_depth=5)
clf.fit(x_train, y_train)
metrics.confusion_matrix(y_train, clf.predict(x_train))
metrics.confusion_matrix(y_test, clf.predict(x_test))