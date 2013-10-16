import random
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
authorship = read_csv('http://people.stern.nyu.edu/jsimonof/AnalCatData/Data/Comma_separated/authorship.csv')
authors = list(set(authorship.Author.values))
le = preprocessing.LabelEncoder()
le.fit(authors)
authorship['Author_num'] = le.transform(authorship['Author'])

#What are some of the stop words we're looking at?
features = list(authorship.columns)
features
features.remove('Author')
features.remove('Author_num')

# Create a random variable (random forests work best with a random variable)
# and create a test and training set
authorship['random'] = [random.random() for i in range(841)]
x_train, x_test, y_train, y_test = train_test_split(authorship[features], authorship.Author_num.values, test_size=0.4, random_state=123)


# Fit Model
etclf = ExtraTreesClassifier(n_estimators=20)
etclf.fit(x_train, y_train)

# Print Confusion Matrix
metrics.confusion_matrix(etclf.predict(x_test), y_test)
print metrics.classification_report(etclf.predict(x_test), y_test)