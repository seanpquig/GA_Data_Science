import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from pandas import DataFrame, read_csv

mammals = read_csv('http://bit.ly/1f2YPsC').sort('body')
lm = svm.SVR(kernel='linear', C=1e1)
lm_rbf = svm.SVR(kernel='rbf', C=1e1)

body = mammals[ ['body'] ].values
brain = mammals.brain.values

lm.fit(body, brain)
lm_rbf.fit(np.log(body), np.log(brain))

## Compare to the original log fit model, as well as other svm kernels:
from sklearn.linear_model import LinearRegression
logfit = LinearRegression().fit(np.log(body), np.log(brain))
mammals['log_regr'] = np.exp(logfit.predict(np.log(body)))
mammals['linear_svm'] = lm.predict(body)
mammals['rbf_svm'] = np.exp(lm_rbf.predict(np.log(body)))

plt.scatter(body, brain)
plt.plot(body, mammals['linear_svm'].values, c='r', label='linear svm')
plt.plot(body, mammals['rbf_svm'].values, c='g', label='gaussian svm')
plt.plot(body, mammals['log_regr'].values, c='b', label='linear regression')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend(loc=2)
plt.show()

for prediction in ('linear_svm', 'rbf_svm', 'log_regr'):
    print 'Mean Squared Error for', prediction, ':', metrics.mean_squared_error(mammals[ [prediction] ].values, mammals[ ['brain'] ].values)
    print 'R-Squared for', prediction, ':', metrics.r2_score(mammals[ [prediction] ].values, mammals[ ['brain'] ].values)
