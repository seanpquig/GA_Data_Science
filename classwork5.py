import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Linear graph
mammals = pd.read_csv('mammals.csv')
plt.scatter(mammals['body'], mammals['brain'])
plt.show()
plt.hist(mammals['body'], bins=range(0, 10000, 100))
plt.show()
plt.hist(mammals['brain'], bins=range(0, 10000, 100))
plt.show()

# Log graph
mammals['log_body'] = np.log(mammals['body'])
mammals['log_brain'] = np.log(mammals['brain'])
plt.scatter(mammals['log_body'], mammals['log_brain'])
plt.show()

# Simple regression of body, brain
from sklearn import linear_model
# Make the model object
regr = linear_model.LinearRegression()
# Fit the data
body = [[x] for x in mammals['body'].values]
brain = mammals['brain'].values
regr.fit(body, brain)
# Display the coefficients:
regr.coef_
# Display our SSE:
np.mean((regr.predict(body) - brain) ** 2)
# Scoring our model (closer to 1 is better!)
regr.score(body, brain)
plt.scatter(body, brain)
plt.plot(body, regr.predict(body), color='blue', linewidth=3)
plt.show()


log_regr = linear_model.LinearRegression()
log_body = [[x] for x in log(mammals['body'].values)]
log_brain = log(mammals['brain'].values)
log_regr.fit(log_body, log_brain)




