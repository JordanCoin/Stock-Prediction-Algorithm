import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm


stock = quandl.get("WIKI/TSLA")

forecast_out = int(30)  # 30 day forecast
stock['Prediction'] = stock[['Adj. Close']].shift(-forecast_out)

# Defining Features & Labels

X = np.array(stock.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]  # set X_forecast equal to last 30
X = X[:-forecast_out]  # remove last 30 from X

y = np.array(stock['Prediction'])
y = y[:-forecast_out]

# Linear Regression Model

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train, y_train)

# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
# print(forecast_prediction)

for forecast in forecast_prediction:
    print("daily forecast: ", forecast)

# TO-DO: Display on matplotlib

# fig = plt.figure()
plt.ylabel('Price')
plt.xlabel('Day')
plt.title('TSLA 30 Day Prediction Chart')
plt.plot(y, X)
plt.axis([0, len(forecast_prediction), 0, 500])
# ax = fig.add_subplot(111)
# ax.plot(y, X)
plt.show()
