# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv("positions.csv")


#print(data.columns)

level = data.iloc[:, 1].values.reshape(-1,1)
salary = data.iloc[:, 2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level, salary)

tahmin = regression.predict(np.array([8.3]).reshape(1,1))



polynomialRegression = PolynomialFeatures(degree = 4)
polinomialLevel = polynomialRegression.fit_transform(level)
regression2 = LinearRegression()
regression2.fit(polinomialLevel, salary)

tahmin2 = regression2.predict(polynomialRegression.fit_transform(np.array([8.3]).reshape(1,1)))

plt.scatter(level, salary, color="red")
plt.plot(level, regression.predict(level), color="blue")
plt.plot(level, regression2.predict(polinomialLevel), color="green")
plt.show()

