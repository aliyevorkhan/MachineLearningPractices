# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data =pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1)
salary  = data.iloc[:,2].values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level, salary)

print(regression.predict(np.array([8.3]).reshape(1,1)))

plt.scatter(level,salary,color="red")
plt.xlabel("level")
plt.ylabel("salary")
plt.title("Decision Tree Model")
x = np.arange(min(level),max(level), 0.01).reshape(-1,1)
plt.plot(x, regression.predict(x),color="blue")
plt.show()