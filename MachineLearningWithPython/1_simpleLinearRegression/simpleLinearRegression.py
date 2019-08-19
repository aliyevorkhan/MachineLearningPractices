import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("hw_25000.csv")
boy = data.Height.values.reshape(-1,1)
kilo = data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(boy,kilo)

print(regression.predict(np.array([60]).reshape(1,1)))
print(regression.predict(np.array([62]).reshape(1,1)))
print(regression.predict(np.array([64]).reshape(1,1)))
print(regression.predict(np.array([66]).reshape(1,1)))
print(regression.predict(np.array([68]).reshape(1,1)))
print(regression.predict(np.array([70]).reshape(1,1)))

plt.scatter(data.Height, data.Weight)
x = np.arange(min(data.Height), max(data.Height)).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")
plt.xlabel("boy(inch)")
plt.ylabel("kilo(pound)")
plt.title("Simple Linear Regression Module")
plt.show()

print(r2_score(kilo,regression.predict(boy)))