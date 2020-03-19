import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import max_error
from sklearn.metrics import accuracy_score

def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)

data = pd.read_csv("data.csv", nrows = 300)
zaman = np.arange(300).reshape(-1,1)
tam_1 = data.tam_1.values.astype(float)
                
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(zaman, tam_1)

plt.scatter(zaman, data.tam_1)
x = np.arange(min(zaman), max(zaman)).reshape(-1,1)
plt.plot(x, y_rbf.predict(x),color="red")
plt.xlabel("zaman")
plt.ylabel("tam_1")
plt.title("Ornek gosterim")
plt.show()
