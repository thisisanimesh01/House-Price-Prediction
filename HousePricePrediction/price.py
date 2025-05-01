import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('/Users/animesh/Desktop/python/house price predtiction/USA_Housing.csv')

print(data.head())

data = data.drop(['Address'], axis=1 )
data.head()

sns.heatmap(data.isnull())
plt.show()

X = data.drop(['Price'], axis=1)
Y = data['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print(predictions)

error = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
print(error)

