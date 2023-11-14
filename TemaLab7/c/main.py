import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("auto-mpg.csv")

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

data = data.dropna(subset=['horsepower', 'mpg'])

df = data[['horsepower', 'mpg']]

X = df[['horsepower']]
y = df['mpg']

model = LinearRegression()

model.fit(X, y)

alpha = model.intercept_
beta = model.coef_[0]

y_pred = model.predict(X)

plt.scatter(X, y, label='Date observate')
plt.plot(X, y_pred, color='red', label=f'Dreapta de regresie: mpg = {alpha:.2f} + {beta:.2f} * horsepower')
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
