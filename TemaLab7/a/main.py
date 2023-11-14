import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("auto-mpg.csv")

data = data.dropna()

plt.scatter(data['horsepower'], data['mpg'])
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.title('Relația dintre horsepower și mpg')
plt.show()
