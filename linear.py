import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,6,8,10])

model = LinearRegression()
model.fit(x,y)

prediksi = model.predict([[6]])
print(f"Prediksi untuk x = 6: {prediksi[0]}")