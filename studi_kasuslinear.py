import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([100,200,150,250,300]).reshape(-1,1)
y = np.array([500,700,600,850,900])

model = LinearRegression()
model.fit(x,y)

prediksi = model.predict([[180]])
print(f"Prediksi harga rumah dengan luas tanah 180 m²: {prediksi[0]:.2f}")