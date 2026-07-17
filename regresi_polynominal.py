import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Membaca dataset
file = pd.read_csv("dataset.handphone.csv")

# Variabel
x = file[["Ram"]]
y = file["Harga"]

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

# Model
model = LinearRegression()
model.fit(X_poly, y)

# Prediksi
y_pred = model.predict(X_poly)

# Menampilkan hasil
print("Koefisien Polynomial Regression:", model.coef_)
print("Intercept:", model.intercept_)

# Mengurutkan data agar garis rapi
urut = x["Ram"].argsort()

plt.scatter(x["Ram"], y, label="Data Aktual")
plt.plot(
    x["Ram"].iloc[urut],
    y_pred[urut],
    label="Model Polynomial"
)

plt.xlabel("RAM (GB)")
plt.ylabel("Harga HP")
plt.title("Polynomial Regression: RAM vs Harga HP")
plt.legend()

plt.show()