import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.DataFrame({
    'luas_Tanah': [100,150,200,250,300,350,400],
    'jumlah_Kamar': [2,3,3,4,4,5,5],
    'harga_Rumah': [300,400,500,600,700,800,900]
})


X = data[['luas_Tanah', 'jumlah_Kamar']]
y = data['harga_Rumah']


model = LinearRegression()
model.fit(X, y)


print("Koefisien Regresi")
print(f"Luas Tanah   : {model.coef_[0]:.2f}")
print(f"Jumlah Kamar : {model.coef_[1]:.2f}")


print("\nIntercept")
print(f"{model.intercept_:.2f}")


print("\nPersamaan Regresi")

print(f"Harga = {model.intercept_:.2f} + "
      f"({model.coef_[0]:.2f} × Luas Tanah) + "
      f"({model.coef_[1]:.2f} × Jumlah Kamar)")


rumah_baru = [[275,4]]

prediksi = model.predict(rumah_baru)

print("\nPrediksi Harga Rumah")
print(f"Luas Tanah    : {rumah_baru[0][0]} m²")
print(f"Jumlah Kamar  : {rumah_baru[0][1]}")
print(f"Harga Prediksi: {prediksi[0]:.2f} juta rupiah")