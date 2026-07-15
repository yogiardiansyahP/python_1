import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



file = pd.read_csv("dataset.handphone.csv")

print("Ukuran data:", file.shape)
print(file.head(5))
print(file.dtypes)


print("\nJumlah Missing Value")
print(file.isnull().sum())

file.fillna(file.select_dtypes(include="number").median(), inplace=True)


print("\nStatistik Data")
print(file.describe())


plt.figure(figsize=(8,5))
sns.histplot(file["Harga"], bins=20, kde=True)
plt.title("Distribusi Harga")
plt.show()


plt.figure(figsize=(8,2))
sns.boxplot(x=file["Harga"])
plt.title("Deteksi Outlier Harga")
plt.show()


Q1 = file["Harga"].quantile(0.25)
Q3 = file["Harga"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = file[
    (file["Harga"] >= lower) &
    (file["Harga"] <= upper)
]

print("\nJumlah data sebelum:", len(file))
print("Jumlah data sesudah:", len(df_clean))


df_corr = file.copy()

le_brand = LabelEncoder()
le_os = LabelEncoder()
le_nama = LabelEncoder()

file["Brand_Code"] = le_brand.fit_transform(file["Brand"])
file["Os_Code"] = le_os.fit_transform(file["Os"])
file["Nama_hp_Code"] = le_nama.fit_transform(file["Nama_hp"])


file["Stok_tersedia"] = file["Stok_tersedia"].astype(int)


file["Resolusi_kamera"] = file["Resolusi_kamera"].str.replace("MP","").astype(int)



kolom = [
    "Harga",
    "Ram",
    "Memori_internal",
    "Ukuran_layar",
    "Resolusi_kamera",
    "Kapasitas_baterai",
    "Rating_pengguna",
    "Brand_Code",
    "Os_Code",
    "Nama_hp_Code"
]

corr = file[kolom].corr(method="pearson")

print("==============================")
print("Matriks Korelasi")
print("==============================")
print(corr)

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Heatmap Korelasi Dataset HP")
plt.show()



X = file[
    [
        "Ram",
        "Memori_internal",
        "Ukuran_layar",
        "Resolusi_kamera",
        "Kapasitas_baterai",
        "Rating_pengguna",
        "Brand_Code",
        "Os_Code"
    ]
]

y = file["Harga"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)



print("\n==============================")
print("HASIL PREDIKSI")
print("==============================")

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²   :", r2_score(y_test, y_pred))



hasil = pd.DataFrame({
    "Harga Asli": y_test.values,
    "Harga Prediksi": y_pred.astype(int)
})

print("\n10 Data Pertama")
print(hasil.head(10))


plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Harga Asli")
plt.ylabel("Harga Prediksi")
plt.title("Prediksi Harga HP")

plt.show()