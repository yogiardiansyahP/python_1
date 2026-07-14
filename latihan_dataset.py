import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



file = pd.read_csv("dataset.handphone.csv")

print("Ukuran data:", file.shape)
print(file.head())
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


df_corr = df_clean.copy()


encoder = LabelEncoder()
df_corr["Nama_hp_Code"] = encoder.fit_transform(df_corr["Nama_hp"])


print(df_corr[["Nama_hp", "Nama_hp_Code"]].drop_duplicates())


corr = df_corr[["Nama_hp_Code", "Harga"]].corr(method="pearson")

print("\nKorelasi Nama_hp dan Harga")
print(corr)


plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

plt.title("Korelasi Nama HP dan Harga")
plt.show()


