import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file = pd.read_csv("ecommerce_dataset_updated.csv")


file.columns = file.columns.str.strip()


print("Nama Kolom:")
print(file.columns.tolist())


print("\nUkuran Dataset :", file.shape)

print("\n5 Data Pertama")
print(file.head())

print("\nTipe Data")
print(file.dtypes)

print("\nMissing Value")
print(file.isnull().sum())


file.fillna(file.select_dtypes(include="number").median(), inplace=True)


print("\nStatistik Deskriptif")
print(file.describe())


plt.figure(figsize=(8,5))
sns.histplot(file["Final_Price(Rs.)"], bins=20, kde=True)
plt.title("Distribusi Final Price")
plt.xlabel("Final Price (Rs.)")
plt.show()


plt.figure(figsize=(8,3))
sns.boxplot(x=file["Final_Price(Rs.)"])
plt.title("Deteksi Outlier Final Price")
plt.show()


Q1 = file["Final_Price(Rs.)"].quantile(0.25)
Q3 = file["Final_Price(Rs.)"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = file[
    (file["Final_Price(Rs.)"] >= lower) &
    (file["Final_Price(Rs.)"] <= upper)
]

print("\nJumlah Data Sebelum :", len(file))
print("Jumlah Data Sesudah :", len(df_clean))


corr = df_clean[
    [
        "Price (Rs.)",
        "Discount (%)",
        "Final_Price(Rs.)"
    ]
].corr(method="pearson")

print("\nMatriks Korelasi")
print(corr)

plt.figure(figsize=(6,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Korelasi Harga dan Diskon")
plt.show()


plt.figure(figsize=(7,5))

sns.scatterplot(
    data=df_clean,
    x="Discount (%)",
    y="Final_Price(Rs.)"
)

plt.title("Hubungan Diskon dengan Final Price")
plt.xlabel("Discount (%)")
plt.ylabel("Final Price (Rs.)")

plt.show()