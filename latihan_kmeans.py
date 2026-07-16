import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

gaji = np.array([
    [1,3],
    [2, 5],
    [3, 8],
    [1, 12],
    [2, 18],
    [3, 25],
    [1, 30],
    [2, 50],

])

kmeans = KMeans(n_clusters=2, random_state=42)

labels = kmeans.fit_predict(gaji)

print(labels)



tinggi = np.array([150, 155, 160, 165, 170, 175, 180, 185]).reshape(-1, 1)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(tinggi)

for t, label in zip(tinggi.flatten(), labels):
    print(f"{t} cm -> Cluster {label}")




df = pd.DataFrame({
    "Kategori": ["elektronik", "fashion", "makanan",
                 "elektronik", "makanan", "fashion"]
})

df_encode = pd.get_dummies(df, columns=["Kategori"], dtype=int)

print(df_encode)

kmeans = KMeans(n_clusters=2, random_state=42)

labels = kmeans.fit_predict(df_encode)

df["Cluster"] = labels

print("\nHasil:")
print(df)


penghasilan = [20,25,30,50,100,200,500,1000]
data_log = np.log1p(penghasilan)
sns.histplot(data_log, kde=True, bins=30)
plt.title("Log Transformasion")
plt.show()

df=pd.DataFrame({
    "pelanggan":["A","B","C","D","E"],
    "jumlah_transaksi":[5,10,15,20,25],
    "Total_pembelian":[500,1200,1800,2500,3200]
})
df["Total_pembelian/jumlah_transaksi"] = df["Total_pembelian"] / df["jumlah_transaksi"]

print(df)
