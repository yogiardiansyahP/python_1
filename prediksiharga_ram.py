import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

file = pd.read_csv("dataset.handphone.csv")


df = pd.DataFrame({
    "Ram": [4, 6, 8, 12],
    "Harga": [2000000, 3000000, 4000000, 6000000]
})

df["Harga_perkiraan"] = df["Harga"]


print(df)

file = pd.DataFrame({
    "Nama_Brand": ["Samsung", "Xiaomi", "Apple"],
    "Harga_hp": [3500000, 5000000, 18000000]
})

def kategori_hp(harga):
    if harga >= 10000000:
        return "Premium"
    elif harga >= 4000000:
        return "Mid Level"
    else:
        return "Entry Level"

file["Kategori_hp"] = file["Harga_hp"].apply(kategori_hp)

print(file)



file2 = np.array([
    [1, 3000000],
    [2, 5000000],
    [3, 12000000],
    [1, 3500000]
])

kmeans = KMeans(n_clusters=2, random_state=42)

labels = kmeans.fit_predict(file2)

print(labels)


