import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

file = pd.read_csv("dataset.handphone.csv")

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

x =  file [['Ram']]
y = file [['Harga']]

X_train,X_test,y_train,y_test= train_test_split(x,y, test_size=0.2,random_state=100)

model = LinearRegression()
model.fit(X_train, y_train)
# Prediksi
y_pred = model.predict(X_test)
# Menampilkan koefisien
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
# Visualisasi
plt.scatter (x, y, color='blue', label='Data Aktual') # Scatter plot data aktual
plt.plot(x, model.predict(x), color='red', label='Garis Regresi') # Garis regresi
plt.xlabel('Ram')
plt.ylabel('Harga')
plt.title('Regresi Linear: Ram vs Harga')
plt.legend()
plt.show()



data1 = pd.DataFrame({
'luas_tanah': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
'jumlah_kamar': [2, 3, 3, 4, 5, 3, 4, 5, 6, 7],
'harga': [300, 400, 500, 600, 700, 550, 650, 750, 850, 950] # Harga dalam juta
})
# Variabel independen (X) dan variabel dependen (y)
X = data1[['luas_tanah', 'jumlah_kamar']]
y = data1['harga']
# Membagi data menjadi 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
#Prediksi harga rumah pada data uji
y_pred = model.predict(X_test)
# Menampilkan koefisien regresi
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
#Scatter plot: hubungan luas tanah dengan harga rumah.
plt.figure(figsize=(8,5))
plt.scatter(data1['luas_tanah'], data1['harga'], color='blue', label='data1 Aktual') # data1 asli
plt.plot(data1['luas_tanah'], model.predict(data1[['luas_tanah', 'jumlah_kamar']]), color='red', label='Garis Regresi') # Garis regresi
plt.xlabel('Luas Tanah')
plt.ylabel('Harga Rumah')
plt.title('Regresi Linear: Luas Tanah & Jumlah Kamar vs Harga')
plt.legend()
plt.show()