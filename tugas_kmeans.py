import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans



data = pd.DataFrame({
    'pelanggan': ['A','B','C','D','E','F','G','H'],
    'pendapatan': [50,60,65,80,90,200,210,220],
    'jumlah_pembelian_per_bulan': [5,7,7,8,10,25,27,30]
})

X =  data[['pendapatan', 'jumlah_pembelian_per_bulan']].values

Kmeans = KMeans(n_clusters= 2, random_state=42)
Kmeans.fit(X)

data['Klaster'] = Kmeans.labels_
print (data)

plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=Kmeans.labels_,cmap='viridis',label="Data")
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=200,c='red', label="Centroid")
plt.xlabel("pendapatan(jt)")
plt.ylabel("jumlah pembelian")
plt.title("hasil clustering dengan kmeans")
plt.legend()
plt.show()