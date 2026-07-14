import pandas as pd
import numpy as np


data = {'Nama': ['andi','budi','siti'],
        'Nilai':[85,90,78]}
df = pd.DataFrame(data)


print(df)

plt.bar(df['Nama'],df['Nilai'],color='skyblue')
plt.xlabel('Nama')
plt.ylabel('Nilai')
plt.title('grafik mahasiswa')

plt.show()

