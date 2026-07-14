import pandas as pd
import matplotlib.pyplot as plt

data = {'fakultas': ['tekhnik','ekonomi','hukum','kedokteran','ilmu komputer'],
        'jumlah_mahasiswa':[250,180,120,90,200]}
df = pd.DataFrame(data)


print(df)

plt.bar(df['fakultas'],df['jumlah_mahasiswa'],color='skyblue')
plt.xlabel('fakultas')
plt.ylabel('jumlah_mahasiswa')
plt.title('grafik mahasiswa')

plt.show()