import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os


file_jual='/penjualan_motor.xlsx'
# Baris ini membuat variabel bernama file_jual.
# Variabel ini menyimpan string yang merupakan path atau lokasi dari file Excel yang ingin Anda baca.
# Path /content/excel/penjualan_motor_bln.xlsx menunjukkan bahwa file tersebut kemungkinan besar berada
# di direktori sementara /content/excel/ di lingkungan Google Colab Anda.

print("Current Working Directory:", os.getcwd())
print("Isi folder:", os.listdir())
print("File ada:", os.path.exists("penjualan_motor.xlsx"))

file_jual = r"D:\xampp\python\penjualan_motor.xlsx"
df = pd.read_excel(file_jual)

# Ini adalah bagian inti yang membaca data.
# Fungsi pd.read_excel() dari pustaka pandas digunakan untuk membaca data dari file Excel.
# Argumen file_jual memberi tahu fungsi ini file mana yang harus dibaca.
# Setelah data dibaca, data tersebut disimpan dalam struktur data khusus pandas yang disebut DataFrame,
# dan DataFrame ini diberi nama variabel df.

df.head(10)

#Metode .head() yang dipanggil pada DataFrame df berfungsi untuk
#menampilkan 5 baris pertama dari DataFrame tersebut.
#Ini sangat berguna untuk melihat sekilas bagaimana data Anda terlihat setelah dibaca,
#memeriksa apakah kolom dan baris sudah benar, dan memastikan data sudah dimuat dengan benar.

from numpy.random.mtrand import f
bulan = ['Januari' , 'Februari' , 'Maret','April']
total_per_bulan = df[bulan].sum()
plt.figure(figsize=(10, 6))
plt.bar(bulan, total_per_bulan)
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan')
for i, v in enumerate(total_per_bulan):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.title('Total Penjualan per Bulan',fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('total_penjualan_per_bulan.png')
plt.show()

df['Total_Penjualan_per_Barang'] = df[bulan].sum(axis=1)

# Untuk pie chart, kita perlu satu seri data. Set 'Barang' sebagai indeks.
pie_data = df.set_index('Barang')['Total_Penjualan_per_Barang']
#Untuk membuat pie chart, kita memerlukan satu seri data numerik dengan label yang sesuai.
#Baris ini mengambil DataFrame df, mengatur kolom 'Barang' sebagai indeksnya, kemudian memilih kolom Total_Penjualan_per_Barang.
# Hasilnya adalah sebuah pandas Series (pie_data) di mana indeksnya adalah nama-nama barang dan nilainya adalah total penjualan masing-masing barang.

plt.figure(figsize=(12, 8))

pie_data.plot(kind='pie', autopct='%1.1f%%', startangle=90)


plt.ylabel('') # Hapus label y default dari pandas Series
plt.title('Proporsi Penjualan per Barang',fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#Menambahkan legenda ke grafik. Legenda ini akan menampilkan nama-nama barang yang sesuai dengan warna irisan pai.
#loc='upper left': Menempatkan legenda di sudut kiri atas kotak legenda itu sendiri.
#bbox_to_anchor=(1, 1): Memosisikan kotak legenda di luar grafik, tepat di sudut kanan atas area plot, untuk mencegah tumpang tindih dengan irisan pai.

plt.tight_layout()
plt.savefig('proporsi_penjualan_per_barang_pie.png')
plt.show()