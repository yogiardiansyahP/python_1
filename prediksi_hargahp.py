import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# Membaca dataset
# ==========================
df = pd.read_csv("dataset.handphone.csv")

# ==========================
# Menampilkan data awal
# ==========================
print("===== DATASET =====")
print(df.head())
print("\nTipe Data:")
print(df.dtypes)

# ==========================
# Membersihkan data numerik
# ==========================

# Harga
df["Harga"] = (
    df["Harga"]
    .astype(str)
    .str.replace("Rp", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# RAM
df["Ram"] = (
    df["Ram"]
    .astype(str)
    .str.replace("GB", "", regex=False)
    .astype(float)
)

# Memori Internal
df["Memori_internal"] = (
    df["Memori_internal"]
    .astype(str)
    .str.replace("GB", "", regex=False)
    .astype(float)
)

# Resolusi Kamera
df["Resolusi_kamera"] = (
    df["Resolusi_kamera"]
    .astype(str)
    .str.replace("MP", "", regex=False)
    .str.strip()
    .astype(float)
)

# Kapasitas Baterai
df["Kapasitas_baterai"] = (
    df["Kapasitas_baterai"]
    .astype(str)
    .str.replace("mAh", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Ukuran Layar
df["Ukuran_layar"] = (
    df["Ukuran_layar"]
    .astype(str)
    .str.replace("inch", "", regex=False)
    .str.replace('"', "", regex=False)
    .astype(float)
)

# ==========================
# Mengubah data kategori menjadi numerik
# ==========================
df = pd.get_dummies(df, columns=["Brand", "Os"], drop_first=True)

# ==========================
# Mengubah Tahun Rilis menjadi angka
# ==========================
df["Tahun_rilis"] = pd.to_datetime(df["Tahun_rilis"])
df["Tahun_rilis"] = df["Tahun_rilis"].dt.year

# ==========================
# Mengubah True/False menjadi 1/0
# ==========================
df["Stok_tersedia"] = df["Stok_tersedia"].astype(int)

# ==========================
# Menentukan fitur dan target
# ==========================
X = df.drop(columns=["Id_hp", "Nama_hp", "Harga"])
y = df["Harga"]

# ==========================
# Membagi data
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Membuat model
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
# Evaluasi
# ==========================
y_pred = model.predict(X_test)

print("\n===== HASIL EVALUASI =====")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))

# ==========================
# Prediksi data baru
# ==========================
data_baru = X.iloc[[0]].copy()

data_baru["Ram"] = 8
data_baru["Memori_internal"] = 256
data_baru["Ukuran_layar"] = 6.7
data_baru["Resolusi_kamera"] = 64
data_baru["Kapasitas_baterai"] = 5000

prediksi = model.predict(data_baru)

print("\n===== PREDIKSI HARGA =====")
print("Harga HP diperkirakan : Rp{:,.0f}".format(prediksi[0]))