import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Agar hasil selalu sama
np.random.seed(42)
tf.random.set_seed(42)

# ==========================
# Dataset
# ==========================
data = pd.DataFrame({
    'Usia_Mobil': [1, 2, 3, 4, 5, 6],
    'Jarak_Tempuh': [10, 20, 30, 40, 50, 60],   # ribu km
    'Harga_Mobil': [500, 450, 400, 350, 300, 250]  # juta rupiah
})

# ==========================
# Memisahkan fitur dan target
# ==========================
X = data[['Usia_Mobil', 'Jarak_Tempuh']]
y = data[['Harga_Mobil']]

# ==========================
# Normalisasi X
# ==========================
X_min = X.min()
X_max = X.max()

X_norm = (X - X_min) / (X_max - X_min)

# ==========================
# Normalisasi Y
# ==========================
y_min = y.min()
y_max = y.max()

y_norm = (y - y_min) / (y_max - y_min)

# ==========================
# Membagi data
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_norm,
    y_norm,
    test_size=0.2,
    random_state=42
)

# ==========================
# Model ANN
# 2 Hidden Layer
# ReLU
# ==========================
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

# ==========================
# Compile
# Optimizer Adam
# ==========================
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# ==========================
# Training
# ==========================
history = model.fit(
    X_train,
    y_train,
    epochs=500,
    verbose=0
)

# ==========================
# Evaluasi
# ==========================
loss, mae = model.evaluate(X_test, y_test, verbose=0)

mae_asli = mae * (y_max.values[0] - y_min.values[0])

print("=" * 40)
print("HASIL EVALUASI MODEL")
print("=" * 40)
print(f"Loss : {loss:.4f}")
print(f"MAE  : {mae_asli:.2f} juta rupiah")

# ==========================
# Prediksi Mobil Baru
# Usia = 3.5 tahun
# Jarak = 35 ribu km
# ==========================

X_new = np.array([[3.5, 35]])

# Normalisasi
X_new_norm = (X_new - X_min.values) / (X_max.values - X_min.values)

# Prediksi
y_pred_norm = model.predict(X_new, verbose=0)
# Perbaikan: gunakan data yang sudah dinormalisasi
y_pred_norm = model.predict(X_new_norm, verbose=0)

# Kembalikan ke skala asli
y_pred = y_pred_norm * (y_max.values - y_min.values) + y_min.values

print("\nHASIL PREDIKSI")
print(f"Usia Mobil     : {X_new[0][0]} tahun")
print(f"Jarak Tempuh   : {X_new[0][1]} ribu km")
print(f"Harga Mobil    : {y_pred[0][0]:.2f} juta rupiah")

# ==========================
# Grafik Loss
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ==========================
# Grafik Aktual vs Prediksi
# ==========================
y_test_pred = model.predict(X_test, verbose=0)

y_test_asli = y_test * (y_max.values - y_min.values) + y_min.values
y_test_pred_asli = y_test_pred * (y_max.values - y_min.values) + y_min.values

plt.figure(figsize=(6,6))
plt.scatter(y_test_asli, y_test_pred_asli)

plt.plot(
    [y_test_asli.min().values[0], y_test_asli.max().values[0]],
    [y_test_asli.min().values[0], y_test_asli.max().values[0]],
    'r--'
)

plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi")
plt.title("Aktual vs Prediksi")
plt.grid(True)
plt.show()