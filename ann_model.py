import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================
# Dataset
# ==========================
data = pd.DataFrame({
    'luas_Tanah': [100,150,200,250,300,350,400,450,500,550],
    'jumlah_Kamar': [2,3,3,4,4,5,5,6,6,7],
    'harga_Rumah': [320,410,480,550,600,680,750,820,870,950]
})

# ==========================
# Memisahkan fitur dan target
# ==========================
X = data[['luas_Tanah', 'jumlah_Kamar']]
y = data[['harga_Rumah']]

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
# Membuat Model
# ==========================
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1)
])

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
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
# Evaluasi Model
# ==========================
loss, mae = model.evaluate(X_test, y_test, verbose=0)

# Mengembalikan MAE ke skala asli
mae_asli = mae * (y_max.values[0] - y_min.values[0])

print("="*40)
print(f"Loss : {loss:.4f}")
print(f"MAE  : {mae_asli:.2f} juta rupiah")
print("="*40)

# ==========================
# Prediksi Rumah Baru
# ==========================
X_new = np.array([
    [250, 4],
    [350, 5]
])

# Normalisasi data baru
X_new_norm = (X_new - X_min.values) / (X_max.values - X_min.values)

# Prediksi (masih dalam bentuk normalisasi)
y_pred_norm = model.predict(X_new_norm)

# Kembalikan ke harga asli
y_pred = y_pred_norm * (y_max.values - y_min.values) + y_min.values

print("\nPrediksi Harga Rumah")
for i in range(len(X_new)):
    print(f"Luas Tanah = {X_new[i][0]} m²")
    print(f"Jumlah Kamar = {X_new[i][1]}")
    print(f"Prediksi Harga = {y_pred[i][0]:.2f} juta rupiah\n")

# ==========================
# Plot Loss
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.close()

# ==========================
# Plot Prediksi vs Aktual
# ==========================
y_test_pred = model.predict(X_test)

# Kembalikan ke skala asli
y_test_asli = y_test * (y_max.values - y_min.values) + y_min.values
y_test_pred_asli = y_test_pred * (y_max.values - y_min.values) + y_min.values

plt.figure(figsize=(6,6))
plt.scatter(y_test_asli, y_test_pred_asli)
plt.plot([300,1000],[300,1000],'r--')
plt.xlabel("Harga Aktual (Juta)")
plt.ylabel("Harga Prediksi (Juta)")
plt.title("Aktual vs Prediksi")
plt.grid(True)
plt.savefig("hasil_prediksi.png")
plt.close()