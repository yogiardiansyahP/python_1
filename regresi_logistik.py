import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn. linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Buat dataset eksak
data = {
'Usia': [22, 25, 30, 35, 40, 45, 50, 55, 27, 37],
'Pendapatan': [30000, 35000, 50000, 60000, 70000, 80000, 90000, 100000, 40000, 65000],
'Beli': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
plt.figure(figsize=(8,6))
plt.scatter(df[df['Beli' ] == 1]['Usia'], df[df['Beli'] == 1]['Pendapatan'], color='blue', label="Membeli (1)")
plt.scatter(df[df['Beli'] == 0]['Usia'], df[df['Beli'] == 0]['Pendapatan'], color='red', label="Tidak Membeli (0)")
plt.xlabel("Usia")
plt.ylabel("Pendapatan Tahunan")
plt.legend()
plt.title("Visualisasi Data: Usia vs Pendapatan Tahunan")
plt.show()
# Pisahkan fitur (X) dan target (y)
X = df[['Usia', 'Pendapatan' ]].values
y = df['Beli' ].values

# Pisahkan menjadi data train (80%) dan test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler. transform(X_test)

# Inisialisasi dan latih model regresi logistik
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediksi hasil
y_pred = model.predict(X_test)
y_prob = model. predict_proba(X_test)[:, 1] # Probabilitas kelas positif

# Akurasi Model
accuracy = accuracy_score(y_test, y_pred)

print("="*40)
print("HASIL EVALUASI MODEL")
print("="*40)

print(f"Akurasi Model : {accuracy*100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Plot keputusan regresi logistik
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Normalisasi meshgrid agar sesuai dengan skala data
grid = np.c_[xx.ravel(), yy.ravel()]
grid = scaler.transform(grid)

# Prediksi untuk setiap titik dalam grid
Z = model.predict(grid)
Z = Z.reshape (xx. shape)

# Plot decision boundary
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(df[df['Beli'] == 1]['Usia' ], df[df['Beli'] == 1]['Pendapatan'], color='blue', label="Membeli (1)")
plt.scatter(df[df['Beli'] == 0]['Usia'], df[df['Beli'] == 0]['Pendapatan'], color='red', label="Tidak Membeli (0)")
plt.xlabel("Usia")
plt.ylabel("Pendapatan Tahunan")
plt.legend()
plt.title("Decision Boundary dari Model Regresi Logistik")
plt.show()

print(y_test)
print(y_pred)