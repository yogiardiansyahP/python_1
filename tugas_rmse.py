from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Contoh nilai aktual dan prediksi
y_actual = np.array([500,600,700,800])
y_pred = np.array([520,590,680,850])

# Hitung metrik evaluasi
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np. sqrt (mse)
r2 = r2_score(y_actual, y_pred)

# Tampilkan hasil
print(f"MAE: {mae : .2f}")
print(f"MSE: {mse : .2f}")
print(f"RMSE: {rmse : .2f}") 
print(f"R-squared (R2): {r2 : .2f}")