import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn. linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset
data = {
'Usia': [22, 25, 30, 35, 40, 45, 50, 55, 27, 37],
'Pendapatan': [30000, 35000, 50000, 60000, 70000, 80000, 90000, 100000, 40000, 65000],
'Makanan_Favorit': ['Burger', 'Burger', 'Pizza', 'Sushi', 'Sushi', 'Sushi', 'Pizza', 'Pizza', 'Burger', 'Sushi']
}
df = pd.DataFrame(data)

# Encode variabel target (Makanan Favorit) menjadi angka: Burger=0, Pizza=1, Sushi=2
le = LabelEncoder()
df['Makanan_Favorit'] = le.fit_transform(df['Makanan_Favorit'])

# Pisahkan fitur (X) dan target (y)
X = df[['Usia', 'Pendapatan' ]].values
y = df['Makanan_Favorit' ].values

# Split data menjadi train (70%) dan test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model dengan multi_class='multinomial' dan solver='lbfgs'
# Model Logistic Regression
model = LogisticRegression(
    solver='lbfgs',
    max_iter=200
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy*100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

df_ordinal = pd.DataFrame(data)

# Initialize OrdinalEncoder
# Specify the order of categories if there is a meaningful order
# For 'Makanan_Favorit', let's assume an arbitrary order for demonstration
# For example: Burger < Pizza < Sushi
category_order = [['Burger', 'Pizza', 'Sushi']]
ordinal_encoder = OrdinalEncoder(categories=category_order)

# Apply OrdinalEncoder to the 'Makanan_Favorit' column
df_ordinal['Makanan_Favorit_Ordinal'] = ordinal_encoder.fit_transform(df_ordinal[['Makanan_Favorit']])

print("DataFrame after Ordinal Encoding:")
print(df_ordinal)