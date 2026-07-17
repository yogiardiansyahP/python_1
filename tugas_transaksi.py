import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



data = {
    'Transaksi': [10, 15, 7, 20, 12, 5, 18, 9, 25, 6],
    'Waktu_akses': [5, 8, 3, 10, 6, 2, 9, 4, 11, 3],
    'Perangkat': [
        'Mobile',
        'Desktop',
        'Tablet',
        'Desktop',
        'Mobile',
        'Tablet',
        'Desktop',
        'Mobile',
        'Desktop',
        'Tablet'
    ]
}

df = pd.DataFrame(data)

print("Dataset Awal")
print(df)



le = LabelEncoder()
df['Perangkat'] = le.fit_transform(df['Perangkat'])

print("\nDataset Setelah Label Encoding")
print(df)

print("\nMapping Label:")
for i, kelas in enumerate(le.classes_):
    print(f"{kelas} = {i}")



X = df[['Transaksi', 'Waktu_akses']].values
y = df['Perangkat'].values



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = LogisticRegression(
    solver='lbfgs',
    max_iter=200
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAkurasi Model")
print(f"{accuracy*100:.2f}%")


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)



print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))



plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



df_ordinal = pd.DataFrame(data)

category_order = [['Mobile', 'Desktop', 'Tablet']]

ordinal_encoder = OrdinalEncoder(categories=category_order)

df_ordinal['Perangkat_Ordinal'] = ordinal_encoder.fit_transform(
    df_ordinal[['Perangkat']]
)

print("\nData Setelah Ordinal Encoding")
print(df_ordinal)