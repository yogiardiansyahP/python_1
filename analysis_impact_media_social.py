import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

file = pd.read_csv("Mental_Health.csv")
print(file.describe())
print(file.dtypes)

print(file.isnull().sum())


file["daily_social_media_hours"] = file["daily_social_media_hours"].fillna(file["daily_social_media_hours"].median())
file = file.drop_duplicates()
file["daily_social_media_hours"] = file["daily_social_media_hours"].astype(int)
print(file.dtypes)


file["platform_usage"] = file["platform_usage"].replace({
    "IG":"Instagram",
    "instagram":"Instagram"
})




le = LabelEncoder()
file["gender"] = le.fit_transform(file["gender"])
file["platform_usage"] = le.fit_transform(file["platform_usage"])
file["social_interaction_level"] = le.fit_transform(file["social_interaction_level"])

scaler = MinMaxScaler()

file[["age","daily_social_media_hours"]] = scaler.fit_transform(
    file[["age","daily_social_media_hours"]]
)

Q1 = file["daily_social_media_hours"].quantile(0.25)
Q3 = file["daily_social_media_hours"].quantile(0.75)

IQR = Q3 - Q1

bawah = Q1 - 1.5 * IQR
atas = Q3 + 1.5 * IQR

file = file[
    (file["daily_social_media_hours"] >= bawah) &
    (file["daily_social_media_hours"] <= atas)
]




X = file.drop("depression_label", axis=1)
y = file["depression_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n===== Accuracy =====")
print(accuracy_score(y_test, y_pred))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("\n===== Confusion Matrix =====")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[" Tidak Depresi", " Depresi"],
    yticklabels=["Tidak Depresi", " Depresi"]
)

plt.title("Confusion Matrix Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()