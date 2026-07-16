import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


file = pd.read_csv("dataset.handphone.csv")

file.columns = file.columns.str.strip()

print(file.columns)

sns.countplot(x=file['Brand'])
plt.show()

sns.barplot(x=file['Brand'],y=file['Ukuran_layar'])
plt.show()

df_encode = pd.get_dummies(file, columns=["Brand"])

print(df_encode)

ordinal_encoder = OrdinalEncoder()

file["Kapasitas_baterai"] = ordinal_encoder.fit_transform(
    file[["Kapasitas_baterai"]]
)

print(file.head())

file["hp_encode"] = file.groupby("Brand")["Harga"].transform("mean")
print(file[["Brand","Harga"]])

file["hp_encode"]=file["Brand"].map(file["Brand"].value_counts())
print(file)

data = file["Harga"]

data = np.random.exponential(scale=2,size=1000)
plt.figure(figsize=(8 , 5))
sns.histplot(data, kde=True, bins=30)
plt.title("Distirbusi data skewed")


data_log = np.log1p(data)
sns.histplot(data_log, kde=True, bins=30)
plt.title("Log Transformasion")


data_sqrt = np.sqrt(data)

sns.histplot(data_sqrt, kde=True, bins=30)
plt.title("Square Root Transformation")


data_cbrt = np.cbrt(data)
sns.histplot(data_cbrt, kde=True, bins=30)
plt.title("Cube rutetransformation")


data_boxcox, lambda_bc = boxcox(data + 1)

sns.histplot(data_boxcox,kde= True, bins=30)
plt.title(f"box cox transformation(lambda =  {lambda_bc:.2f})")

data = file["Harga"]

pt = PowerTransformer(method="yeo-johnson")

data_yeo = pt.fit_transform(data.to_numpy().reshape(-1, 1))

sns.histplot(data_yeo.flatten(), kde=True, bins=30)
plt.title("Yeo-Johnson Transformation")


fc = file[["Harga", "Ram"]].copy()

fc["Pendapatan_penjualan/ram"] = fc["Harga"] / fc["Ram"]

print(fc)

fc["Log_pendapatan"]= np.log(fc["Ram"])
print(fc)

df = pd.DataFrame({
    "Spek_hp": pd.date_range(start="2025-01-01", periods=5, freq="D")
})


df["Memori_internal"] = df["Spek_hp"].dt.unit
df["Ukuran_layar"] = df["Spek_hp"].dt.unit
df["Resolusi_kamera"] = df["Spek_hp"].dt.unit

print(df)

text_data = [
    "I like learning machine learning",
    "machine learning is very interesting",
    "I want to understand more deeply about machine learning"
]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Analisis Sentimen dengan TextBlob
sentiments = [TextBlob(review).sentiment.polarity for review in text_data]

df_sentiment = pd.DataFrame({
    "Review": text_data,
    "Sentiment Score": sentiments
})

print("Data Sentimen")
print(df_sentiment)

print("\nTF-IDF")
print(df_tfidf)

# Fungsi memberi label sentimen
def label_sentimen(text):
    text = text.lower()

    if "like" in text or "interesting" in text:
        return "Positif"
    elif "doesnt" in text or "don't" in text or "dont" in text:
        return "Negatif"
    else:
        return "Netral"

# Menambahkan label sentimen
df_sentiment["Sentimen"] = df_sentiment["Review"].apply(label_sentimen)

print("\nHasil Akhir")
print(df_sentiment)