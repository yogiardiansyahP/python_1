import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


file = pd.read_csv("dataset.handphone.csv")


sns.countplot(x=file['Brand'])
plt.show()

sns.barplot(x=file['Brand'],y=file['Ukuran_layar'])
plt.show()