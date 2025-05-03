import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
df = pd.read_csv("train.csv")

# Ambil hanya fitur numerik
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Buang kolom ID dan target (misalnya "SalePrice")
X = df_numeric.drop(columns=["Id", "SalePrice"])

# Buat folder penyimpanan visualisasi
os.makedirs("scaling_boxplots", exist_ok=True)

# ========== 1. Original (Tanpa Scaling) ==========
plt.figure(figsize=(12, 4))
sns.boxplot(data=X)
plt.title("Boxplot - Original (Tanpa Scaling)")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("scaling_boxplots/boxplot_original.png")
plt.close()

# ========== 2. StandardScaler ==========
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

plt.figure(figsize=(12, 4))
sns.boxplot(data=pd.DataFrame(X_standard, columns=X.columns))
plt.title("Boxplot - StandardScaler")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("scaling_boxplots/boxplot_standardscaler.png")
plt.close()

# ========== 3. MinMaxScaler ==========
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

plt.figure(figsize=(12, 4))
sns.boxplot(data=pd.DataFrame(X_minmax, columns=X.columns))
plt.title("Boxplot - MinMaxScaler")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("scaling_boxplots/boxplot_minmaxscaler.png")
plt.close()

print("Visualisasi scaling selesai dan disimpan di folder 'scaling_boxplots'.")
