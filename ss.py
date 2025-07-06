
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("boston.csv")

print("📊 Dataset Shape:", df.shape)
print("\n🧼 Missing Values:\n", df.isnull().sum())

imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

if 'RM' in df.columns and 'DIS' in df.columns:
    df_imputed['RM_per_DIS'] = df_imputed['RM'] / (df_imputed['DIS'] + 1e-5)

if 'LSTAT' in df.columns and 'CRIM' in df.columns:
    df_imputed['LSTAT_CRIM'] = df_imputed['LSTAT'] * df_imputed['CRIM']

scaler = StandardScaler()
numerical_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed[numerical_cols]), columns=numerical_cols)

plt.figure(figsize=(12, 10))
corr_matrix = df_scaled.corr()
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("boston_correlation.png")
plt.show()  
if 'MEDV' in df_scaled.columns:
    top_corr = df_scaled.corr()['MEDV'].sort_values(ascending=False)
    print("\n🔥 Top Features Correlated with MEDV:\n")
    print(top_corr.head(10))

input("\n✅ Script complete. Press ENTER to exit...")
