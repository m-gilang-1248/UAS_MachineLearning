import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

# --- 1. Persiapan Data (Sama seperti sebelumnya) ---
file_path = 'fake_real_job/fake_real_job_postings_3000x25.csv'
df = pd.read_csv(file_path)

# Preprocessing
cols_to_drop = ['job_id', 'fraud_reason', 'contact_email', 'posting_date', 'application_deadline']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# Feature Engineering
df_clean['has_company_logo'] = df_clean['has_logo']
df_clean['has_company_website'] = df_clean['company_website'].apply(lambda x: 0 if pd.isnull(x) else 1)
df_clean['has_company_profile'] = df_clean['company_profile'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Simpan kolom kategorikal sebelum drop untuk visualisasi tertentu jika perlu, 
# tapi untuk training kita drop teks
text_cols = ['company_website', 'company_profile', 'job_description', 'requirements', 'benefits', 'company_name', 'job_title', 'location', 'salary_range', 'department', 'job_function']
df_clean = df_clean.drop(columns=text_cols, errors='ignore')

# Encoding
education_map = {np.nan: 0, 'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4, 'Unspecified': 0}
df_clean['education_level_encoded'] = df_clean['education_level'].map(education_map).fillna(0)
df_clean = df_clean.drop(columns=['education_level'])
df_clean = pd.get_dummies(df_clean, columns=['industry', 'employment_type'], drop_first=True)

# Split
X = df_clean.drop('is_fake', axis=1)
y = df_clean['is_fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# --- 2. Pembuatan Visualisasi ---
# Buat folder untuk menyimpan gambar jika belum ada
if not os.path.exists('plots'):
    os.makedirs('plots')

# Visualisasi 1: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title('Confusion Matrix: Random Forest (Akurasi 100%)')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.tight_layout()
plt.savefig('plots/1_confusion_matrix.png')
print("Disimpan: plots/1_confusion_matrix.png")

# Visualisasi 2: Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Top 10 Fitur Terpenting (Feature Importance)')
plt.xlabel('Tingkat Kepentingan (Importance Score)')
plt.tight_layout()
plt.savefig('plots/2_feature_importance.png')
print("Disimpan: plots/2_feature_importance.png")

# Visualisasi 3: Analisis 'text_length' (Kunci Pemisah Data)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='text_length', hue='is_fake', element='step', palette=['blue', 'red'], bins=30)
plt.title('Distribusi Panjang Teks: Real vs Fake')
plt.xlabel('Jumlah Karakter (Text Length)')
plt.ylabel('Jumlah Data')
plt.legend(title='Kategori', labels=['Fake', 'Real'])
plt.tight_layout()
plt.savefig('plots/3_text_length_distribution.png')
print("Disimpan: plots/3_text_length_distribution.png")

# Visualisasi 4: Korelasi Antar Fitur Utama
plt.figure(figsize=(8, 6))
# Ambil fitur top 5 + target untuk korelasi
top_features = feat_df['Feature'].head(5).tolist()
top_features.append('is_fake')
corr_matrix = df_clean[top_features].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Korelasi Fitur Utama dengan Target (is_fake)')
plt.tight_layout()
plt.savefig('plots/4_correlation_heatmap.png')
print("Disimpan: plots/4_correlation_heatmap.png")

print("\nSemua visualisasi selesai dibuat.")
