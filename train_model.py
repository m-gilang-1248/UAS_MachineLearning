import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Menggunakan Gradient Boosting (XGBoost like) dari scikit-learn jika xgboost tidak terinstall, 
# atau bisa menggunakan HistGradientBoostingClassifier yang sangat mirip LightGBM/XGBoost.
from sklearn.ensemble import GradientBoostingClassifier 

# 1. Load Data
file_path = 'fake_real_job/fake_real_job_postings_3000x25.csv'
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Preprocessing & Cleaning
print("\n--- Preprocessing ---")

# A. Hapus Data Leakage & ID
# 'fraud_reason' memberitahu kita langsung bahwa itu penipuan, harus dihapus.
cols_to_drop = ['job_id', 'fraud_reason', 'contact_email', 'posting_date', 'application_deadline'] 
df_clean = df.drop(columns=cols_to_drop, errors='ignore')
print(f"Dropped columns: {cols_to_drop}")

# B. Feature Engineering dari Missing Values (PENTING berdasarkan EDA)
# Pola missing value sangat informatif. Fake job seringkali tidak punya website/profil.
df_clean['has_company_logo'] = df_clean['has_logo'] # Rename for clarity
df_clean['has_company_website'] = df_clean['company_website'].apply(lambda x: 0 if pd.isnull(x) else 1)
df_clean['has_company_profile'] = df_clean['company_profile'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Hapus kolom teks asli yang sudah diekstrak fiturnya (untuk model tabular ini)
# Kita simpan text_length yang sudah ada
text_cols = ['company_website', 'company_profile', 'job_description', 'requirements', 'benefits', 'company_name', 'job_title', 'location', 'salary_range', 'department', 'job_function']
df_clean = df_clean.drop(columns=text_cols, errors='ignore')

# C. Encoding Kategorikal
# Ordinal Encoding untuk Education
education_map = {
    np.nan: 0,
    'High School': 1,
    'Bachelor': 2,
    'Master': 3,
    'PhD': 4,
    'Unspecified': 0
}
df_clean['education_level_encoded'] = df_clean['education_level'].map(education_map).fillna(0)
df_clean = df_clean.drop(columns=['education_level'])

# One-Hot Encoding untuk Nominal (Industry, Employment Type)
df_clean = pd.get_dummies(df_clean, columns=['industry', 'employment_type'], drop_first=True)

print(f"Fitur final ({df_clean.shape[1]} fitur):")
print(df_clean.columns.tolist())

# 3. Split Data
X = df_clean.drop('is_fake', axis=1)
y = df_clean['is_fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Training Model 1: Random Forest (Bagging)
print("\n--- Training Random Forest (Bagging) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Akurasi RF:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report RF:")
print(classification_report(y_test, y_pred_rf))

# 5. Training Model 2: Gradient Boosting (Boosting)
# Ini adalah representasi dari keluarga Boosting (mirip XGBoost)
print("\n--- Training Gradient Boosting (Boosting) ---")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("Akurasi Boosting:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report Boosting:")
print(classification_report(y_test, y_pred_gb))

# 6. Feature Importance (dari Random Forest)
print("\n--- Top 5 Feature Importance (Random Forest) ---")
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_importance_df.sort_values(by='importance', ascending=False).head(5))