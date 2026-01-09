import pandas as pd
import os

file_path = 'fake_real_job/fake_real_job_postings_3000x25.csv'

if not os.path.exists(file_path):
    print(f"File not found at: {file_path}")
else:
    try:
        df = pd.read_csv(file_path)
        
        print("--- 1. Struktur Data ---")
        print(f"Dimensi: {df.shape}")
        print("\nTipe Data:")
        print(df.dtypes)
        
        print("\n--- 2. Analisis Target (is_fake) ---")
        print(df['is_fake'].value_counts())
        print("\nPersentase:")
        print(df['is_fake'].value_counts(normalize=True) * 100)
        
        print("\n--- 3. Missing Values (Top 10) ---")
        print(df.isnull().sum().sort_values(ascending=False).head(10))
        
        print("\n--- 4. Statistik Numerik ---")
        print(df.describe())
        
        print("\n--- 5. Kardinalitas Kolom Kategorikal ---")
        cat_cols = ['industry', 'employment_type', 'required_experience_years', 'education_level', 'fraud_reason']
        for col in cat_cols:
            if col in df.columns:
                print(f"{col}: {df[col].nunique()} unik values")
                if df[col].nunique() < 10:
                    print(f"   Values: {df[col].unique()}")

    except Exception as e:
        print(f"Error reading CSV: {e}")
