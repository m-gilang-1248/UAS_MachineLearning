# Checklist Iterasi Implementasi: Deteksi Lowongan Palsu

Dokumen ini berfungsi sebagai log aktivitas dan checklist untuk implementasi teknis dalam proyek deteksi lowongan pekerjaan palsu.

## Tahap 1: Eksplorasi & Pemahaman Data (EDA)
- [x] **Load Dataset:** Memuat file `fake_real_job_postings_3000x25.csv` menggunakan Pandas.
- [x] **Cek Struktur Data:** Memeriksa dimensi (3000 baris, 25 kolom) dan tipe data.
- [x] **Analisis Target (`is_fake`):** Memastikan keseimbangan kelas (Real: 50.9%, Fake: 49.1%).
- [x] **Analisis Missing Values:** Mengidentifikasi kolom dengan nilai kosong tinggi (`company_website`, `company_profile`).
- [x] **Deteksi Kebocoran Data (Data Leakage):** Menemukan bahwa `fraud_reason` hanya terisi untuk data palsu (harus dihapus).
- [x] **Analisis Statistik Deskriptif:** Memeriksa rentang nilai numerik (`text_length`) dan kardinalitas fitur kategorikal.

## Tahap 2: Preprocessing Data (Pembersihan & Transformasi)
- [x] **Hapus Kolom Bermasalah:**
    - `job_id` (Identifier, tidak relevan).
    - `fraud_reason` (Data leakage, jawaban langsung).
    - `contact_email`, `posting_date`, `application_deadline` (Metadata yang belum akan diproses).
- [x] **Feature Engineering (Metadata Flagging):**
    - Konversi `company_website` menjadi biner `has_company_website` (1=Ada, 0=Tidak).
    - Konversi `company_profile` menjadi biner `has_company_profile`.
    - Rename `has_logo` menjadi `has_company_logo` untuk konsistensi.
- [x] **Drop Fitur Teks Mentah:** Menghapus kolom teks panjang (`description`, `requirements`) karena fokus pada model tabular dan fitur metadata.
- [x] **Encoding Variabel Kategorikal:**
    - **Ordinal Encoding:** `education_level` (Unspecified -> 0, High School -> 1, ..., PhD -> 4).
    - **One-Hot Encoding:** `industry` dan `employment_type` (menggunakan `pd.get_dummies`).

## Tahap 3: Pengembangan Model (Training)
- [x] **Data Splitting:** Membagi data menjadi Train (80%) dan Test (20%) dengan `stratify=y` agar proporsi kelas tetap seimbang.
- [x] **Implementasi Random Forest (Bagging):**
    - Inisialisasi model dengan `n_estimators=100`.
    - Set `random_state=42` untuk hasil konsisten.
    - Latih model dengan data training.
- [x] **Implementasi Gradient Boosting (Boosting):**
    - Inisialisasi model dengan `learning_rate=0.1`, `n_estimators=100`.
    - Latih model dengan data training sebagai pembanding.

## Tahap 4: Evaluasi & Validasi Kinerja
- [x] **Prediksi Data Uji:** Menghasilkan prediksi (`y_pred`) menggunakan data test (`X_test`).
- [x] **Hitung Metrik Evaluasi:**
    - Mengukur *Accuracy*.
    - Mengukur *Precision*, *Recall*, dan *F1-Score* untuk kedua kelas.
- [x] **Confusion Matrix:** Membuat matriks untuk melihat True Positive, True Negative, False Positive, dan False Negative.
- [x] **Analisis Feature Importance:** Mengambil skor kepentingan fitur dari model Random Forest untuk memahami faktor dominan (Hasil: `text_length` & `has_company_website` dominan).

## Tahap 5: Visualisasi & Pelaporan
- [x] **Generate Plot Confusion Matrix:** Disimpan sebagai `plots/1_confusion_matrix.png`.
- [x] **Generate Plot Feature Importance:** Bar chart top 10 fitur (`plots/2_feature_importance.png`).
- [x] **Generate Plot Distribusi:** Histogram `text_length` pemisah kelas (`plots/3_text_length_distribution.png`).
- [x] **Generate Heatmap Korelasi:** Korelasi fitur utama dengan target (`plots/4_correlation_heatmap.png`).
- [x] **Penyusunan Laporan Akhir:** Integrasi semua temuan ke dalam `Laporan_UAS_Machine_Learning.md`.

## Tahap 6: Dokumentasi Bab 4 (Hasil & Pembahasan)
- [x] **Detail Teknis Workflow Preprocessing:**
    - [x] Jelaskan logika penghapusan `fraud_reason` (Leakage Prevention).
    - [x] Jelaskan strategi *Metadata Flagging* untuk `company_website` (Handling Missing Values).
    - [x] Jelaskan penggunaan *Ordinal Encoding* vs *One-Hot Encoding*.
    - [x] Justifikasi keputusan **tidak** melakukan normalisasi (sifat Tree-based models).
- [x] **Spesifikasi Metrik Evaluasi:**
    - [x] Definisikan Accuracy, Precision, Recall, dan F1-Score yang digunakan.
- [x] **Detail Dataset & Fitur:**
    - [x] Tuliskan dimensi final (3000 data, 19 fitur hasil olahan).
    - [x] List fitur kunci (`text_length`, `has_company_website`, dll).
- [x] **Justifikasi Pemilihan Algoritma:**
    - [x] Alasan memilih Random Forest (Bagging, Robustness).
    - [x] Alasan memilih Gradient Boosting (Boosting, SOTA Benchmark).
- [x] **Parameter Model:**
    - [x] Dokumentasikan `n_estimators=100` dan `random_state=42`.
- [x] **Pengujian & Analisis Kinerja:**
    - [x] Tampilkan tabel skor evaluasi (Akurasi 100%).
    - [x] Interpretasi Confusion Matrix (0 kesalahan).
    - [x] **Analisis Kritis:** Jelaskan *mengapa* akurasi sempurna (peran dominan `text_length` dan `has_company_website` dalam memisahkan data).

## Tahap 7: Pelaporan Hasil Praktikum (Output & Visualisasi)
*Checklist untuk memastikan bagian 'Hasil dan Pembahasan' menyajikan bukti empiris:*

- [x] **Dokumentasi Output Algoritma:**
    - [x] Tampilkan tabel perbandingan performa (Random Forest vs Gradient Boosting).
    - [x] Cantumkan *Classification Report* lengkap (Precision, Recall, F1 per kelas).
- [x] **Analisis Visualisasi:**
    - [x] **Confusion Matrix:** Jelaskan arti diagonal utama (TP/TN) yang sempurna dan ketiadaan error (FP/FN) pada plot `1_confusion_matrix.png`.
    - [x] **Feature Importance:** Hubungkan grafik `2_feature_importance.png` dengan hipotesis awal (pentingnya metadata vs konten).
- [x] **Pembahasan Metrik Evaluasi:**
    - [x] Jelaskan implikasi skor Akurasi 1.0 (Apakah overfitting? Atau data terlalu mudah?).
    - [x] Bahas implikasi skor Recall 1.0 (Tidak ada *False Negative*, artinya semua penipuan terdeteksi).

## Tahap 8: Dokumentasi Bab 6 (Pembahasan)
- [x] **Kesesuaian dengan Ekspektasi:**
    - [x] Bahas apakah akurasi 100% sudah diprediksi atau mengejutkan.
- [x] **Faktor Pengaruh Utama:**
    - [x] Peran krusial Feature Engineering (`text_length`, `has_company_website`).
    - [x] Kestabilan algoritma Ensemble (Random Forest/Boosting).
- [x] **Kendala & Tantangan:**
    - [x] Identifikasi Data Leakage (`fraud_reason`).
    - [x] Tantangan dalam tidak menggunakan NLP berat (memilih fitur tabular sederhana).

## Tahap 9: Dokumentasi Bab 7 (Kesimpulan & Saran)
- [x] **Kesimpulan:**
    - [x] Ringkas temuan utama (Performa model & faktor penentu).
    - [x] Jawab tujuan praktikum.
- [x] **Saran:**
    - [x] Rekomendasi pengembangan (Analisis teks NLP untuk data yang lebih sulit).
    - [x] Rekomendasi deployment (Integrasi ke sistem web/browser extension).
