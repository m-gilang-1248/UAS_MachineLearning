# Naskah Video Presentasi UAS Machine Learning (Full Comprehensive Version)

**Judul Proyek:** Analisis dan Deteksi Lowongan Pekerjaan Palsu (*Fake Job Posting Detection*) Menggunakan Pendekatan Ensemble Learning
**Nama:** [Nama Anda]
**NIM:** [NIM Anda]
**Referensi Utama:** Laporan UAS Machine Learning (Berdasarkan Dataset EMSCAD)
**Estimasi Durasi:** 10 - 12 Menit

---

## Persiapan & Setup
*   **Scene 1 (Intro):** Slide Judul / Wajah Full (Webcam).
*   **Scene 2 (Teori):** Slide Latar Belakang & Tinjauan Pustaka.
*   **Scene 3 (Dataset):** Tampilan Dataset di Excel/CSV atau Browser.
*   **Scene 4 (Demo):** VS Code (Code & Terminal).
*   **Scene 5 (Hasil):** Tampilan Folder `plots/`.

---

## Segmen 1: Pendahuluan & Latar Belakang Masalah (0:00 - 2:30)
**(Visual: Wajah Full atau Slide Latar Belakang)**

"Assalamualaikum Warahmatullahi Wabarakatuh.
Salam sejahtera bagi kita semua.

Perkenalkan, nama saya **[Nama Anda]**. Pada kesempatan kali ini, saya mempresentasikan proyek akhir mata kuliah Machine Learning saya dengan judul **'Deteksi Lowongan Pekerjaan Palsu Menggunakan Algoritma Ensemble Learning'**.

**Latar Belakang Masalah:**
Topik ini saya angkat karena fenomena penipuan kerja (*employment scam*) yang semakin meresahkan. Berdasarkan studi dari **Vidros et al. (2017)**, kemudahan platform rekrutmen *online* sering disalahgunakan untuk mencuri data pribadi dan uang pelamar.
Konteks ini sangat relevan di Indonesia. Data Kementerian Luar Negeri RI mencatat lonjakan kasus penipuan kerja daring hingga **3.703 kasus** antara tahun 2020 hingga 2024. Dampaknya bukan hanya kerugian finansial, tetapi juga apa yang disebut oleh **Talroo (2023)** sebagai *'Erosion of Trust'* atau hilangnya kepercayaan masyarakat terhadap ekosistem digital.

**Tujuan Penelitian:**
Oleh karena itu, penelitian ini bertujuan untuk:
1.  Membangun model deteksi otomatis yang cerdas.
2.  Mengeksplorasi algoritma berbasis pohon keputusan (*Tree-based*) untuk membedakan pola lowongan asli dan palsu.
3.  Mengidentifikasi fitur metadata apa yang menjadi 'Red Flag' utama dalam sebuah lowongan kerja."

---

## Segmen 2: Tinjauan Pustaka & Metodologi (2:30 - 4:30)
**(Visual: Slide Metodologi atau File `Laporan_UAS_Machine_Learning.md`)**

"Dalam proyek ini, saya tidak hanya menggunakan satu, tetapi membandingkan dua pendekatan algoritma *Ensemble Learning* yang sering disebut sebagai *State-of-the-Art* untuk data tabular:

1.  **Random Forest (Bagging):**
    Metode ini bekerja dengan membangun ratusan pohon keputusan secara paralel (*Bagging*). Keunggulannya, seperti dikutip dari **Breiman (2001)**, adalah ketangguhannya terhadap *overfitting* dan kemampuannya menangani data berdimensi tinggi tanpa perlu normalisasi yang rumit.

2.  **Gradient Boosting (Boosting):**
    Sebagai pembanding, saya menggunakan konsep Boosting. Berbeda dengan Random Forest, metode ini membangun model secara berurutan, di mana model baru memperbaiki kesalahan model sebelumnya. Ini seringkali memberikan akurasi lebih tinggi pada pola data yang kompleks.

Namun, untuk demonstrasi hari ini, saya akan berfokus pada implementasi **Random Forest** karena stabilitasnya yang sangat baik pada dataset yang saya gunakan."

---

## Segmen 3: Dataset & Preprocessing (4:30 - 6:30)
**(Visual: Pindah ke VS Code, buka `analyze_data.py` atau CSV)**

"Mari kita bedah datanya.
Dataset yang saya gunakan adalah subset dari **EMSCAD (Employment Scam Aegean Dataset)** yang tersedia publik di Kaggle.
Data ini terdiri dari **3.000 sampel** dengan **25 fitur**.
Yang menarik, dataset ini **Balanced (Seimbang)**—50% Asli dan 50% Palsu. Ini menghilangkan kebutuhan akan teknik *resampling* seperti SMOTE yang biasanya wajib di kasus *fraud*.

**Strategi Preprocessing (CRISP-DM):**
Tantangan terbesar di data ini ada dua, dan inilah solusi teknis yang saya terapkan:

1.  **Pencegahan Data Leakage:**
    Saat Exploratory Data Analysis (EDA), saya menemukan kolom `fraud_reason`. Kolom ini berisi alasan kenapa lowongan itu palsu. Ini adalah 'bocoran jawaban'. Maka, kolom ini saya hapus total agar model valid.

2.  **Feature Engineering dari Missing Values:**
    Saya menemukan bahwa lowongan palsu memiliki banyak *Missing Values* di kolom `company_profile` dan `company_website`.
    Hipotesis saya: *'Penipu cenderung malas mengisi profil perusahaan.'*
    Maka, saya mengubah kolom teks kosong ini menjadi fitur biner: `has_company_website` (1 jika ada, 0 jika tidak). Ternyata, fitur sederhana ini memiliki korelasi negatif yang sangat kuat dengan label penipuan."

---

## Segmen 4: Demo Program & Implementasi (6:30 - 8:30)
**(Visual: Split Screen VS Code - Code `train_model.py` & Terminal)**

"Mari kita jalankan kodenya."

**(Aksi: Jalankan `python train_model.py`)**

"Sambil menunggu training berjalan, saya jelaskan alur kodenya:
1.  **Data Loading:** Membaca CSV.
2.  **Cleaning:** Menghapus fitur teks panjang yang berat (seperti deskripsi lengkap) dan menggantinya dengan fitur `text_length` (panjang karakter). Ini trik efisiensi komputasi yang saya ambil dari referensi penelitian Lal et al. (2019).
3.  **Encoding:** Menggunakan *Ordinal Encoding* untuk tingkat pendidikan (SMA ke S3) dan *One-Hot Encoding* untuk jenis industri.
4.  **Modeling:** Melatih Random Forest dengan 100 *estimators*."

**(Aksi: Tunjuk Output Terminal)**

"Dan inilah hasilnya. Proses training selesai dalam hitungan detik."

---

## Segmen 5: Pembahasan Hasil & Evaluasi (8:30 - 10:30)
**(Visual: Buka gambar di folder `plots/` secara berurutan)**

"Hasil evaluasi menunjukkan performa yang **Sempurna (Akurasi 100%)**. Mari kita bahas secara kritis apakah ini wajar?"

**(Buka `1_confusion_matrix.png`)**
"Pada Confusion Matrix, kita lihat:
*   **0 False Negatives:** Tidak ada penipuan yang lolos.
*   **0 False Positives:** Tidak ada lowongan asli yang salah dituduh.
Nilai **Precision, Recall, dan F1-Score** semuanya 1.0.

**(Buka `2_feature_importance.png` & `3_text_length_distribution.png`)**
"Mengapa bisa 100%? Apakah modelnya curang?
Jawabannya ada di grafik **Feature Importance** dan **Distribusi Teks** ini.
1.  **`text_length` (Panjang Teks):** Grafik merah ini menunjukkan lowongan palsu selalu memiliki deskripsi yang sangat pendek (kurang dari 1000 karakter). Sedangkan yang asli (biru) sangat panjang.
2.  **Metadata:** Hampir semua lowongan palsu tidak punya website.

Kombinasi dua fitur ini menciptakan *Separating Hyperplane* (garis pemisah) yang sangat tegas. Algoritma Random Forest dengan mudah menarik garis batas ini, sehingga akurasinya mencapai 100%."

**(Buka `4_correlation_heatmap.png`)**
"Heatmap ini mengonfirmasi bahwa korelasi antara `is_fake` dengan `has_company_profile` sangat negatif. Artinya, jika profil ada, kemungkinan palsu sangat kecil."

---

## Segmen 6: Kesimpulan & Diskusi Kritis (10:30 - 12:00)
**(Visual: Wajah Full)**

"Sebagai penutup, saya ingin menyampaikan Diskusi Kritis seperti yang tertuang dalam Bab Pembahasan laporan saya.

Meskipun akurasi 100% sangat memuaskan, kita harus waspada. Hasil ini menunjukkan bahwa dataset sampel ini memiliki pola penipuan yang 'stereotipikal' atau terlalu mudah ditebak (penipu malas).
Di dunia nyata (*In-the-wild*), penipu mungkin akan berevolusi dengan membuat website palsu atau deskripsi panjang.
Oleh karena itu, saran pengembangan selanjutnya adalah:
1.  Mengintegrasikan **NLP (Natural Language Processing)** untuk menganalisis semantik kata, bukan hanya panjang teks.
2.  Menguji model pada dataset tahun terbaru (2024-2025) untuk validasi robust-ness.

Secara keseluruhan, proyek ini berhasil membuktikan bahwa dengan **Feature Engineering** yang cerdas pada metadata, kita bisa mendeteksi penipuan dengan sangat efektif tanpa memerlukan komputasi yang berat.

Demikian presentasi UAS saya. Semoga bermanfaat.
Terima kasih. Wassalamualaikum Wr. Wb."