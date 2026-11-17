# 🍌 CAVENDISH VISION

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-KNN-F7931E?logo=scikitlearn&logoColor=white)

> **"Deteksi Tingkat Kematangan Buah Pisang Cavendish Menggunakan Analisis Warna pada Citra Digital"**

**CAVENDISH VISION** adalah aplikasi berbasis web (*Computer Vision & Machine Learning*) untuk mendeteksi dan mengklasifikasikan tingkat kematangan buah **Pisang Cavendish** secara otomatis, cepat, dan objektif melalui citra digital (dari unggahan **Galeri** maupun tangkapan **Kamera** langsung).

Sistem ini menggunakan ekstraksi fitur ruang warna **HSV (Hue, Saturation, Value)**, analisis geometri bentuk kontur objek, serta algoritma klasifikasi **K-Nearest Neighbors (KNN) Dua Tahap**.

---

## ✨ Fitur Utama Aplikasi

1. **Input Fleksibel (Galeri & Kamera Langsung):** Mendukung unggahan file gambar (`JPG`, `JPEG`, `PNG`) dari perangkat maupun pemotretan langsung melalui *webcam*/kamera.
2. **Segmentasi Objek Otomatis:** Memisahkan objek utama buah pisang dari latar belakang (*background*) secara otomatis menggunakan *adaptive thresholding* dan operasi morfologi citra.
3. **Klasifikasi KNN Dua Tahap (*Two-Stage Classification*):**
   - **Tahap 1 (Detektor Validitas Objek):** Memfilter apakah objek pada foto adalah buah **Pisang** atau **Bukan Pisang** berdasarkan 16 fitur bentuk kontur, tekstur, dan spektrum warna.
   - **Tahap 2 (Klasifikasi Tingkat Kematangan):** Mengklasifikasikan buah pisang ke dalam 4 tingkat kematangan berdasarkan 12 fitur statistik ruang warna HSV dan persentase warna kulit.
4. **Visualisasi Komposisi Warna Kulit:** Menampilkan persentase proporsi warna kulit pisang (**Hijau**, **Kuning**, dan **Cokelat**) beserta penjelasan diagnostik otomatis.

---

## 📊 Empat Kategori Kematangan Pisang Cavendish

| Tahap | Kategori | Profil Warna HSV | Karakteristik Fisik |
| :--- | :--- | :--- | :--- |
| **Tahap 1** | **Mentah** (`mentah`) | Dominan Hijau (`Hue 28°–75°`) | Kulit hijau segar merata, tekstur masih keras, dan belum manis. |
| **Tahap 2** | **Setengah Matang** (`setengah_matang`) | Hijau Kekuningan (`Hue 24°–30°`) | Warna kulit bergradasi hijau kekuningan dan tekstur mulai melunak. |
| **Tahap 3** | **Matang** (`matang`) | Kuning Cerah (`Hue 18°–24°`) | Kulit kuning cerah merata, rasa manis optimal, dan siap dikonsumsi. |
| **Tahap 4** | **Terlalu Matang** (`terlalu_matang`) | Kuning Tua Bercak Cokelat (`Hue < 18°`) | Kulit kuning tua disertai banyak bercak cokelat dan tekstur sangat lunak. |

---

## 🛠️ Teknologi yang Digunakan

- **Python & Streamlit** — Antarmuka aplikasi web interaktif dan responsif.
- **OpenCV, Pillow, NumPy & SciPy** — *Preprocessing* citra, reduksi *noise* (*Gaussian Blur*), konversi ruang warna BGR ke HSV, segmentasi *background*, dan ekstraksi geometri kontur.
- **Scikit-learn & Joblib** — Pipeline normalisasi fitur (`StandardScaler`), optimasi hyperparameter (`GridSearchCV` + `StratifiedKFold`), dan serialisasi model `KNeighborsClassifier`.
- **Matplotlib** — Pembuatan visualisasi grafik *Confusion Matrix* pada modul evaluasi.

---

## 📁 Struktur Proyek

```text
CAVENDISH VISION/
├── .streamlit/config.toml        # Konfigurasi tema warna Streamlit
├── dataset/
│   ├── training/                 # Data latih (mentah, setengah_matang, matang, terlalu_matang, bukan_pisang)
│   └── testing/                  # Data uji (mentah, setengah_matang, matang, terlalu_matang, bukan_pisang)
├── models/                       # Artefak model KNN (.joblib) hasil pelatihan
├── results/                      # Laporan metrik evaluasi & visualisasi Confusion Matrix (.png)
├── docs/                         # Dokumentasi laporan penelitian
├── app.py                        # Aplikasi utama website Streamlit
├── style.css                     # Stylesheet CSS kustom antarmuka web
├── image_processing.py           # Modul preprocessing citra, segmentasi kontur, ekstraksi HSV, & inferensi KNN
├── train.py                      # Modul pelatihan model KNN Dua Tahap
├── evaluate.py                   # Modul pengujian dan evaluasi performa model
├── requirements.txt              # Daftar pustaka/dependensi Python
└── README.md                     # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/Juherman203/CAVENDISH-VISION.git
   cd "CAVENDISH-VISION"
   ```

2. **Instal pustaka Python yang dibutuhkan:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi website Streamlit:**
   ```bash
   python -m streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

4. **(Opsional) Melatih ulang & mengevaluasi model KNN:**
   ```bash
   python train.py
   python evaluate.py
   ```
