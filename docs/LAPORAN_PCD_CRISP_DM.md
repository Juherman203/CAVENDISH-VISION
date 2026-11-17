# DETEKSI TINGKAT KEMATANGAN BUAH PISANG CAVENDISH MENGGUNAKAN ANALISIS WARNA PADA CITRA DIGITAL (CAVENDISH VISION)

*(Dapat disisipkan Logo STT Wastukancana)*

**Disusun oleh :**

| Nama Mahasiswa | NIM | Jabatan | Kelas |
| :--- | :---: | :---: | :---: |
| Idhsan Nur Ramadhan | 231351062 | Ketua | Malam A |
| Anjas pebrian | 231351025 | Anggota | Malam A |
| Juherman | 231351069 | Anggota | Malam A |

**PROGRAM STUDI TEKNIK INFORMATIKA**  
**SEKOLAH TINGGI TEKNOLOGI WASTUKANCANA**  
**PURWAKARTA**  
**2026**

---

## DAFTAR ISI
- **DAFTAR ISI** ............................................................................................................ i
- **DAFTAR TABEL** ........................................................................................................ ii
- **DAFTAR GAMBAR** ..................................................................................................... iii
- **I. Business Understanding** ...................................................................................... 1
- **II. Data Understanding** .......................................................................................... 2
- **III. Data Preparation** ............................................................................................. 4
- **IV. Modeling** ......................................................................................................... 6
- **V. Evaluation** ........................................................................................................ 8
- **VI. Deployment** .................................................................................................... 10

---

## DAFTAR TABEL
- **Tabel 1.1** Pihak pengguna dan manfaat yang diperoleh ............................................ 2
- **Tabel 2.1** Jumlah citra pada setiap kelas kematangan Pisang Cavendish ..................... 3
- **Tabel 2.2** Karakteristik sebaran warna HSV pada empat fase kematangan ................. 4
- **Tabel 3.1** Pembagian dataset citra pisang (80% Training : 20% Testing) ..................... 4
- **Tabel 3.2** Rincian 12 fitur HSV dan fitur validasi geometri kontur ............................ 5
- **Tabel 4.1** Hasil pelatihan dan validasi silang model KNN (5-Fold CV) ......................... 7
- **Tabel 5.1** Hasil evaluasi performa model pada data testing (Classification Report) ..... 8
- **Tabel 5.2** Matriks kebingungan (Confusion Matrix) klasifikasi kematangan ................ 9

---

## DAFTAR GAMBAR
- **Gambar 1** Struktur folder dataset citra Pisang Cavendish ......................................... 3
- **Gambar 2** Contoh citra setiap kelas: (a) Mentah, (b) Setengah Matang, (c) Matang, (d) Terlalu Matang ................................................................................. 4
- **Gambar 3** Alur tahapan preprocessing citra dan segmentasi kontur objek ................ 5
- **Gambar 4** Visualisasi Confusion Matrix pengujian pada data testing .......................... 9
- **Gambar 5** Tampilan antarmuka website CAVENDISH VISION ................................. 10

---

# I. Business Understanding

### Latar Belakang Masalah
Bagi masyarakat Indonesia, buah pisang bukan sekadar hidangan pencuci mulut biasa. Pisang hadir hampir di setiap meja makan keluarga, pasar tradisional, minimarket, hingga hotel berbintang. Dari berbagai varietas pisang yang ada, Pisang Cavendish menjadi salah satu yang paling banyak dicari karena memiliki kulit yang mulus, rasa manis yang konsisten, dan daging buah yang empuk. 

Namun, Pisang Cavendish memiliki sifat klimakterik, yaitu buahnya akan terus matang dan mengalami perubahan fisik meski sudah dipetik dari tangkainya. Perubahan fase kematangannya berlangsung cukup cepat: berawal dari hijau keras (*mentah*), perlahan menguning (*setengah matang*), kuning cerah merata (*matang sempurna*), hingga akhirnya muncul bercak-bercak cokelat dan teksturnya lembek (*terlalu matang*). Setiap fase kematangan ini memiliki kegunaan yang berbeda-beda:
1. Pisang yang masih **mentah** lebih tahan lama dan cocok untuk pengiriman jarak jauh atau diolah menjadi keripik.
2. Pisang **setengah matang** paling dicari pedagang buah karena masih memiliki masa pajang beberapa hari di etalase toko.
3. Pisang **matang** adalah kondisi terbaik untuk langsung dimakan karena rasa manis dan aromanya sudah keluar sempurna.
4. Pisang **terlalu matang** harus segera diolah (misalnya dibuat bolu pisang atau jus) agar tidak keburu membusuk dan terbuang sia-sia.

Di sinilah letak masalahnya. Membedakan kematangan pisang di lapangan selama ini masih mengandalkan mata telanjang manusia (inspeksi manual). Cara ini sangat subjektif dan rentan salah, terutama saat petugas gudang atau pedagang sudah mulai lelah setelah menyortir ratusan buah, atau ketika kondisi cahaya di ruangan remang-remang. Salah memperkirakan kematangan buah bisa berujung rugi: pisang yang seharusnya sudah harus segera dijual malah tertahan di gudang sampai keburu busuk.

Di sisi lain, hampir semua pedagang, petani, maupun pembeli saat ini sudah memegang ponsel pintar berkamera. Pada bidang Pengolahan Citra Digital (PCD), perubahan kematangan pisang ditandai oleh pergeseran warna kulit dari hijau ke kuning lalu ke cokelat. Ruang warna **HSV (Hue, Saturation, Value)** terbukti sangat efektif membaca pola warna alami ini karena kanal warnanya tidak mudah tertipu oleh redup atau terangnya cahaya lampu ruangan. Dari situ muncul gagasan proyek ini: memanfaatkan foto buah pisang untuk mendeteksi tingkat kematangannya secara objektif, cepat, dan otomatis menggunakan algoritma **K-Nearest Neighbors (KNN)**.

### Tujuan
Tujuan umum proyek ini adalah membangun aplikasi berbasis web yang mampu mendeteksi tingkat kematangan buah Pisang Cavendish secara akurat hanya dari foto buah.

Secara lebih rinci, tujuan yang ingin dicapai ada tiga:
1. Membangun alur pengolahan citra (*preprocessing*) yang mampu memisahkan buah pisang dari latar belakangnya (*background*) secara bersih menggunakan segmentasi kontur OpenCV.
2. Mengekstraksi ciri warna kulit pisang murni berbasis ruang warna HSV (proporsi hijau, kuning, dan cokelat) serta melatih model klasifikasi KNN dua tahap (membedakan pisang vs bukan pisang, lalu mengelompokkan 4 tingkat kematangan).
3. Mengukur kemampuan model secara jujur pada data uji (*testing*) menggunakan metrik ilmiah seperti *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*, serta menyajikannya dalam antarmuka web modern yang mudah digunakan siapa saja.

### Solusi
Solusi yang ditawarkan adalah aplikasi web cerdas bernama **CAVENDISH VISION**. Alur kerjanya dirancang sangat sederhana bagi pengguna:
1. Pengguna memotret buah pisang menggunakan kamera HP/laptop atau mengunggah foto yang tersimpan di galeri.
2. Citra diperkecil secara proporsional (*resize* maksimal 320 piksel) dan diperhalus menggunakan *Gaussian Blur (5×5)* untuk menghilangkan pantulan cahaya yang menyilaukan pada kulit pisang.
3. Sistem secara otomatis memisahkan objek pisang dari latar belakang mejanya menggunakan teknik segmentasi kontur OpenCV.
4. Sistem menghitung persentase warna kulit (berapa persen warna hijau, kuning, dan cokelat) pada area buah yang sudah dipisahkan tadi.
5. Model KNN memprediksi apakah buah tersebut masuk ke dalam kategori **Mentah**, **Setengah Matang**, **Matang**, atau **Terlalu Matang**.
6. Layar website menampilkan hasil deteksi dengan tulisan besar yang tegas, dilengkapi *progress bar* persentase warna kulit serta penjelasan singkat mengenai kondisi pisang tersebut.

Perlu dicatat bahwa sistem ini dirancang sebagai alat bantu penyortiran cepat yang praktis dan objektif, sehingga dapat mempermudah pekerjaan sehari-hari tanpa memerlukan alat laboratorium yang mahal.

### Pengguna
Pihak yang diharapkan menerima manfaat dari proyek ini dirangkum pada Tabel 1.1.

**Tabel 1.1 Pihak pengguna dan manfaat yang diperoleh.**

| Pihak | Peran | Manfaat yang Diperoleh |
| :--- | :--- | :--- |
| **Petani & Pengepul Pisang** | Penyortir hasil panen | Terbantu memilah pisang yang siap kirim jauh (mentah) dan yang harus segera dijual ke pasar terdekat, sehingga penyortiran lebih seragam dan cepat. |
| **Pedagang Buah & Supermarket** | Pengelola etalase toko | Membantu memantau kondisi stok buah di etalase setiap hari, sehingga pisang yang mulai terlalu matang bisa segera dipromosikan atau dipajang lebih awal sebelum membusuk. |
| **Industri Olahan Makanan** | Manufaktur olahan pisang | Mempermudah pemilihan bahan baku yang tepat (misalnya pisang mentah untuk keripik pisang, atau pisang matang untuk selai dan kue). |
| **Konsumen Rumah Tangga** | Pembeli langsung | Memastikan pisang yang dibeli sesuai dengan rencana konsumsi keluarga di rumah (untuk langsung dimakan atau disimpan 2–3 hari ke depan). |
| **Mahasiswa & Peneliti** | Pengembang lanjutan | Memiliki contoh nyata penerapan teori PCD murni (ruang warna HSV dan segmentasi kontur) yang dihubungkan dengan machine learning (KNN) ke dalam aplikasi web nyata. |

---

# II. Data Understanding

### Sumber Data
Data yang digunakan dalam proyek ini bersumber dari dataset citra ilmiah publik berjudul **Banana Ripeness Classification (BananaID)** yang dipublikasikan melalui repositori **Mendeley Data** (*DOI: 10.17632/h6n5srjjyw.1*) dan tersedia secara paralel di platform **Kaggle**. 

Dataset ini berisi foto buah Pisang Cavendish asli dari berbagai sudut pemotretan di bawah pencahayaan terstandarisasi. Kode sumber proyek dapat diakses pada repositori lokal `c:\CAVENDISH VISION\`.

### Deskripsi Data
Dataset yang digunakan terdiri dari **380 citra digital** berformat `.jpg`. Seluruh berkas citra dikelompokkan ke dalam empat subfolder berdasarkan fase kematangannya, yaitu `mentah`, `setengah_matang`, `matang`, dan `terlalu_matang`. Struktur folder tersebut dapat dilihat pada Gambar 1.

```
dataset/
├── training/
│   ├── mentah/             (68 citra)
│   ├── setengah_matang/    (76 citra)
│   ├── matang/             (100 citra)
│   └── terlalu_matang/     (60 citra)
└── testing/
    ├── mentah/             (16 citra)
    ├── setengah_matang/    (20 citra)
    ├── matang/             (24 citra)
    └── terlalu_matang/     (16 citra)
```
*Gambar 1 Struktur folder dataset citra Pisang Cavendish.*

Jumlah citra pada setiap kelas kematangan disajikan pada Tabel 2.1.

**Tabel 2.1 Jumlah citra pada setiap kelas kematangan Pisang Cavendish.**

| Kelas Kematangan | Nama Folder | Jumlah Citra | Persentase | Karakteristik Utama |
| :--- | :--- | :---: | :---: | :--- |
| **Mentah (*Unripe*)** | `mentah` | 84 | 22,11% | Kulit hijau segar merata, buah masih keras dan kaku. |
| **Setengah Matang (*Semi-Ripe*)** | `setengah_matang` | 96 | 25,26% | Kulit bergradasi kuning kehijauan, bagian ujung masih hijau. |
| **Matang (*Ripe*)** | `matang` | 124 | 32,63% | Kulit kuning cerah merata, rasa manis optimal. |
| **Terlalu Matang (*Overripe*)** | `terlalu_matang` | 76 | 20,00% | Kulit kuning tua disertai bintik-bintik cokelat merata. |
| **Total** | | **380** | **100%** | |

Dari tabel di atas terlihat bahwa jumlah citra relatif seimbang di setiap kelasnya, dengan kelas Matang memiliki jumlah terbanyak (124 citra) dan kelas Terlalu Matang memiliki jumlah paling sedikit (76 citra).

### Explore Data (Karakteristik Visual Tiap Kelas)
Untuk mengenali karakteristik visual masing-masing kelas, diambil perwakilan citra seperti yang diperlihatkan pada Gambar 2.

| (a) Mentah | (b) Setengah Matang | (c) Matang | (d) Terlalu Matang |
| :---: | :---: | :---: | :---: |
| *Kulit hijau klorofil merata* | *Gradasi hijau dan kuning* | *Kuning keemasan cerah* | *Kuning tua dengan bintik cokelat* |

*Gambar 2 Contoh citra setiap kelas kematangan Pisang Cavendish.*

Berdasarkan analisis nilai warna pada ruang HSV, keempat kategori memiliki karakteristik nilai warna kulit yang sangat khas:
1. **Mentah:** Didominasi pigmen hijau segar dengan nilai *Hue* berkisar antara 28° hingga 35°. Nilai saturasi cukup tinggi dan belum ada bercak cokelat sama sekali.
2. **Setengah Matang:** Memperlihatkan percampuran antara hijau dan kuning. Nilai *Hue* bergeser ke rentang 24°–27°. Bagian badan buah mulai menguning cerah sedangkan area pangkal dan ujung tangkai masih mempertahankan warna hijau.
3. **Matang:** Didominasi warna kuning cerah dengan nilai *Hue* 18°–23° dan nilai *Value* (kecerahan) yang sangat tinggi (di atas 150). Bintik cokelat masih sangat sedikit atau bahkan belum muncul.
4. **Terlalu Matang:** Nilai kecerahan mulai menurun dan permukaan kulit mulai dipenuhi bintik atau bercak cokelat gelap (*senescence spots*) dengan nilai *Hue* cokelat kemerahan di bawah 18°.

Karakteristik sebaran warna pada setiap kelas kematangan dirangkum pada Tabel 2.2.

**Tabel 2.2 Karakteristik sebaran warna HSV pada empat fase kematangan.**

| Kategori Kematangan | Rentang Nilai Hue (0–180) | Karakteristik Visual Kulit | Persentase Bintik Cokelat |
| :--- | :--- | :--- | :---: |
| **Mentah** | 28 – 38 (Domain Hijau Pekat) | Hijau segar homogen, buah padat dan kaku | 0,0% |
| **Setengah Matang** | 24 – 28 (Kuning Kehijauan) | Gradasi warna hijau di ujung dan kuning di badan | < 1,5% |
| **Matang** | 18 – 24 (Kuning Cerah Merata) | Kuning keemasan cerah merata, aroma manis keluar | 1,0% – 5,0% |
| **Terlalu Matang** | 8 – 22 (Kuning Berbintik Cokelat) | Kuning tua kusam dengan bintik cokelat menyebar | > 8,0% hingga > 30% |

### Kualitas Data
Secara umum kualitas dataset ini sangat baik untuk keperluan pembelajaran dan riset pengolahan citra digital:
- Foto diambil dengan pencahayaan yang cukup stabil dan fokus yang tajam pada buah.
- Objek buah pisang diletakkan di tengah bingkai sehingga kontur tepian buah mudah dikenali dan dipisahkan dari latar belakang.
- Seluruh 380 berkas citra beroperasi normal tanpa ada file yang rusak (*corrupted*) atau terpotong secara ekstrem, sehingga 100% citra berhasil diproses ke tahap berikutnya.

---

# III. Data Preparation

### Pemilihan Data & Pembagian Dataset
Dari 380 foto pisang yang tersedia, dataset dibagi menjadi dua bagian dengan perbandingan **80% untuk data pelatihan (*training*)** dan **20% untuk data pengujian (*testing*)**. Pembagian dilakukan menggunakan metode *Stratified Sampling* dengan nilai *random state* 42 agar proporsi tiap kelas tetap seimbang dan hasilnya dapat diuji ulang secara konsisten. Rincian pembagian disajikan pada Tabel 3.1.

**Tabel 3.1 Pembagian dataset citra pisang (80% Training : 20% Testing).**

| Kategori Kematangan | Data Training (80%) | Data Testing (20%) | Total Sampel |
| :--- | :---: | :---: | :---: |
| Mentah | 68 | 16 | 84 |
| Setengah Matang | 76 | 20 | 96 |
| Matang | 100 | 24 | 124 |
| Terlalu Matang | 60 | 16 | 76 |
| **Total Keseluruhan** | **304** | **76** | **380** |

Data *testing* sebanyak 76 citra ini disimpan di folder terpisah dan **sama sekali tidak dilibatkan saat pelatihan**, sehingga murni digunakan untuk menguji seberapa pintar model saat melihat foto baru.

### Data Preprocessing
Tahap *preprocessing* bertujuan menyiapkan citra agar seragam dan bebas dari gangguan latar belakang. Seluruh tahapan ini dijalankan menggunakan pustaka OpenCV pada berkas `image_processing.py`, dengan urutan langkah sebagai berikut:

```
[Foto Asli] ──> [Resize Proporsional (Maks. 320 px)] ──> [Gaussian Blur (5x5)]
     │
     └───> [Konversi Ruang Warna BGR ke HSV]
                 │
                 └───> [Segmentasi Kontur & Masking Kulit Buah]
                             │
                             └───> [Ekstraksi 12 Fitur Warna HSV]
```
*Gambar 3 Alur tahapan preprocessing citra dan segmentasi kontur objek.*

1. **Pemeriksaan Citra:** Memeriksa apakah file gambar dapat dibaca dengan baik. Jika file menggunakan format PNG dengan latar transparan, sistem secara otomatis memberi warna dasar terang netral agar area transparan tidak berubah menjadi hitam pekat.
2. **Pengecilan Ukuran Proporsional (*Proportional Resize*):** Mengubah ukuran foto dengan batas sisi terpanjang maksimal 320 piksel, dengan tetap mempertahankan rasio panjang-lebar aslinya agar bentuk pisang tidak mengalami distorsi (tidak gepeng atau melar).
3. **Pereduksian Noise (*Gaussian Blur 5×5*):** Menerapkan filter pelembut berukuran 5×5 piksel untuk meredam kilatan cahaya lampu (*glare*) pada permukaan kulit buah yang licin.
4. **Konversi Ruang Warna (BGR ke HSV):** Mengubah format warna standar kamera (BGR) menjadi ruang warna HSV untuk memisahkan komponen warna murni (*Hue*) dan kemurnian (*Saturation*) dari intensitas cahaya (*Value*).
5. **Segmentasi Kontur Objek:** Memisahkan buah pisang dari latar belakangnya menggunakan perbedaan warna tepi (*border background estimation*) yang diperhalus melalui operasi morfologi penutup lubang (*closing* dan *opening*).
6. **Ekstraksi Fitur Warna HSV:** Menghitung nilai statistik (*mean*, *median*, kuartil *Hue*, serta persentase warna hijau, kuning, dan cokelat) **khusus pada area kulit pisang yang telah tersegmentasi**. Latar belakang meja atau dinding sama sekali tidak diikutsertakan.

Rincian 12 fitur HSV yang digunakan untuk klasifikasi kematangan serta fitur pendukungnya dirangkum pada Tabel 3.2.

**Tabel 3.2 Rincian 12 fitur HSV dan fitur validasi geometri kontur.**

| Kategori Fitur | Jumlah | Nama Fitur yang Diekstraksi | Fungsi dan Kegunaan |
| :--- | :---: | :--- | :--- |
| **Fitur Klasifikasi Kematangan (Model KNN)** | 12 | `mean_h`, `median_h`, `q25_h`, `q75_h`, `mean_v`, `std_v`, `green_ratio`, `yellow_green_ratio`, `yellow_ratio`, `brown_ratio`, `peel_green_pct`, `peel_yellow_pct` | Mengukur derajat warna hijau klorofil, dominasi warna kuning, serta proporsi bintik cokelat pada area kulit pisang murni. |
| **Fitur Validasi Bentuk (Pra-pemrosesan)** | 7 | `aspect_ratio_bbox`, `elongation_ratio`, `circularity`, `solidity`, `extent`, `banana_spectrum_ratio`, `non_banana_hue_ratio` | Memeriksa apakah bentuk objek memanjang khas pisang dan menyaring benda bulat (seperti apel/jeruk) atau warna asing di luar spektrum pisang. |

### Data Issue
Dalam proses pengolahan citra digital di dunia nyata, terdapat beberapa tantangan yang berhasil diatasi oleh sistem:
1. **Warna Latar Belakang Ikut Terhitung:** Jika latar belakang meja (misalnya meja cokelat atau taplak hijau) ikut dihitung, warna pisang akan terbaca salah. Masalah ini diselesaikan melalui **segmentasi kontur adaptif dan erosi tepi 2 piksel**, sehingga piksel yang dihitung murni hanya kulit pisang.
2. **Pantulan Cahaya Lampu (*Specular Glare*):** Kilatan lampu pada kulit pisang yang licin menghasilkan piksel putih terang (saturasi sangat rendah dan nilai kecerahan tinggi). Sistem secara otomatis menyaring piksel kilatan tersebut agar tidak merusak rata-rata warna pisang.
3. **Ujung Tangkai yang Berwarna Gelap Alami:** Ujung pangkal tangkai pisang sering berwarna hitam atau cokelat alami, yang bisa keliru dianggap sebagai pisang busuk. Sistem mengatasinya dengan memfokuskan deteksi bintik cokelat pada area badan buah (*body region*), sehingga ujung tangkai gelap tidak menipu model.
4. **Objek Bukan Pisang atau Gambar Rusak:** Jika pengguna mengunggah gambar yang terlalu gelap, buram, atau foto buah lain yang bulat padat (seperti apel atau jeruk), sistem akan membangkitkan pesan penolakan yang ramah sebelum proses klasifikasi kematangan dijalankan.

---

# IV. Modeling

### Algoritma yang Digunakan
Model klasifikasi pada proyek ini menggunakan algoritma **K-Nearest Neighbors (KNN)**. Cara kerja KNN sangat intuitif: model menyimpan vektor fitur dari 304 foto pisang yang sudah dipelajari saat pelatihan. Ketika ada foto pisang baru yang masuk, model akan mengukur jarak kemiripan warna pisang tersebut ke seluruh data pelatihan, lalu mengambil sejumlah $K$ tetangga terdekat yang paling mirip warnanya untuk menentukan tingkat kematangannya.

Sebelum jarak dihitung, seluruh data dinormalisasi menggunakan **`StandardScaler`** ($z = (x - \mu) / \sigma$) agar semua fitur memiliki skala nilai yang seimbang dan fitur dengan angka besar tidak menenggelamkan fitur lainnya.

Jarak kemiripan dihitung menggunakan formula jarak **Manhattan Distance**:

$$d(p, q) = \sum_{i=1}^n |p_i - q_i|$$

Metrik Manhattan terbukti lebih tangguh terhadap kemunculan bintik cokelat kecil pada kulit pisang. Selain itu, sistem menggunakan pembobotan berbasis jarak (*weights='distance'*), di mana tetangga yang jarak warnanya paling dekat mendapat pengaruh suara yang lebih besar.

### Proses Pelatihan & Pencarian Parameter Terbaik
Pelatihan dijalankan secara otomatis melalui skrip `train.py`. Untuk menemukan kombinasi parameter terbaik, dilakukan proses **Grid Search** yang dipadukan dengan **5-Fold Stratified Cross-Validation** pada 304 data *training*. Sistem menguji berbagai nilai $K$ ganjil (1, 3, 5, 7, 9, 11) dan membandingkan fungsi bobot serta metrik jarak. Hasil parameter terbaik dirangkum pada Tabel 4.1.

**Tabel 4.1 Hasil pelatihan dan validasi silang model KNN (5-Fold CV).**

| Model Klasifikasi | Jumlah Fitur Masukan | Parameter Terbaik Terpilih | Metode Validasi | Skor Validasi Silang (F1-Macro) | Akurasi Data Latih |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Klasifikasi 4 Kematangan Pisang** | 12 Fitur HSV | $K = 11$, Manhattan Distance, Distance-weighted | 5-Fold Stratified CV | **89,73% – 94,79%** | **100,00%** |

Model KNN yang sudah terlatih kemudian disimpan secara permanen ke dalam format berkas `.joblib` di folder `models/` agar dapat langsung dipanggil oleh website tanpa harus melatih ulang setiap saat.

---

# V. Evaluation

### Hasil Evaluasi Model pada Data Testing
Pengujian performa akhir dilakukan terhadap **76 data pengujian (*testing*)** yang belum pernah dilihat model saat pelatihan. Hasil pengujian pada modul `evaluate.py` disajikan pada Tabel 5.1.

**Tabel 5.1 Hasil evaluasi performa model pada data testing (Classification Report).**

| Kategori Kematangan | Presisi (*Precision*) | Kepekaan (*Recall*) | *F1-Score* | Jumlah Sampel Uji (*Support*) |
| :--- | :---: | :---: | :---: | :---: |
| **Mentah (*Unripe*)** | 100,00% | 93,75% | 96,77% | 16 citra |
| **Setengah Matang (*Semi-Ripe*)** | 95,24% | 100,00% | 97,56% | 20 citra |
| **Matang (*Ripe*)** | 100,00% | 100,00% | 100,00% | 24 citra |
| **Terlalu Matang (*Overripe*)** | 100,00% | 100,00% | 100,00% | 16 citra |
| **Rata-rata Makro (*Macro Avg*)** | **98,81%** | **98,44%** | **98,58%** | **76 citra** |
| **Rata-rata Tertimbang (*Weighted Avg*)** | **98,75%** | **98,68%** | **98,68%** | **76 citra** |
| **Akurasi Keseluruhan (*Accuracy*)** | **-** | **-** | **98,68%** | **76 citra** |

### Analisis Confusion Matrix & Kesalahan Prediksi
Untuk melihat secara detail apakah ada kelas yang sempat tertukar, hasil pengujian dipetakan ke dalam *Confusion Matrix* pada Tabel 5.2.

**Tabel 5.2 Matriks kebingungan (Confusion Matrix) klasifikasi kematangan.**

| Aktual \ Prediksi | Prediksi Mentah | Prediksi Setengah Matang | Prediksi Matang | Prediksi Terlalu Matang | Total Aktual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Aktual Mentah** | **15** | 1 | 0 | 0 | 16 |
| **Aktual Setengah Matang** | 0 | **20** | 0 | 0 | 20 |
| **Aktual Matang** | 0 | 0 | **24** | 0 | 24 |
| **Aktual Terlalu Matang** | 0 | 0 | 0 | **16** | 16 |
| **Total Prediksi** | **15** | **21** | **24** | **16** | **76** |

Dari total 76 foto uji, model berhasil menebak **75 foto dengan benar**, dan hanya terjadi **1 kesalahan prediksi**:
- **Tingkat Akurasi Sangat Tinggi (98,68%):** Model mencapai akurasi hampir sempurna pada data uji.
- **Analisis Kesalahan (1 Citra Mentah tertebak Setengah Matang):** Kesalahan ini terjadi pada salah satu pisang mentah yang baru mulai memasuki masa peralihan, di mana warna hijau tua di dekat pangkal buah mulai sedikit memudar ke arah kuning kehijauan sehingga rasio kuning-hijaunya sedikit melampaui batas ambang kelas mentah murni.
- **Presisi 100% pada Kelas Matang dan Terlalu Matang:** Kedua kelas yang paling penting bagi pedagang dan konsumen, yaitu pisang siap konsumsi (Matang) dan pisang yang harus segera diolah (Terlalu Matang), berhasil ditebak dengan benar tanpa ada kesalahan sama sekali (*F1-Score 100%*).

---

# VI. Deployment

### Penerapan Project dalam Dunia Nyata
Sistem **CAVENDISH VISION** telah diwujudkan ke dalam bentuk aplikasi berbasis web interaktif menggunakan framework **Streamlit** (file `app.py`) dan dipercantik dengan lembar gaya CSS modern (`style.css`). Desain website dibuat responsif sehingga nyaman dibuka baik dari layar laptop maupun ponsel pintar (*smartphone*).

Aplikasi web ini memiliki tiga halaman utama:
1. **Halaman Beranda:** Menampilkan perkenalan aplikasi serta ilustrasi pisang interaktif yang dapat diklik untuk melihat penjelasan warna dari keempat tingkat kematangan.
2. **Halaman Deteksi Pisang:** Halaman utama tempat pengguna memasukkan foto pisang, baik dengan memilih file dari galeri maupun dengan memotret langsung menggunakan kamera HP/laptop. Setelah tombol *Mulai Deteksi* ditekan, halaman ini menampilkan:
   - Status keberhasilan identifikasi buah pisang.
   - Kategori tingkat kematangan dalam huruf besar dan jelas (misalnya: **`Pisang Matang`**).
   - Penjelasan singkat yang ramah dan mudah dipahami pengguna awam.
   - Tiga *progress bar* rapi yang memperlihatkan komposisi persentase warna kulit pisang (**Hijau**, **Kuning**, dan **Cokelat**).
   - Pesan penolakan yang informatif jika pengguna mengunggah foto selain pisang atau foto yang tidak jelas.
3. **Halaman Tentang Aplikasi:** Memberikan panduan praktis mengenai cara mengambil foto pisang yang baik (pencahayaan merata dan buah berada di tengah bingkai) agar hasil deteksi optimal.

### Panduan Pengoperasian Sistem
Aplikasi dapat dijalankan secara mudah di komputer lokal menggunakan terminal (PowerShell atau Command Prompt) melalui langkah-langkah berikut:
1. Pastikan paket pendukung sudah terpasang dengan menjalankan:
   ```powershell
   pip install -r requirements.txt
   ```
2. Jalankan pelatihan model jika ingin memperbarui bobot model KNN (opsional):
   ```powershell
   python train.py
   ```
3. Jalankan pengujian performa model untuk menghasilkan laporan evaluasi dan grafik *Confusion Matrix* (opsional):
   ```powershell
   python evaluate.py
   ```
4. Buka aplikasi web CAVENDISH VISION:
   ```powershell
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka pada browser default di alamat `http://localhost:8501`.
