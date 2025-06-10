# Laporan Proyek Machine Learning - Carmen Emanuela Dwiva Lisapaly

## Project Overview

### Latar Belakang:
**Masalah yang ingin diselesaikan dalam proyek ini adalah banyaknya pilihan film yang tersedia sehingga pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka.** Hal ini menjadi penting karena tanpa bantuan sistem yang tepat, pengguna bisa mengalami kebingungan atau kehilangan minat akibat kelebihan informasi. Untuk mengatasi hal tersebut, dibutuhkan sistem rekomendasi yang mampu menyaring dan menampilkan film-film yang paling relevan bagi setiap pengguna.

**Sistem rekomendasi yang dikembangkan dalam proyek ini bekerja dengan menganalisis preferensi pengguna berdasarkan riwayat interaksi atau rating film sebelumnya.** Dengan pendekatan ini, sistem dapat mengenali pola kesukaan pengguna dan merekomendasikan film yang serupa. Metode yang digunakan bersifat kolaboratif, yaitu dengan mencocokkan pengguna dengan pengguna lain yang memiliki preferensi serupa, atau dengan mengidentifikasi kemiripan antarfilm berdasarkan rating historis.

Namun, tantangan utama dari pendekatan ini adalah kelangkaan data (data sparsity), terutama ketika jumlah pengguna dan film terus meningkat. Hal ini membuat proses pencarian 'tetangga' yang relevan menjadi lebih sulit dan memengaruhi kualitas rekomendasi. Oleh karena itu, **penyempurnaan algoritma pemilihan tetangga dan penanganan data sparsity menjadi kunci dalam meningkatkan kinerja sistem rekomendasi ini.**

### Referensi:
- J. Salter dan N. Antonopoulos, “Resolving Data Sparsity and Cold Start in Recommender Systems,” Springer, 2012.
- Yohon dkk., “Improving Data Sparsity in Recommender Systems Using Matrix Reconstruction Method,” MDPI Mathematics, vol. 11, no. 2, Mar. 2023.
- P. Li, S. A. M. Noah, dan H. Sarim, “A Survey on Deep Neural Networks in Collaborative Filtering Recommendation Systems,” arXiv:2412.01378, Dec. 2024.
- T. Zhang dkk., “Deep Learning based Recommender System: A Survey and New Perspectives,” arXiv:1707.07435, Jul. 2017.
- L. Fallahi dan J. Mohammadzadeh, “Leveraging Deep Learning Techniques on Collaborative Filtering Recommender Systems,” arXiv:2304.09282, Apr. 2023.
- A. C. Althbiti dkk., “Addressing Data Sparsity in CF Using Clustering and ANN (CANNBCF),” ResearchGate, 2022.
- J. Doe dkk., “A Comprehensive Review of Recommender Systems (2017–2024),” Journal of Big Data, 2022.

## Business Understanding

### Problem Statements:
Bagaimana sistem dapat menyarankan film kepada seorang pengguna dengan mengacu pada preferensi atau kesukaan pengguna lain yang memiliki selera serupa?

### Goals
Mengembangkan sistem rekomendasi yang mampu memberikan hasil yang relevan dan tepat, dengan memanfaatkan data histori seperti rating dan pola aktivitas pengguna sebelumnya.

### Solution Approach
Solusi yang diusulkan dalam proyek ini adalah dengan memanfaatkan dua pendekatan algoritma *Machine Learning* untuk sistem rekomendasi, yaitu:
- **Content-Based Filtering**, yaitu metode yang memberikan rekomendasi film dengan mempertimbangkan kesamaan karakteristik dari film-film yang sebelumnya disukai atau telah ditonton oleh pengguna. Algoritma ini berfokus pada preferensi individual berdasarkan riwayat aktivitas pengguna.
- **Collaborative Filtering**, yaitu pendekatan yang menggunakan informasi dari pengguna lain untuk memberikan saran. Algoritma ini tidak membutuhkan detail khusus dari masing-masing film, melainkan mengandalkan pola rating yang diberikan oleh banyak pengguna.
Dalam implementasinya, algoritma Content-Based Filtering digunakan untuk menyarankan film yang mirip dengan yang telah disukai pengguna sebelumnya, sedangkan Collaborative Filtering digunakan untuk menyarankan film yang mendapat penilaian tinggi dari banyak pengguna.

## Data Understanding

### Dataset:
Dataset yang digunakan adalah: Movie Recomendation Data. Sumber: [Movie Recomendarion Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)

### Jumlah Data:
- Jumlah data link movie: **9742**
- Jumlah data movie: **9742**
- Jumlah data ratings dari user (user unik): **610**
- Jumlah data ratings dari user (movie yang dirating): **9724**
- Jumlah data movie yang diberi tag: **81572**
  
### Variabel-variabel pada Movie Recomendation Dataset adalah Sebagai Berikut:
Dataset ini terdiri dari empat file utama: `movies.csv`, `ratings.csv`, `tags.csv`, dan `links.csv`. Masing-masing memiliki struktur dan fungsi sebagai berikut:
#### 1. `movies.csv` – Daftar film beserta informasi genre

Jumlah data: 9.742 baris

**Fitur-fitur:**

* `movieId` : ID unik untuk setiap film.
* `title` : Judul lengkap film, termasuk tahun rilis dalam tanda kurung.
* `genres` : Daftar genre film, dipisahkan dengan tanda `|` jika lebih dari satu (contoh: "Action|Adventure|Fantasy").

#### 2. `ratings.csv` – Data penilaian dari pengguna terhadap film

Jumlah data: 100.836 baris

**Fitur-fitur:**

* `userId` : ID unik untuk setiap pengguna yang memberikan rating.
* `movieId` : ID film yang dirating (berelasi dengan `movies.csv`).
* `rating` : Nilai rating yang diberikan pengguna, dalam skala 0.5 sampai 5.0.
* `timestamp` : Waktu (dalam format Unix timestamp) saat rating diberikan.

#### 3. `tags.csv` – Kata kunci atau tag yang diberikan pengguna pada film

Jumlah data: 3.683 baris

**Fitur-fitur:**

* `userId` : ID pengguna yang memberikan tag.
* `movieId` : ID film yang diberi tag.
* `tag` : Kata kunci atau frasa bebas yang digunakan pengguna untuk menggambarkan film.
* `timestamp` : Waktu saat tag tersebut diberikan.
  
#### 4. `links.csv` – Relasi antara `movieId` dengan ID dari database film eksternal

Jumlah data: 9.742 baris

**Fitur-fitur:**

* `movieId` : ID unik film (relasi utama antar dataset).
* `imdbId` : ID film di Internet Movie Database (IMDb).
* `tmdbId` : ID film di The Movie Database (TMDb). Terdapat beberapa nilai kosong.

### Tahapan Pengolahan Data:
- Melakukan **Exploratory Data Analysis (EDA)** untuk memahami struktur dan hubungan antar variabel dalam dataset.
- Mengamati **hubungan antar variabel berdasarkan `movieId` dan `userId`** guna memahami keterkaitan data.
- **Menggabungkan seluruh variabel yang berkaitan** dengan film ke dalam satu DataFrame `movie_all` berdasarkan `movieId`.
- **Menggabungkan seluruh variabel yang berkaitan dengan user** ke dalam satu DataFrame `user_all` berdasarkan `userId`.

## Data Preparation

### Teknik:

* **Penanganan Missing Value**: Mengecek dan menghapus data yang kosong untuk menjaga kualitas data.
* **Pengurutan Data**: Mengurutkan data berdasarkan `movieId` untuk menjaga konsistensi.
* **Penghapusan Duplikasi**: Menghapus data ganda berdasarkan `movieId` agar setiap film hanya muncul satu kali.
* **Konversi ke List**: Mengubah kolom-kolom penting (`movieId`, `title`, `genres`) ke dalam bentuk list untuk kebutuhan selanjutnya.
* **Pembuatan Dictionary**: Menyatukan list menjadi struktur dictionary menggunakan DataFrame baru untuk proses pemodelan.

### Proses Data Preparation:

1. **Pengecekan Missing Value**
   Dataset `all_movie` diperiksa menggunakan `isnull().sum()` dan ditemukan 52.549 nilai kosong pada kolom `tag`.
2. **Penghapusan Missing Value**
   Nilai kosong dibuang dengan `dropna()`. Jumlah baris berkurang dari 285.762 menjadi 233.213.
3. **Validasi Pembersihan**
   Dataset `all_movie_clean` dicek kembali menggunakan `isnull().sum()` dan dipastikan tidak ada missing value tersisa.
4. **Pengurutan Berdasarkan `movieId`**
   Dataset diurutkan berdasarkan `movieId` secara ascending dan disimpan sebagai `fix_movie`.
5. **Pemeriksaan Jumlah Film Unik**
   Jumlah film unik dihitung dengan `len(fix_movie.movieId.unique())`.
6. **Penetapan Dataset Final**
   Dataset yang telah diurutkan disimpan sebagai variabel `preparation`.
7. **Penghapusan Duplikasi**
   Dilakukan `drop_duplicates('movieId')` untuk memastikan tidak ada data film yang berulang.
8. **Konversi ke List**
   Kolom `movieId`, `title`, dan `genres` dikonversi ke dalam bentuk list menggunakan `.tolist()`:
   * `movie_id`: List ID film
   * `movie_name`: List judul film
   * `movie_genre`: List genre film
9. **Validasi Panjang List**
   Dicek panjang elemen dari masing-masing list untuk memastikan konsistensi data.
10. **Pembuatan Dictionary**
    List yang sudah dibuat digabungkan ke dalam sebuah DataFrame baru bernama `movie_new` sebagai struktur dictionary untuk keperluan pemodelan.
    
## Modeling and Result

1. **Pembuatan Sistem Rekomendasi**
   Untuk menyelesaikan permasalahan dalam merekomendasikan film kepada pengguna, dibangun sistem rekomendasi menggunakan dua pendekatan algoritma machine learning:
    * **Content-Based Filtering**, yang merekomendasikan film berdasarkan kesamaan konten (genre).
    * **Collaborative Filtering**, yang menggunakan pola rating dari banyak pengguna.
   
2. **Content-Based Filtering**
   Pendekatan ini merekomendasikan film yang mirip dengan film yang disukai sebelumnya, berdasarkan kemiripan genre.
   Langkah-langkah:
   * **TF-IDF Vectorizer** digunakan untuk mengubah kolom `genre` menjadi representasi numerik.
      ```
      python
      from sklearn.feature_extraction.text import TfidfVectorizer
    
      tf = TfidfVectorizer()
      tfidf_matrix = tf.fit_transform(movie_new['genre'])
      ```
   * **Cosine Similarity** digunakan untuk mengukur kesamaan antar film.
     ```
     python
     from sklearn.metrics.pairwise import cosine_similarity
      
     cosine_sim = cosine_similarity(tfidf_matrix)
     cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_new['movie_name'], columns=movie_new['movie_name'])
     ```
   * Fungsi Rekomendasi:
     ```
     python
     def movie_recommendations(nama_movie, similarity_data=cosine_sim_df, items=movie_new[['movie_name', 'genre']], k=5):
     index = similarity_data.loc[:, nama_movie].to_numpy().argpartition(range(-1, -k, -1))
     closest = similarity_data.columns[index[-1:-(k+2):-1]]
     closest = closest.drop(nama_movie, errors='ignore')
     return pd.DataFrame(closest).merge(items).head(k)
      ```
   Berdasarkan histori pengguna, sistem mendeteksi bahwa pengguna menyukai film Jumanji (1995) yang bergenre Adventure, Children, Fantasy.

   ![Jumanji 1995](images/Jumanji1995.png)
   
   Sebagai hasilnya, berikut adalah Top-5 rekomendasi berdasarkan Content-Based Filtering:
   
   ![Top-5](images/Top5.png)

   Terlihat bahwa film-film dengan genre serupa (Adventure, Children, Fantasy) mendominasi hasil rekomendasi. Hal ini mencerminkan pendekatan content-based yang menggunakan kemiripan konten sebagai dasar rekomendasi.
  
4. **Collaborative Filtering**
   Berbeda dengan metode sebelumnya, pendekatan ini menggunakan rating yang diberikan pengguna lain terhadap film, dan menemukan pola kesamaan antar pengguna.
   * Data Preparation:
     ```
     python
     df = ratings.copy()
      
     # Encoding userId dan movieId
     user_ids = df['userId'].unique().tolist()
     user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
     movie_ids = df['movieId'].unique().tolist()
     movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
      
     df['user'] = df['userId'].map(user_to_user_encoded)
     df['movie'] = df['movieId'].map(movie_to_movie_encoded)
     df['ratings'] = df['rating'].astype(np.float32)
     
     # Statistik dasar
     num_users = len(user_to_user_encoded)
     num_movies = len(movie_to_movie_encoded)
     min_rating = df['rating'].min()
     max_rating = df['rating'].max()
      
     print(f'Users: {num_users}, Movies: {num_movies}, Rating Range: {min_rating} - {max_rating}')
      ```
   * Data Splitting:
     ```
     python
     from sklearn.model_selection import train_test_split
      
     x = df[['user', 'movie']].values
     y = df['ratings'].values
      
     x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
     ```
   Setalah semua persiapan selesai, dilakukan Model Development Collaborative Filtering dengan Neural Network TensorFlow
   
   Sistem menganalisis data rating dan menemukan bahwa genre yang paling tinggi ratingnya adalah Comedy dan Drama.
   Berdasarkan hasil ini, berikut adalah Top-10 film yang direkomendasikan menggunakan Collaborative Filtering:
   
   ![Top-10](images/Top10.png)
   
   Rekomendasi ini menyajikan film-film dengan genre yang paling banyak disukai oleh komunitas pengguna lain, tanpa melihat isi konten film itu sendiri.
   
6. **Perbandingan Pendekatan**
   - **Kelebihan Content-Based Filtering:**
     * Tidak bergantung pada data pengguna lain.
     * Cocok untuk pengguna baru dengan histori preferensi yang sudah diketahui.
   - **Kekurangan Content-Based Filtering:**
     * Kurang mampu menangkap variasi atau kejutan dalam rekomendasi.
     * Rekomendasi cenderung terlalu sempit (genre mirip terus menerus).
   - **Kelebihan Collaborative Filtering:**
     * Dapat memberikan rekomendasi yang lebih beragam karena melihat pola dari banyak pengguna.
     * Tidak butuh detail konten film secara eksplisit.
   - **Kekurangan Collaborative Filtering:**
     * Sulit bekerja jika data rating sedikit (cold start).
     * Butuh data pengguna dalam jumlah besar agar akurat. 

## Evaluation
1. **Hasil Evaluasi untuk Content-Based Filtering**
   Untuk mengevaluasi kinerja algoritma Content-Based Filtering, digunakan metrik evaluasi **Precision**.
Precision mengukur seberapa banyak rekomendasi yang benar-benar relevan dibandingkan dengan seluruh rekomendasi yang diberikan.
   Dalam eksperimen ini, sistem memberikan **Top-5 rekomendasi** berdasarkan film favorit pengguna, yaitu **Jumanji (1995)** dengan genre *Adventure, Children, Fantasy*. Berikut hasil rekomendasinya:
   Dari 5 film yang direkomendasikan, terdapat **3 film yang memiliki genre serupa**, sehingga dapat dihitung:
   
   ![Top-5](images/Top5.png)
   
   Dari 5 film yang direkomendasikan, terdapat 3 film yang memiliki genre yang sama atau serupa.
   > **Precision = (Jumlah item relevan yang direkomendasikan) / (Jumlah total item yang direkomendasikan)**
   > **Precision = 3 / 5 = 0.60 atau 60%**
   
   Dengan demikian, sistem memiliki akurasi rekomendasi sebesar 60%, yang menunjukkan bahwa mayoritas rekomendasi sesuai dengan preferensi genre pengguna sebelumnya.
2. **Hasil Evaluasi untuk Collaborative Filtering**
   Untuk **Collaborative Filtering**, digunakan metrik **Root Mean Squared Error (RMSE)**, yang mengukur rata-rata kuadrat selisih antara rating aktual dengan rating hasil prediksi. untuk mengevaluasi kinerja model dalam memprediksi rating.
   RMSE mengukur selisih rata-rata kuadrat antara rating aktual dan prediksi dari model. Metrik ini sangat umum digunakan dalam sistem rekomendasi karena mampu memberikan penalti lebih besar terhadap prediksi yang jauh dari nilai sebenarnya.
   **Berikut Formula RMSE:**
   
   ![Formula RMSE](images/RumusRMSE.png)

   
   **Di mana:**
   > * $A_t$ adalah nilai aktual
   > * $f_t$ adalah nilai prediksi
   > * $N$ adalah jumlah data

   - **Kelebihan RMSE:**
     * Sensitif terhadap kesalahan besar, sehingga dapat mendeteksi outlier prediksi yang jauh dari kenyataan.
   - **Kekurangan RMSE:**
     * Memberikan bobot besar pada error yang ekstrem, sehingga dapat terlalu keras pada kasus tertentu.


  Implementasi RMSE pada model dilakukan dengan menambahkan parameter `metrics=[tf.keras.metrics.RootMeanSquaredError()]` pada tahap  kompilasi model.

  ![Gambar Model Compile](images/ModelCompile.png)
  
  - **Hasil akhir setelah 100 epoch training:**
      * **Training RMSE**: sekitar **0.0307**
      * **Validation RMSE**: sekitar **0.2379**
    Nilai tersebut diambil dari log training terakhir:
    ```
    Epoch 100/100
    ...
    root_mean_squared_error: 0.0307 - val_root_mean_squared_error: 0.2379
    ```
  - **Visualisasi plot training dan validation RMSE**
    
    ![Visualisasi](images/DiagramRMSE.png)

    Berdasarkan grafik, terlihat bahwa model mencapai kestabilan setelah beberapa epoch. RMSE pada data training sangat rendah, sementara RMSE pada data validasi berada di kisaran 0.23. Hal ini dapat mengindikasikan adanya perbedaan performa yang masih bisa dioptimalkan antara training dan validasi

3. **Kesimpulan Evaluasi**
   * Sistem **Content-Based** memberikan Precision sebesar **60%**, yang menunjukkan bahwa mayoritas rekomendasi sesuai dengan preferensi pengguna.
    * Sistem **Collaborative Filtering** menghasilkan RMSE **0.0307 (training)** dan **0.2379 (validation)**, yang menunjukkan model cukup baik dalam mempelajari data pelatihan, namun ada ruang untuk perbaikan dalam generalisasi.
