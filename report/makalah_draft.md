# Analisis Serangan Adversarial Attack terhadap Model Machine Learning pada Klasifikasi Gambar

**Nama:** Muhammad Adam Mirza  
**NIM:** 18223015  
**Mata Kuliah:** II3230 Keamanan Informasi  
**Program Studi:** Sistem dan Teknologi Informasi, Institut Teknologi Bandung

## Abstrak

Perkembangan machine learning telah membuat model klasifikasi gambar banyak digunakan dalam berbagai sistem, seperti pengenalan objek, deteksi wajah, sistem keamanan, dan analisis citra medis. Walaupun model dapat mencapai akurasi tinggi pada data normal, sejumlah penelitian menunjukkan bahwa model tersebut masih dapat dimanipulasi melalui adversarial attack. Serangan ini dilakukan dengan menambahkan perturbasi kecil pada gambar input sehingga perubahan visualnya sulit dikenali oleh manusia, tetapi dapat menyebabkan model menghasilkan klasifikasi yang salah. Makalah ini membahas analisis serangan adversarial attack terhadap model machine learning pada tugas klasifikasi gambar. Fokus pembahasan diarahkan pada konsep adversarial example, cara perturbasi memengaruhi proses prediksi model, serta dampaknya terhadap akurasi klasifikasi. Metode serangan yang dikaji adalah Fast Gradient Sign Method (FGSM). Analisis dilakukan dengan membandingkan performa model pada gambar normal dan gambar yang telah diberi perturbasi adversarial.

**Kata kunci:** adversarial attack, klasifikasi gambar, machine learning, CNN, keamanan informasi

## 1. Pendahuluan

### 1.1 Latar Belakang

Isi bagian ini dengan pengembangan dari proposal. Tekankan bahwa model machine learning tidak hanya perlu akurat, tetapi juga perlu aman terhadap input yang sengaja dimanipulasi.

### 1.2 Rumusan Masalah

1. Bagaimana konsep adversarial attack pada model klasifikasi gambar?
2. Bagaimana metode FGSM dapat menghasilkan adversarial example?
3. Bagaimana pengaruh nilai epsilon terhadap akurasi model klasifikasi gambar?
4. Apa strategi mitigasi yang dapat diterapkan untuk mengurangi risiko adversarial attack?

### 1.3 Tujuan

1. Menjelaskan konsep adversarial attack dalam konteks keamanan informasi.
2. Menganalisis cara kerja FGSM pada model klasifikasi gambar.
3. Membandingkan akurasi model pada data normal dan data hasil serangan.
4. Mengidentifikasi strategi mitigasi terhadap adversarial attack.

### 1.4 Batasan Masalah

Makalah ini membatasi pembahasan pada dataset MNIST, model CNN sederhana, dan metode serangan FGSM. Serangan yang dibahas bersifat white-box, yaitu penyerang diasumsikan memiliki akses terhadap informasi model yang diperlukan untuk menghitung gradien.

## 2. Tinjauan Pustaka

### 2.1 Klasifikasi Gambar

Jelaskan konsep klasifikasi gambar dan hubungan input gambar dengan label kelas.

### 2.2 Convolutional Neural Network

Jelaskan CNN sebagai model yang umum digunakan untuk klasifikasi gambar.

### 2.3 Adversarial Example

Jelaskan bahwa adversarial example adalah input yang telah dimodifikasi secara kecil tetapi dapat menyebabkan model salah prediksi.

### 2.4 Fast Gradient Sign Method

FGSM membuat perturbasi berdasarkan arah gradien dari loss terhadap input. Secara umum, input adversarial dapat ditulis sebagai:

```text
x_adv = x + epsilon * sign(gradient_x(loss))
```

## 3. Metodologi

### 3.1 Dataset

Dataset yang digunakan adalah MNIST, yaitu dataset gambar angka tulisan tangan dengan 10 kelas, mulai dari angka 0 sampai 9.

### 3.2 Model

Model yang digunakan adalah CNN sederhana dengan dua convolution layer, max pooling, dropout, dan fully connected layer.

### 3.3 Skenario Eksperimen

1. Model dilatih menggunakan data train MNIST.
2. Akurasi model dihitung pada data test normal.
3. Data test dimodifikasi menggunakan FGSM dengan beberapa nilai epsilon.
4. Akurasi model pada data adversarial dihitung.
5. Hasil dibandingkan untuk melihat pengaruh perturbasi terhadap performa model.

### 3.4 Metrik Evaluasi

Metrik yang digunakan adalah akurasi setelah serangan dan attack success rate.

## 4. Hasil dan Pembahasan

Bagian ini diisi setelah program dijalankan.

Gunakan file `results/metrics.csv` untuk membuat tabel seperti berikut:

| Epsilon | Akurasi Setelah Serangan | Attack Success Rate |
|---:|---:|---:|
| 0.00 | ... | ... |
| 0.05 | ... | ... |
| 0.10 | ... | ... |
| 0.15 | ... | ... |
| 0.20 | ... | ... |
| 0.30 | ... | ... |

Pembahasan yang perlu ditulis:

1. Apakah akurasi menurun ketika epsilon meningkat?
2. Pada epsilon berapa model mulai mengalami penurunan signifikan?
3. Apakah perubahan gambar masih terlihat kecil bagi manusia?
4. Mengapa model dapat salah walaupun gambar tampak mirip?
5. Apa dampaknya terhadap keamanan sistem berbasis machine learning?

## 5. Strategi Mitigasi

Beberapa mitigasi yang dapat dibahas:

1. Adversarial training.
2. Pembatasan nilai perturbasi dan validasi input.
3. Evaluasi robustness selain evaluasi akurasi biasa.
4. Monitoring terhadap input yang mencurigakan.
5. Penggunaan model yang lebih tahan terhadap adversarial example.

## 6. Penutup

Tuliskan kesimpulan berdasarkan hasil eksperimen. Intinya, model yang akurat pada data normal belum tentu aman terhadap input adversarial.

## Referensi

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. International Conference on Learning Representations (ICLR).
2. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). *Intriguing Properties of Neural Networks*. International Conference on Learning Representations (ICLR).
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. International Conference on Learning Representations (ICLR).
