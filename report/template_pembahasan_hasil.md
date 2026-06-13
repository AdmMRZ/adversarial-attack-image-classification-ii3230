# Template Pembahasan Hasil Eksperimen

Setelah file `results/metrics.csv` dihasilkan, bagian ini bisa dipakai untuk menulis pembahasan.

## Narasi Pembuka

Berdasarkan hasil pengujian, model CNN menunjukkan akurasi yang tinggi pada data normal. Namun, setelah gambar input diberi perturbasi menggunakan FGSM, akurasi model mengalami penurunan. Penurunan ini menunjukkan bahwa model memiliki kerentanan terhadap adversarial example, yaitu input yang dimodifikasi secara kecil tetapi dirancang untuk menaikkan loss dan mengubah hasil klasifikasi.

## Kalimat Analisis Epsilon

Pada nilai epsilon yang kecil, penurunan akurasi masih relatif terbatas karena perubahan piksel belum terlalu besar. Namun, ketika epsilon meningkat, perturbasi yang diberikan menjadi lebih kuat sehingga model semakin sering menghasilkan prediksi yang salah. Pola ini menunjukkan bahwa tingkat keberhasilan serangan FGSM dipengaruhi oleh besar kecilnya perturbasi yang ditambahkan pada input.

## Kalimat Hubungan dengan Keamanan Informasi

Dalam konteks keamanan informasi, adversarial attack dapat dipahami sebagai ancaman terhadap integritas sistem. Input yang telah dimodifikasi dapat menyebabkan sistem menghasilkan keputusan yang tidak sesuai, walaupun perubahan visual pada input tidak selalu mudah dikenali oleh pengguna. Hal ini penting diperhatikan pada sistem yang menggunakan machine learning untuk proses pengambilan keputusan, terutama pada sistem keamanan, pengenalan wajah, kendaraan otonom, dan analisis citra medis.

## Kalimat Mitigasi

Mitigasi yang dapat diterapkan antara lain adversarial training, evaluasi robustness secara berkala, validasi input, dan penggunaan metode deteksi input anomali. Dengan adanya pengujian terhadap adversarial example, keamanan model tidak hanya dinilai dari akurasi pada data normal, tetapi juga dari kemampuannya mempertahankan prediksi ketika menerima input yang sengaja dimanipulasi.
