# Analisis Serangan Adversarial Attack terhadap Model Machine Learning pada Klasifikasi Gambar

Repository ini dibuat untuk tugas **II3230 Keamanan Informasi**. Topik yang dianalisis adalah serangan **adversarial attack** terhadap model klasifikasi gambar, dengan eksperimen sederhana menggunakan dataset MNIST, model CNN, dan metode **Fast Gradient Sign Method (FGSM)**.

## Tujuan

1. Melatih model CNN sederhana untuk klasifikasi gambar angka MNIST.
2. Menguji akurasi model pada data normal.
3. Membuat adversarial example menggunakan FGSM.
4. Membandingkan akurasi model sebelum dan sesudah serangan.
5. Menyimpan tabel hasil dan contoh gambar untuk bahan pembahasan makalah.

## Struktur Folder

```text
.
├── docs/
│   └── Proposal_18223015.docx
├── report/
│   ├── makalah_draft.md
│   └── template_pembahasan_hasil.md
├── src/
│   ├── data.py
│   ├── fgsm.py
│   ├── main.py
│   ├── model.py
│   ├── plotting.py
│   └── utils.py
├── results/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── requirements.txt
└── .gitignore
```

## Cara Menjalankan

### 1. Buat virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

### 3. Jalankan eksperimen cepat

```bash
python src/main.py --epochs 1 --max-train-samples 12000 --max-test-samples 2000
```

Program akan:

- mengunduh dataset MNIST secara otomatis,
- melatih model CNN sederhana,
- menguji model pada data normal,
- menjalankan FGSM pada beberapa nilai epsilon,
- menyimpan hasil ke folder `results/`,
- menyimpan model ke folder `models/`.

### 4. Output yang dihasilkan

Setelah program selesai, cek:

```text
results/metrics.csv
results/metrics.json
results/fgsm_examples.png
models/mnist_cnn.pt
```

File `metrics.csv` dipakai untuk tabel hasil makalah. File `fgsm_examples.png` dipakai sebagai gambar contoh perbandingan input normal dan input yang sudah diberi perturbasi.

## Cara Push ke GitHub

Kalau repo ini sudah ada di laptop:

```bash
git init
git add .
git commit -m "Initial adversarial attack experiment"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA_REPO.git
git push -u origin main
```

Ganti `USERNAME` dan `NAMA_REPO` sesuai akun GitHub dan nama repo yang dibuat.

## Catatan Etika

Eksperimen ini digunakan untuk pembelajaran keamanan informasi dan evaluasi robustness model. Repository ini tidak ditujukan untuk menyerang sistem nyata atau menyalahgunakan model milik pihak lain.
