# Adversarial Attack Image Classification

Repo ini dibuat untuk tugas **II3230 Keamanan Informasi**. Isi repo digunakan untuk menganalisis serangan **adversarial attack** terhadap model klasifikasi gambar, khususnya dengan metode **Fast Gradient Sign Method (FGSM)** pada dataset MNIST.

Eksperimen dilakukan dengan melatih model CNN sederhana untuk mengenali angka tulisan tangan, lalu menguji ulang model tersebut menggunakan gambar yang sudah diberi perturbasi adversarial. Hasilnya dipakai untuk melihat penurunan akurasi model ketika nilai epsilon pada FGSM dinaikkan.

## Isi Repo

```text
docs/       makalah final
src/        kode eksperimen CNN dan FGSM
data/       dataset MNIST
models/     model hasil pelatihan
results/    metrik, grafik, dan contoh gambar adversarial
```

## Cara Menjalankan

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan eksperimen:

```bash
python src/main.py --epochs 1 --max-train-samples 12000 --max-test-samples 2000
```

Output utama akan tersimpan di:

```text
results/metrics.csv
results/metrics.json
results/fgsm_examples.png
results/fgsm_metrics_plot.png
models/mnist_cnn.pt
```

## Laporan

Makalah final tersedia di:

```text
docs/Makalah_18223015.pdf
```
