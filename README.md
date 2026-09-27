<!-- README.md -->
**Pembuat:** Laurensius Brian Prayoga (24060124130077), Haikal Imam Ridha (24060124130097)

# pearson-cuda
Implementasi paralel perhitungan korelasi Pearson antara dua kolom
numerik dataset CSV menggunakan CUDA, dibandingkan dengan baseline CPU
sekuensial.

Proyek tugas 2, mata kuliah **Komputasi Tersebar Paralel**.

## 1. Deskripsi Proyek
Aplikasi CLI yang menghitung korelasi Pearson (`r`) antara dua kolom
numerik dari file CSV, dengan dua implementasi:

- **CPU** — akumulasi sekuensial menggunakan `double`.
- **CUDA** — akumulasi paralel menggunakan grid-stride loop, shared
  memory, dan parallel reduction di level block, dengan final reduction
  di CPU.

Program melaporkan nilai `r`, waktu eksekusi CPU, waktu eksekusi GPU,
speedup, dan validasi kedekatan hasil CPU vs CUDA.

## 2. Tujuan

- Membandingkan performa implementasi sekuensial (CPU) vs paralel (CUDA)
  untuk komputasi statistik sederhana namun representatif (akumulasi
  reduction).
- Mendemonstrasikan penerapan konsep CUDA inti: grid-stride loop, shared
  memory, dan parallel reduction, dalam kasus nyata.

**Di luar cakupan** (sengaja tidak diimplementasikan): matriks korelasi
N×N, multi-GPU, CUDA streams, GUI/web app, machine learning, library
statistik eksternal, algoritma korelasi selain Pearson.

## 3. Requirement

- Compiler C++17.
- CUDA Toolkit 11.x atau 12.x (`nvcc`) beserta driver NVIDIA yang sesuai.
- GPU NVIDIA dengan compute capability yang didukung toolkit Anda.
- CMake ≥ 3.18 (jika memakai jalur build CMake) **atau** GNU Make (jika
  memakai jalur build Makefile) — tidak perlu keduanya.

## 4. Struktur Folder

```
pearson-cuda/
├── CMakeLists.txt
├── Makefile
├── README.md
├── .gitignore
├── data/
│   ├── sample_small.csv       (100 baris, korelasi target 0.8)
│   ├── sample_medium.csv      (100.000 baris, korelasi target 0.8)
│   ├── generate_data.py
│   └── benchmark_datasets/    (dibuat otomatis oleh run_benchmark.sh)
├── include/
│   ├── csv_loader.h
│   ├── pearson_cpu.h
│   ├── pearson_cuda.h
│   └── utils.h
├── src/
│   ├── main.cpp
│   ├── csv_loader.cpp
│   ├── pearson_cpu.cpp
│   ├── pearson_cuda.cu
│   └── utils.cpp
├── tests/
│   └── test_pearson.cpp
├── scripts/
│   └── run_benchmark.sh
└── docs/
    ├── laporan.md
    └── benchmark_results.md
```

## 5. Cara Generate Dataset

```bash
# Dataset kustom
python3 data/generate_data.py --rows 1000000 --corr 0.8 --out data/big.csv

# Parameter:
#   --rows  : jumlah baris (wajib, >= 2)
#   --corr  : korelasi target, -1.0 s/d 1.0 (wajib)
#   --out   : path file output (wajib)
#   --seed  : seed RNG untuk reproducibility (opsional)
```

Setelah generate, script menampilkan **korelasi empiris aktual** dari
data yang dihasilkan (bisa berbeda sedikit dari target, terutama untuk
`--rows` kecil).

`data/sample_small.csv` dan `data/sample_medium.csv` sudah disediakan
sebagai dataset uji cepat siap pakai.

## 6. Cara Build dengan CMake

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<compute_capability_gpu_anda>
cmake --build build
# Binary ada di: build/pearson
```

Cek compute capability GPU Anda:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## 7. Cara Build dengan Makefile

```bash
make CUDA_ARCH=sm_XX   # ganti XX sesuai compute capability GPU Anda, mis. sm_86
# Binary ada di: ./pearson
```

Contoh pemetaan compute capability ke flag `sm_XX`:

| Compute Capability | Flag      | Contoh GPU (indikatif)     |
|---------------------|-----------|-----------------------------|
| 7.0                  | `sm_70`   | Tesla V100                  |
| 7.5                  | `sm_75`   | RTX 20-series, GTX 16-series|
| 8.0                  | `sm_80`   | A100                        |
| 8.6                  | `sm_86`   | RTX 30-series                |

> Tabel di atas bersifat indikatif umum, bukan hasil verifikasi pada
> perangkat spesifik Anda — selalu cek dengan `nvidia-smi` di mesin Anda.

## 8. Cara Menjalankan Mode CPU

```bash
./pearson --input data/sample_small.csv --col-x 0 --col-y 1 --mode cpu
```

## 9. Cara Menjalankan Mode CUDA

```bash
./pearson --input data/sample_small.csv --col-x 0 --col-y 1 --mode cuda
```

## 10. Cara Menjalankan Mode Both

```bash
./pearson --input data/sample_small.csv --col-x 0 --col-y 1 --mode both
```

Mode ini menjalankan CPU dan CUDA, membandingkan hasil, dan menghitung
speedup (`CPU time / GPU time`).

## 11. Cara Menjalankan Test

```bash
# Dengan CMake
cmake --build build --target pearson_tests
./build/pearson_tests

# Dengan Makefile
make test
```

Test yang melibatkan CUDA (`test_cpu_vs_cuda`) akan **di-skip secara
informatif** (bukan gagal atau dipalsukan lulus) jika tidak ada CUDA
device terdeteksi.

## 12. Cara Melakukan Benchmark

```bash
./scripts/run_benchmark.sh ./pearson     # jika build dengan Makefile
./scripts/run_benchmark.sh ./build/pearson  # jika build dengan CMake
```

Hasil otomatis ditulis ke `docs/benchmark_results.md`, berdasarkan
eksekusi aktual pada mesin Anda — **bukan angka yang di-hardcode**.

## 13. Cara Membaca Output

```
=== Pearson Correlation Result ===
Dataset      : data/sample_small.csv
Rows         : 100
Column X     : 0
Column Y     : 1
----------------
Pearson r    : 0.807639
CPU time     : 0.123 ms
GPU time     : 1.456 ms
Speedup      : 0.084x
Error        : 3.201e-06 (dalam toleransi 1.000e-04)
======================
```

- **Pearson r**: nilai korelasi, rentang [-1, 1]. Jika data tidak layak
  dikorelasikan (n<2, salah satu kolom konstan), akan tertulis
  `TIDAK TERDEFINISI` beserta alasannya.
- **Speedup < 1**: **wajar** untuk dataset kecil — overhead transfer
  data ke GPU dan launch kernel bisa lebih besar dari waktu komputasi
  itu sendiri pada data kecil. Speedup GPU umumnya baru terlihat pada
  dataset besar.
- **Error**: `|r_cpu - r_cuda|`, dibandingkan terhadap toleransi validasi
  (default `1e-4`). Nilai kecil namun bukan nol adalah normal karena
  perbedaan urutan floating-point antara CPU sekuensial dan GPU paralel.

## 14. Troubleshooting CUDA

| Gejala | Kemungkinan Penyebab | Solusi |
|---|---|---|
| `nvcc: command not found` | CUDA Toolkit belum ter-install atau tidak ada di `PATH` | Install CUDA Toolkit, tambahkan `/usr/local/cuda/bin` ke `PATH` |
| Error saat runtime: "CUDA device tidak tersedia" | Driver NVIDIA tidak terpasang, atau tidak ada GPU NVIDIA | Cek `nvidia-smi`; install/update driver NVIDIA |
| Error kompilasi terkait arsitektur (`nvcc fatal: Unsupported gpu architecture`) | Flag `-arch`/`CMAKE_CUDA_ARCHITECTURES` tidak sesuai compute capability GPU Anda atau tidak didukung versi CUDA toolkit Anda | Cek compute capability dengan `nvidia-smi --query-gpu=compute_cap --format=csv`, sesuaikan flag build |
| `CUDA_ERROR_OUT_OF_MEMORY` pada dataset besar | VRAM GPU tidak cukup untuk `cudaMalloc` pada `n` besar | Kurangi ukuran dataset, atau gunakan GPU dengan VRAM lebih besar |
| Hasil `r_cpu` dan `r_cuda` berbeda melebihi toleransi | Kemungkinan bug pada kernel (indexing, sinkronisasi) — bukan sekadar isu presisi | Periksa `pearson_cuda.cu`, khususnya batas grid-stride loop dan reduction; laporkan sebagai bug jika deviasi besar |
| Kompilasi `g++`/`nvcc` gagal karena versi compiler tidak kompatibel | Versi CUDA toolkit membatasi versi `g++` host compiler yang didukung | Cek dokumentasi kompatibilitas NVIDIA untuk versi CUDA Anda, gunakan `g++` versi yang didukung |