<!-- laporan.md -->
## Bab 1 — Pendahuluan

### 1.1 Latar Belakang

Komputasi statistik dasar seperti korelasi Pearson sering menjadi
langkah awal dalam analisis data, namun pada dataset berukuran besar
(jutaan baris), perhitungan sekuensial di CPU dapat menjadi bottleneck.
GPU, dengan arsitektur many-core-nya, menawarkan potensi percepatan
signifikan untuk operasi yang bersifat *data-parallel* seperti akumulasi
sum yang menjadi dasar rumus Pearson.

### 1.2 Rumusan Masalah

1. Bagaimana perhitungan korelasi Pearson dapat didekomposisi menjadi
   operasi yang dapat dijalankan paralel di GPU?
2. Seberapa besar percepatan (speedup) yang dapat dicapai implementasi
   CUDA dibandingkan implementasi CPU sekuensial, pada berbagai ukuran
   dataset?
3. Bagaimana konsistensi numerik antara hasil CPU dan CUDA dapat
   divalidasi, mengingat perbedaan urutan operasi floating-point antara
   komputasi sekuensial dan paralel?

### 1.3 Tujuan

Membangun dan mengevaluasi implementasi CUDA untuk perhitungan korelasi
Pearson dua kolom dataset CSV, dibandingkan dengan baseline CPU, dari
segi korektnes (kedekatan numerik) dan performa (speedup).

### 1.4 Batasan Masalah

Proyek ini dibatasi pada:
- Korelasi Pearson **dua kolom** saja (bukan matriks korelasi N×N).
- Satu GPU (bukan multi-GPU).
- Tidak menggunakan CUDA streams, mixed precision, atau library
  statistik eksternal.
- Tidak mencakup antarmuka grafis (GUI) atau aplikasi web.

### 1.5 Manfaat

- Memberikan pemahaman praktis penerapan konsep CUDA (grid-stride loop,
  shared memory, parallel reduction) pada kasus nyata di luar contoh
  buku teks.
- Menjadi referensi baseline untuk perbandingan performa CPU vs GPU pada
  operasi reduction sederhana.

---

## Bab 2 — Landasan Teori

### 2.1 Korelasi Pearson

**Definisi**: Korelasi Pearson (`r`) mengukur kekuatan dan arah hubungan
linear antara dua variabel numerik.

**Rumus**:

```
r = (n·Σxy − Σx·Σy) / √[(n·Σx² − (Σx)²)(n·Σy² − (Σy)²)]
```

**Interpretasi nilai r**:
- `r = 1`: korelasi linear positif sempurna.
- `r = -1`: korelasi linear negatif sempurna.
- `r = 0`: tidak ada korelasi linear (bisa saja ada hubungan non-linear).
- `|r|` mendekati 1: hubungan linear kuat; mendekati 0: hubungan linear lemah.

### 2.2 Komputasi Paralel dan CUDA

CUDA adalah platform komputasi paralel NVIDIA yang mengorganisasikan
eksekusi dalam hierarki:
- **Grid**: kumpulan seluruh block yang di-launch untuk satu kernel.
- **Block**: kumpulan thread yang dijadwalkan bersama pada satu
  Streaming Multiprocessor (SM), dan dapat berbagi *shared memory*.
- **Thread**: unit eksekusi terkecil, masing-masing menjalankan kode
  kernel yang sama (SIMT — Single Instruction, Multiple Thread) tetapi
  bisa memproses data berbeda.

Pada proyek ini, setiap thread awalnya memproses elemen dataset yang
berbeda (kolom X dan Y), sehingga hierarki ini memetakan langsung ke
pembagian data: grid mencakup seluruh dataset, block adalah kelompok
thread yang bekerja sama untuk mereduksi hasil parsial mereka.

### 2.3 Parallel Reduction

Parallel reduction adalah pola untuk menggabungkan banyak nilai (mis.
hasil dari ribuan thread) menjadi satu nilai (mis. total sum), secara
bertahap dan paralel — bukan satu per satu secara sekuensial. Pola umum
yang dipakai proyek ini adalah **tree reduction**: pada tiap langkah,
separuh elemen aktif dijumlahkan dengan pasangannya, sehingga jumlah
elemen aktif berkurang separuh setiap langkah, hingga tersisa satu nilai.

### 2.4 Shared Memory

Shared memory adalah memori on-chip yang dapat diakses bersama oleh
seluruh thread dalam satu block, dengan latensi jauh lebih rendah
dibanding memori global GPU. Pada proyek ini, shared memory dipakai
sebagai "ruang kerja" untuk melakukan tree reduction dalam block, karena
mengakses shared memory berulang kali (log2(blockDim.x) kali) jauh lebih
efisien dibanding mengakses memori global berulang kali.

### 2.5 Grid-Stride Loop

Pola di mana setiap thread memproses lebih dari satu elemen data dengan
"melompat" sejauh `total_threads` (jumlah seluruh thread aktif) setiap
iterasi, alih-alih meluncurkan sebanyak elemen data thread (yang bisa
saja melebihi batas hardware atau tidak efisien untuk dataset sangat
besar). Pola ini membuat kernel bekerja benar untuk ukuran dataset
berapa pun, independen dari konfigurasi jumlah block/thread yang dipilih.

---

## Bab 3 — Metodologi

### 3.1 Arsitektur Sistem

Program terdiri dari tiga lapisan: (1) CSV Loader untuk membaca dan
memvalidasi data, (2) lapisan komputasi (CPU dan CUDA) yang menghasilkan
akumulator Pearson yang identik strukturnya, dan (3) lapisan CLI yang
mengorkestrasi alur dan mencetak hasil. Detail lengkap ada di README.md
bagian struktur folder.

### 3.2 Alur Kerja Program

```
CSV → CSV Loader → vector<float> x, y
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
       CPU (sekuensial)      CUDA (paralel)
       akumulasi double      Kernel 1: partial sum per block
            │                (grid-stride + shared memory + reduction)
            │                       │
            │                Final reduction (CPU)
            │                       │
            ▼                       ▼
        r_cpu, t_cpu           r_cuda, t_cuda
                        │
                        ▼
              Perbandingan & validasi toleransi
                        │
                        ▼
        Output: r, CPU time, GPU time, Speedup, Error
```

### 3.3 Desain CPU

Implementasi CPU (`pearson_cpu.cpp`) melakukan iterasi sekuensial atas
seluruh elemen, mengakumulasi enam nilai (`n, Σx, Σy, Σx², Σy², Σxy`)
menggunakan tipe `double` untuk mengurangi akumulasi galat pembulatan.
Waktu diukur dengan `std::chrono::high_resolution_clock`, hanya
mencakup bagian akumulasi (bukan validasi/finalisasi rumus).

### 3.4 Desain Kernel CUDA

Kernel 1 (`pearson_partial_sums_kernel`) memakai grid-stride loop agar
setiap thread dapat memproses banyak elemen tanpa bergantung pada jumlah
total thread yang di-launch. Konfigurasi: 256 thread per block,
maksimum 2048 block (lihat pembahasan lengkap di respons Tahap 4
pengerjaan proyek dan komentar kode `pearson_cuda.cu`).

### 3.5 Partial Sum

Setiap thread mengakumulasi lima sum lokal (`Σx, Σy, Σx², Σy², Σxy`) di
register selama grid-stride loop, lalu menuliskannya ke shared memory
untuk direduksi bersama thread lain dalam block yang sama.

### 3.6 Final Reduction

Dipilih dilakukan di **CPU**, bukan kernel GPU kedua, karena jumlah
partial sum (sebanyak jumlah block, dibatasi maksimum 2048) sudah cukup
kecil untuk direduksi sekuensial tanpa overhead berarti. Justifikasi
teknis lengkap ada di komentar `pearson_cuda.h` dan respons Tahap 4.

### 3.7 Strategi Shared Memory

Setiap block mengalokasikan shared memory dinamis sebesar
`5 × blockDim.x × sizeof(double)` byte (lima array untuk lima sum,
masing-masing sepanjang jumlah thread per block), dialokasikan melalui
parameter ketiga saat kernel launch (`<<<grid, block, shared_bytes>>>`).

### 3.8 Dataset Uji

Dataset sintetis dihasilkan `data/generate_data.py` dengan tiga skenario
korelasi target (independen ~0, positif ~0.8, negatif ~−0.8), memakai
model bivariate normal. Dataset siap pakai: `sample_small.csv` (100
baris) dan `sample_medium.csv` (100.000 baris), keduanya dengan target
korelasi 0.8.

### 3.9 Metode Pengukuran Waktu

- **CPU**: `std::chrono::high_resolution_clock`, mengukur durasi fungsi
  akumulasi murni.
- **GPU**: CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`) untuk
  setiap komponen (transfer host→device, eksekusi kernel, transfer
  device→host) secara terpisah, ditambah `std::chrono` untuk final
  reduction di CPU. "GPU time" yang dilaporkan adalah **total end-to-end**
  dari keempat komponen ini — bukan hanya waktu kernel murni. Lihat
  `pearson_cuda.h` (`CudaTimingBreakdown`) untuk rincian per-komponen.

---

## Bab 4 — Hasil dan Pembahasan

### 4.1 Lingkungan Pengujian

> **PLACEHOLDER — isi dengan spesifikasi mesin aktual Anda.**

- CPU: `<CPU_NAME>`
- GPU: `<GPU_NAME>`
- CUDA: `<CUDA_VERSION>`
- OS: `<OS>`
- Compiler: `<COMPILER>`

### 4.2 Validasi CPU vs CUDA

**Status: BELUM DIVALIDASI DENGAN GPU NYATA.**

Selama pengembangan, komponen CPU (`pearson_cpu.cpp`) telah diverifikasi
secara aktual melalui kompilasi dan eksekusi langsung (lihat Lampiran,
"Log Verifikasi Selama Pengembangan"), dengan hasil yang sesuai
perhitungan manual (contoh: `x={1,2,3}, y={2,1,4}` menghasilkan
`r = 0.654654`, sesuai perhitungan tangan). Namun, **komponen CUDA belum
pernah dijalankan di GPU nyata**, sehingga baris perbandingan
`r_cpu` vs `r_cuda` di bawah ini masih placeholder:

| Dataset | n | r (CPU) | r (CUDA) | \|Δr\| | Status |
|---|---|---|---|---|---|
| `<isi>` | `<isi>` | `<isi>` | `<isi>` | `<isi>` | `<isi>` |

Isi tabel ini dengan menjalankan:
```bash
./pearson --input data/sample_small.csv --col-x 0 --col-y 1 --mode both
./pearson --input data/sample_medium.csv --col-x 0 --col-y 1 --mode both
```
pada mesin dengan GPU NVIDIA, lalu salin nilai `Pearson r`, dan hitung
selisihnya secara manual jika program tidak dijalankan mode `both`
(mode `both` sudah menghitung dan menampilkan `Error` secara otomatis).

### 4.3 Benchmark

**Status: BELUM ADA DATA — lihat `docs/benchmark_results.md`.**

| Rows      | CPU (ms) | GPU (ms) | Speedup |
| --------- | -------- | -------- | ------- |
| 100       |          |          |         |
| 1.000     |          |          |         |
| 10.000    |          |          |         |
| 100.000   |          |          |         |
| 1.000.000 |          |          |         |

Jalankan `./scripts/run_benchmark.sh ./pearson` untuk mengisi tabel ini
dengan data aktual (lihat `docs/benchmark_results.md` untuk hasil
lengkap beserta metadata eksekusi).

### 4.4 Analisis Bottleneck

**Analisis teoritis** (belum dikonfirmasi hasil aktual):

Untuk dataset kecil, waktu transfer host-device dan overhead peluncuran
kernel (fixed cost) kemungkinan mendominasi total waktu GPU, membuat
`speedup < 1` (CPU lebih cepat). Untuk dataset besar, waktu komputasi
(yang berskala dengan `n`) seharusnya mendominasi, dan paralelisme GPU
seharusnya memberikan speedup > 1. **Titik potong (crossover point)**
antara kedua rezim ini **hanya dapat ditentukan dari data benchmark
aktual** (Bab 4.3), bukan diasumsikan.

Final reduction di CPU (bagian dari `GPU time` total, lihat 3.9) adalah
komponen serial yang **tidak** ikut dipercepat GPU — namun karena jumlah
block dibatasi kecil (maks. 2048), kontribusinya terhadap total waktu
seharusnya kecil pada dataset besar. Ini perlu dikonfirmasi dengan
melihat rincian `kernel_ms` vs `final_reduction_cpu_ms` dari
`CudaTimingBreakdown` pada hasil aktual.

### 4.5 Analisis Efisiensi Paralel

**Status: BELUM DAPAT DIANALISIS — memerlukan data benchmark aktual.**

Setelah tabel 4.3 terisi, efisiensi paralel dapat didekati dengan
membandingkan speedup aktual terhadap jumlah core CUDA GPU yang
digunakan (idealnya speedup linear terhadap jumlah core, meski dalam
praktik jarang tercapai karena overhead memori dan sinkronisasi).
**Jangan menuliskan kesimpulan kuantitatif di sini sebelum data tersedia.**

---

## Bab 5 — Kesimpulan dan Saran

**Status: kesimpulan sementara, berdasarkan apa yang benar-benar sudah
diverifikasi selama pengembangan** (bukan hasil eksperimen performa
penuh, yang belum tersedia):

- Modul CPU (`csv_loader`, `pearson_cpu`, `utils`, `main`) telah
  **dikompilasi dan diuji secara aktual**, mencakup kasus manual,
  korelasi positif/negatif/independen, n=1, n=2, X/Y konstan, dan error
  handling CSV — seluruhnya lulus sesuai ekspektasi.
- Modul CUDA (`pearson_cuda.cu`) telah ditulis mengikuti prinsip desain
  CUDA yang benar (grid-stride loop, shared memory, parallel reduction,
  error checking lengkap), namun **belum diverifikasi dengan kompilasi
  atau eksekusi nyata** karena keterbatasan lingkungan pengembangan.
- **Kesimpulan mengenai speedup aktual CUDA vs CPU TIDAK DAPAT
  dinyatakan** pada tahap ini — ini memerlukan eksekusi benchmark nyata
  yang menjadi tanggung jawab pengguna proyek untuk melengkapi Bab 4.

**Saran untuk melengkapi proyek**:
1. Jalankan build (CMake/Makefile) di mesin dengan CUDA toolkit + GPU,
   verifikasi tidak ada error kompilasi.
2. Jalankan unit test (`pearson_tests`), pastikan `test_cpu_vs_cuda`
   tidak lagi di-skip dan lulus dalam toleransi.
3. Jalankan `scripts/run_benchmark.sh`, lengkapi Bab 4 laporan ini
   dengan hasil aktual.
4. (Opsional, pengembangan masa depan — TIDAK diimplementasikan di
   proyek ini) pertimbangkan final reduction kernel kedua di GPU untuk
   dataset yang sangat besar, atau dukungan multi-GPU, jika dibutuhkan
   skala yang lebih besar dari cakupan tugas ini.

---

## Lampiran

### Source Code

Seluruh source code ada di `src/` dan `include/` — lihat struktur folder
lengkap di `README.md`.

### Cara Build dan Menjalankan

Lihat `README.md` bagian 6–13.

### Tabel Benchmark

Lihat `docs/benchmark_results.md`.

### Contoh Output (Berlabel CONTOH — Bukan Hasil Eksperimen)

**CONTOH** (dibuat manual untuk ilustrasi format, BUKAN dari eksekusi
CUDA nyata):

```
=== Pearson Correlation Result ===
Dataset      : data/sample_small.csv
Rows         : 100
Column X     : 0
Column Y     : 1
----------------
Pearson r    : 0.8xxxxx
CPU time     : X.XXX ms
GPU time     : X.XXX ms
Speedup      : X.XXXx
Error        : X.XXXe-XX (dalam toleransi 1.000e-04)
======================
```

### Log Verifikasi Selama Pengembangan (Hasil Aktual, CPU-only)

Cuplikan hasil eksekusi **nyata** unit test CPU-only (memakai stub
pengganti CUDA, dijalankan selama pengembangan proyek ini):

```
=== Menjalankan Unit Test Pearson CPU/CUDA ===

[PASS] manual_small_dataset: is_defined
[PASS] manual_small_dataset: nilai r
[PASS] positive_correlation: is_defined
[PASS] positive_correlation: r mendekati 1.0
[PASS] negative_correlation: is_defined
[PASS] negative_correlation: r mendekati -1.0
[PASS] independent_data: is_defined
[PASS] independent_data: r dalam rentang [-1,1]
[SKIP] cpu_vs_cuda -- Tidak ada CUDA device terdeteksi pada sistem ini
[PASS] n_equals_1: tidak terdefinisi
[PASS] n_equals_2: terdefinisi
[PASS] n_equals_2: r = 1.0 (garis lurus naik)
[PASS] x_constant: tidak terdefinisi
[PASS] y_constant: tidak terdefinisi
[PASS] empty_csv_dataset: CsvLoadError dilempar

=== Ringkasan ===
Passed  : 14
Failed  : 0
Skipped : 1
```

Ini adalah hasil eksekusi sungguhan (bukan contoh ilustratif) yang
dijalankan tanpa GPU, membuktikan korektnes logika CPU dan penanganan
error, tetapi **tidak mencakup verifikasi CUDA**.
