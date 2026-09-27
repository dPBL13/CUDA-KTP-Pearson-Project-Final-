<!-- benchmark_results.md -->
# Hasil Benchmark

> **STATUS: TEMPLATE KOSONG — BELUM ADA EKSPERIMEN AKTUAL.**
> Tabel di bawah ini sengaja dikosongkan. Angka apa pun yang muncul di
> sini HARUS berasal dari eksekusi nyata program `pearson` pada mesin
> dengan CUDA toolkit dan GPU NVIDIA, BUKAN dikarang atau diasumsikan.

## Cara Mengisi Tabel Ini

Jalankan setelah proyek berhasil di-build (lihat `README.md`, bagian
"Cara build"):

```bash
./scripts/run_benchmark.sh ./pearson
```

Script ini akan otomatis:
1. Menghasilkan dataset sintetis untuk setiap ukuran baris di bawah
   (memakai `data/generate_data.py`).
2. Menjalankan `./pearson --mode both` pada setiap dataset.
3. Mem-parsing output program (CPU time, GPU time, Speedup, Error).
4. Menulis ulang file ini dengan hasil aktual.

## Definisi Speedup

```
Speedup = CPU time / GPU time
```

Di mana `CPU time` dan `GPU time` adalah **waktu end-to-end** sebagaimana
dilaporkan program (untuk GPU: mencakup transfer host-device, eksekusi
kernel, transfer device-host, dan final reduction di CPU — lihat
penjelasan rinci definisi waktu GPU di `docs/laporan.md`, Bab 3.9, dan
di respons Tahap 4 pengerjaan proyek ini).

## Definisi Kolom "Error"

`Error` adalah nilai absolut `|r_cpu − r_cuda|`, dibandingkan terhadap
toleransi validasi (`1e-4` secara default — lihat `include/utils.h`).
Nilai ini **tidak diharapkan nol** karena perbedaan urutan operasi
floating-point antara akumulasi sekuensial (CPU) dan parallel reduction
(GPU) — lihat penjelasan lengkap di `docs/laporan.md` Bab 2.3 dan
respons Tahap 4.

## Tabel Hasil

| Rows      | CPU (ms) | GPU (ms) | Speedup | Error |
| --------- | -------- | -------- | ------- | ----- |
| 100       |          |          |         |       |
| 1.000     |          |          |         |       |
| 10.000    |          |          |         |       |
| 100.000   |          |          |         |       |
| 1.000.000 |          |          |         |       |

## Lingkungan Pengujian

Isi bagian ini secara manual setelah menjalankan benchmark, dengan
spesifikasi mesin AKTUAL (bukan asumsi):

- **CPU**: `<isi manual, mis. hasil dari 'lscpu' atau 'cat /proc/cpuinfo'>`
- **GPU**: `<isi manual, mis. hasil dari 'nvidia-smi --query-gpu=name --format=csv'>`
- **CUDA Toolkit**: `<isi manual, mis. hasil dari 'nvcc --version'>`
- **OS**: `<isi manual>`
- **Compiler**: `<isi manual, mis. hasil dari 'g++ --version'>`
- **Tanggal eksekusi**: `<isi manual>`

## Catatan Analisis (Isi Setelah Ada Data Aktual)

Bagian ini sengaja dikosongkan. Setelah tabel di atas terisi hasil
aktual, tuliskan observasi berdasarkan angka tersebut, misalnya:
- Pada ukuran berapa GPU mulai lebih cepat dari CPU (jika ada)?
- Apakah speedup meningkat, menurun, atau stagnan seiring ukuran data
  bertambah? Mengapa (kaitkan dengan overhead transfer vs komputasi)?
- Apakah nilai Error konsisten berada dalam toleransi di semua ukuran?

**Jangan menuliskan kesimpulan performa di sini sebelum tabel di atas
benar-benar terisi dari hasil eksekusi nyata.**
