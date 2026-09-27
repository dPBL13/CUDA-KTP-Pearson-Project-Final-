"""
How to use:
    python generate_data.py --rows 1000000 --corr 0.8 --out data/big.csv
    python generate_data.py --rows 100 --corr 0.0 --out data/independen.csv
    python generate_data.py --rows 100 --corr -0.8 --out data/negatif.csv
"""

import argparse
import csv
import math
import random
import sys


def parse_arguments():
    """Mem-parsing argumen command-line untuk generator dataset."""
    parser = argparse.ArgumentParser(
        description="Generator dataset CSV sintetis untuk uji korelasi Pearson."
    )
    parser.add_argument(
        "--rows", type=int, required=True,
        help="Jumlah baris data yang akan dihasilkan (harus >= 2)."
    )
    parser.add_argument(
        "--corr", type=float, required=True,
        help="Korelasi TARGET (antara -1.0 dan 1.0). Korelasi aktual pada "
             "data yang dihasilkan akan mendekati, TAPI TIDAK PASTI SAMA "
             "DENGAN, nilai ini -- terutama untuk jumlah baris kecil."
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Path file CSV output."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed untuk random number generator (opsional, untuk "
             "reproducibility). Jika tidak diberikan, hasil akan berbeda "
             "setiap kali dijalankan."
    )
    return parser.parse_args()


def validate_arguments(args):
    """Memvalidasi argumen yang sudah diparsing sebelum generate dimulai."""
    if args.rows < 2:
        print(f"Error: --rows harus >= 2 (diterima: {args.rows}).",
              file=sys.stderr)
        sys.exit(1)
    if not (-1.0 <= args.corr <= 1.0):
        print(f"Error: --corr harus berada di rentang [-1.0, 1.0] "
              f"(diterima: {args.corr}).", file=sys.stderr)
        sys.exit(1)


def generate_correlated_data(rows, target_corr, rng):
    """
    Menghasilkan `rows` pasang (x, y) dengan korelasi target menggunakan
    pendekatan bivariate normal standar.

    Parameter:
        rows        : jumlah pasangan data yang dihasilkan.
        target_corr : korelasi target (populasi), bukan jaminan hasil sampel.
        rng         : instance random.Random untuk pembangkitan angka acak.

    Mengembalikan:
        (list_x, list_y) -- dua list float dengan panjang `rows`.
    """
    list_x = []
    list_y = []

    # sqrt(1 - corr^2) adalah bobot komponen noise independen. Untuk
    # |corr| mendekati 1.0, komponen ini mendekati 0 (Y hampir sepenuhnya
    # mengikuti X); untuk corr=0, komponen ini = 1 (Y sepenuhnya independen).
    noise_weight = math.sqrt(max(0.0, 1.0 - target_corr * target_corr))

    for _ in range(rows):
        z1 = rng.gauss(0.0, 1.0)
        z2 = rng.gauss(0.0, 1.0)
        x = z1
        y = target_corr * z1 + noise_weight * z2
        list_x.append(x)
        list_y.append(y)

    return list_x, list_y


def compute_empirical_pearson(list_x, list_y):
    """
    Menghitung korelasi Pearson EMPIRIS AKTUAL dari data yang dihasilkan,
    menggunakan rumus akumulator standar (n, Sx, Sy, Sx2, Sy2, Sxy) --
    rumus yang SAMA dengan yang dipakai program C++/CUDA utama, agar nilai
    yang ditampilkan di sini bisa langsung dibandingkan dengan hasil
    program tersebut.

    Ini BUKAN nilai target yang diasumsikan -- ini dihitung langsung dari
    data yang benar-benar dihasilkan.
    """
    n = len(list_x)
    if n < 2:
        return None  # Tidak terdefinisi untuk n < 2, konsisten dengan modul CPU/CUDA.

    sum_x = sum(list_x)
    sum_y = sum(list_y)
    sum_x2 = sum(v * v for v in list_x)
    sum_y2 = sum(v * v for v in list_y)
    sum_xy = sum(list_x[i] * list_y[i] for i in range(n))

    numerator = n * sum_xy - sum_x * sum_y
    var_x_term = n * sum_x2 - sum_x * sum_x
    var_y_term = n * sum_y2 - sum_y * sum_y

    if var_x_term <= 0.0 or var_y_term <= 0.0:
        return None  # Data konstan, korelasi tidak terdefinisi.

    denominator = math.sqrt(var_x_term * var_y_term)
    if denominator == 0.0:
        return None

    return numerator / denominator


def write_csv(path, list_x, list_y):
    """Menulis data (x, y) ke file CSV dengan header 'x,y'."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for xi, yi in zip(list_x, list_y):
            writer.writerow([xi, yi])


def main():
    args = parse_arguments()
    validate_arguments(args)

    rng = random.Random(args.seed)  # Jika seed=None, pakai entropy sistem.

    print(f"Menghasilkan {args.rows} baris data dengan korelasi target "
          f"{args.corr:.4f} ...")

    list_x, list_y = generate_correlated_data(args.rows, args.corr, rng)

    write_csv(args.out, list_x, list_y)
    print(f"Dataset ditulis ke: {args.out}")

    # WAJIB: hitung dan tampilkan korelasi empiris aktual, JANGAN
    # mengasumsikan korelasi aktual == target.
    actual_corr = compute_empirical_pearson(list_x, list_y)
    if actual_corr is None:
        print("Korelasi empiris AKTUAL: TIDAK TERDEFINISI "
              "(data konstan atau n < 2).")
    else:
        print(f"Korelasi empiris AKTUAL pada data yang dihasilkan: "
              f"{actual_corr:.6f}")
        print(f"(Target yang diminta: {args.corr:.6f} -- nilai aktual di "
              f"atas adalah hasil sesungguhnya dari sampel yang dibuat, "
              f"BUKAN dijamin identik dengan target, terutama untuk "
              f"jumlah baris kecil.)")


if __name__ == "__main__":
    main()
