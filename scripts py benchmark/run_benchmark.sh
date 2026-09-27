# Penggunaan:
#   ./scripts/run_benchmark.sh [path_ke_binary_pearson]
#   (default path binary: ./pearson, relatif dari root proyek)
# =============================================================================

set -euo pipefail

PEARSON_BIN="${1:-./pearson}"
SIZES=(100 1000 10000 100000 1000000)
TARGET_CORR=0.8
DATA_DIR="data/benchmark_datasets"
RESULTS_FILE="docs/benchmark_results.md"

if [ ! -x "$PEARSON_BIN" ]; then
    echo "Error: executable '$PEARSON_BIN' tidak ditemukan atau tidak dapat" >&2
    echo "dieksekusi. Build proyek terlebih dahulu (lihat README.md, bagian" >&2
    echo "'Cara build') sebelum menjalankan benchmark ini." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 tidak ditemukan, dibutuhkan untuk generate_data.py." >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Menulis hasil ke: $RESULTS_FILE"
echo ""

{
    echo "# Hasil Benchmark"
    echo ""
    echo "Dihasilkan otomatis oleh \`scripts/run_benchmark.sh\` pada: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
    echo "Speedup dihitung sebagai: \`Speedup = CPU time / GPU time\`."
    echo ""
    echo "| Rows      | CPU (ms) | GPU (ms) | Speedup | Error |"
    echo "| --------- | -------- | -------- | ------- | ----- |"
} > "$RESULTS_FILE"

for size in "${SIZES[@]}"; do
    csv_path="$DATA_DIR/bench_${size}.csv"

    echo ">> Menghasilkan dataset $size baris ..."
    python3 data/generate_data.py --rows "$size" --corr "$TARGET_CORR" \
        --out "$csv_path" --seed 123 > /dev/null

    echo ">> Menjalankan benchmark untuk $size baris ..."
    # Tangkap output; jangan biarkan 'set -e' menghentikan script jika satu
    # ukuran gagal (misalnya kehabisan memori GPU pada ukuran besar) --
    # kita ingin tahu ukuran mana yang gagal, bukan menghentikan seluruh
    # proses benchmark.
    set +e
    output=$("$PEARSON_BIN" --input "$csv_path" --col-x 0 --col-y 1 --mode both 2>&1)
    exit_code=$?
    set -e

    if [ $exit_code -ne 0 ]; then
        echo "   [GAGAL] Ukuran $size baris gagal dijalankan (exit code $exit_code)." >&2
        echo "| $size |  GAGAL  |  GAGAL  |  GAGAL  | GAGAL |" >> "$RESULTS_FILE"
        continue
    fi

    # Parsing output berdasarkan format tetap dari utils.cpp:
    #   "CPU time     : X ms"
    #   "GPU time     : X ms"
    #   "Speedup      : Xx"
    #   "Error        : X (...)"
    cpu_ms=$(echo "$output" | grep "CPU time" | sed -E 's/.*: *([0-9.]+) ms/\1/')
    gpu_ms=$(echo "$output" | grep "GPU time" | sed -E 's/.*: *([0-9.]+) ms/\1/')
    speedup=$(echo "$output" | grep "Speedup" | sed -E 's/.*: *([0-9.]+)x/\1/')
    error_val=$(echo "$output" | grep "Error" | sed -E 's/.*: *([0-9.eE+-]+).*/\1/')

    echo "   CPU=${cpu_ms}ms GPU=${gpu_ms}ms Speedup=${speedup}x Error=${error_val}"

    printf "| %-9s | %-8s | %-8s | %-7s | %-5s |\n" \
        "$size" "$cpu_ms" "$gpu_ms" "$speedup" "$error_val" >> "$RESULTS_FILE"
done

echo ""
echo "Benchmark selesai. Lihat hasil di: $RESULTS_FILE"
