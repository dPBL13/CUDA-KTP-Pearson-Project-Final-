// Nama: main.cpp
// Desc: Program Perhitungan Korelasi Pearson CPU dan CUDA

// Pembuat:
// 1. Laurensius Brian Prayoga (24060124130077)
// 2. Haikal Imam Ridha (24060124130097)

#include <iostream>
#include <cmath>

#include "csv_loader.h"
#include "pearson_cpu.h"
#include "pearson_cuda.h"
#include "utils.h"

namespace {

int run_cpu_only(const pearson::CliArguments& args,
                 const pearson::CsvColumns& data) {
    pearson::PearsonResult result;

    const double cpu_ms =
        pearson::measure_pearson_cpu_ms(data.x, data.y, result);

    pearson::print_result_header(
        args.input_path,
        data.x.size(),
        args.col_x,
        args.col_y
    );

    pearson::print_pearson_r_line(result);
    pearson::print_cpu_time_line(cpu_ms);
    pearson::print_result_footer();

    return 0;
}

int run_cuda_only(const pearson::CliArguments& args,
                  const pearson::CsvColumns& data) {
    if (!pearson::is_cuda_device_available()) {
        std::cerr
            << "Error: CUDA device tidak tersedia pada sistem ini. "
               "Pastikan driver NVIDIA terpasang dan GPU terdeteksi "
               "(cek dengan 'nvidia-smi').\n";

        return 1;
    }

    pearson::CudaTimingBreakdown timing;

    pearson::PearsonResult result =
        pearson::compute_pearson_cuda(
            data.x,
            data.y,
            timing
        );

    pearson::print_result_header(
        args.input_path,
        data.x.size(),
        args.col_x,
        args.col_y
    );

    pearson::print_pearson_r_line(result);
    pearson::print_gpu_time_line(timing.total_ms());
    pearson::print_result_footer();

    return 0;
}

int run_both(const pearson::CliArguments& args,
             const pearson::CsvColumns& data) {
    if (!pearson::is_cuda_device_available()) {
        std::cerr
            << "Error: CUDA device tidak tersedia pada sistem ini, "
               "sehingga mode 'both' tidak dapat dijalankan. "
               "Gunakan '--mode cpu' untuk menjalankan hanya CPU, "
               "atau pastikan driver NVIDIA/GPU tersedia.\n";

        return 1;
    }

    pearson::PearsonResult cpu_result;

    const double cpu_ms =
        pearson::measure_pearson_cpu_ms(
            data.x,
            data.y,
            cpu_result
        );

    pearson::CudaTimingBreakdown timing;

    const pearson::PearsonResult cuda_result =
        pearson::compute_pearson_cuda(
            data.x,
            data.y,
            timing
        );

    const double gpu_ms = timing.total_ms();

    pearson::print_result_header(
        args.input_path,
        data.x.size(),
        args.col_x,
        args.col_y
    );

    if (!cpu_result.is_defined || !cuda_result.is_defined) {
        std::cout
            << "Pearson r (CPU)  : "
            << (cpu_result.is_defined
                    ? std::to_string(cpu_result.r)
                    : "TIDAK TERDEFINISI (" +
                          std::string(cpu_result.reason) + ")")
            << "\n";

        std::cout
            << "Pearson r (CUDA) : "
            << (cuda_result.is_defined
                    ? std::to_string(cuda_result.r)
                    : "TIDAK TERDEFINISI (" +
                          std::string(cuda_result.reason) + ")")
            << "\n";

        pearson::print_cpu_time_line(cpu_ms);
        pearson::print_gpu_time_line(gpu_ms);

        pearson::print_speedup_line(
            gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0
        );

        std::cout
            << "Error        : TIDAK DIHITUNG "
               "(salah satu hasil tidak terdefinisi)\n";

        pearson::print_result_footer();

        return 0;
    }

    const double abs_error =
        std::fabs(cpu_result.r - cuda_result.r);

    const bool within_tolerance =
        abs_error <= pearson::kDefaultValidationTolerance;

    pearson::print_pearson_r_line(cpu_result);
    pearson::print_cpu_time_line(cpu_ms);
    pearson::print_gpu_time_line(gpu_ms);

    pearson::print_speedup_line(
        gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0
    );

    pearson::print_error_line(
        abs_error,
        pearson::kDefaultValidationTolerance,
        within_tolerance
    );

    pearson::print_result_footer();

    if (!within_tolerance) {
        std::cerr
            << "\nPERINGATAN: selisih hasil CPU dan CUDA melebihi "
               "toleransi validasi ("
            << pearson::kDefaultValidationTolerance
            << "). Ini bisa mengindikasikan bug pada implementasi "
               "CUDA -- periksa kernel dan konfigurasi grid/block.\n";
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    pearson::CliArguments args;

    try {
        args = pearson::parse_cli_arguments(argc, argv);
    }
    catch (const pearson::CliArgumentError& e) {
        std::cerr
            << "Error argumen CLI: "
            << e.what()
            << "\n\n";

        pearson::print_usage(
            argc > 0 ? argv[0] : "pearson"
        );

        return 1;
    }

    pearson::CsvColumns data;

    try {
        data = pearson::load_two_columns(
            args.input_path,
            args.col_x,
            args.col_y
        );
    }
    catch (const pearson::CsvLoadError& e) {
        std::cerr
            << "Error memuat CSV: "
            << e.what()
            << "\n";

        return 1;
    }

    try {
        switch (args.mode) {

            case pearson::RunMode::kCpuOnly:
                return run_cpu_only(args, data);

            case pearson::RunMode::kCudaOnly:
                return run_cuda_only(args, data);

            case pearson::RunMode::kBoth:
                return run_both(args, data);
        }
    }
    catch (const pearson::CudaRuntimeError& e) {
        std::cerr
            << "Error CUDA runtime: "
            << e.what()
            << "\n";

        return 1;
    }
    catch (const std::invalid_argument& e) {
        std::cerr
            << "Error: "
            << e.what()
            << "\n";

        return 1;
    }
    catch (const std::exception& e) {
        std::cerr
            << "Error tak terduga: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}


/*
===============================================================================
// BAGIAN TESTING

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>

#include "pearson_cpu.h"
#include "pearson_cuda.h"
#include "csv_loader.h"

namespace {

int g_tests_passed = 0;
int g_tests_failed = 0;
int g_tests_skipped = 0;

void report_pass(const std::string& name) {
    std::cout << "[PASS] " << name << "\n";
    ++g_tests_passed;
}

void report_fail(
    const std::string& name,
    const std::string& detail
) {
    std::cout
        << "[FAIL] "
        << name
        << " -- "
        << detail
        << "\n";

    ++g_tests_failed;
}

void report_skip(
    const std::string& name,
    const std::string& reason
) {
    std::cout
        << "[SKIP] "
        << name
        << " -- "
        << reason
        << "\n";

    ++g_tests_skipped;
}

constexpr double kExactMathTolerance = 1e-6;
constexpr double kCpuVsCudaTolerance = 1e-4;

void check_close(
    const std::string& name,
    double actual,
    double expected,
    double tolerance
) {
    const double diff =
        std::fabs(actual - expected);

    if (diff <= tolerance) {
        report_pass(name);
    }
    else {
        report_fail(
            name,
            "actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected) +
            " diff=" + std::to_string(diff) +
            " (toleransi=" +
            std::to_string(tolerance) +
            ")"
        );
    }
}

void check_true(
    const std::string& name,
    bool condition,
    const std::string& detail_if_false
) {
    if (condition) {
        report_pass(name);
    }
    else {
        report_fail(name, detail_if_false);
    }
}

void test_manual_small_dataset() {
    std::vector<float> x = {1, 2, 3};
    std::vector<float> y = {2, 1, 4};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "manual_small_dataset: is_defined",
        result.is_defined,
        "Seharusnya terdefinisi untuk data ini."
    );

    if (result.is_defined) {
        check_close(
            "manual_small_dataset: nilai r",
            result.r,
            0.6546536707,
            kExactMathTolerance
        );
    }
}

void test_positive_correlation() {
    std::vector<float> x =
        {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<float> y;

    for (float v : x) {
        y.push_back(3.0f * v + 1.0f);
    }

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "positive_correlation: is_defined",
        result.is_defined,
        ""
    );

    if (result.is_defined) {
        check_close(
            "positive_correlation: r mendekati 1.0",
            result.r,
            1.0,
            kExactMathTolerance
        );
    }
}

void test_negative_correlation() {
    std::vector<float> x =
        {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<float> y;

    for (float v : x) {
        y.push_back(-2.0f * v + 5.0f);
    }

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "negative_correlation: is_defined",
        result.is_defined,
        ""
    );

    if (result.is_defined) {
        check_close(
            "negative_correlation: r mendekati -1.0",
            result.r,
            -1.0,
            kExactMathTolerance
        );
    }
}

void test_independent_data() {
    std::vector<float> x =
        {1, 5, 2, 8, 3, 9, 4, 7, 6, 0};

    std::vector<float> y =
        {7, 2, 9, 1, 6, 0, 8, 3, 4, 5};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "independent_data: is_defined",
        result.is_defined,
        ""
    );

    if (result.is_defined) {
        check_true(
            "independent_data: r dalam rentang [-1,1]",
            result.r >= -1.0 - kExactMathTolerance &&
            result.r <= 1.0 + kExactMathTolerance,
            "r = " +
            std::to_string(result.r) +
            " di luar rentang valid."
        );
    }
}

void test_cpu_vs_cuda() {
    if (!pearson::is_cuda_device_available()) {
        report_skip(
            "cpu_vs_cuda",
            "Tidak ada CUDA device terdeteksi pada sistem ini "
            "-- test di-skip, BUKAN dianggap lulus."
        );

        return;
    }

    const int n = 500000;

    std::vector<float> x(n);
    std::vector<float> y(n);

    for (int i = 0; i < n; ++i) {
        x[i] =
            static_cast<float>(i % 1000) *
            0.37f -
            100.0f;

        y[i] =
            static_cast<float>((i * 7) % 1000) *
            0.21f +
            5.0f;
    }

    const auto cpu_result =
        pearson::compute_pearson_cpu(x, y);

    pearson::CudaTimingBreakdown timing;

    const auto cuda_result =
        pearson::compute_pearson_cuda(
            x,
            y,
            timing
        );

    check_true(
        "cpu_vs_cuda: kedua hasil terdefinisi",
        cpu_result.is_defined &&
        cuda_result.is_defined,
        "Salah satu hasil tidak terdefinisi."
    );

    if (cpu_result.is_defined &&
        cuda_result.is_defined) {

        check_close(
            "cpu_vs_cuda: r_cpu mendekati r_cuda "
            "(toleransi 1e-4)",
            cuda_result.r,
            cpu_result.r,
            kCpuVsCudaTolerance
        );
    }
}

void test_n_equals_1() {
    std::vector<float> x = {5.0f};
    std::vector<float> y = {10.0f};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "n_equals_1: tidak terdefinisi",
        !result.is_defined,
        "n=1 seharusnya menghasilkan is_defined=false."
    );
}

void test_n_equals_2() {
    std::vector<float> x =
        {1.0f, 3.0f};

    std::vector<float> y =
        {2.0f, 8.0f};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "n_equals_2: terdefinisi",
        result.is_defined,
        "n=2 dengan X tidak konstan seharusnya terdefinisi."
    );

    if (result.is_defined) {
        check_close(
            "n_equals_2: r = 1.0 (garis lurus naik)",
            result.r,
            1.0,
            kExactMathTolerance
        );
    }
}

void test_x_constant() {
    std::vector<float> x =
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    std::vector<float> y =
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "x_constant: tidak terdefinisi",
        !result.is_defined,
        "X konstan seharusnya menghasilkan is_defined=false."
    );
}

void test_y_constant() {
    std::vector<float> x =
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::vector<float> y =
        {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

    const auto result =
        pearson::compute_pearson_cpu(x, y);

    check_true(
        "y_constant: tidak terdefinisi",
        !result.is_defined,
        "Y konstan seharusnya menghasilkan is_defined=false."
    );
}

void test_empty_csv_dataset() {
    const std::string path =
        "/tmp/test_empty_dataset_unit_test.csv";

    {
        std::ofstream f(path);
    }

    bool threw_expected_error = false;

    try {
        pearson::load_two_columns(
            path,
            0,
            1
        );
    }
    catch (const pearson::CsvLoadError&) {
        threw_expected_error = true;
    }

    check_true(
        "empty_csv_dataset: CsvLoadError dilempar",
        threw_expected_error,
        "File CSV kosong seharusnya melempar CsvLoadError."
    );

    std::remove(path.c_str());
}

} // namespace

int main() {
    std::cout
        << "=== Menjalankan Unit Test Pearson CPU/CUDA ===\n\n";

    test_manual_small_dataset();
    test_positive_correlation();
    test_negative_correlation();
    test_independent_data();
    test_cpu_vs_cuda();
    test_n_equals_1();
    test_n_equals_2();
    test_x_constant();
    test_y_constant();
    test_empty_csv_dataset();

    std::cout
        << "\n=== Ringkasan ===\n";

    std::cout
        << "Passed  : "
        << g_tests_passed
        << "\n";

    std::cout
        << "Failed  : "
        << g_tests_failed
        << "\n";

    std::cout
        << "Skipped : "
        << g_tests_skipped
        << "\n";

    if (g_tests_skipped > 0) {
        std::cout
            << "\nCATATAN: "
            << g_tests_skipped
            << " test di-skip "
               "(kemungkinan karena tidak ada CUDA device). "
               "Ini BUKAN berarti test tersebut lulus -- "
               "jalankan ulang di mesin dengan GPU untuk "
               "mendapatkan cakupan penuh.\n";
    }

    return g_tests_failed > 0 ? 1 : 0;
}

*/
