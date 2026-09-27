#include "utils.h"

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

namespace pearson {

namespace {


std::optional<int> try_parse_int(const std::string& s) {
    if (s.empty()) {
        return std::nullopt;
    }
    char* end_ptr = nullptr;
    const long value = std::strtol(s.c_str(), &end_ptr, 10);
    if (end_ptr == s.c_str() || *end_ptr != '\0') {
        return std::nullopt;
    }
    return static_cast<int>(value);
}


std::optional<RunMode> parse_mode(const std::string& s) {
    if (s == "cpu") return RunMode::kCpuOnly;
    if (s == "cuda") return RunMode::kCudaOnly;
    if (s == "both") return RunMode::kBoth;
    return std::nullopt;
}

}

CliArguments parse_cli_arguments(int argc, char** argv) {
    std::optional<std::string> input_path;
    std::optional<int> col_x;
    std::optional<int> col_y;
    RunMode mode = RunMode::kBoth;


    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const std::string& flag_name) -> std::string {
            if (i + 1 >= argc) {
                throw CliArgumentError(
                    "Argumen '" + flag_name + "' membutuhkan value, tetapi "
                    "tidak ada value yang diberikan.");
            }
            return argv[++i];
        };

        if (arg == "--input") {
            input_path = require_value(arg);
        } else if (arg == "--col-x") {
            const std::string value_str = require_value(arg);
            const auto parsed = try_parse_int(value_str);
            if (!parsed) {
                throw CliArgumentError(
                    "Nilai '--col-x' harus berupa bilangan bulat, "
                    "diterima: '" + value_str + "'.");
            }
            col_x = parsed;
        } else if (arg == "--col-y") {
            const std::string value_str = require_value(arg);
            const auto parsed = try_parse_int(value_str);
            if (!parsed) {
                throw CliArgumentError(
                    "Nilai '--col-y' harus berupa bilangan bulat, "
                    "diterima: '" + value_str + "'.");
            }
            col_y = parsed;
        } else if (arg == "--mode") {
            const std::string value_str = require_value(arg);
            const auto parsed = parse_mode(value_str);
            if (!parsed) {
                throw CliArgumentError(
                    "Nilai '--mode' harus salah satu dari 'cpu', 'cuda', "
                    "atau 'both', diterima: '" + value_str + "'.");
            }
            mode = *parsed;
        } else {
            throw CliArgumentError("Argumen tidak dikenali: '" + arg + "'.");
        }
    }

    if (!input_path) {
        throw CliArgumentError(
            "Argumen '--input' wajib diberikan (path ke file CSV).");
    }
    if (!col_x) {
        throw CliArgumentError(
            "Argumen '--col-x' wajib diberikan (indeks kolom X, 0-based).");
    }
    if (!col_y) {
        throw CliArgumentError(
            "Argumen '--col-y' wajib diberikan (indeks kolom Y, 0-based).");
    }

    CliArguments result;
    result.input_path = *input_path;
    result.col_x = *col_x;
    result.col_y = *col_y;
    result.mode = mode;
    return result;
}

void print_usage(const std::string& program_name) {
    std::cout
        << "Penggunaan:\n"
        << "  " << program_name
        << " --input <file.csv> --col-x <int> --col-y <int> "
           "--mode <cpu|cuda|both>\n\n"
        << "Contoh:\n"
        << "  " << program_name
        << " --input data.csv --col-x 0 --col-y 1 --mode both\n";
}

void print_result_header(const std::string& dataset_path,
                          std::size_t rows,
                          int col_x,
                          int col_y) {
    std::cout << "=== Pearson Correlation Result ===\n";
    std::cout << "Dataset      : " << dataset_path << "\n";
    std::cout << "Rows         : " << rows << "\n";
    std::cout << "Column X     : " << col_x << "\n";
    std::cout << "Column Y     : " << col_y << "\n";
    std::cout << "----------------\n";
}

void print_pearson_r_line(const PearsonResult& result) {
    std::cout << std::fixed << std::setprecision(6);
    if (result.is_defined) {
        std::cout << "Pearson r    : " << result.r << "\n";
    } else {
        std::cout << "Pearson r    : TIDAK TERDEFINISI (" << result.reason
                   << ")\n";
    }
}

void print_cpu_time_line(double cpu_ms) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time     : " << cpu_ms << " ms\n";
}

void print_gpu_time_line(double gpu_ms) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU time     : " << gpu_ms << " ms\n";
}

void print_speedup_line(double speedup) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Speedup      : " << speedup << "x\n";
}

void print_error_line(double abs_error, double tolerance, bool within_tolerance) {
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Error        : " << abs_error
               << (within_tolerance ? " (dalam toleransi " : " (MELEBIHI toleransi ")
               << std::scientific << tolerance << ")\n";
    std::cout << std::fixed;
}

void print_result_footer() {
    std::cout << "======================\n";
}

}
