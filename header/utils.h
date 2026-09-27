#pragma once


#include <string>
#include <optional>
#include <stdexcept>
#include "pearson_cpu.h"
#include "pearson_cuda.h"

namespace pearson {


enum class RunMode {
    kCpuOnly,
    kCudaOnly,
    kBoth
};


constexpr double kDefaultValidationTolerance = 1e-4;


struct CliArguments {
    std::string input_path;
    int col_x = 0;
    int col_y = 0;
    RunMode mode = RunMode::kBoth;
};


class CliArgumentError : public std::runtime_error {
public:
    explicit CliArgumentError(const std::string& message)
        : std::runtime_error(message) {}
};


CliArguments parse_cli_arguments(int argc, char** argv);


void print_usage(const std::string& program_name);


void print_result_header(const std::string& dataset_path,
                          std::size_t rows,
                          int col_x,
                          int col_y);


void print_pearson_r_line(const PearsonResult& result);


void print_cpu_time_line(double cpu_ms);


void print_gpu_time_line(double gpu_ms);


void print_speedup_line(double speedup);


void print_error_line(double abs_error, double tolerance, bool within_tolerance);


void print_result_footer();

}
