#include "csv_loader.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <algorithm>

namespace pearson {

namespace {


std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    return tokens;
}


std::string trim(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}


bool try_parse_float(const std::string& raw, float& out_value) {
    const std::string trimmed = trim(raw);
    if (trimmed.empty()) {
        return false;
    }
    char* end_ptr = nullptr;
    const float parsed = std::strtof(trimmed.c_str(), &end_ptr);
    if (end_ptr == trimmed.c_str() || *end_ptr != '\0') {


        return false;
    }
    out_value = parsed;
    return true;
}


bool row_parses_as_numeric_at(const std::vector<std::string>& tokens,
                               int col_x, int col_y) {
    if (static_cast<int>(tokens.size()) <= col_x ||
        static_cast<int>(tokens.size()) <= col_y) {
        return false;
    }
    float tmp;
    return try_parse_float(tokens[col_x], tmp) &&
           try_parse_float(tokens[col_y], tmp);
}

}

CsvColumns load_two_columns(const std::string& file_path,
                             int col_x,
                             int col_y) {
    if (col_x < 0 || col_y < 0) {
        throw CsvLoadError(
            "Indeks kolom tidak valid: col_x dan col_y harus >= 0 "
            "(diterima col_x=" + std::to_string(col_x) +
            ", col_y=" + std::to_string(col_y) + ").");
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw CsvLoadError("File tidak ditemukan atau tidak dapat dibuka: " +
                            file_path);
    }

    std::vector<std::string> raw_lines;
    {
        std::string line;
        while (std::getline(file, line)) {

            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            raw_lines.push_back(line);
        }
    }


    while (!raw_lines.empty() && trim(raw_lines.back()).empty()) {
        raw_lines.pop_back();
    }

    if (raw_lines.empty()) {
        throw CsvLoadError("File CSV kosong (tidak ada baris data): " +
                            file_path);
    }


    size_t start_index = 0;
    {
        std::vector<std::string> first_tokens = split_csv_line(raw_lines[0]);
        if (!row_parses_as_numeric_at(first_tokens, col_x, col_y)) {
            start_index = 1;
        }
    }

    if (start_index >= raw_lines.size()) {
        throw CsvLoadError(
            "File CSV hanya berisi header, tidak ada baris data: " +
            file_path);
    }

    CsvColumns result;
    result.x.reserve(raw_lines.size() - start_index);
    result.y.reserve(raw_lines.size() - start_index);

    for (size_t i = start_index; i < raw_lines.size(); ++i) {
        const std::string& line = raw_lines[i];
        if (trim(line).empty()) {
            throw CsvLoadError(
                "Ditemukan baris kosong di tengah data pada baris ke-" +
                std::to_string(i + 1) + " (baris kosong hanya diizinkan di "
                "akhir file).");
        }

        std::vector<std::string> tokens = split_csv_line(line);

        const int required_cols = std::max(col_x, col_y) + 1;
        if (static_cast<int>(tokens.size()) < required_cols) {
            throw CsvLoadError(
                "Jumlah kolom kurang pada baris ke-" + std::to_string(i + 1) +
                ": dibutuhkan minimal " + std::to_string(required_cols) +
                " kolom (col_x=" + std::to_string(col_x) +
                ", col_y=" + std::to_string(col_y) + "), tetapi baris ini "
                "hanya memiliki " + std::to_string(tokens.size()) + " kolom.");
        }

        float value_x = 0.0f;
        float value_y = 0.0f;

        if (!try_parse_float(tokens[col_x], value_x)) {
            throw CsvLoadError(
                "Data non-numerik pada baris ke-" + std::to_string(i + 1) +
                ", kolom X (indeks " + std::to_string(col_x) +
                "): nilai = '" + tokens[col_x] + "'.");
        }
        if (!try_parse_float(tokens[col_y], value_y)) {
            throw CsvLoadError(
                "Data non-numerik pada baris ke-" + std::to_string(i + 1) +
                ", kolom Y (indeks " + std::to_string(col_y) +
                "): nilai = '" + tokens[col_y] + "'.");
        }

        if (std::isnan(value_x) || std::isinf(value_x)) {
            throw CsvLoadError(
                "Nilai X pada baris ke-" + std::to_string(i + 1) +
                " adalah NaN/Inf, yang tidak didukung untuk perhitungan "
                "korelasi Pearson.");
        }
        if (std::isnan(value_y) || std::isinf(value_y)) {
            throw CsvLoadError(
                "Nilai Y pada baris ke-" + std::to_string(i + 1) +
                " adalah NaN/Inf, yang tidak didukung untuk perhitungan "
                "korelasi Pearson.");
        }

        result.x.push_back(value_x);
        result.y.push_back(value_y);
    }


    if (result.x.size() != result.y.size()) {
        throw CsvLoadError(
            "Kesalahan internal: jumlah data X (" +
            std::to_string(result.x.size()) + ") dan Y (" +
            std::to_string(result.y.size()) + ") tidak sama setelah "
            "parsing. Ini seharusnya tidak terjadi — mohon laporkan sebagai "
            "bug.");
    }

    return result;
}

}
