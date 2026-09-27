#pragma once


#include <string>
#include <vector>
#include <stdexcept>

namespace pearson {


class CsvLoadError : public std::runtime_error {
public:
    explicit CsvLoadError(const std::string& message)
        : std::runtime_error(message) {}
};


struct CsvColumns {
    std::vector<float> x;
    std::vector<float> y;
};


CsvColumns load_two_columns(const std::string& file_path,
                             int col_x,
                             int col_y);

}
