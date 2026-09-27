#pragma once


#include <vector>
#include <cstddef>
#include <string_view>

namespace pearson {


struct PearsonResult {
    double r = 0.0;
    bool is_defined = false;
    std::string_view reason;
};


struct PearsonAccumulators {
    double n = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    double sum_xy = 0.0;
};


PearsonAccumulators accumulate_cpu(const std::vector<float>& x,
                                    const std::vector<float>& y);


PearsonResult finalize_pearson(const PearsonAccumulators& acc);


PearsonResult compute_pearson_cpu(const std::vector<float>& x,
                                   const std::vector<float>& y);


double measure_pearson_cpu_ms(const std::vector<float>& x,
                               const std::vector<float>& y,
                               PearsonResult& out_result);

}
