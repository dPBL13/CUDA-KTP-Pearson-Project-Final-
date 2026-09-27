#include "pearson_cpu.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pearson {

namespace {


constexpr double kDenominatorEpsilon = 1e-12;

}

PearsonAccumulators accumulate_cpu(const std::vector<float>& x,
                                    const std::vector<float>& y) {
    PearsonAccumulators acc;
    const std::size_t n = x.size();
    acc.n = static_cast<double>(n);


    for (std::size_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        acc.sum_x += xi;
        acc.sum_y += yi;
        acc.sum_x2 += xi * xi;
        acc.sum_y2 += yi * yi;
        acc.sum_xy += xi * yi;
    }

    return acc;
}

PearsonResult finalize_pearson(const PearsonAccumulators& acc) {
    PearsonResult result;


    if (acc.n < 2.0) {
        result.is_defined = false;
        result.reason = "Jumlah data (n) kurang dari 2; korelasi Pearson "
                         "membutuhkan minimal 2 titik data.";
        return result;
    }


    const double numerator = acc.n * acc.sum_xy - acc.sum_x * acc.sum_y;
    const double var_x_term = acc.n * acc.sum_x2 - acc.sum_x * acc.sum_x;
    const double var_y_term = acc.n * acc.sum_y2 - acc.sum_y * acc.sum_y;


    if (var_x_term <= kDenominatorEpsilon) {
        result.is_defined = false;
        result.reason = "Data X konstan (varians X = 0); korelasi dengan "
                         "variabel yang tidak bervariasi tidak terdefinisi.";
        return result;
    }
    if (var_y_term <= kDenominatorEpsilon) {
        result.is_defined = false;
        result.reason = "Data Y konstan (varians Y = 0); korelasi dengan "
                         "variabel yang tidak bervariasi tidak terdefinisi.";
        return result;
    }

    const double denominator = std::sqrt(var_x_term * var_y_term);


    if (denominator <= kDenominatorEpsilon) {
        result.is_defined = false;
        result.reason = "Denominator perhitungan Pearson mendekati nol; "
                         "korelasi tidak dapat dihitung secara stabil.";
        return result;
    }

    result.r = numerator / denominator;
    result.is_defined = true;
    result.reason = "";
    return result;
}

PearsonResult compute_pearson_cpu(const std::vector<float>& x,
                                   const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "Ukuran vector X (" + std::to_string(x.size()) +
            ") dan Y (" + std::to_string(y.size()) + ") tidak sama.");
    }
    const PearsonAccumulators acc = accumulate_cpu(x, y);
    return finalize_pearson(acc);
}

double measure_pearson_cpu_ms(const std::vector<float>& x,
                               const std::vector<float>& y,
                               PearsonResult& out_result) {
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "Ukuran vector X (" + std::to_string(x.size()) +
            ") dan Y (" + std::to_string(y.size()) + ") tidak sama.");
    }


    const auto t_start = std::chrono::high_resolution_clock::now();
    const PearsonAccumulators acc = accumulate_cpu(x, y);
    const auto t_end = std::chrono::high_resolution_clock::now();

    out_result = finalize_pearson(acc);

    const std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
    return elapsed.count();
}

}
