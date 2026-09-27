#pragma once


#include <vector>
#include <string>
#include <stdexcept>
#include "pearson_cpu.h"

namespace pearson {


struct CudaTimingBreakdown {

    double host_to_device_ms = 0.0;

    double kernel_ms = 0.0;


    double device_to_host_ms = 0.0;


    double final_reduction_cpu_ms = 0.0;


    double total_ms() const {
        return host_to_device_ms + kernel_ms + device_to_host_ms +
               final_reduction_cpu_ms;
    }
};


bool is_cuda_device_available();


PearsonResult compute_pearson_cuda(const std::vector<float>& x,
                                    const std::vector<float>& y,
                                    CudaTimingBreakdown& out_timing);


class CudaRuntimeError : public std::runtime_error {
public:
    explicit CudaRuntimeError(const std::string& message)
        : std::runtime_error(message) {}
};

}
