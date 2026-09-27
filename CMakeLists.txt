# CMakeLists.txt
# Pembuat: Laurensius Brian Prayoga (24060124130077), Haikal Imam Ridha (24060124130097)

# =============================================================================
# CMakeLists.txt
# -----------------------------------------------------------------------------
# Build system utama proyek pearson-cuda. Mengompilasi source C++ (.cpp)
# dengan compiler host dan source CUDA (.cu) dengan nvcc, lalu me-link
# keduanya menjadi satu executable "pearson".
# =============================================================================

cmake_minimum_required(VERSION 3.18)  # 3.18+ dibutuhkan untuk CMAKE_CUDA_ARCHITECTURES modern

project(pearson_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# PENTING -- WAJIB DISESUAIKAN PENGGUNA: ARSITEKTUR GPU
# =============================================================================
# CMAKE_CUDA_ARCHITECTURES menentukan compute capability target kompilasi.
# Daftar di bawah ("60 70 75 80 86") adalah rentang umum yang mencakup
# beberapa generasi GPU NVIDIA (Pascal hingga Ampere), TAPI ini BUKAN
# jaminan mencakup GPU Anda, dan mengompilasi untuk banyak arsitektur
# sekaligus memperlambat build.
#
# Cek compute capability GPU Anda dengan:
#   nvidia-smi --query-gpu=compute_cap --format=csv
#
# Lalu build dengan arsitektur spesifik GPU Anda, contoh untuk compute
# capability 8.6:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
#
# Jika tidak diberikan, default di bawah ini akan dipakai.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# -----------------------------------------------------------------------
# Executable utama
# -----------------------------------------------------------------------
add_executable(pearson
    src/main.cpp
    src/csv_loader.cpp
    src/pearson_cpu.cpp
    src/utils.cpp
    src/pearson_cuda.cu
)

set_target_properties(pearson PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(pearson PRIVATE CUDA::cudart)

# Warning flags hanya untuk kompilasi C++ (nvcc punya himpunan flag warning
# yang berbeda dan tidak selalu kompatibel dengan -Wall/-Wextra langsung).
target_compile_options(pearson PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# -----------------------------------------------------------------------
# Unit test executable -- hanya ditambahkan jika file test sudah ada
# (dibuat pada Tahap 8 proyek ini). Pengecekan EXISTS membuat CMakeLists
# ini tetap valid dipakai sebelum maupun sesudah Tahap 8 selesai.
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_pearson.cpp")
    add_executable(pearson_tests
        tests/test_pearson.cpp
        src/csv_loader.cpp
        src/pearson_cpu.cpp
        src/pearson_cuda.cu
    )
    set_target_properties(pearson_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(pearson_tests PRIVATE CUDA::cudart)
    target_compile_options(pearson_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()
