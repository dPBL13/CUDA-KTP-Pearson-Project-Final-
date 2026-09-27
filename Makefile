# Makefile
# Pembuat: Laurensius Brian Prayoga (24060124130077), Haikal Imam Ridha (24060124130097)

# =============================================================================
# Makefile
# -----------------------------------------------------------------------------
# Alternatif build system tanpa CMake. Mengompilasi source .cpp dengan g++
# dan source .cu dengan nvcc secara terpisah, lalu me-link keduanya.
# =============================================================================

CXX      := g++
NVCC     := nvcc

CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -Iinclude
NVCCFLAGS := -std=c++17 -O2 -Iinclude

# =============================================================================
# PENTING -- WAJIB DISESUAIKAN PENGGUNA: ARSITEKTUR GPU
# =============================================================================
# CUDA_ARCH menentukan compute capability target kompilasi nvcc.
# Default di bawah ("sm_75") adalah asumsi umum (compute capability 7.5,
# generasi RTX 20-series/GTX 16-series) dan KEMUNGKINAN BESAR TIDAK COCOK
# dengan GPU Anda.
#
# Cek compute capability GPU Anda dengan:
#   nvidia-smi --query-gpu=compute_cap --format=csv
#
# Lalu override saat build, contoh untuk compute capability 8.6:
#   make CUDA_ARCH=sm_86
CUDA_ARCH ?= sm_75
NVCCFLAGS += -arch=$(CUDA_ARCH)

SRC_DIR := src
OBJ_DIR := build
TARGET  := pearson
TEST_TARGET := pearson_tests

CPP_SOURCES := $(SRC_DIR)/main.cpp $(SRC_DIR)/csv_loader.cpp \
               $(SRC_DIR)/pearson_cpu.cpp $(SRC_DIR)/utils.cpp
CU_SOURCES  := $(SRC_DIR)/pearson_cuda.cu

CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS  := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SOURCES))

.PHONY: all clean test

# Target default: build executable utama.
all: $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Aturan kompilasi file .cpp (host code) dengan g++.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Aturan kompilasi file .cu (device code) dengan nvcc.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link seluruh object file menjadi executable. Menggunakan nvcc sebagai
# linker (bukan g++) karena nvcc otomatis menautkan CUDA runtime library.
$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Target untuk mengompilasi unit test (Tahap 8). Hanya berfungsi setelah
# tests/test_pearson.cpp dibuat.
$(TEST_TARGET): tests/test_pearson.cpp $(SRC_DIR)/csv_loader.cpp \
                $(SRC_DIR)/pearson_cpu.cpp $(SRC_DIR)/pearson_cuda.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGET)
