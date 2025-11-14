#include <torch/extension.h>

// Forward declarations
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);

// Future implementations will be added here

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "Naive GEMM");
    m.def("gemm_tiled", &gemm_tiled, "Tiled GEMM");

}