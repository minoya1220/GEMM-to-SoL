#include <torch/extension.h>

// Forward declarations
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_register_blocked(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_warptiled(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_vectorized(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_double_buffered(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_transposed(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_swizzled(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_final(torch::Tensor A, torch::Tensor B);



// Future implementations will be added here

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "Naive GEMM");
    m.def("gemm_tiled", &gemm_tiled, "Tiled GEMM");
    m.def("gemm_register_blocked", &gemm_register_blocked, "Register Blocked GEMM");
    m.def("gemm_warptiled", &gemm_warptiled, "Warptiled GEMM");
    m.def("gemm_vectorized", &gemm_vectorized, "Vectorized GEMM");
    m.def("gemm_double_buffered", &gemm_double_buffered, "Double Buffered GEMM");
    m.def("gemm_transposed", &gemm_transposed, "Transposed GEMM");
    m.def("gemm_swizzled", &gemm_swizzled, "Swizzled GEMM");
    m.def("gemm_final", &gemm_final, "Final GEMM");


}