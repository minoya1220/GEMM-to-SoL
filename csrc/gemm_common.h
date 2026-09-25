#pragma once
#include <torch/extension.h>
#include <cstdint>

struct gemm_setup_t {
    float* A;
    float* B;
    float* C;
    torch::Tensor A_tensor, B_tensor, C_tensor;
    int64_t M, N, K;
};

inline gemm_setup_t prep_tensors(torch::Tensor A, torch::Tensor B) {
    gemm_setup_t out;

    TORCH_CHECK(A.defined() && B.defined(), "undefined tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expected 2D");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Incompatible datatype, must be float32");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Both tensors must be contiguous");

    const int64_t M = A.size(-2);
    const int64_t K = A.size(-1);
    const int64_t N = B.size(-1);
    TORCH_CHECK(K == B.size(-2), "Incompatible dimension: ", K, " does not equal ", B.size(-2));

    out.M = M;
    out.K = (K + 3) / 4 * 4;
    out.N = (N + 3) / 4 * 4;

    const int64_t k_padding = out.K - K;
    const int64_t n_padding = out.N - N;
    if (k_padding || n_padding) {
        namespace F = torch::nn::functional;
        A = F::pad(A, F::PadFuncOptions({0, k_padding}).value(0));
        B = F::pad(B, F::PadFuncOptions({0, n_padding, 0, k_padding}).value(0));
    }

    out.A_tensor = A;
    out.B_tensor = B;

    torch::Tensor C_padded = torch::empty({out.M, out.N}, A.options());
    out.C_tensor = C_padded.narrow(1, 0, N);

    out.A = A.data_ptr<float>();
    out.B = B.data_ptr<float>();
    out.C = C_padded.data_ptr<float>();

    return out;

}

