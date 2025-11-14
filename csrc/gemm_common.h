#pragma once
#include <torch/extension.h>
#include <cstdint>

struct gemm_setup_t {
    float* A;
    float* B;
    float* C;
    torch::Tensor C_tensor;
    int64_t M, N, K;
};

inline gemm_setup_t prep_tensors(torch::Tensor A, torch::Tensor B) {
    gemm_setup_t out;
    
    out.M = A.size(-2);
    out.K = A.size(-1);
    out.N = B.size(-1);
    
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Incompatible datatype, must be float32");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Both tensors must be contiguous");
    TORCH_CHECK(out.K == B.size(-2), "Incompatible dimension: ", out.K, " does not equal ", B.size(-2));
    
    out.C_tensor = torch::empty({out.M, out.N}, A.options());
    
    out.A = A.data_ptr<float>();
    out.B = B.data_ptr<float>();
    out.C = out.C_tensor.data_ptr<float>();

    return out;
}

