#include <torch/extension.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= N) {
        return;
    }
    
    float accum = 0;
    for (int k = 0; k < K; k++) {
        accum += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = accum;

    
}

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(-2);
    auto K = A.size(-1);
    auto N = B.size(-1);

    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Incompatible datatype, must be float32");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Both tensors must be contiguous");
    TORCH_CHECK(K == B.size(-2), "Incompatible dimension: ", K, " does not equal ", B.size(-2));

    auto C = torch::empty({M, N}, A.options());
    
    auto A_ = A.data_ptr<float>();
    auto B_ = B.data_ptr<float>();
    auto C_ = C.data_ptr<float>();

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y); 

    gemm_kernel<<<grid, block>>>(A_, B_, C_, M, N, K);
    cudaDeviceSynchronize();
    
    return C;

}

