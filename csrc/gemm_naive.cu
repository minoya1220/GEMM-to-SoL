#include <torch/extension.h>
#include "gemm_common.h"


__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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
    auto t = prep_tensors(A, B);

    dim3 block(16, 16);
    dim3 grid((t.M + block.x - 1) / block.x, (t.N + block.y - 1) / block.y); 

    gemm_naive_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

