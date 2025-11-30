#include <torch/extension.h>
#include "gemm_common.h"

constexpr int TILE_SIZE = 32;

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int bdimx = blockDim.x;
    int bdimy = blockDim.y;
    
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    // int tid = tidy * bdimx * tidx;
    
    
    // Layouts
    // A (M, K) : (K, 1)
    // B (K, N) : (N, 1)
    
    // for this version bdimx, y, tile_size are all 32 
    __shared__ float tileA[TILE_SIZE * TILE_SIZE]; // Block M * Block K
    __shared__ float tileB[TILE_SIZE * TILE_SIZE]; // Block K * Block N 
    
    
    float accum = 0;
    for (int kt = 0; kt < K; kt += TILE_SIZE) {

        tileA[tidx * TILE_SIZE + tidy] = bidx * bdimx < M &&  kt + tidx < K ? 
            A[kt + (bidx * bdimx * K) + (tidx * K + tidy)] : 0; 
        
        tileB[tidy * TILE_SIZE + tidx] = bidy * bdimy < N && kt + tidy < K ? 
            B[(bidy * bdimy) + kt * N + (tidy * N + tidx)] : 0;
        

        __syncthreads();
        

        for (int k = 0; k < TILE_SIZE; k++) {
            accum += tileA[tidx * TILE_SIZE + k] * tileB[k * bdimy + tidy];
        }
        __syncthreads(); // I FORGOT THIS 

    }
    C[(bidx * bdimx * N) + (bidy * bdimy) + (tidx * N + tidy)] = accum;


}

torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(32, 32);
    dim3 grid((t.M + block.x - 1) / block.x, (t.N + block.y - 1) / block.y); 

    gemm_tiled_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

