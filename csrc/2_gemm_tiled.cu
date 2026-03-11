#include <torch/extension.h>
#include "gemm_common.h"

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = 256;

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    
    // allocate SMEM for input tiles
    __shared__ float tileA[TILE_SIZE * TILE_SIZE]; 
    __shared__ float tileB[TILE_SIZE * TILE_SIZE]; 
    
    float accum = 0;
    
    int in_tile_m = tid / TILE_SIZE; // } same process as in naive but instead of 
    int in_tile_n = tid % TILE_SIZE; // } the whole matrix, its within the tile

    int num_blks_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    int mt = bid / num_blks_n * TILE_SIZE; // } this time we want our indices quantized to the tiles
    int nt = bid % num_blks_n * TILE_SIZE; // } 
    for (int kt = 0; kt < K; kt += TILE_SIZE) { 
        // boundary checking
        bool maskA = mt + in_tile_m < M && kt + tid % TILE_SIZE < K;
        bool maskB = kt + tid / TILE_SIZE < K && nt + in_tile_n < N;
        
        // loading from GMEM -> SMEM, if boundary check fails fill with 0
        tileA[tid] = maskA ? A[(mt + in_tile_m) * K + (kt + tid % TILE_SIZE)] : 0; 
        tileB[tid] = maskB ? B[(kt + tid / TILE_SIZE) * N + (nt + in_tile_n)] : 0;
        

        __syncthreads();
        
        // computes dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            accum += tileA[in_tile_m * TILE_SIZE + k] * tileB[k * TILE_SIZE + in_tile_n];
        }
        __syncthreads();  // forgetting this creates a race condition

    }
    // write result from registers -> GMEM
    if ((mt + in_tile_m) < M && (nt + in_tile_n) < N) {
        C[(mt + in_tile_m) * N + (nt + in_tile_n)] = accum;
    }
}

torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BLOCK_SIZE); // = 256
    dim3 grid((t.M + TILE_SIZE - 1) / TILE_SIZE * (t.N + TILE_SIZE - 1) / TILE_SIZE); 

    gemm_tiled_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

