#include <torch/extension.h>
#include "gemm_common.h"

constexpr int WARP_SIZE = 32; // constant for all nvidia gpus
constexpr int BDIM = 256;
constexpr int WARPS_PER_BLOCK = BDIM / WARP_SIZE;

constexpr int TILE_M = 128; // block sizes along each dimension
constexpr int TILE_N = TILE_M; 
constexpr int TILE_K = 8; // small K and larger M and N boosts arithmetic intensity
constexpr int FRAG_SIZE = 8;

// for laying out warps within a block
constexpr int WARP_PER_ROW = 2; // can be 2 or 4
constexpr int WARP_TILE_N = TILE_N / WARP_PER_ROW; // 128 / 2 = 64
constexpr int WARP_TILE_M = TILE_M / (BDIM / WARP_SIZE / WARP_PER_ROW); // (NUM_WARPS / WARPS_PER_ROW) is warps per col, // (256 / 32 / 2) = 4

constexpr int T_PER_WTILE_ROW = WARP_TILE_N / FRAG_SIZE;


__global__ void gemm_warptiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE; // warp idx within a block, WARP_SIZE = 32 (universal across all nvidia gpus)
    int lane_id = tid % WARP_SIZE; // lane idx within a warp
    
    
    __shared__ float tileA[TILE_M * TILE_K]; // 128 x 8
    __shared__ float tileB[TILE_K * TILE_N]; // 8 x 128

    
    float output[FRAG_SIZE][FRAG_SIZE] = {0}; // preallocate our output tile on to registers
    
    int num_blks_n = (N + TILE_N - 1) / TILE_N; // number of blocks in the n dimension  
    int mt = bid / num_blks_n * TILE_M; // m tile idx
    int nt = bid % num_blks_n * TILE_N; // n tile idx
    for (int kt = 0; kt < K; kt += TILE_K) {
        // Load from GMEM to SMEM
        #pragma unroll
        for (int i = 0; i < TILE_M * TILE_K / BDIM; i++) { // since TILE_M = TILE_N both tiles can be loaded in a single loop
            int idx = tid + i * BDIM; // strided to allow coalescing for GMEM loads
            bool maskA = mt + idx / TILE_K < M && kt + idx % TILE_K < K; 
            bool maskB = kt + idx / TILE_N < K && nt + idx % TILE_N < N; 
            tileA[idx] = maskA ? A[(mt + idx / TILE_K) * K + (kt + idx % TILE_K)] : 0; // (m index) * K + (k index)
            tileB[idx] = maskB ? B[(kt + idx / TILE_N) * N + (nt + idx % TILE_N)] : 0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float fragA[FRAG_SIZE];
            float fragB[FRAG_SIZE];

            // Load from SMEM to registers
            #pragma unroll
            for (int i = 0; i < FRAG_SIZE; i++) {
                int in_tile_m = warp_id / WARP_PER_ROW * WARP_TILE_M + lane_id / T_PER_WTILE_ROW * FRAG_SIZE;
                int in_tile_n = warp_id % WARP_PER_ROW * WARP_TILE_N + lane_id % T_PER_WTILE_ROW * FRAG_SIZE;
                fragA[i] = tileA[(in_tile_m + i) * TILE_K + (k)];
                fragB[i] = tileB[(k) * TILE_N + (in_tile_n + i)];
            }

            // compute outer product (matmul of our two vector fragments)
            #pragma unroll
            for (int m = 0; m < FRAG_SIZE; m++) {
                #pragma unroll
                for (int n = 0; n < FRAG_SIZE; n++) {
                    output[m][n] += fragA[m] * fragB[n];
                }
            }
            
        }
        __syncthreads(); 

    }
    // write output to GMEM, this is uncoalesced atm
    #pragma unroll
    for (int m = 0; m < FRAG_SIZE; m++) {
        #pragma unroll
        for (int n = 0; n < FRAG_SIZE; n++) {
            int in_tile_m = warp_id / WARP_PER_ROW * WARP_TILE_M + lane_id / T_PER_WTILE_ROW * FRAG_SIZE;
            int in_tile_n = warp_id % WARP_PER_ROW * WARP_TILE_N + lane_id % T_PER_WTILE_ROW * FRAG_SIZE;
            C[(mt + in_tile_m + m) * N + (nt + in_tile_n + n)] = output[m][n];
        }
    }
}

torch::Tensor gemm_warptiled(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BDIM);
    dim3 grid(((t.M + TILE_M - 1) / TILE_M) * ((t.N + TILE_N - 1) / TILE_N)); 

    gemm_warptiled_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

