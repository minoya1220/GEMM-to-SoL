#include <torch/extension.h>
#include "gemm_common.h"

constexpr int WARP_SIZE = 32; // constant for all nvidia gpus
constexpr int BDIM = 256;
constexpr int WARPS_PER_BLOCK = BDIM / WARP_SIZE;

constexpr int BLK_M = 128; // block sizes along each dimension
constexpr int BLK_N = BLK_M; 
constexpr int BLK_K = 8; // small K and larger M and N boosts arithmetic intensity
constexpr int FRAG_SIZE = 8;

// for laying out warps within a block
constexpr int WARP_PER_ROW = 2; // can be 2 or 4
constexpr int WARP_TILE_W = BLK_N / WARP_PER_ROW;
constexpr int WARP_TILE_H = BLK_M / (BDIM / WARP_SIZE / WARP_PER_ROW); // (NUM_WARPS / WARPS_PER_ROW) is warps per col

constexpr int T_PER_WTILE_ROW = WARP_TILE_W / FRAG_SIZE;




__global__ void gemm_vectorized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int m_blks = (M + BLK_M - 1) / BLK_M;
    int n_blks = (N + BLK_N - 1) / BLK_N;  
    
    // preprocess address calculations for SMEM -> reg and reg -> GMEM
    int tile_offset_m = warp_id / WARP_PER_ROW * WARP_TILE_H + lane_id / T_PER_WTILE_ROW * FRAG_SIZE/2;
    int tile_offset_n = warp_id % WARP_PER_ROW * WARP_TILE_W + lane_id % T_PER_WTILE_ROW * FRAG_SIZE/2;
    
    __shared__ float tileA[BLK_M * BLK_K]; // 128 x 8
    __shared__ float tileB[BLK_K * BLK_N]; // 8 x 128

    
    float output[4][FRAG_SIZE/2][FRAG_SIZE/2] = {0}; // if we stride our output tiles well be able to coalesce our store

    
    int mt = bid / n_blks * BLK_M; // m tile idx
    int nt = bid % n_blks * BLK_N; // n tile idx
    for (int kt = 0; kt < K; kt += BLK_K) {
        // Load from GMEM to SMEM
        int idx = tid * 4;
        bool maskA = mt + idx / BLK_K < M && kt + idx % BLK_K < K;
        bool maskB = kt + idx / BLK_N < K && nt + idx % BLK_N < N;
        float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *(float4*)&tileA[idx] = maskA ? __ldcg((float4*)&A[(mt + idx / BLK_K) * K + (kt + idx % BLK_K)]) : zero; // (m index) * K + (k index)
        *(float4*)&tileB[idx] = maskB ? __ldcg((float4*)&B[(kt + idx / BLK_N) * N + (nt + idx % BLK_N)]) : zero;
        
        
        __syncthreads();
        
        
        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            float fragA[FRAG_SIZE];
            float fragB[FRAG_SIZE];


            // Load from SMEM to registers
            #pragma unroll
            for (int i = 0; i < FRAG_SIZE/2; i++) {
                fragA[i] = tileA[(tile_offset_m + i) * BLK_K + (k)];
                fragA[i + 4] = tileA[(tile_offset_m + i + WARP_TILE_H/2) * BLK_K + (k)];

            }
            *(float4*)&fragB[0] = *(float4*)&tileB[(k) * BLK_N + (tile_offset_n)];
            *(float4*)&fragB[4] = *(float4*)&tileB[(k) * BLK_N + (tile_offset_n + WARP_TILE_W/2)];

            // compute outer product (matmul for our two fragments)
            #pragma unroll
            for (int tile = 0; tile < 4; tile++) {
                #pragma unroll
                for (int m = 0; m < FRAG_SIZE/2; m++) {
                    #pragma unroll
                    for (int n = 0; n < FRAG_SIZE/2; n++) {
                        output[tile][m][n] += fragA[tile / 2 * FRAG_SIZE/2 + m] * fragB[tile % 2 * FRAG_SIZE/2 + n];
                    }
                }
            }
        }
        __syncthreads(); 

    }
    // write output to GMEM
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {    
        #pragma unroll
        for (int m = 0; m < FRAG_SIZE/2; m++) {
            int tile_coord_m = tile_offset_m + tile / 2 * WARP_TILE_H/2 + m;
            int tile_coord_n = tile_offset_n + tile % 2 * WARP_TILE_W/2;
            __stwb((float4*)&C[(mt + tile_coord_m) * N + (nt + tile_coord_n)], *(float4*)&output[tile][m]); // __stwb is the same as the default store 
        }
    }
}

torch::Tensor gemm_vectorized(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BDIM);
    dim3 grid(((t.M + BLK_M - 1) / BLK_M) * ((t.N + BLK_N - 1) / BLK_N)); 

    gemm_vectorized_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

