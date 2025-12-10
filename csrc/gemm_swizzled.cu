#include <torch/extension.h>
#include "gemm_common.h"

constexpr int WARP_SIZE = 32; // constant for all nvidia gpus
constexpr int BDIM = 256;
constexpr int WARPS_PER_BLOCK = BDIM / WARP_SIZE;

constexpr int BLK_M = 128; // block sizes along each dimension
constexpr int BLK_N = 128; 
constexpr int BLK_K = 8; // small K and larger M and N boosts arithmetic intensity
constexpr int FRAG_SIZE = 8;

// for laying out warps within a block
constexpr int WARP_PER_ROW = 2; // can be 2 or 4
constexpr int WARP_TILE_W = BLK_N / WARP_PER_ROW;
constexpr int WARP_TILE_H = BLK_M / (BDIM / WARP_SIZE / WARP_PER_ROW); // (NUM_WARPS / WARPS_PER_ROW) is warps per col

constexpr int T_PER_WTILE_ROW = WARP_TILE_W / FRAG_SIZE;

__device__ __forceinline__ int swizzle_A(int row, int col) {
    return row * BLK_K + (col ^ ((row >> 2) & 0b111));
}

__global__ __launch_bounds__(BDIM, 2) void gemm_swizzled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int m_blks = (M + BLK_M - 1) / BLK_M;
    int n_blks = (N + BLK_N - 1) / BLK_N;  
    __shared__ float tileA[2][BLK_M * BLK_K]; // 128 x 8
    __shared__ float tileB[2][BLK_K * BLK_N]; // 8 x 128
    int read = 0;
    int write = 1;

    
    float output[4][FRAG_SIZE/2][FRAG_SIZE/2] = {0}; // if we stride our output tiles well be able to coalesce our store

    
    // preprocess address calculations for SMEM -> reg and reg -> GMEM
    int tile_offset_m = warp_id / WARP_PER_ROW * WARP_TILE_H + lane_id / T_PER_WTILE_ROW * FRAG_SIZE/2;
    int tile_offset_n = warp_id % WARP_PER_ROW * WARP_TILE_W + lane_id % T_PER_WTILE_ROW * FRAG_SIZE/2;
    
    int mt = bid / n_blks * BLK_M; // m tile idx
    int nt = bid % n_blks * BLK_N; // n tile idx
    int idx = tid * 4;

    // Load first iteration tiles from GMEM to SMEM
    bool maskA_0 = mt + idx / BLK_K < M && idx % BLK_K < K;
    bool maskB_0 = idx / BLK_N < K && nt + idx % BLK_N < N;
    float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 firstA = maskA_0 ? __ldcg((float4*)&A[(mt + idx / BLK_K) * K + (idx % BLK_K)]) : zero; // (m index) * K + (k index)
    tileA[0][(swizzle_A(idx / BLK_K, idx % BLK_K))] = firstA.x;
    tileA[0][(swizzle_A(idx / BLK_K, idx % BLK_K + 1))] = firstA.y;
    tileA[0][(swizzle_A(idx / BLK_K, idx % BLK_K + 2))] = firstA.z;
    tileA[0][(swizzle_A(idx / BLK_K, idx % BLK_K + 3))] = firstA.w;

    *(float4*)&tileB[0][idx] = maskB_0 ? __ldcg((float4*)&B[(idx / BLK_N) * N + (nt + idx % BLK_N)]) : zero;

    __syncthreads();

    for (int kt = 0; kt < K; kt += BLK_K) {
        // Begin the load for the next iteration if it exists
        bool maskA = mt + idx / BLK_K < M && (kt + BLK_K) + idx % BLK_K < K && (kt + BLK_K) < K;
        bool maskB = (kt + BLK_K) + idx / BLK_N < K && nt + idx % BLK_N < N && (kt + BLK_K) < K;

        // start GMEM load for next iterations 
        float4 nextA = maskA ? __ldcg((float4*)&A[(mt + idx / BLK_K) * K + ((kt + BLK_K) + idx % BLK_K)]) : zero; // (m index) * K + (k index)
        float4 nextB = maskB ? __ldcg((float4*)&B[((kt + BLK_K) + idx / BLK_N) * N + (nt + idx % BLK_N)]) : zero;
                
        
        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            float fragA[FRAG_SIZE];
            float fragB[FRAG_SIZE];

            // Load from SMEM to registers
            #pragma unroll
            for (int i = 0; i < FRAG_SIZE/2; i++) {
                fragA[i] = tileA[read][swizzle_A(tile_offset_m + i, k)];
                fragA[i + 4] = tileA[read][swizzle_A(tile_offset_m + i + WARP_TILE_H/2, k)];

            }
            *(float4*)&fragB[0] = *(float4*)&tileB[read][(k) * BLK_N + (tile_offset_n)];
            *(float4*)&fragB[4] = *(float4*)&tileB[read][(k) * BLK_N + (tile_offset_n + WARP_TILE_W/2)];

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

        // Store next iterations GMEM load into SMEM
        // tileA stores are now scalar because of swizzling
        tileA[write][(swizzle_A(idx / BLK_K, idx % BLK_K))] = nextA.x;
        tileA[write][(swizzle_A(idx / BLK_K, idx % BLK_K + 1))] = nextA.y;
        tileA[write][(swizzle_A(idx / BLK_K, idx % BLK_K + 2))] = nextA.z;
        tileA[write][(swizzle_A(idx / BLK_K, idx % BLK_K + 3))] = nextA.w;
        *(float4*)&tileB[write][idx] = nextB;

        // swap read and write buffers
        read ^= 1; 
        write ^= 1;

        __syncthreads(); 

    }
    // write output to GMEM
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {    
        #pragma unroll
        for (int m = 0; m < FRAG_SIZE/2; m++) {
            int tile_coord_m = tile_offset_m + tile / 2 * WARP_TILE_H/2 + m;
            int tile_coord_n = tile_offset_n + tile % 2 * WARP_TILE_W/2;
            __stwb((float4*)&C[(mt + tile_coord_m) * N + (nt + tile_coord_n)], *(float4*)&output[tile][m]); 
        }
    }
}

torch::Tensor gemm_swizzled(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BDIM);
    dim3 grid(((t.M + BLK_M - 1) / BLK_M) * ((t.N + BLK_N - 1) / BLK_N)); 

    gemm_swizzled_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

