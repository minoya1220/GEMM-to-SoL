#include <torch/extension.h>
#include "gemm_common.h"

constexpr int WARP_SIZE = 32; // constant for all nvidia gpus
constexpr int BDIM = 256;

constexpr int TILE_M = 128; // block sizes along each dimension
constexpr int TILE_N = TILE_M; 
constexpr int TILE_K = 8; // small K and larger M and N boosts arithmetic intensity
constexpr int FRAG_SIZE = 8;

// for laying out warps within a block
constexpr int WARP_PER_ROW = 2; // can be 2 or 4
constexpr int WARP_TILE_N = TILE_N / WARP_PER_ROW;
constexpr int WARP_TILE_M = TILE_M / (BDIM / WARP_SIZE / WARP_PER_ROW); // (NUM_WARPS / WARPS_PER_ROW) is warps per col

constexpr int T_PER_WTILE_ROW = WARP_TILE_N / FRAG_SIZE;



__global__ void gemm_transposed_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    __shared__ __align__(16) float tileA[2][TILE_M * TILE_K]; // 128 x 8
    __shared__ __align__(16) float tileB[2][TILE_K * TILE_N]; // 8 x 128
    int read = 0;
    int write = 1;

    
    float output[4][FRAG_SIZE/2][FRAG_SIZE/2] = {0}; 

    
    // preprocess address calculations for SMEM -> reg and reg -> GMEM
    int tile_offset_m = warp_id / WARP_PER_ROW * WARP_TILE_M + lane_id / T_PER_WTILE_ROW * FRAG_SIZE/2;
    int tile_offset_n = warp_id % WARP_PER_ROW * WARP_TILE_N + lane_id % T_PER_WTILE_ROW * FRAG_SIZE/2;
    
    int n_blks = (N + TILE_N - 1) / TILE_N;  
    int mt = bid / n_blks * TILE_M; // m tile idx
    int nt = bid % n_blks * TILE_N; // n tile idx
    int idx = tid * 4;

    // Load first iteration tiles from GMEM to SMEM
    bool maskA_0 = mt + idx / TILE_K < M && idx % TILE_K < K;
    bool maskB_0 = idx / TILE_N < K && nt + idx % TILE_N < N;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 firstA = maskA_0 ? __ldcg((float4*)&A[(mt + idx / TILE_K) * K + (idx % TILE_K)]) : zero; // (m index) * K + (k index)
    tileA[0][(idx % TILE_K) * TILE_M + (idx / TILE_K)]     = firstA.x; // transpose A tile while storing
    tileA[0][(idx % TILE_K + 1) * TILE_M + (idx / TILE_K)] = firstA.y;
    tileA[0][(idx % TILE_K + 2) * TILE_M + (idx / TILE_K)] = firstA.z;
    tileA[0][(idx % TILE_K + 3) * TILE_M + (idx / TILE_K)] = firstA.w;
    *(float4*)&tileB[0][idx] = maskB_0 ? __ldcg((float4*)&B[(idx / TILE_N) * N + (nt + idx % TILE_N)]) : zero;

    __syncthreads();

    for (int kt = 0; kt < K; kt += TILE_K) {
        // Begin the load for the next iteration if it exists
        bool maskA = mt + idx / TILE_K < M && (kt + TILE_K) + idx % TILE_K < K && (kt + TILE_K) < K;
        bool maskB = (kt + TILE_K) + idx / TILE_N < K && nt + idx % TILE_N < N && (kt + TILE_K) < K;

        // start GMEM load for next iterations 
        float4 nextA = maskA ? __ldcg((float4*)&A[(mt + idx / TILE_K) * K + ((kt + TILE_K) + idx % TILE_K)]) : zero; // (m index) * K + (k index)
        float4 nextB = maskB ? __ldcg((float4*)&B[((kt + TILE_K) + idx / TILE_N) * N + (nt + idx % TILE_N)]) : zero;
                
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float fragA[FRAG_SIZE];
            float fragB[FRAG_SIZE];

            // Load from SMEM to registers
            *(float4*)&fragA[0] = *(float4*)&tileA[read][(k) * TILE_M + (tile_offset_m)];
            *(float4*)&fragA[4] = *(float4*)&tileA[read][(k) * TILE_M + (tile_offset_m + WARP_TILE_M/2)];
           
            *(float4*)&fragB[0] = *(float4*)&tileB[read][(k) * TILE_N + (tile_offset_n)];
            *(float4*)&fragB[4] = *(float4*)&tileB[read][(k) * TILE_N + (tile_offset_n + WARP_TILE_N/2)];

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
        
        tileA[write][(idx % TILE_K) * TILE_M + (idx / TILE_K)]     = nextA.x; // transpose A tile while storing
        tileA[write][(idx % TILE_K + 1) * TILE_M + (idx / TILE_K)] = nextA.y;
        tileA[write][(idx % TILE_K + 2) * TILE_M + (idx / TILE_K)] = nextA.z;
        tileA[write][(idx % TILE_K + 3) * TILE_M + (idx / TILE_K)] = nextA.w;


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
            int tile_coord_m = tile_offset_m + tile / 2 * WARP_TILE_M/2 + m;
            int tile_coord_n = tile_offset_n + tile % 2 * WARP_TILE_N/2;
            __stwb((float4*)&C[(mt + tile_coord_m) * N + (nt + tile_coord_n)], *(float4*)&output[tile][m]); 
        }
    }
}

torch::Tensor gemm_transposed(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BDIM);
    dim3 grid(((t.M + TILE_M - 1) / TILE_M) * ((t.N + TILE_N - 1) / TILE_N)); 

    gemm_transposed_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

