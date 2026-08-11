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
constexpr int NUM_TILES = 4;


__device__ __forceinline__ int swizzleA(int row, int col){
    return row * TILE_M + (col ^ (row << 2));
}

__global__ void gemm_final_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    __shared__ __align__(16) float tileA[2][TILE_K * TILE_M]; // 128 x 8
    __shared__ __align__(16) float tileB[2][TILE_K * TILE_N]; // 8 x 128
    int read = 0;
    int write = 1;

    
    float output[NUM_TILES * FRAG_SIZE/2 * FRAG_SIZE/2] = {0}; // if we stride our output tiles well be able to coalesce our store

    
    // preprocess address calculations for SMEM -> reg and reg -> GMEM
    int tile_offset_m = warp_id / WARP_PER_ROW * WARP_TILE_M + lane_id / T_PER_WTILE_ROW * FRAG_SIZE/2;
    int tile_offset_n = warp_id % WARP_PER_ROW * WARP_TILE_N + lane_id % T_PER_WTILE_ROW * FRAG_SIZE/2;
    
    int num_blks_n = (N + TILE_N - 1) / TILE_N;  
    int mt = bid / num_blks_n * TILE_M; // m tile idx
    int nt = bid % num_blks_n * TILE_N; // n tile idx
    int idx = tid * 4;

    // Load first iteration tiles from GMEM to SMEM
    bool maskB_0 = idx / TILE_N < K && nt + idx % TILE_N < N;
    bool maskA_0 = mt + idx / TILE_K < M && idx % TILE_K < K;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    float4 firstA = maskA_0 ? __ldcg((float4*)&A[(mt + idx / TILE_K) * K + (idx % TILE_K)]) : zero; // (m index) * K + (k index)
    tileA[0][swizzleA((idx % TILE_K),     (idx / TILE_K))] = firstA.x; // transpose A while storing
    tileA[0][swizzleA((idx % TILE_K + 1), (idx / TILE_K))] = firstA.y;
    tileA[0][swizzleA((idx % TILE_K + 2), (idx / TILE_K))] = firstA.z;
    tileA[0][swizzleA((idx % TILE_K + 3), (idx / TILE_K))] = firstA.w;

    *(float4*)&tileB[0][idx] = maskB_0 ? __ldcg((float4*)&B[(idx / TILE_N) * N + (nt + idx % TILE_N)]) : zero;

    __syncthreads();

    for (int kt = 0; kt < K; kt += TILE_K) {
        // Begin the load for the next iteration if it exists
        bool maskB = (kt + TILE_K) + idx / TILE_N < K && nt + idx % TILE_N < N && (kt + TILE_K) < K;
        bool maskA = mt + idx / TILE_K < M && (kt + TILE_K) + idx % TILE_K < K && (kt + TILE_K) < K;

        // start GMEM load for next iterations
        float4 nextB = maskB ? __ldcg((float4*)&B[((kt + TILE_K) + idx / TILE_N) * N + (nt + idx % TILE_N)]) : zero;
        float4 nextA = maskA ? __ldcg((float4*)&A[(mt + idx / TILE_K) * K + ((kt + TILE_K) + idx % TILE_K)]) : zero; // (m index) * K + (k index)

        
        int reg_read = 0;
        int reg_write = 1;
        float currA_lo[FRAG_SIZE/2];
        float currA_hi[FRAG_SIZE/2];
        float currB_lo[FRAG_SIZE/2];
        float currB_hi[FRAG_SIZE/2];


        *(float4*)&currA_lo[0] = *(float4*)&tileA[read][swizzleA(0, tile_offset_m)];
        *(float4*)&currA_hi[0] = *(float4*)&tileA[read][swizzleA(0, tile_offset_m + WARP_TILE_M/2)];
           
        *(float4*)&currB_lo[0] = *(float4*)&tileB[read][(0) * TILE_N + (tile_offset_n)];
        *(float4*)&currB_hi[0] = *(float4*)&tileB[read][(0) * TILE_N + (tile_offset_n + WARP_TILE_N/2)];

        #pragma unroll 
        for (int k = 0; k < TILE_K; k++) {
            int kv = k;
            asm("" : "+r"(kv)); 

            float nextA_lo[FRAG_SIZE/2];
            float nextA_hi[FRAG_SIZE/2];
            float nextB_lo[FRAG_SIZE/2];
            float nextB_hi[FRAG_SIZE/2];

            // Load from SMEM to registers
            bool r_mask = k < TILE_K - 1;
            *(float4*)&nextA_lo[0] = r_mask ? *(float4*)&tileA[read][swizzleA(kv + 1, tile_offset_m)] : zero;
            *(float4*)&nextA_hi[0] = r_mask ? *(float4*)&tileA[read][swizzleA(kv + 1, tile_offset_m + WARP_TILE_M/2)] : zero;
            
            *(float4*)&nextB_lo[0] = r_mask ? *(float4*)&tileB[read][(k + 1) * TILE_N + (tile_offset_n)] : zero;
            *(float4*)&nextB_hi[0] = r_mask ? *(float4*)&tileB[read][(k + 1) * TILE_N + (tile_offset_n + WARP_TILE_N/2)] : zero;
            
            // compute outer product (matmul for our two fragments)
            // #pragma unroll
            // for (int tile = 0; tile < NUM_TILES; tile++) {
            //     #pragma unroll
            //     for (int m = 0; m < FRAG_SIZE/2; m++) {
            //         #pragma unroll
            //         for (int n = 0; n < FRAG_SIZE/2; n++) {
            //             output[tile * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2 + n] += 
            //                 fragA[reg_read * FRAG_SIZE + tile / 2 * FRAG_SIZE/2 + m] * fragB[reg_read * FRAG_SIZE + tile % 2 * FRAG_SIZE/2 + n];
            //         }
            //     }
            // }
            float4 tmp;

            #pragma unroll
            for (int m = 0; m < FRAG_SIZE/2; m++) {
                #pragma unroll
                for (int n = 0; n < FRAG_SIZE/2; n++) {
                    output[0 * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2 + n] += 
                        currA_lo[m] * currB_lo[n];
                }
            }
            #pragma unroll
            for (int m = 0; m < FRAG_SIZE/2; m++) {
                #pragma unroll
                for (int n = 0; n < FRAG_SIZE/2; n++) {
                    output[1 * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2 + n] += 
                        currA_hi[m] * currB_lo[n];
                
                }
            }

            tmp = *(float4*)&currB_lo;
            *(float4*)&currB_lo[0] = *(float4*)&nextB_lo[0];
            *(float4*)&nextB_lo[0] = tmp;

            #pragma unroll
            for (int m = 0; m < FRAG_SIZE/2; m++) {
                #pragma unroll
                for (int n = 0; n < FRAG_SIZE/2; n++) {
                    output[2 * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2 + n] += 
                        currA_hi[m] * currB_hi[n];
                
                }
            }

            tmp = *(float4*)&currA_hi;
            *(float4*)&currA_hi[0] = *(float4*)&nextA_hi[0];
            *(float4*)&nextA_hi[0] = tmp;

            #pragma unroll
            for (int m = 0; m < FRAG_SIZE/2; m++) {
                #pragma unroll
                for (int n = 0; n < FRAG_SIZE/2; n++) {
                    output[3 * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2 + n] += 
                        currA_lo[m] * currB_hi[n];
                
                }
            }
            
            tmp = *(float4*)&currA_lo;
            *(float4*)&currA_lo[0] = *(float4*)&nextA_lo[0];
            *(float4*)&nextA_lo[0] = tmp;
            
            tmp = *(float4*)&currB_hi;
            *(float4*)&currB_hi[0] = *(float4*)&nextB_hi[0];
            *(float4*)&nextB_hi[0] = tmp;
            
        }

        


        // Store next iterations GMEM load into SMEM
        tileA[write][swizzleA((idx % TILE_K), (idx / TILE_K))]     = nextA.x; // transpose A tile while storing
        tileA[write][swizzleA((idx % TILE_K + 1), (idx / TILE_K))] = nextA.y;
        tileA[write][swizzleA((idx % TILE_K + 2), (idx / TILE_K))] = nextA.z;
        tileA[write][swizzleA((idx % TILE_K + 3), (idx / TILE_K))] = nextA.w;

        *(float4*)&tileB[write][idx] = nextB;

        // swap read and write buffers
        read ^= 1; 
        write ^= 1;

        __syncthreads(); // try moving this to the beginning of the loop

    }
    // write output to GMEM
    #pragma unroll
    for (int tile = 0; tile < NUM_TILES; tile++) {    
        #pragma unroll
        for (int m = 0; m < FRAG_SIZE/2; m++) { // add boundary check
            int tile_coord_m = tile_offset_m + tile / 2 * WARP_TILE_M/2 + m;
            int tile_coord_n = tile_offset_n + tile % 2 * WARP_TILE_N/2;
            __stwb((float4*)&C[(mt + tile_coord_m) * N + (nt + tile_coord_n)], *(float4*)&output[tile * NUM_TILES * FRAG_SIZE/2 + m * FRAG_SIZE/2]); 
        }
    }
}

torch::Tensor gemm_final(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(BDIM);
    dim3 grid(((t.M + TILE_M - 1) / TILE_M) * ((t.N + TILE_N - 1) / TILE_N)); 

    gemm_final_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}
