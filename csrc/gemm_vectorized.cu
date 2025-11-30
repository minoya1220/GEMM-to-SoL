#include <torch/extension.h>
#include "gemm_common.h"

constexpr int BLK_M = 128; // block sizes along each dimension
constexpr int BLK_N = 128; // small K and larger M and N boosts arithmetic intensity
constexpr int BLK_K = 8; 
constexpr int WARP_SIZE = 32;
constexpr int WARP_TILE_W = BLK_N / 4;
constexpr int WARP_TILE_H = BLK_M / 2;

//   REWRITE WITH PTX LOAD STORE INSTEAD OF FLOAT4
__global__ void gemm_vectorized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int blk_id = blockIdx.x;
    int blk_dim = blockDim.x;
    int thr_id = threadIdx.x;
    int warp_id = thr_id / WARP_SIZE; 
    int lane_id = thr_id % WARP_SIZE; // thread idx within warp
    int m_blks = (M + BLK_M - 1) / BLK_M;
    int n_blks = (N + BLK_N - 1) / BLK_N;
    
    
    // Layouts
    // A (M, K) : (K, 1)
    // B (K, N) : (N, 1)
    
    __shared__ float4 tileA4[BLK_M * BLK_K / 4]; // 128 x 8
    __shared__ float4 tileB4[BLK_K * BLK_N / 4]; // 8 x 128 

    float* tileA = (float*)tileA4;

    float4* A4 = (float4*)A;
    float4* B4 = (float4*)B;
    float4* C4 = (float4*)C;
    
    __syncthreads();
    
    float output[4][4][4] = {0};

    // precalculated offsets for address calculation
    int bmA = blk_id / n_blks * BLK_M; 
    int tmA = thr_id / (BLK_K / 4); // thread offset along the m dimension for tile A
    int tkA = thr_id % (BLK_K / 4);

    int bnB = blk_id % n_blks * (BLK_N / 4); // block offset along the n dimension for tile B
    int tnB = thr_id % (BLK_N / 4);
    int tkB = thr_id / (BLK_N / 4);

    for (int kt = 0; kt < K; kt += BLK_K) {
         
        bool maskA = (bmA + tmA < M) && (kt/4 + tkA < K/4);
        tileA4[thr_id] = maskA ? A4[(bmA * K/4 + kt/4) + (tmA * K/4 + tkA)] : make_float4(0,0,0,0);

        bool maskB = (bnB + tnB < N/4) && (kt + tkB < K);      
        tileB4[thr_id] = maskB ? B4[(kt * N/4 + bnB) + (tkB * N/4 + tnB)] : make_float4(0,0,0,0); 
        // TODO: handle if mnk arent divisible by 4 and also profile a layout where warps are 4*8 instead of 8*4 layout in the tile division

        __syncthreads();
        
        
        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            
            float fragA[8]; // to get a full 128 bit load on this fragment our A tile would have to be M major (col major)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int offsetfA = warp_id / 4 * BLK_K * WARP_TILE_H + i * BLK_K + k ; // add wthr_id 
                fragA[i] = tileA[offsetfA];
                fragA[4 + i] = tileA[offsetfA + (WARP_TILE_H / 2) * BLK_K];
            }

            int offsetfB = k * (BLK_N / 4) + warp_id % 4 * (WARP_TILE_W / 4) + wthr_id % 4; 
            float4 fragB4[2]; 
            fragB4[0] = tileB4[offsetfB];
            fragB4[1] = tileB4[offsetfB + (WARP_TILE_W / 4 / 2)]; // + (WARP_TILE_W / 4 / 2) is half of a warp tile width in float4
            float* fragB = (float*)fragB4;
            
            
            // outer product of each vector and tile
            #pragma unroll
            for (int tile = 0; tile < 4; tile++) { 
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    #pragma unroll
                    for(int n = 0; n < 4; n++) {
                        output[tile][m][n] += fragA[tile % 2 * 4 + m] * fragB[tile / 2 * 4 + n];
                    }
                }
            }


        }



        __syncthreads(); 

    }

    float4* output4 = (float4*)output;
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            int m_addr = blk_id / n_blks + warp_id / 4 * WARP_TILE_H + wthr_id / 4 + tile / 2 * (WARP_TILE_H / 2) + m;
            int n_addr = blk_id % n_blks + warp_id % 4 * WARP_TILE_W + wthr_id % 4 + tile % 2 * (WARP_TILE_W / 2);
            C4[(m_addr) * (N/4) + (n_addr) / 4] = output4[tile * 4 + m]; // damn
        }
    }


}

torch::Tensor gemm_vectorized(torch::Tensor A, torch::Tensor B) {
    auto t = prep_tensors(A, B);

    dim3 block(256);
    dim3 grid(((t.M + BLK_M - 1) / BLK_M) * ((t.N + BLK_N - 1) / BLK_N)); 

    gemm_vectorized_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

