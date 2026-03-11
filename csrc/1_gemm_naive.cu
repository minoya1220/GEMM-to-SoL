#include <torch/extension.h>
#include "gemm_common.h"

constexpr int NUM_THREADS = 256;

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id

    if (tid >= M * N) { // boundary check
        return;
    }

    // For a row major matrix which is size MxN, element (row, col) is at
    // a true array index of: row * N + col. The end result is an array in
    // which the rows of the matrix are laid out one after another. 
    // In this array, to move up to the next col idx we add 1 and to 
    // move to the next row idx we add N.
    // 
    // We have to do this because we need the row and col indices separately for 
    // accessing the correct elements in the A & B input matrices.
    // To do the opposite conversion using / and % with the row width (N)
    // to create row and col indices.
    int m = tid / N; // row 
    int n = tid % N; // col
    
    float accum = 0;
    for (int k = 0; k < K; k++) {
        // compute dot product by iterating along k
        accum += A[m * K + k] * B[k * N + n]; 
    }
    // write back result
    C[tid] = accum; // equivalent to C[m * N + n] 
}

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    // creates output tensor, checks dtype, contiguity, gets pointers, and gets M, N, K
    auto t = prep_tensors(A, B); // check

    dim3 block(NUM_THREADS);
    dim3 grid((t.M * t.N + NUM_THREADS - 1) / NUM_THREADS); 
    // (A + B - 1) / B is ceiling division

    gemm_naive_kernel<<<grid, block>>>(t.A, t.B, t.C, t.M, t.N, t.K);
    cudaDeviceSynchronize();
    
    return t.C_tensor;

}

