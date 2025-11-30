# GEMM to SoL
GEMM (General Matrix Multiply) to SoL (Speed of Light) are a collection of matrix multiplication kernels that build up optimizations upon each other to get as close as possible to the theoretical speed limit of matrix multiplication on gpus, the "speed of light."

### Naive Implementation
The most basic GEMM implementation is three nested for loops:
```c++
for(int m = 0; m < M; m++) {     // } these loops get parallelized in the kernel
    for(int n = 0; n < N; n++) { // }
        for(int k = 0; k < K; k++) {
            C[m][n] += A[m][k] * B[k][n];
        }
    }
}
```
Given input matrices A (dims: M,K) and B (dims: K,N) we can create an output matrix C (dims: M,N) where each C output element (m,n) we compute the dot product along the k dimension of A(m,...) and B(...,n).

In our naive implementation kernel we parallelize the outer two loops iterating over the M and N dimensions because each m and n outputs can be calculated without needing the output of any other element. If we were to parallelize along k we would have to add multiple outputs with same (m,n) together before we could get the final result. It would create a lot of overhead so we avoid parallelizing it unless the situation calls for it.

Naive kernel implementation:
```c++
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; // } parallelized loops are now thread indices
    int n = blockIdx.y * blockDim.y + threadIdx.y; // } 

    if (m >= M || n >= N) { 
        return;
    }
    
    float sum = 0; 
    for (int k = 0; k < K; k++) {
        sum += A[m*K + k] * B[k*N + n];
    }
    C[m*N + n] = sum;
}
```
There are a few differences between this and the original serial naive implementation. First, we have removed the two outer loops and replaced them with parallel threads for each m and n. There is now boundary checking incase there are more threads than elements. We also have a sum variable and 1D array indexing instead of 2D. Using a sum variable is important because it ensures that we only write to global memory once and use a register for storing our intermediate computation for efficiency.  Since CUDA doesnt support 2D memory accesses for global memory we also have to flatten our 2D index into 1D. We asssume a row major memory layout so to traverse a row we add can increment by 1 and to traverse a column we increment by the row size.

% NAIVE SPEED HERE %

### Tiled GEMM

### Vectorized GEMM