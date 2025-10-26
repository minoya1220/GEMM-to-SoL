from gc import enable
import torch
from torch.testing import assert_close
import gemm

def test(M, N, K):
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')

    C = A @ B

    assert_close(gemm.gemm_naive(A, B), C, rtol=1e-4, atol=1e-3)
    
    print("test passed")
    


def benchmark(M, N, K):
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')

    # warmup
    for _ in range(0,5): 
        gemm.gemm_naive(A, B)
    
    times = []
    for _ in range(0,10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        gemm.gemm_naive()
        end.record()

        times.append(start.elapsed_time(end))
    
    times = torch.Tensor(times)
    tflops = 2 * M * N * K / 1e12
    print(f"Results:")
    print(f"    mean: {times.mean()}")
    print(f"    std: {times.std()}")
    print(f"    max: {times.max()}")
    print(f"    min: {times.min()}")
    print(f"    avg_tflops/s: { tflops / times.mean() }")
    


        

if __name__ == "__main__":
    M = N = 1024
    K = 2048
    test(M, N, K)
    benchmark(M, N, K)