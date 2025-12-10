import torch
from torch.testing import assert_close
from functools import partial
from typing import Callable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gemm

def test(A, B, func: Callable):
    C = A @ B
    C_act = func()
    C_act2 = func()

    

    assert_close(C_act, C_act2)
    print("determinate results")
    assert_close(C_act, C, rtol=1e-4, atol=1e-3)
    print("test passed")


def benchmark(M, N, K, func: Callable):
    # warmup
    for _ in range(0,20): 
        func()
    
    times = []
    for _ in range(0,100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        func()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        
    times = torch.Tensor(times)
    tflops = 2 * M * N * K / 1e12 
    print(f"Results:")
    print(f"    mean: {times.mean()}")
    print(f"    std: {times.std()}")
    print(f"    max: {times.max()}")
    print(f"    min: {times.min()}")
    print(f"    avg_tflops/s: { tflops / times.mean() * 1000 }") # converts from flops/ms to flops/s
    

def plot( func: Callable):
    fig, ax = plt.subplots(4, 4, figsize=(24, 24), dpi=120)
    
    def prepare_row(n, A, B):
        C = A @ B
        C_act = func(A, B)
        C_act2 = func(A, B)
        ax[n, 0].imshow(C.cpu())
        ax[n, 1].imshow(C_act.cpu())
        ax[n, 2].imshow((C-C_act).cpu())
        ax[n, 3].imshow((C_act - C_act2).cpu())

    prepare_row(0, torch.rand(32, 32, device='cuda'),    torch.rand(32, 32, device='cuda'))
    prepare_row(1, torch.eye(64, 64, device='cuda'),     torch.eye(64, 64, device='cuda'))
    prepare_row(2, torch.rand(64, 64, device='cuda'),    torch.rand(64, 64, device='cuda'))
    prepare_row(3, torch.randn(128, 128, device='cuda'), torch.randn(128, 128, device='cuda'))

    plt.tight_layout()
    plt.savefig('compare.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    M = N = 4096
    K = 4096

    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')

    # test(M, N, K, partial(gemm.gemm_naive, A, B))
    # benchmark(M, N, K, partial(gemm.gemm_naive, A, B))
    # plot(gemm.gemm_naive)

    func = gemm.gemm_double_buffered

    # func(A, B)
    # test(A, B, partial(func, A, B))
    benchmark(M, N, K, partial(func, A, B))
    # plot(func)
    

    # benchmark(M, N, K, partial(torch.matmul, A, B))
