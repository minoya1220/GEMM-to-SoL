import torch
from torch.testing import assert_close
from functools import partial
from typing import Callable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gemm
import pynvml
import threading
import time
# from torch.profiler import profile, ProfilerActivity

def test(A, B, func: Callable):
    C = A @ B
    C_act = func()
    C_act2 = func()

    assert_close(C_act, C_act2)
    print("determinate results")
    assert_close(C_act, C, rtol=1e-4, atol=1e-3)
    print("test passed")


def benchmark(M, N, K, func: Callable, warmup=20, iters=100, print_results: bool=False):
    # warmup
    for _ in range(0,warmup): 
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(0,iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        func()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = torch.Tensor(times)
    tflops = 2 * M * N * K / 1e12 
    if print_results:
        print(f"Results:")
        print(f"    mean: {times.mean()}")
        print(f"    std: {times.std()}")
        print(f"    max: {times.max()}")
        print(f"    min: {times.min()}")
        print(f"    avg_tflops/s: { tflops / times.mean() * 1000 }") # converts from flops/ms to flops/s
    
def benchmark_clock(M, N, K, func: Callable, warmup=20, iters=100, print_results: bool=False):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    clock_end = threading.Event()
    def sample_clocks():
        while not clock_end.wait(0.061):
            clocks.append(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
    clock_thread = threading.Thread(target=sample_clocks)        
    
    # warmup
    for _ in range(0,warmup): 
        func()
    torch.cuda.synchronize()

    clock_thread.start()
    clocks = []
    times = []
    for _ in range(0,iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        func()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    clock_end.set()
    clock_thread.join()
    pynvml.nvmlShutdown()
    
    avg_clock = torch.Tensor(clocks).mean()
    times = torch.Tensor(times)
    tflops = 2 * M * N * K / 1e12 
    theoretical_flops_per_second = 64 * 40 * 2 * avg_clock / 1e6 # 64 cores per sm, 40 sms, 2 flops per fma
    actual_flops_per_second = tflops / times.mean() * 1000 # * 1000 for ms to s
    compute_throughput = actual_flops_per_second / theoretical_flops_per_second * 100
    
    if print_results:
        print(f"Results:")
        print(f"    mean: {times.mean()}")
        print(f"    std: {times.std()}")
        print(f"    max: {times.max()}")
        print(f"    min: {times.min()}")
        print(f"    avg_tflops/s: { actual_flops_per_second }") 
        print(f"    avg_clock: {avg_clock}")
        print(f"    compute throughput: {compute_throughput}")

def bench_all(M, N, K, A, B, print_results: bool=True):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    base_clock = pynvml.nvmlDeviceGetDefaultApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
    if (clock != base_clock):
        print(f"Must lock clock to base clock ({base_clock}). Run 'sudo nvidia-smi -lgc {base_clock}'")
    pynvml.nvmlShutdown()
    
    benchmark(M, N, K, partial(gemm.gemm_naive, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_tiled, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_register_blocked, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_vectorized, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_double_buffered, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_transposed, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(gemm.gemm_swizzled, A, B), warmup=5, iters=10, print_results=print_results)
    benchmark(M, N, K, partial(torch.matmul, A, B), warmup=5, iters=10, print_results=print_results)
    

def plot(func: Callable):
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

    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')

    func = gemm.gemm_warptiled
    func2 = torch.matmul
    # func(A, B)
    # test(A, B, partial(func, A, B))
    # plot(func)
    
    # time.sleep(100)
    # benchmark_clock(M, N, K, partial(func2, A, B))
    # time.sleep(100)
    benchmark(M, N, K, partial(func, A, B), warmup=0,iters=1,print_results=True)
    # benchmark_clock(M, N, K, partial(func2, A, B), print_results=True)
    



