import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['CUDAHOSTCXX'] = 'g++'


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm',
    ext_modules=[
        CUDAExtension(
            name='gemm',
            sources=[
                'csrc/1_gemm_naive.cu',
                'csrc/2_gemm_tiled.cu',
                'csrc/3_gemm_register_blocked.cu',
                'csrc/4_gemm_warptiled.cu',
                'csrc/5_gemm_vectorized.cu',
                'csrc/6_gemm_double_buffered.cu',
                'csrc/7_gemm_transposed.cu',
                'csrc/8_gemm_swizzled.cu',
                'csrc/bindings.cpp'
            ],
            extra_compile_args={
                'nvcc': [
                    '-gencode=arch=compute_75,code=sm_75',  # T4 only
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}  
)