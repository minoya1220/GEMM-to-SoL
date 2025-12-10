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
                'csrc/gemm_naive.cu',
                'csrc/gemm_tiled.cu',
                'csrc/gemm_coarsened.cu',
                'csrc/gemm_vectorized.cu',
                'csrc/gemm_double_buffered.cu',
                'csrc/gemm_swizzled.cu',
                'csrc/gemm_transposed.cu',
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