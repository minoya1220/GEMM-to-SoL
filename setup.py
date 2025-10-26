from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm',
    ext_modules=[
        CUDAExtension(
            name='gemm',
            sources=[
                'csrc/gemm_naive.cu',
                'csrc/bindings.cpp' 
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}  # Move this here
)