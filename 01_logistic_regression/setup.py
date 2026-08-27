import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Optimization flags for NVCC
nvcc_flags = ["-O3", "--use_fast_math", "-std=c++17"]

setup(
    name="cuda_logistic_regression",
    ext_modules=[
        CUDAExtension(
            name="cuda_logistic_regression",
            sources=[
                "csrc/binding.cpp",
                "csrc/logistic_regression_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
