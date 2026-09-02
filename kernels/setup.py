import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Paths
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
src_dir = os.path.join(os.path.dirname(__file__), "src")
include_dir = os.path.join(os.path.dirname(__file__), "include")
common_dir = os.path.join(os.path.dirname(__file__), "..", "00_common", "include")

sources = [
    os.path.join(csrc_dir, "binding.cpp"),
    os.path.join(src_dir, "gemm.cu"),
    os.path.join(src_dir, "convolution.cu"),
    os.path.join(src_dir, "reduction.cu"),
    os.path.join(src_dir, "softmax.cu"),
    os.path.join(src_dir, "normalization.cu"),
    os.path.join(src_dir, "activation.cu"),
    os.path.join(src_dir, "pooling.cu"),
    os.path.join(src_dir, "elementwise.cu"),
    os.path.join(src_dir, "optimizers.cu"),
]


extra_compile_args = {
    "cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas=-v",
        "-std=c++17",
    ],
}

setup(
    name="cuda_kernels_engine",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernels_engine",
            sources=sources,
            include_dirs=[include_dir, common_dir],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
