import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
kernels_dir = os.path.join(root_dir, "kernels")
common_dir = os.path.join(root_dir, "00_common", "include")

sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "rnn_cell.cu"),
    os.path.join(current_dir, "csrc", "sequence.cu"),
    os.path.join(kernels_dir, "src", "gemm.cu"),
    os.path.join(kernels_dir, "src", "softmax.cu"),
    os.path.join(kernels_dir, "src", "reduction.cu"),
    os.path.join(kernels_dir, "src", "elementwise.cu"),
    os.path.join(kernels_dir, "src", "optimizers.cu"),
]

setup(
    name="cuda_rnn",
    ext_modules=[
        CUDAExtension(
            name="cuda_rnn",
            sources=sources,
            include_dirs=[
                os.path.join(kernels_dir, "include"),
                common_dir,
                os.path.join(current_dir, "csrc"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
