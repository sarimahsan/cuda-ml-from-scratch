from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "conv2d.cu"),
    os.path.join(current_dir, "csrc", "pool.cu"),
    os.path.join(current_dir, "csrc", "linear.cu"),
    os.path.join(current_dir, "csrc", "activations.cu"),
    os.path.join(current_dir, "csrc", "softmax_loss.cu"),
    os.path.join(current_dir, "csrc", "optimizers.cu"),
]

setup(
    name="cuda_cnn",
    version="0.1.0",
    author="Sarim Ahsan",
    description="Convolutional Neural Network (CNN) with modular CUDA C++ kernels",
    ext_modules=[
        CUDAExtension(
            name="cuda_cnn",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
