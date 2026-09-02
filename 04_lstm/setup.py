from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
kernels_dir = os.path.join(root_dir, "kernels")
common_dir = os.path.join(root_dir, "00_common", "include")

sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "input_gate.cu"),
    os.path.join(current_dir, "csrc", "forget_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_candidate_gate.cu"),
    os.path.join(current_dir, "csrc", "output_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_state.cu"),
    os.path.join(current_dir, "csrc", "fused_gates.cu"),
    os.path.join(current_dir, "csrc", "sequence.cu"),
    os.path.join(kernels_dir, "src", "gemm.cu"),
    os.path.join(kernels_dir, "src", "softmax.cu"),
    os.path.join(kernels_dir, "src", "reduction.cu"),
    os.path.join(kernels_dir, "src", "elementwise.cu"),
    os.path.join(kernels_dir, "src", "optimizers.cu"),
]

setup(
    name="cuda_lstm",
    ext_modules=[
        CUDAExtension(
            name="cuda_lstm",
            sources=sources,
            include_dirs=[os.path.join(kernels_dir, "include"), common_dir],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

