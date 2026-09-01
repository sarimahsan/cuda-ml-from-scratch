from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "input_gate.cu"),
    os.path.join(current_dir, "csrc", "forget_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_candidate_gate.cu"),
    os.path.join(current_dir, "csrc", "output_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_state.cu"),
    os.path.join(current_dir, "csrc", "fused_gates.cu"),
    os.path.join(current_dir, "csrc", "linear.cu"),
    os.path.join(current_dir, "csrc", "softmax_loss.cu"),
    os.path.join(current_dir, "csrc", "optimizers.cu"),
    os.path.join(current_dir, "csrc", "sequence.cu"),
]

setup(
    name="cuda_lstm",
    ext_modules=[
        CUDAExtension(
            name="cuda_lstm",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
