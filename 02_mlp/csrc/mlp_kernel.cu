// This monolithic file has been split into dedicated modular files:
// - csrc/linear.cu (Linear / GEMM forward and backward)
// - csrc/activations.cu (ReLU, GELU, Sigmoid activations)
// - csrc/softmax_loss.cu (Numerically stable Softmax & Cross-Entropy loss)
// - csrc/optimizers.cu (SGD with Momentum & Adam)
// - csrc/binding.cpp (Pybind11 extension interface)
