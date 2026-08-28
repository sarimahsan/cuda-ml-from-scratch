// This monolithic file has been split into dedicated modular files:
// - src/linear.cu (Linear / GEMM forward and backward)
// - src/activations.cu (ReLU, GELU, Sigmoid, LeakyReLU activations)
// - src/softmax_loss.cu (Softmax & Cross-Entropy loss reduction)
// - src/optimizers.cu (SGD with Momentum & Adam)
// - src/mlp.cu (MLPCUDA class coordinator)
