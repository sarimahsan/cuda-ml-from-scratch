#include "../include/cnn.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    std::cout << "=======================================================\n";
    std::cout << "   CUDA ML From Scratch: Modular CNN (03_cnn)\n";
    std::cout << "=======================================================\n";

    const int N = 64;
    const int C_in = 1;
    const int H_in = 28;
    const int W_in = 28;
    const int C1_out = 8;
    const int C2_out = 16;
    const int hidden_dim = 64;
    const int num_classes = 10;
    const int num_steps = 100;

    std::cout << "[INFO] Initializing Standalone CUDACNN Architecture:\n";
    std::cout << "  - Input: [" << N << ", " << C_in << ", " << H_in << ", " << W_in << "]\n";
    std::cout << "  - Conv1: 3x3 (pad 1, stride 1) -> [" << N << ", " << C1_out << ", 28, 28] -> ReLU\n";
    std::cout << "  - MaxPool1: 2x2 (stride 2) -> [" << N << ", " << C1_out << ", 14, 14]\n";
    std::cout << "  - Conv2: 3x3 (pad 1, stride 1) -> [" << N << ", " << C2_out << ", 14, 14] -> ReLU\n";
    std::cout << "  - MaxPool2: 2x2 (stride 2) -> [" << N << ", " << C2_out << ", 7, 7] -> Flatten (784)\n";
    std::cout << "  - FC1: Linear(784 -> " << hidden_dim << ") -> ReLU\n";
    std::cout << "  - FC2: Linear(" << hidden_dim << " -> " << num_classes << ") -> Softmax Cross-Entropy\n\n";

    CUDACNN model(N, C_in, H_in, W_in, C1_out, C2_out, hidden_dim, num_classes, 0.02f, 0.9f);
    model.init_weights(1337);

    // Generate synthetic dummy batch
    std::vector<float> h_X(N * C_in * H_in * W_in);
    std::vector<float> h_targets(N * num_classes, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);

    for (size_t i = 0; i < h_X.size(); ++i) {
        h_X[i] = norm_dist(rng);
    }
    for (int i = 0; i < N; ++i) {
        int c = class_dist(rng);
        h_targets[i * num_classes + c] = 1.0f;
    }

    std::cout << "[INFO] Running " << num_steps << " Training Iterations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= num_steps; ++step) {
        model.forward(h_X.data());
        float loss = model.backward_and_step(h_targets.data());

        if (step % 20 == 0 || step == 1) {
            std::cout << "  Step [" << step << "/" << num_steps << "] | Categorical Cross-Entropy Loss: " << loss << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "\n[SUCCESS] Completed " << num_steps << " training steps in "
              << elapsed_ms << " ms (" << (elapsed_ms / num_steps) << " ms/step, "
              << (num_steps * N / (elapsed_ms / 1000.0)) << " samples/sec)\n";

    return 0;
}
