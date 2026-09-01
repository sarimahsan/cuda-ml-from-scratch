#include "../include/lstm.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "=================================================================\n";
    std::cout << "  🚀 Pure CUDA C++ Modular LSTM: Standalone Sequence Benchmark   \n";
    std::cout << "=================================================================\n";

    LSTMConfig cfg;
    cfg.input_dim = 64;      // D
    cfg.hidden_dim = 128;    // H
    cfg.vocab_size = 64;     // V
    cfg.seq_len = 32;        // T
    cfg.batch_size = 32;     // N
    cfg.learning_rate = 0.005f;
    cfg.clip_norm = 1.0f;
    cfg.use_fused_gate = true;

    std::cout << "• Configuration:\n";
    std::cout << "  - Input Dim (D)   : " << cfg.input_dim << "\n";
    std::cout << "  - Hidden Dim (H)  : " << cfg.hidden_dim << "\n";
    std::cout << "  - Vocab Size (V)  : " << cfg.vocab_size << "\n";
    std::cout << "  - Sequence Len (T): " << cfg.seq_len << "\n";
    std::cout << "  - Batch Size (N)  : " << cfg.batch_size << "\n";
    std::cout << "  - Learning Rate   : " << cfg.learning_rate << "\n";
    std::cout << "-----------------------------------------------------------------\n";

    try {
        CUDALSTM lstm(cfg);
        lstm.init_weights(42);

        // Generate synthetic sequence data: one-hot or random vectors
        int total_inputs = cfg.seq_len * cfg.batch_size * cfg.input_dim;
        int total_targets = cfg.seq_len * cfg.batch_size;

        std::vector<float> h_X(total_inputs);
        std::vector<int> h_targets(total_targets);

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> target_dist(0, cfg.vocab_size - 1);

        for (int i = 0; i < total_inputs; ++i) h_X[i] = x_dist(rng);
        for (int i = 0; i < total_targets; ++i) h_targets[i] = target_dist(rng);

        GpuTimer timer;

        std::cout << "\n>>> Running 20 Training Iterations (Forward + BPTT + Adam Step)...\n\n";
        std::cout << "  Step  |   Loss   | Iteration Time (ms) | Speed (tokens/sec)\n";
        std::cout << " -------|----------|---------------------|--------------------\n";

        for (int step = 1; step <= 20; ++step) {
            timer.start();
            float loss = lstm.forward(h_X.data(), h_targets.data());
            lstm.backward();
            lstm.step();
            float ms = timer.stop();

            float tokens_per_sec = (cfg.seq_len * cfg.batch_size) / (ms / 1000.0f);

            std::cout << "  " << std::setw(5) << step << " | "
                      << std::fixed << std::setprecision(4) << std::setw(8) << loss << " | "
                      << std::setprecision(2) << std::setw(19) << ms << " | "
                      << std::setprecision(1) << std::setw(18) << tokens_per_sec << "\n";
        }

        std::cout << "\n=================================================================\n";
        std::cout << " [SUCCESS] Pure CUDA LSTM training loop executed with 0 errors! \n";
        std::cout << "=================================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
