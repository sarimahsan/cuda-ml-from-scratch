#include "../include/mlp.cuh"
#include "../../00_common/include/cuda_utils.cuh"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

// Generates synthetic multi-class classification dataset
void generate_multiclass_data(
    std::vector<float>& X,
    std::vector<float>& y,
    int N,
    int D,
    int C,
    unsigned int seed = 42
) {
    X.resize(N * D);
    y.resize(N);

    std::mt19937 gen(seed);
    std::normal_distribution<float> feat_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, C - 1);

    // Generate cluster centers for each class
    std::vector<std::vector<float>> centers(C, std::vector<float>(D));
    std::uniform_real_distribution<float> center_dist(-2.5f, 2.5f);
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < D; ++d) {
            centers[c][d] = center_dist(gen);
        }
    }

    for (int i = 0; i < N; ++i) {
        int label = class_dist(gen);
        y[i] = static_cast<float>(label);
        for (int d = 0; d < D; ++d) {
            X[i * D + d] = centers[label][d] + feat_dist(gen) * 0.8f;
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=======================================================\n";
    std::cout << "          CUDA ML Models: Module 02 - MLP              \n";
    std::cout << "=======================================================\n\n";

    // Hyperparameters & Architecture Configuration
    const int NUM_SAMPLES     = 60000;  // 60k samples
    const int INPUT_DIM       = 64;     // 64 input features
    const int HIDDEN_DIM      = 128;    // 128 hidden neurons (Layer 1)
    const int NUM_CLASSES     = 10;     // 10 output classes (Softmax)
    const int BATCH_SIZE      = 256;    // Mini-batch size
    const float LEARNING_RATE = 0.05f;  // Learning rate
    const float MOMENTUM      = 0.9f;   // SGD Momentum
    const int EPOCHS          = 20;

    std::cout << "[INFO] Network Architecture: [" << INPUT_DIM << " -> " 
              << HIDDEN_DIM << " (ReLU) -> " << NUM_CLASSES << " (Softmax)]\n";
    std::cout << "[INFO] Dataset Configuration:\n";
    std::cout << "       Total Samples (N): " << NUM_SAMPLES << "\n";
    std::cout << "       Input Dim (D_in):  " << INPUT_DIM << "\n";
    std::cout << "       Hidden Dim (H):    " << HIDDEN_DIM << "\n";
    std::cout << "       Output Dim (C):    " << NUM_CLASSES << "\n";
    std::cout << "       Batch Size:        " << BATCH_SIZE << "\n";
    std::cout << "       Epochs:            " << EPOCHS << "\n";
    std::cout << "       Learning Rate:     " << LEARNING_RATE << "\n";
    std::cout << "       SGD Momentum:      " << MOMENTUM << "\n\n";

    // Split: 80% Train, 20% Test
    int N_train = static_cast<int>(NUM_SAMPLES * 0.8f);
    int N_test  = NUM_SAMPLES - N_train;

    std::vector<float> X_all, y_all;
    generate_multiclass_data(X_all, y_all, NUM_SAMPLES, INPUT_DIM, NUM_CLASSES, 42);

    const float* X_train = X_all.data();
    const float* y_train = y_all.data();
    const float* X_test  = X_all.data() + (N_train * INPUT_DIM);
    const float* y_test  = y_all.data() + N_train;

    // Initialize CUDA MLP Model
    MLPCUDA model(BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, LEARNING_RATE, MOMENTUM);

    GpuTimer timer;
    GpuTimer total_timer;

    std::cout << "-------------------------------------------------------\n";
    std::cout << std::setw(8)  << "Epoch"
              << std::setw(15) << "CE Loss"
              << std::setw(15) << "Train Acc"
              << std::setw(16) << "Time (ms)"
              << "\n";
    std::cout << "-------------------------------------------------------\n";

    int num_batches = (N_train + BATCH_SIZE - 1) / BATCH_SIZE;

    total_timer.start();

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        timer.start();
        float epoch_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            int offset = b * BATCH_SIZE;
            int cur_batch = std::min(BATCH_SIZE, N_train - offset);

            // 1. Transfer batch to GPU
            model.load_batch(X_train + offset * INPUT_DIM, y_train + offset, cur_batch);

            // 2. Forward pass: X -> Z1 -> A1 (ReLU) -> Z2 -> A2 (Softmax)
            model.forward(cur_batch);

            // 3. Compute Cross-Entropy Loss
            float batch_loss = model.compute_loss(cur_batch);
            epoch_loss += batch_loss * cur_batch;

            // 4. Backward pass: dZ2 -> dW2, db2 -> dA1 -> dZ1 -> dW1, db1
            model.backward(cur_batch);

            // 5. Optimizer step (SGD + Momentum)
            model.step();
        }

        float epoch_time = timer.stop();
        epoch_loss /= N_train;

        if (epoch % 2 == 0 || epoch == 1 || epoch == EPOCHS) {
            float train_acc = model.evaluate(X_train, y_train, std::min(5000, N_train));
            std::cout << std::setw(8)  << epoch
                      << std::setw(15) << std::fixed << std::setprecision(4) << epoch_loss
                      << std::setw(14) << std::fixed << std::setprecision(2) << train_acc << "%"
                      << std::setw(15) << std::fixed << std::setprecision(2) << epoch_time
                      << "\n";
        }
    }

    float total_ms = total_timer.stop();
    std::cout << "-------------------------------------------------------\n";
    std::cout << "[INFO] Total Training Time: " << total_ms << " ms ("
              << (total_ms / EPOCHS) << " ms/epoch)\n\n";

    // Test Evaluation
    std::cout << "[INFO] Evaluating on Test Set (" << N_test << " samples)...\n";
    float test_acc = model.evaluate(X_test, y_test, N_test);
    std::cout << "[RESULT] Final Test Accuracy: " << std::fixed << std::setprecision(2)
              << test_acc << " %\n";

    std::cout << "\n=======================================================\n";
    std::cout << "                  Training Completed!                  \n";
    std::cout << "=======================================================\n";

    return 0;
}
