#include "../include/logistic_regression.cuh"
#include "../../00_common/include/data_utils.cuh"
#include "../../00_common/include/cuda_utils.cuh"

#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "=======================================================\n";
    std::cout << "       CUDA ML Models: Module 01 - Logistic Regression  \n";
    std::cout << "=======================================================\n\n";

    // Hyperparameters & Dataset Configuration
    const int NUM_SAMPLES     = 100000; // 100k samples
    const int NUM_FEATURES    = 16;     // 16 input features
    const float LEARNING_RATE = 0.2f;
    const int EPOCHS          = 50;
    const int THREADS_PER_BLK = 256;

    std::cout << "[INFO] Generating synthetic dataset:\n";
    std::cout << "       Samples (N):  " << NUM_SAMPLES << "\n";
    std::cout << "       Features (D): " << NUM_FEATURES << "\n";
    std::cout << "       Epochs:       " << EPOCHS << "\n";
    std::cout << "       LR (eta):     " << LEARNING_RATE << "\n\n";

    // Split: 80% Train, 20% Test
    int N_train = static_cast<int>(NUM_SAMPLES * 0.8f);
    int N_test  = NUM_SAMPLES - N_train;

    std::vector<float> X_all, y_all;
    ml_utils::generate_synthetic_data(X_all, y_all, NUM_SAMPLES, NUM_FEATURES, 42);

    const float* X_train = X_all.data();
    const float* y_train = y_all.data();
    const float* X_test  = X_all.data() + (N_train * NUM_FEATURES);
    const float* y_test  = y_all.data() + N_train;

    // Initialize CUDA Model
    LogisticRegressionCUDA model(N_train, NUM_FEATURES, LEARNING_RATE, THREADS_PER_BLK);
    model.load_data(X_train, y_train);

    GpuTimer timer;
    GpuTimer total_timer;

    std::cout << "-------------------------------------------------------\n";
    std::cout << std::setw(8)  << "Epoch"
              << std::setw(15) << "BCE Loss"
              << std::setw(18) << "Train Time (ms)"
              << "\n";
    std::cout << "-------------------------------------------------------\n";

    total_timer.start();

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        timer.start();

        // 1. Forward Pass
        model.forward();

        // 2. Compute Loss
        float loss = model.compute_loss();

        // 3. Backward Pass
        model.backward();

        // 4. Optimizer Step
        model.step();

        float epoch_time = timer.stop();

        if (epoch % 5 == 0 || epoch == 1 || epoch == EPOCHS) {
            std::cout << std::setw(8)  << epoch
                      << std::setw(15) << std::fixed << std::setprecision(5) << loss
                      << std::setw(18) << std::fixed << std::setprecision(3) << epoch_time
                      << "\n";
        }
    }

    float total_ms = total_timer.stop();
    std::cout << "-------------------------------------------------------\n";
    std::cout << "[INFO] Total Training Time: " << total_ms << " ms ("
              << (total_ms / EPOCHS) << " ms/epoch)\n\n";

    // Test Evaluation
    std::cout << "[INFO] Evaluating on Test Set (" << N_test << " samples)...\n";
    std::vector<float> test_preds(N_test);
    model.predict(X_test, test_preds.data(), N_test);

    float test_accuracy = ml_utils::compute_accuracy(test_preds.data(), y_test, N_test);
    std::cout << "[RESULT] Test Accuracy: " << std::fixed << std::setprecision(2)
              << test_accuracy << " %\n";

    // Print learned weights sample
    std::vector<float> learned_w;
    float learned_b = 0.0f;
    model.get_weights(learned_w, learned_b);

    std::cout << "[RESULT] Learned Bias (b): " << learned_b << "\n";
    std::cout << "[RESULT] First 4 Learned Weights: [";
    for (int j = 0; j < std::min(4, NUM_FEATURES); ++j) {
        std::cout << learned_w[j] << (j < 3 ? ", " : "");
    }
    std::cout << "...]\n";

    std::cout << "\n=======================================================\n";
    std::cout << "                  Training Completed!                  \n";
    std::cout << "=======================================================\n";

    return 0;
}
