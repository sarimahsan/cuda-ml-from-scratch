#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ml_utils {

// Generates synthetic binary classification dataset linearly separable with noise
// X: N x D matrix (flattened row-major)
// y: N-element binary labels {0.0f, 1.0f}
inline void generate_synthetic_data(
    std::vector<float>& X,
    std::vector<float>& y,
    int N,
    int D,
    unsigned int seed = 42
) {
    X.resize(N * D);
    y.resize(N);

    std::mt19937 gen(seed);
    std::normal_distribution<float> feature_dist(0.0f, 1.5f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);

    // Random true weights and bias for generating labels
    std::vector<float> true_weights(D);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    for (int j = 0; j < D; ++j) {
        true_weights[j] = weight_dist(gen);
    }
    float true_bias = 0.35f;

    for (int i = 0; i < N; ++i) {
        float score = true_bias;
        for (int j = 0; j < D; ++j) {
            float val = feature_dist(gen);
            X[i * D + j] = val;
            score += val * true_weights[j];
        }
        score += noise_dist(gen);
        // Sigmoid probability threshold at 0.5 (i.e., score >= 0)
        y[i] = (score >= 0.0f) ? 1.0f : 0.0f;
    }
}

// Compute classification accuracy on CPU
inline float compute_accuracy(const float* preds, const float* targets, int N, float threshold = 0.5f) {
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        float binary_pred = (preds[i] >= threshold) ? 1.0f : 0.0f;
        if (std::abs(binary_pred - targets[i]) < 1e-5f) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(N) * 100.0f;
}

} // namespace ml_utils
