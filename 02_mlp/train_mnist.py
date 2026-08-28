import time
import torch
import numpy as np
from mlp import CUDAMLP

def load_mnist_data():
    """
    Loads the MNIST handwritten digit dataset via torchvision or synthetic fallback.
    Returns (X_train, y_train, X_test, y_test) as PyTorch tensors.
    """
    print("[INFO] Fetching MNIST Handwritten Digits Dataset...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        X_train = train_dataset.data.float().view(-1, 28 * 28) / 255.0
        y_train = train_dataset.targets.long()

        X_test = test_dataset.data.float().view(-1, 28 * 28) / 255.0
        y_test = test_dataset.targets.long()
        print(f"[INFO] Loaded real MNIST dataset: {len(X_train)} train, {len(X_test)} test images.")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"[WARNING] Could not load torchvision MNIST ({e}). Falling back to synthetic 784-dim digits.")
        np.random.seed(42)
        N_train, N_test, D, C = 60000, 10000, 784, 10
        centers = np.random.randn(C, D).astype(np.float32)
        
        y_tr = np.random.randint(0, C, size=N_train)
        X_tr = centers[y_tr] + np.random.randn(N_train, D).astype(np.float32) * 0.5
        
        y_te = np.random.randint(0, C, size=N_test)
        X_te = centers[y_te] + np.random.randn(N_test, D).astype(np.float32) * 0.5

        return torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(X_te), torch.from_numpy(y_te)


def main():
    print("=" * 65)
    print("   CUDA ML Models: Multi-Layer Perceptron on MNIST Digits   ")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Please run this script with a GPU (e.g., Google Colab).")
        return

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # 2. Hyperparameters
    LAYER_SIZES   = [784, 256, 128, 10]  # 3-Layer MLP: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (Softmax)
    LEARNING_RATE = 1e-3
    BATCH_SIZE    = 128
    EPOCHS        = 15
    OPTIMIZER     = "adam"

    print(f"\n[INFO] Initializing CUDAMLP Architecture: {LAYER_SIZES}")
    print(f"[INFO] Optimizer: {OPTIMIZER.upper()} | LR: {LEARNING_RATE} | Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}\n")

    # 3. Instantiate Custom CUDA MLP Model
    model = CUDAMLP(
        layer_sizes=LAYER_SIZES,
        lr=LEARNING_RATE,
        optimizer=OPTIMIZER,
        device="cuda"
    )

    # 4. Train Model
    t_start = time.perf_counter()
    model.fit(
        X=X_train,
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=True
    )
    total_time = time.perf_counter() - t_start

    # 5. Final Test Evaluation
    print("-" * 65)
    print(f"[INFO] Total Training Duration: {total_time:.2f} s ({total_time / EPOCHS * 1000:.1f} ms/epoch)")
    print(f"[INFO] Evaluating Test Accuracy on {len(X_test)} samples...")

    test_accuracy = model.score(X_test, y_test)
    print(f"\n[RESULT] >>> Final MNIST Test Accuracy: {test_accuracy:.2f}% <<<\n")

    # 6. Per-Class Accuracy Analysis
    preds = model.predict(X_test).cpu()
    y_test_cpu = y_test.cpu()

    print("Class-wise Accuracy:")
    for c in range(10):
        mask = (y_test_cpu == c)
        c_acc = (preds[mask] == c).float().mean().item() * 100.0
        print(f"  Digit '{c}': {c_acc:6.2f}% ({mask.sum().item()} samples)")

    print("\n=======================================================")
    print("              MLP Training Complete! 🚀                 ")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
