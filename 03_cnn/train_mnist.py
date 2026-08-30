import time
import torch
import numpy as np
from cnn import CUDACNN

def load_mnist_data():
    """
    Loads the MNIST handwritten digit dataset via torchvision or synthetic fallback.
    Returns (X_train, y_train, X_test, y_test) as 4D PyTorch tensors [N, 1, 28, 28].
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

        X_train = train_dataset.data.float().unsqueeze(1) / 255.0  # [60000, 1, 28, 28]
        y_train = train_dataset.targets.long()

        X_test = test_dataset.data.float().unsqueeze(1) / 255.0    # [10000, 1, 28, 28]
        y_test = test_dataset.targets.long()
        print(f"[INFO] Loaded real MNIST dataset: {len(X_train)} train, {len(X_test)} test images.")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"[WARNING] Could not load torchvision MNIST ({e}). Falling back to synthetic 4D images.")
        np.random.seed(42)
        N_train, N_test, C, H, W = 6000, 1000, 1, 28, 28
        num_classes = 10

        patterns = np.random.randn(num_classes, C, H, W).astype(np.float32)
        y_tr = np.random.randint(0, num_classes, size=N_train)
        X_tr = patterns[y_tr] + np.random.randn(N_train, C, H, W).astype(np.float32) * 0.4

        y_te = np.random.randint(0, num_classes, size=N_test)
        X_te = patterns[y_te] + np.random.randn(N_test, C, H, W).astype(np.float32) * 0.4

        return torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(X_te), torch.from_numpy(y_te)


def main():
    print("=" * 68)
    print("   CUDA ML Models: Modular Convolutional Neural Network (CNN)   ")
    print("=" * 68)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Please run this script with a GPU (e.g., Google Colab).")
        return

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # 2. Hyperparameters
    BATCH_SIZE    = 64
    EPOCHS        = 5
    LEARNING_RATE = 0.001
    OPTIMIZER     = "adam"  # or "momentum"

    print(f"\n[INFO] Initializing CUDACNN Architecture:")
    print("  - Layer 1: Conv2d(1 -> 16, 3x3, pad 1) -> ReLU -> MaxPool2d(2x2) -> [16, 14, 14]")
    print("  - Layer 2: Conv2d(16 -> 32, 3x3, pad 1) -> ReLU -> MaxPool2d(2x2) -> [32, 7, 7]")
    print("  - Layer 3: Linear(1568 -> 128) -> ReLU")
    print("  - Layer 4: Linear(128 -> 10) -> Softmax Cross-Entropy Loss")
    print(f"[INFO] Optimizer: {OPTIMIZER.upper()} | LR: {LEARNING_RATE} | Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}\n")

    # 3. Model instantiation
    model = CUDACNN(
        in_channels=1,
        in_height=28,
        in_width=28,
        conv1_channels=16,
        conv2_channels=32,
        fc_hidden=128,
        num_classes=10,
        lr=LEARNING_RATE,
        momentum=0.9
    ).cuda()

    N_train = len(X_train)
    num_batches = N_train // BATCH_SIZE

    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        indices = torch.randperm(N_train)
        running_loss = 0.0
        correct_train = 0

        epoch_start = time.perf_counter()

        for b in range(num_batches):
            batch_idx = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            X_b = X_train[batch_idx].cuda()
            y_b = y_train[batch_idx].cuda()

            # Forward pass
            logits = model(X_b)
            targets_onehot = torch.nn.functional.one_hot(y_b, num_classes=10).float()
            loss, probs, _ = model.compute_loss_and_probs(logits, targets_onehot)

            # Backward pass via autograd
            loss.backward()

            # Custom CUDA in-place optimizer step
            model.step_optimizer(method=OPTIMIZER)

            running_loss += loss.item()
            preds = torch.argmax(probs, dim=1)
            correct_train += (preds == y_b).sum().item()

            if (b + 1) % 200 == 0 or (b + 1) == num_batches:
                cur_acc = 100.0 * correct_train / ((b + 1) * BATCH_SIZE)
                cur_loss = running_loss / (b + 1)
                print(f"Epoch [{epoch}/{EPOCHS}] | Step [{b+1}/{num_batches}] | Loss: {cur_loss:.4f} | Train Acc: {cur_acc:.2f}%")

        epoch_time = time.perf_counter() - epoch_start
        epoch_loss = running_loss / num_batches
        epoch_acc = 100.0 * correct_train / (num_batches * BATCH_SIZE)

        # Validation evaluation
        with torch.no_grad():
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
            val_loss, val_acc = model.evaluate(test_loader)

        print(f"--> Epoch {epoch} Completed in {epoch_time:.2f}s | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100.0:.2f}%\n")

    total_time = time.perf_counter() - t_start
    print("-" * 68)
    print(f"[INFO] Total Training Duration: {total_time:.2f}s ({total_time / EPOCHS:.2f}s/epoch)")

    # Final Evaluation
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        _, final_test_acc = model.evaluate(test_loader)

    print(f"\n[RESULT] >>> Final MNIST Test Accuracy: {final_test_acc * 100.0:.2f}% <<<\n")


if __name__ == "__main__":
    main()
