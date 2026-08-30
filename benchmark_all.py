"""
=============================================================================
CUDA ML Models: Master Benchmark Runner
=============================================================================
Unified CLI interface to run benchmarks across all models in the repository:
  - 01_logistic_regression (Custom CUDA vs PyTorch Native)
  - 02_mlp (Custom Modular CUDA vs PyTorch Native)

Usage:
  python benchmark_all.py --all
  python benchmark_all.py --model logistic
  python benchmark_all.py --model mlp
  python benchmark_all.py --quick
=============================================================================
"""

import argparse
import os
import shutil
import subprocess
import sys
import torch


def ensure_dependencies():
    if shutil.which("ninja") is None:
        try:
            import ninja  # type: ignore
        except ImportError:
            print("[INFO] 'ninja' build tool not detected. Auto-installing ninja for fast JIT compilation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "--quiet"])
                print("[INFO] ninja installed successfully.")
            except Exception as e:
                print(f"[WARNING] Auto-installing ninja failed: {e}")


def print_banner():
    print("=" * 85)
    print("   ⚡ CUDA ML FROM SCRATCH: GPU BENCHMARK SUITE (CUDA vs PyTorch Native) ⚡")
    print("=" * 85)
    if torch.cuda.is_available():
        print(f"• Detected GPU     : {torch.cuda.get_device_name(0)}")
        print(f"• Compute Device   : CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
        print(f"• Available VRAM   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("• [WARNING] CUDA is NOT detected on this host. Please run in Google Colab / GPU server.")
    print("=" * 85 + "\n")


def run_benchmark(model_dir: str, script_name: str = "benchmark.py", extra_args: list = None):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(root_dir, model_dir, script_name)

    if not os.path.exists(target_path):
        print(f"[ERROR] Script not found: {target_path}")
        return

    cmd = [sys.executable, target_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n>>> Running Benchmark: {model_dir}/{script_name} ...")
    print(f">>> Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(root_dir, model_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=os.path.join(root_dir, model_dir), env=env)
    if result.returncode != 0:
        print(f"[ERROR] Benchmark failed with return code {result.returncode}")
    else:
        print(f"[DONE] Benchmark for {model_dir} completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Master CUDA ML Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "logistic", "mlp", "cnn"],
        default="all",
        help="Which model benchmark to execute (logistic, mlp, cnn, or all)",
    )
    parser.add_argument("--quick", action="store_true", help="Execute faster benchmarks with reduced iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (default: cuda)")
    args = parser.parse_args()

    print_banner()
    ensure_dependencies()

    extra_args = []
    if args.quick:
        extra_args.append("--quick")
    if args.device:
        extra_args.extend(["--device", args.device])

    if args.model in ["all", "logistic"]:
        run_benchmark("01_logistic_regression", "benchmark.py", extra_args)

    if args.model in ["all", "mlp"]:
        run_benchmark("02_mlp", "benchmark.py", extra_args)

    if args.model in ["all", "cnn"]:
        run_benchmark("03_cnn", "benchmark.py", extra_args)


if __name__ == "__main__":
    main()
