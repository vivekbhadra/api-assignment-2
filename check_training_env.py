#!/usr/bin/env python3
"""
Environment verification for fine-tuning setup.
Run:
    python check_training_env.py
"""

def main():
    try:
        import torch
        import transformers
        import pandas
    except ImportError as e:
        print(f"[ERROR] Missing package: {e.name}.")
        print("Run: pip install torch transformers pandas")
        return

    print("=== Fine-Tuning Environment Check ===")
    print(f"Torch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Pandas version: {pandas.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=====================================")

if __name__ == "__main__":
    main()

