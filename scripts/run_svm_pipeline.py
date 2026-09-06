import os
import sys
import json
import yaml
import argparse

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.svm_classifier import run_svm_pipeline, resolve_dataset_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Support Vector Machine (SVM) Pipeline on 5-Class Marine Datasets")
    parser.add_argument("--dataset", type=str, default="both", choices=["MADOS_5Class", "MARIDA_5Class", "both"],
                        help="Dataset to evaluate on (MADOS_5Class, MARIDA_5Class, or both)")
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"],
                        help="SVM Kernel type")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter C")
    parser.add_argument("--max_iter", type=int, default=3000, help="Maximum solver iterations")
    parser.add_argument("--max_train_pixels", type=int, default=250000, help="Maximum training pixels sampled")
    parser.add_argument("--max_test_pixels", type=int, default=250000, help="Maximum test pixels sampled")
    args = parser.parse_args()

    # Load default configs if available
    config_path = os.path.join(repo_root, "configs", "svm.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg:
                kernel = cfg.get("kernel", args.kernel)
                C_val = cfg.get("C", args.C)
                max_iter = cfg.get("max_iter", args.max_iter)
                max_train_px = cfg.get("max_train_pixels", args.max_train_pixels)
            else:
                kernel, C_val, max_iter, max_train_px = args.kernel, args.C, args.max_iter, args.max_train_pixels
    else:
        kernel, C_val, max_iter, max_train_px = args.kernel, args.C, args.max_iter, args.max_train_pixels

    # Command-line flags override config if explicitly provided
    if args.kernel != "linear":
        kernel = args.kernel
    if args.C != 1.0:
        C_val = args.C
    if args.max_iter != 3000:
        max_iter = args.max_iter
    if args.max_train_pixels != 250000:
        max_train_px = args.max_train_pixels

    mados_path, marida_path = resolve_dataset_paths(repo_root)
    print("=" * 65)
    print(" 5-CLASS MARINE DEBRIS & WATER SEGMENTATION — SVM BENCHMARK")
    print("=" * 65)
    print(f"Repository Root    : {repo_root}")
    print(f"MADOS Dataset Path : {mados_path}")
    print(f"MARIDA Dataset Path: {marida_path}")
    print(f"SVM Kernel         : {kernel}")
    print(f"Regularization (C) : {C_val}")
    print(f"Max Solver Iter    : {max_iter}")
    print(f"Max Train Pixels   : {max_train_px:,}")
    print("=" * 65 + "\n")

    results = {}
    
    # Run MADOS
    if args.dataset in ["MADOS_5Class", "both"]:
        if mados_path and os.path.exists(mados_path):
            mados_summary, mados_txt = run_svm_pipeline(
                "MADOS_5Class",
                mados_path,
                max_train_pixels=max_train_px,
                max_test_pixels=args.max_test_pixels,
                kernel=kernel,
                C=C_val,
                max_iter=max_iter
            )
            results["MADOS_5Class"] = mados_summary
        else:
            print(f"[ERROR] MADOS_5Class dataset directory not found!")

    # Run MARIDA
    if args.dataset in ["MARIDA_5Class", "both"]:
        if marida_path and os.path.exists(marida_path):
            marida_summary, marida_txt = run_svm_pipeline(
                "MARIDA_5Class",
                marida_path,
                max_train_pixels=max_train_px,
                max_test_pixels=args.max_test_pixels,
                kernel=kernel,
                C=C_val,
                max_iter=max_iter
            )
            results["MARIDA_5Class"] = marida_summary
        else:
            print(f"[ERROR] MARIDA_5Class dataset directory not found!")

    print("\n" + "=" * 65)
    print(" FINAL SUMMARY OF SUPPORT VECTOR MACHINE (SVM) EXPERIMENTS")
    print("=" * 65)
    print(json.dumps(results, indent=2))
    print("=" * 65 + "\n")

