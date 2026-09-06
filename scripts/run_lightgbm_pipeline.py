import os
import sys
import json

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.lightgbm_classifier import run_lightgbm_pipeline

if __name__ == "__main__":
    mados_path = os.path.join(repo_root, "data", "MADOS_5Class")
    if not os.path.exists(mados_path):
        mados_path = os.path.join(repo_root, "MADOS_5Class")
        
    marida_path = os.path.join(repo_root, "data", "MARIDA_5Class")
    if not os.path.exists(marida_path):
        marida_path = os.path.join(repo_root, "MARIDA_5Class")
        
    results = {}
    if os.path.exists(mados_path):
        mados_summary, _ = run_lightgbm_pipeline("MADOS_5Class", mados_path, max_train_pixels=500000, n_estimators=100, max_depth=6, num_leaves=31)
        results["MADOS"] = mados_summary
    else:
        print(f"MADOS dataset not found at {mados_path}")

    if os.path.exists(marida_path):
        marida_summary, _ = run_lightgbm_pipeline("MARIDA_5Class", marida_path, max_train_pixels=500000, n_estimators=100, max_depth=6, num_leaves=31)
        results["MARIDA"] = marida_summary
    else:
        print(f"MARIDA dataset not found at {marida_path}")

    print("\n=======================================================")
    print(" SUMMARY OF LIGHTGBM EXPERIMENTS")
    print("=======================================================")
    print(json.dumps(results, indent=2))
