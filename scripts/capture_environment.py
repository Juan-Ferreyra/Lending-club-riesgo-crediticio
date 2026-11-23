"""Save exact versions and data provenance without bundling the original dataset."""
import importlib.metadata as metadata
import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm", "xgboost",
            "optuna", "pyarrow", "matplotlib", "seaborn", "joblib", "threadpoolctl",
            "nbformat", "nbclient", "ipykernel", "SQLAlchemy", "alembic", "shap", "slicer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    versions = {name: metadata.version(name) for name in PACKAGES}
    out = ROOT/args.output
    out.mkdir(exist_ok=True, parents=True)
    (ROOT/"requirements-lock.txt").write_text(
        "# Executed environment: Python "+platform.python_version()+" / "+platform.system()+" "+platform.machine()+"\n"+
        "\n".join(f"{name}=={version}" for name, version in versions.items())+"\n", encoding="utf-8")
    (out/"execution_environment.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "libraries": versions,
        "training_threads": 3, "seed": 42}, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT/args.data_dir/"manifest.json", out/"data_manifest.json")
    print("Environment versions and data provenance captured.")
