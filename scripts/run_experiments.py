"""Run the research pipeline from the repository root, without writing a report."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to original accepted_2007_to_2018Q4.csv.gz")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="outputs/reproduction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt_path = root/args.output/"final"/"evaluation_receipt.json"
    if receipt_path.exists():
        raise SystemExit("That output already contains an external evaluation. Choose a fresh output directory to reproduce; never tune using an evaluated test.")
    run("scripts/run_integrity_checks.py", "--output", args.output)
    if not (root/args.data_dir/"manifest.json").exists():
        if not args.source:
            parser.error("--source is required when prepared data are absent")
        run("-m", "src.prepare", "--source", Path(args.source).resolve(), "--out", args.data_dir)
    run("scripts/capture_environment.py", "--output", args.output, "--data-dir", args.data_dir)
    run("-m", "src.train", "--stage", "all", "--data-dir", args.data_dir, "--output", args.output)
    run("-m", "src.final_test", "--execute-final", "--data-dir", args.data_dir, "--output", args.output)
    run("-m", "src.analyze", "--data-dir", args.data_dir, "--output", args.output)
    run("scripts/build_research_notebooks.py", "--output", args.output)
