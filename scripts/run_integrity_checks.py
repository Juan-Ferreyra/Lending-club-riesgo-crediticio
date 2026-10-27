"""Persist integrity-test evidence and return a failing exit status on errors."""
import argparse
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import write_json, utc_now

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover("tests")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    out = Path(args.output)/"checks"
    out.mkdir(parents=True, exist_ok=True)
    (out/"integrity_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    write_json(out/"integrity_tests.json", {"utc": utc_now(), "tests_run": result.testsRun,
        "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors)})
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
