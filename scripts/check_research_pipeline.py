"""Development-only smoke checks and shuffled-label sanity experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import json
import io
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
from src.models import LoanModel
from src.analyze import component_shap
from src.utils import write_json

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/development.parquet")
    train = df[df.issue_date.le("2013-12-31")].sample(2500, random_state=42)
    valid = df[df.issue_date.between("2014-01-01", "2014-06-30")].sample(1000, random_state=42)
    results = []
    with threadpool_limits(limits=1):
        for family in ["logistic", "random_forest", "lightgbm", "xgboost", "catboost"]:
            params = {"n_estimators": 20} if family in ["random_forest", "lightgbm", "xgboost"] else ({"iterations": 20} if family == "catboost" else {"max_iter": 500})
            start = time.time()
            model = LoanModel(family, params, threads=1).fit(train, train.target,
                validation=(valid, valid.target) if family in ["lightgbm", "xgboost", "catboost"] else None)
            p = model.predict_proba(valid)[:, 1]
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            recovered = joblib.load(stream)
            np.testing.assert_allclose(recovered.predict_proba(valid)[:, 1], p, atol=1e-12)
            unknown = valid.iloc[:5].copy()
            unknown["addr_state"] = "__UNSEEN_STATE__"
            assert np.isfinite(recovered.predict_proba(unknown)).all()
            row = {"family": family, "validation_n": len(valid), "AP": average_precision_score(valid.target, p),
                   "finite_probabilities": bool(np.isfinite(p).all()), "seconds": time.time()-start}
            if family in ["lightgbm", "xgboost", "catboost", "logistic", "random_forest"]:
                _, _, _, units, error = component_shap(model, valid.iloc[:20])
                row.update(explanation_units=units, explanation_max_additivity_error=error)
            results.append(row)
            write_json("outputs/checks/pipeline_smoke.json", results)
            print(f"Development-only smoke: {family} OK", flush=True)
        train = df[df.issue_date.le("2013-12-31")].sample(20000, random_state=42)
        valid = df[df.issue_date.between("2014-01-01", "2014-06-30")].sample(15000, random_state=42)
        shuffled = []
        for seed in [42, 43, 44]:
            y = np.random.default_rng(seed).permutation(train.target)
            model = LoanModel("lightgbm", {"n_estimators": 150}, seed=seed, threads=1).fit(train, y)
            p = model.predict_proba(valid)[:, 1]
            shuffled.append({"seed": seed, "AP_shuffled_training_labels": average_precision_score(valid.target, p),
                             "validation_prevalence": float(valid.target.mean())})
        write_json("outputs/checks/pipeline_smoke.json", results)
        write_json("outputs/checks/shuffled_label_sanity.json", {"evaluation_cohort": "2014 H1 development selection",
            "train_n": len(train), "validation_n": len(valid), "results": shuffled,
            "interpretation": "Diagnostic for gross target leakage; similarity to prevalence does not prove absence of every kind of leakage."})
        print("Development-only sanity checks saved.", flush=True)
