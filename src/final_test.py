"""One frozen external evaluation; never used for model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models import LoanEnsemble
from src.evaluate import metrics, paired_bootstrap
from src.prepare import file_sha256, ids_sha256
from src.utils import write_json, utc_now


def evaluate_frozen(data_dir, output, resume=False):
    data_dir, output = Path(data_dir), Path(output)
    lock_path = output/"selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_dir/"manifest.json").read_text(encoding="utf-8"))
    assert lock["model_sha256"] == file_sha256(output/"models"/"frozen_bundle.joblib")
    assert lock["protocol_sha256"] == file_sha256("experiments/protocol.json")
    assert lock["data_manifest_sha256"] == file_sha256(data_dir/"manifest.json")
    assert lock["selection_sha256"] == file_sha256(output/"selection.json")
    for path, expected in lock["code_sha256"].items():
        assert file_sha256(path) == expected, f"Experiment code changed after freeze: {path}"
    final_dir = output/"final"
    final_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = final_dir/"evaluation_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not resume or receipt["state"] == "complete":
            raise FileExistsError("Final evaluation already started or completed. Do not reopen it for tuning.")
        assert receipt["lock_sha256"] == file_sha256(lock_path), "Frozen experiment changed."
    else:
        receipt = {"state": "in_progress", "started_utc": utc_now(),
                   "lock_sha256": file_sha256(lock_path), "completed_cohorts": [],
                   "model_selection_already_frozen": True}
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
    bundle = joblib.load(output/"models"/"frozen_bundle.joblib")
    names = ["baseline19", "logistic", "random_forest", "lightgbm", "xgboost", "catboost"]
    models = {name: bundle["models"][name] for name in names}
    models["selected"] = LoanEnsemble([bundle["models"][n] for n in bundle["selected_weights"]], list(bundle["selected_weights"].values()))
    initial = pd.read_csv("data/lending_club_muestra.csv.gz", usecols=["id", "issue_d", "term", "loan_status"], dtype=str)
    initial_ids = set(initial.id)
    preview_ids = set(manifest["schema_preview_excluded_ids"])
    results, uncertainties, policy_rows = [], {}, []
    for year in [2015, 2016]:
        path = data_dir/f"holdout_{year}.parquet"
        assert manifest["partitions"][f"holdout_{year}"]["file_sha256"] == file_sha256(path)
        df = pd.read_parquet(path)
        assert df.issue_year.eq(year).all() and df.id.is_unique
        assert not set(df.id).intersection(initial_ids)
        assert not set(df.id).intersection(preview_ids)
        assert ids_sha256(df.id) == manifest["partitions"][f"holdout_{year}"]["ids_sha256"]
        print(f"External {year} opened after freeze: {len(df):,} loans, prevalence={df.target.mean():.5f}", flush=True)
        predictions = df[["id", "issue_date", "issue_year", "term", "target"]].copy()
        for name, model in models.items():
            raw_p = model.predict_proba(df)[:, 1]
            p = bundle["calibrators"][name].predict(raw_p)
            predictions[f"{name}_raw"] = raw_p
            predictions[name] = p
            threshold = bundle["thresholds"][name]["f1"]
            for scope, mask in [("all_resolved", np.ones(len(df), dtype=bool)),
                                ("36_month", df.term.str.strip().eq("36 months").to_numpy()),
                                ("60_month", df.term.str.strip().eq("60 months").to_numpy())]:
                if mask.sum() == 0:
                    continue
                row = {"cohort": year, "scope": scope, "model": name,
                       "selected_before_testing": name == "selected", **metrics(df.target[mask], p[mask], threshold)}
                row["AP_raw"] = metrics(df.target[mask], raw_p[mask])["AP"]
                row["calibrator"] = bundle["calibrators"][name].method
                results.append(row)
            if name == "selected":
                for policy_name, t in bundle["thresholds"][name].items():
                    policy_rows.append({"cohort": year, "policy": policy_name, **metrics(df.target, p, t)})
        predictions.to_parquet(output/"predictions"/f"external_{year}.parquet", index=False)
        if year == 2015:
            for scope, mask in [("all_resolved", np.ones(len(df), dtype=bool)),
                                ("36_month", df.term.str.strip().eq("36 months").to_numpy())]:
                print(f"Paired uncertainty estimation: {year} / {scope}", flush=True)
                uncertainties[f"{year}_{scope}"] = paired_bootstrap(
                    df.target.to_numpy()[mask], predictions.selected.to_numpy()[mask],
                    predictions.baseline19.to_numpy()[mask],
                    df.issue_date.dt.strftime("%Y-%m").to_numpy()[mask])
        receipt["completed_cohorts"].append(year)
        write_json(receipt_path, receipt)
    # Release censoring coverage only after the entire predictor and policies are frozen.
    coverage = pd.read_parquet(data_dir/"sealed_coverage.parquet")
    coverage.to_csv(final_dir/"source_cohort_status_counts.csv", index=False)
    initial["issue_year"] = pd.to_datetime(initial.issue_d, format="%b-%Y").dt.year
    initial_counts = initial.groupby(["issue_year", "term", "loan_status"]).size().reset_index(name="initial_sample_n")
    external_cov = coverage.merge(initial_counts, how="left", on=["issue_year", "term", "loan_status"])
    external_cov["n"] = external_cov.n-external_cov.initial_sample_n.fillna(0)
    preview_counts = pd.read_parquet(data_dir/"sealed_preview_counts.parquet")
    if not preview_counts.empty:
        external_cov = external_cov.merge(preview_counts.rename(columns={"n": "schema_preview_n"}), how="left", on=["issue_year", "term", "loan_status"])
        external_cov["n"] = external_cov.n-external_cov.schema_preview_n.fillna(0)
    external_cov = external_cov[external_cov.issue_year.isin([2015, 2016])]
    coverage_rows = []
    for (year, term), group in external_cov.groupby(["issue_year", "term"]):
        total = int(group.n.sum())
        resolved = int(group.loc[group.loan_status.isin(["Fully Paid", "Charged Off"]), "n"].sum())
        coverage_rows.append({"year": int(year), "term": term.strip(), "external_population_n": total,
                              "resolved_n": resolved, "excluded_nonterminal_n": total-resolved,
                              "terminal_fraction": resolved/total})
    pd.DataFrame(coverage_rows).to_csv(final_dir/"maturity_coverage.csv", index=False)
    for year in [2015, 2016]:
        assert sum(row["resolved_n"] for row in coverage_rows if row["year"] == year) == manifest["partitions"][f"holdout_{year}"]["rows"], "Coverage does not match external cohort eligibility."
    write_json(final_dir/"metrics.json", results)
    pd.DataFrame(results).drop(columns="confusion_matrix").to_csv(final_dir/"metrics.csv", index=False)
    write_json(final_dir/"uncertainty.json", uncertainties)
    pd.DataFrame(policy_rows).drop(columns="confusion_matrix").to_csv(final_dir/"threshold_policies.csv", index=False)
    receipt.update(state="complete", completed_utc=utc_now(),
                   selected_model_unchanged=True, selected_name=bundle["selected_name"],
                   metrics_sha256=file_sha256(final_dir/"metrics.json"))
    write_json(receipt_path, receipt)
    print(pd.DataFrame(results).query("scope == 'all_resolved'")[["cohort", "model", "AP", "ROC_AUC", "Brier", "F1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--execute-final", action="store_true", help="Confirm selection is frozen before opening external outcomes")
    parser.add_argument("--resume", action="store_true", help="Resume a failed evaluation with exactly the same locked model")
    args = parser.parse_args()
    if not args.execute_final:
        parser.error("--execute-final is required")
    with threadpool_limits(limits=3):
        evaluate_frozen(args.data_dir, args.output, args.resume)
