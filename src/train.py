"""Temporal hyperparameter search, independent model selection, calibration, freeze.

Only development.parquet is read in this module. External test access lives in
src.final_test, which requires matching frozen artifact hashes.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.evaluate import metrics, choose_thresholds, ProbabilityCalibrator
from src.features import FeatureBuilder, FORBIDDEN
from src.models import LoanModel, LoanEnsemble
from src.prepare import file_sha256, ids_sha256
from src.utils import write_json, utc_now

FAMILIES = ["logistic", "random_forest", "lightgbm", "xgboost", "catboost"]


def subset(df, start=None, end=None):
    mask = np.ones(len(df), dtype=bool)
    if start:
        mask &= df.issue_date.ge(pd.Timestamp(start)).to_numpy()
    if end:
        mask &= df.issue_date.le(pd.Timestamp(end)).to_numpy()
    return df.loc[mask].copy()


def sample_development(frame, limit, seed=42):
    if len(frame) <= limit:
        return frame.copy()
    rows, _ = train_test_split(np.arange(len(frame)), train_size=limit, stratify=frame.target, random_state=seed)
    return frame.iloc[np.sort(rows)].copy()


def suggest(trial, family):
    if family == "logistic":
        return {"C": trial.suggest_float("C", 1e-4, 5, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if family == "random_forest":
        return {"n_estimators": 120,
                "max_depth": trial.suggest_categorical("max_depth", [8, 12, 18, None]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10, 25, 60, 120]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", .2, .4]),
                "max_samples": trial.suggest_float("max_samples", .55, .8),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"])}
    if family == "lightgbm":
        return {"n_estimators": 1400,
                "learning_rate": trial.suggest_float("learning_rate", .02, .12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 40, 300, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", .5, 40, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", .65, 1),
                "subsample": trial.suggest_float("subsample", .7, 1), "subsample_freq": 1,
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1., 2., 4.])}
    if family == "xgboost":
        return {"n_estimators": 1400,
                "learning_rate": trial.suggest_float("learning_rate", .025, .12, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_weight": trial.suggest_float("min_child_weight", 5, 80, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 40, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3, log=True),
                "subsample": trial.suggest_float("subsample", .7, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", .65, 1),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1., 2., 4.])}
    if family == "catboost":
        return {"iterations": 1200,
                "learning_rate": trial.suggest_float("learning_rate", .025, .1, log=True),
                "depth": trial.suggest_int("depth", 4, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 30, log=True),
                "random_strength": trial.suggest_float("random_strength", .1, 3, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
                "rsm": trial.suggest_float("rsm", .7, 1),
                "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"])}
    raise ValueError(family)


def search(df, protocol, output, families):
    folds, fold_manifest = [], []
    for i, spec in enumerate(protocol["cv"]):
        train = sample_development(subset(df, end=spec["train_end"]), protocol["search_max_train_rows_per_fold"])
        valid = sample_development(subset(df, spec["validation_start"], spec["validation_end"]), protocol["search_max_validation_rows_per_fold"])
        assert train.issue_date.max() < valid.issue_date.min()
        assert set(train.id).isdisjoint(valid.id)
        folds.append((train, valid))
        fold_manifest.append({**spec, "train_n": len(train), "validation_n": len(valid),
                              "train_ids_sha256": ids_sha256(train.id), "validation_ids_sha256": ids_sha256(valid.id)})
    write_json(output/"cv_folds.json", fold_manifest)
    study_dir = output/"studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in families:
        started = time.time()
        study_name = {"random_forest": "random_forest_dense_v3", "xgboost": "xgboost_native_v2"}.get(family, family)
        study = optuna.create_study(direction="maximize", study_name=study_name,
                    storage=f"sqlite:///{(study_dir/'search.sqlite3').as_posix()}", load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
        family_folds = folds
        if family == "random_forest":
            family_folds = [(sample_development(t, 30000), sample_development(v, 20000)) for t, v in folds]
            write_json(output/"cv_folds_random_forest.json", [
                {"train_n": len(t), "validation_n": len(v), "train_ids_sha256": ids_sha256(t.id),
                 "validation_ids_sha256": ids_sha256(v.id), "same_dates_as_global_CV": True}
                for t, v in family_folds])

        def objective(trial):
            params = suggest(trial, family)
            scores, iterations, durations = [], [], []
            trial.set_user_attr("model_params", params)
            for i, (train, valid) in enumerate(family_folds):
                t0 = time.time()
                model = LoanModel(family, params).fit(train, train.target,
                            validation=(valid, valid.target) if family in ["lightgbm", "xgboost", "catboost"] else None)
                p = model.predict_proba(valid)[:, 1]
                score = float(average_precision_score(valid.target, p))
                scores.append(score)
                iterations.append(model.best_iteration_)
                durations.append(round(time.time()-t0, 2))
                trial.set_user_attr("fold_AP", scores)
                trial.set_user_attr("fold_best_iterations", iterations)
                trial.set_user_attr("fold_seconds", durations)
                trial.report(float(np.mean(scores)), i)
                del model
                gc.collect()
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        def progress(study, trial):
            frame = study.trials_dataframe()
            frame.to_csv(output/f"search_{family}.csv", index=False)
            value = f"{trial.value:.5f}" if trial.value is not None else "failed"
            print(f"{family} trial {trial.number+1}/{protocol['trials'][family]} {trial.state.name}: AP={value}; best={study.best_value:.5f}", flush=True)

        remaining = max(0, protocol["trials"][family]-len(study.trials))
        study.optimize(objective, n_trials=remaining, callbacks=[progress], gc_after_trial=True)
        best = study.best_trial
        params = best.user_attrs["model_params"]
        search_params = dict(params)
        if family == "random_forest":
            params["n_estimators"] = 250
        trees = [v for v in best.user_attrs["fold_best_iterations"] if v is not None]
        if trees:
            # A fixed rule set before family selection; external outcomes are absent.
            key = "iterations" if family == "catboost" else "n_estimators"
            params[key] = min(params[key], max(50, int(np.median(trees)*1.15)))
        write_json(output/f"best_{family}.json", {"family": family, "params": params, "search_params": search_params,
                   "CV_AP": best.value, "best_trial": best.number, "fold_AP": best.user_attrs["fold_AP"],
                   "fold_best_iterations": best.user_attrs["fold_best_iterations"],
                   "trial_count": len(study.trials), "search_seconds_this_call": round(time.time()-started, 2)})
        print(f"Search complete: {family}, CV AP={best.value:.5f}", flush=True)


def candidate(df, name, family, params, extended, seed, output):
    path = output/"models"/f"candidate_{name}.joblib"
    selection = subset(df, "2014-01-01", "2014-06-30")
    if path.exists():
        model = joblib.load(path)
    else:
        train = subset(df, end="2013-12-31")
        t0 = time.time()
        model = LoanModel(family, params, extended=extended, seed=seed).fit(train, train.target)
        joblib.dump(model, path, compress=3)
        print(f"Candidate fitted: {name}, n={len(train):,}, seconds={time.time()-t0:.1f}", flush=True)
    p = model.predict_proba(selection)[:, 1]
    row = {"name": name, "family": family, "extended": extended, "seed": seed,
           **metrics(selection.target, p)}
    row.pop("confusion_matrix")
    return model, p, row


def select(df, output):
    (output/"models").mkdir(parents=True, exist_ok=True)
    (output/"predictions").mkdir(parents=True, exist_ok=True)
    rows, predictions, specs = [], {}, {}
    for family in FAMILIES:
        cfg = json.loads((output/f"best_{family}.json").read_text(encoding="utf-8"))
        name = family
        model, p, row = candidate(df, name, family, cfg["params"], True, 42, output)
        rows.append(row)
        predictions[name] = p
        specs[name] = {"family": family, "params": cfg["params"], "extended": True, "seed": 42}
        del model
        gc.collect()
    baseline_params = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
    model, p, row = candidate(df, "baseline19", "logistic", baseline_params, False, 42, output)
    rows.append(row)
    predictions["baseline19"] = p
    specs["baseline19"] = {"family": "logistic", "params": baseline_params, "extended": False, "seed": 42}
    del model
    top_families = sorted([r for r in rows if r["name"] != "baseline19"], key=lambda x: x["AP"], reverse=True)[:2]
    for base in top_families:
        family = base["family"]
        for seed in [43, 44]:
            name = f"{family}_seed{seed}"
            model, p, row = candidate(df, name, family, specs[family]["params"], True, seed, output)
            rows.append(row)
            predictions[name] = p
            specs[name] = {**specs[family], "seed": seed}
            del model
            gc.collect()
    # The best family's baseline-feature ablation is itself a selectable candidate.
    family = top_families[0]["family"]
    name = f"{family}_19features"
    model, p, row = candidate(df, name, family, specs[family]["params"], False, 42, output)
    rows.append(row)
    predictions[name] = p
    specs[name] = {**specs[family], "extended": False}
    del model
    selection = subset(df, "2014-01-01", "2014-06-30")
    y = selection.target.to_numpy()
    base_rows = rows.copy()
    groups = {}
    for family in FAMILIES:
        members = [name for name, spec in specs.items() if spec["family"] == family and spec["extended"]]
        groups[family] = {name: 1/len(members) for name in members}
    groups["baseline19"] = {"baseline19": 1.0}
    groups[f"{top_families[0]['family']}_19features"] = {f"{top_families[0]['family']}_19features": 1.0}
    group_predictions = {name: sum(w*predictions[member] for member, w in members.items())
                         for name, members in groups.items()}
    group_scores = {name: float(average_precision_score(y, p)) for name, p in group_predictions.items()}
    best_single = max(base_rows, key=lambda r: r["AP"])
    best_name, best_score = best_single["name"], best_single["AP"]
    best_weights = {best_name: 1.0}
    ensemble_trials = []
    group_top = sorted(group_scores, key=group_scores.get, reverse=True)[:3]
    for a in range(11):
        for b in range(11-a):
            weights = np.array([a, b, 10-a-b], dtype=float)/10
            p = sum(w*group_predictions[name] for name, w in zip(group_top, weights))
            score = float(average_precision_score(y, p))
            ensemble_trials.append({"groups": group_top, "weights": weights.tolist(), "AP": score})
    best_ensemble = max(ensemble_trials, key=lambda r: r["AP"])
    if best_ensemble["AP"] >= best_score + .001:
        best_name, best_score = "ensemble", best_ensemble["AP"]
        best_weights = {}
        for group, weight in zip(best_ensemble["groups"], best_ensemble["weights"]):
            for member, member_weight in groups[group].items():
                if weight > 0:
                    best_weights[member] = best_weights.get(member, 0) + weight*member_weight
    selected_p = sum(w*predictions[name] for name, w in best_weights.items())
    rows.append({"name": "selected", "family": best_name, "extended": True, "seed": -1, **metrics(y, selected_p)})
    pd.DataFrame(rows).drop(columns=["confusion_matrix"], errors="ignore").sort_values("AP", ascending=False).to_csv(output/"selection_leaderboard.csv", index=False)
    for group, members in groups.items():
        if len(members) > 1:
            scores = [r["AP"] for r in rows if r["name"] in members]
            print(f"Seed stability {group}: AP range {min(scores):.5f}..{max(scores):.5f}", flush=True)
    pd.DataFrame({"id": selection.id.to_numpy(), "target": y, **predictions, "selected": selected_p}).to_parquet(output/"predictions"/"selection.parquet", index=False)
    write_json(output/"selection.json", {"selected": best_name, "selection_AP": best_score,
               "weights": best_weights, "specs": specs, "single_best": best_single,
               "ensemble_search": ensemble_trials, "group_AP": group_scores,
               "selected_utc": utc_now(), "test_outcomes_accessed": False})
    print(f"SELECTED: {best_name}, AP={best_score:.5f}; weights={best_weights}", flush=True)


def freeze(df, output, data_dir):
    lock_path = output/"selection_lock.json"
    if lock_path.exists():
        raise FileExistsError("Selection already frozen. Existing external-test experiment cannot be retuned.")
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    train = subset(df, end="2014-06-30")
    calibration = subset(df, "2014-07-01", "2014-09-30")
    policy = subset(df, "2014-10-01", "2014-12-31")
    assert train.issue_date.max() < calibration.issue_date.min() <= calibration.issue_date.max() < policy.issue_date.min()
    names = set(FAMILIES + ["baseline19"] + list(selection["weights"]))
    models, calibrators, thresholds, calibration_rows = {}, {}, {}, []
    for name in sorted(names):
        spec = selection["specs"][name]
        path = output/"models"/f"final_{name}.joblib"
        t0 = time.time()
        if path.exists():
            model = joblib.load(path)
        else:
            model = LoanModel(**spec).fit(train, train.target)
            joblib.dump(model, path, compress=3)
        models[name] = model
        print(f"Final development fit: {name}, {len(train):,} rows, {time.time()-t0:.1f}s", flush=True)
    ensemble = LoanEnsemble([models[n] for n in selection["weights"]], list(selection["weights"].values()))
    all_models = {**{name: models[name] for name in FAMILIES+["baseline19"]}, "selected": ensemble}
    calibration_predictions = {"id": calibration.id.to_numpy(), "target": calibration.target.to_numpy()}
    policy_predictions = {"id": policy.id.to_numpy(), "target": policy.target.to_numpy()}
    for name, model in all_models.items():
        cp = model.predict_proba(calibration)[:, 1]
        pp = model.predict_proba(policy)[:, 1]
        options = []
        for method in ["identity", "sigmoid", "isotonic"]:
            calibrator = ProbabilityCalibrator(method).fit(cp, calibration.target)
            p = calibrator.predict(pp)
            row = {"model": name, "calibration": method, **metrics(policy.target, p)}
            options.append((row, calibrator, p))
            calibration_rows.append(row)
        row, calibrator, p = min(options, key=lambda r: (r[0]["Brier"], r[0]["log_loss"]))
        calibrators[name] = calibrator
        thresholds[name] = choose_thresholds(policy.target, p)
        calibration_predictions[f"{name}_raw"] = cp
        policy_predictions[f"{name}_raw"] = pp
        policy_predictions[name] = p
        print(f"Calibration {name}: {calibrator.method}, Brier={row['Brier']:.5f}, F1 threshold={thresholds[name]['f1']:.4f}", flush=True)
    pd.DataFrame(calibration_rows).drop(columns="confusion_matrix").to_csv(output/"calibration_selection.csv", index=False)
    pd.DataFrame(calibration_predictions).to_parquet(output/"predictions"/"calibration.parquet", index=False)
    pd.DataFrame(policy_predictions).to_parquet(output/"predictions"/"policy.parquet", index=False)
    bundle = {"models": models, "selected_weights": selection["weights"], "calibrators": calibrators,
              "thresholds": thresholds, "selected_name": selection["selected"],
              "training_end": "2014-06-30", "protocol_version": "1.0"}
    path = output/"models"/"frozen_bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    lock = {"frozen_utc": utc_now(), "model_sha256": file_sha256(path),
            "protocol_sha256": file_sha256("experiments/protocol.json"),
            "data_manifest_sha256": file_sha256(data_dir/"manifest.json"),
            "selection_sha256": file_sha256(output/"selection.json"),
            "selected": selection["selected"], "weights": selection["weights"],
            "calibration": {name: c.method for name, c in calibrators.items()},
            "thresholds": thresholds, "test_outcomes_accessed": False,
            "training_rows": len(train), "calibration_rows": len(calibration), "policy_rows": len(policy),
            "training_ids_sha256": ids_sha256(train.id), "calibration_ids_sha256": ids_sha256(calibration.id),
            "policy_ids_sha256": ids_sha256(policy.id)}
    lock["code_sha256"] = {str(p).replace("\\", "/"): file_sha256(p) for p in Path("src").glob("*.py")}
    write_json(lock_path, lock)
    print("Frozen model, calibrators, policies and hashes. External testing is now permitted.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "select", "freeze", "all"], default="all")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    output, data_dir = Path(args.output), Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path("experiments/protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == file_sha256("experiments/protocol.json"), "Protocol changed after preparation."
    assert manifest["partitions"]["development"]["file_sha256"] == file_sha256(data_dir/"development.parquet"), "Development data changed."
    assert not (output/"final"/"evaluation_receipt.json").exists(), "External testing has started; do not retune this experiment."
    df = pd.read_parquet(data_dir/"development.parquet")
    assert df.issue_date.max() < pd.Timestamp("2015-01-01")
    with threadpool_limits(limits=3):
        if args.stage in ["search", "all"]:
            search(df, protocol, output, args.families)
        if args.stage in ["select", "all"]:
            select(df, output)
        if args.stage in ["freeze", "all"]:
            freeze(df, output, data_dir)


if __name__ == "__main__":
    main()
