"""Post-freeze error analysis, component SHAP, robustness and diagnostic plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits

from src.features import FeatureBuilder, BASE_NUMERIC, BUREAU_NUMERIC, CATEGORICAL, RAW_PREDICTORS, FORBIDDEN
from src.models import LoanEnsemble
from src.evaluate import metrics
from src.prepare import file_sha256
from src.utils import write_json

COLOR = {"selected": "#096B68", "baseline19": "#C05935"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
                     "savefig.facecolor": "white", "axes.titleweight": "bold"})


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def wilson(successes, n):
    if n == 0:
        return np.nan, np.nan
    p, z = successes/n, 1.95996398454
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., center-half), min(1., center+half)


def segments(df, p, threshold, year):
    fico = (pd.to_numeric(df.fico_range_low, errors="coerce")+pd.to_numeric(df.fico_range_high, errors="coerce"))/2
    income = pd.to_numeric(df.annual_inc, errors="coerce")
    dti = pd.to_numeric(df.dti, errors="coerce")
    groups = {"term": df.term.str.strip(), "purpose": df.purpose, "home_ownership": df.home_ownership,
              "state": df.addr_state, "employment": df.emp_length,
              "application_type": df.application_type, "verification_status": df.verification_status,
              "issue_month": df.issue_date.dt.strftime("%Y-%m"),
              "FICO": pd.cut(fico, [-np.inf, 640, 680, 720, 760, np.inf], right=False).astype(str),
              "income": pd.cut(income, [-np.inf, 40000, 80000, 120000, np.inf], right=False).astype(str),
              "DTI": pd.cut(dti, [-np.inf, 10, 20, 30, np.inf], right=False).astype(str)}
    rows = []
    for dimension, values in groups.items():
        for value in values.fillna("MISSING").unique():
            mask = values.fillna("MISSING").eq(value).to_numpy()
            row = {"cohort": year, "dimension": dimension, "segment": str(value), **metrics(df.target.to_numpy()[mask], p[mask], threshold)}
            tn, fp = row["confusion_matrix"][0]
            fn, tp = row["confusion_matrix"][1]
            row["FNR_low"], row["FNR_high"] = wilson(fn, fn+tp)
            row["FPR_low"], row["FPR_high"] = wilson(fp, fp+tn)
            row["low_support"] = bool(row["n"] < 300 or min(fn+tp, fp+tn) < 30)
            row["mean_predicted_risk"] = float(np.mean(p[mask]))
            rows.append(row)
    return rows


def reliability(y, p):
    frame = pd.DataFrame({"y": np.asarray(y), "p": p})
    frame["bin"] = pd.qcut(frame.p.rank(method="first"), 10, labels=False)
    rows = []
    for i, group in frame.groupby("bin"):
        lo, hi = wilson(int(group.y.sum()), len(group))
        rows.append({"bin": int(i), "n": len(group), "mean_prediction": group.p.mean(),
                     "observed_rate": group.y.mean(), "low": lo, "high": hi})
    return pd.DataFrame(rows)


def grouped_permutation(model, calibrator, raw, y, out):
    rng = np.random.default_rng(42)
    groups = {
        "FICO": ["fico_range_low", "fico_range_high"],
        "Ingresos": ["annual_inc", "annual_inc_joint"],
        "Monto y plazo": ["loan_amnt", "term"],
        "Deuda / ingreso": ["dti", "dti_joint"],
        "Deuda revolvente": ["revol_bal", "revol_util", "total_rev_hi_lim"],
        "Historia crediticia": ["earliest_cr_line", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op"],
        "Consultas recientes": ["inq_last_6mths", "inq_last_12m", "mths_since_recent_inq"],
        "Moras anteriores": ["delinq_2yrs", "mths_since_last_delinq", "num_accts_ever_120_pd", "num_tl_90g_dpd_24m"],
        "Vivienda y estado": ["home_ownership", "addr_state"],
        "Cuentas y saldos": ["open_acc", "total_acc", "tot_cur_bal", "mort_acc", "total_bal_ex_mort"],
        "Propósito y empleo": ["purpose", "emp_length"],
        "Verificación": ["verification_status"],
    }
    base = calibrator.predict(model.predict_proba(raw)[:, 1])
    base_ap = average_precision_score(y, base)
    rows = []
    for name, cols in groups.items():
        cols = [col for col in cols if col in raw]
        for repeat in range(5):
            perturbed = raw.copy()
            indices = rng.permutation(len(raw))
            for col in cols:
                perturbed[col] = raw[col].to_numpy()[indices]
            p = calibrator.predict(model.predict_proba(perturbed)[:, 1])
            rows.append({"group": name, "repeat": repeat, "AP_drop": base_ap-average_precision_score(y, p)})
        print(f"Permutation group complete: {name}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out/"grouped_permutation_raw.csv", index=False)
    summary = result.groupby("group").AP_drop.agg(["mean", "std"]).sort_values("mean", ascending=False)
    summary.to_csv(out/"grouped_permutation.csv")
    return summary


def component_shap(model, raw):
    X = model.features.transform(raw)
    if model.family == "catboost":
        from catboost import Pool
        values = model.estimator.get_feature_importance(Pool(X, cat_features=CATEGORICAL), type="ShapValues", thread_count=3)
        names = list(X.columns)
        units = "log_odds"
        expected = np.asarray(model.estimator.predict(X, prediction_type="RawFormulaVal")).reshape(-1)
    elif model.family == "lightgbm":
        values = model.estimator.predict(model.transform(raw), pred_contrib=True)
        names = list(model.preprocessor.get_feature_names_out())
        units = "log_odds"
        expected = model.estimator.booster_.predict(model.transform(raw), raw_score=True)
    elif model.family == "xgboost":
        from xgboost import DMatrix
        booster = model.estimator.get_booster()
        try:
            end_iteration = model.estimator.best_iteration + 1
        except AttributeError:
            end_iteration = booster.num_boosted_rounds()
        matrix = DMatrix(model.transform(raw), enable_categorical=True, feature_types=booster.feature_types)
        values = booster.predict(matrix, pred_contribs=True,
                                 iteration_range=(0, end_iteration))
        names = list(model.preprocessor.get_feature_names_out())
        units = "log_odds"
        expected = booster.predict(matrix, output_margin=True, iteration_range=(0, end_iteration))
    elif model.family == "logistic":
        Z = model.transform(raw)
        # Exact additive linear log-odds decomposition around zero transformed inputs.
        contributions = np.asarray(Z.multiply(model.estimator.coef_[0]).toarray()) if hasattr(Z, "multiply") else Z*model.estimator.coef_[0]
        values = np.column_stack([contributions, np.full(len(raw), model.estimator.intercept_[0])])
        names = list(model.preprocessor.get_feature_names_out())
        units = "log_odds_linear_decomposition_not_SHAP"
        expected = model.estimator.decision_function(Z)
    else:
        import shap
        explainer = shap.TreeExplainer(model.estimator)
        Z = model.transform(raw)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        Z = np.asarray(Z, dtype=np.float32)
        explanation = explainer(Z)
        values = np.column_stack([explanation.values[:, :, 1], explanation.base_values[:, 1]])
        names = list(model.preprocessor.get_feature_names_out())
        units = "probability"
        expected = model.predict_proba(raw)[:, 1]
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values)
    # Group one-hot and missing indicators back to their corresponding engineered feature.
    base_names = sorted(X.columns, key=len, reverse=True)
    mapped = []
    for name in names:
        candidate = name.removeprefix("missingindicator_")
        mapped.append(next((b for b in base_names if candidate == b or candidate.startswith(b+"_")), name))
    aggregate = pd.DataFrame(values[:, :-1], columns=mapped).T.groupby(level=0, sort=False).sum().T
    additive_error = float(np.max(np.abs(aggregate.sum(axis=1).to_numpy()+values[:, -1]-expected)))
    assert additive_error < 2e-4, f"Component SHAP additivity failed: {additive_error}"
    return aggregate, values[:, -1], X, units, additive_error


def interpret(bundle, df, prediction, out, figures):
    weights = bundle["selected_weights"]
    selected = LoanEnsemble([bundle["models"][n] for n in weights], list(weights.values()))
    calibrator = bundle["calibrators"]["selected"]
    threshold = bundle["thresholds"]["selected"]["f1"]
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(df), size=min(10000, len(df)), replace=False)
    importance = grouped_permutation(selected, calibrator, df.iloc[chosen].copy(), df.target.iloc[chosen].to_numpy(), out)
    fig, ax = plt.subplots(figsize=(8, 5))
    d = importance.sort_values("mean")
    ax.barh(d.index, d["mean"], xerr=d["std"], color="#096B68")
    ax.set(xlabel="Disminución de AP al permutar (media ± DE, 5 repeticiones)", title="Importancia por grupos del predictor seleccionado")
    save(fig, figures/"importancia_permutacion.png")
    p = prediction.selected.to_numpy()
    y = df.target.to_numpy()
    case_indices, case_types = [], []
    for label, mask, descending in [("FP", (y == 0)&(p >= threshold), True),
                                    ("FN", (y == 1)&(p < threshold), False),
                                    ("TP", (y == 1)&(p >= threshold), True),
                                    ("TN", (y == 0)&(p < threshold), False)]:
        indices = np.flatnonzero(mask)
        ordered = indices[np.argsort(-p[indices] if descending else p[indices], kind="stable")[:3]]
        case_indices.extend(ordered.tolist())
        case_types.extend([label]*len(ordered))
    global_indices = rng.choice(len(df), size=min(1200, len(df)), replace=False)
    explain_indices = np.unique(np.r_[global_indices, case_indices])
    explain = df.iloc[explain_indices].copy()
    position = {int(v): i for i, v in enumerate(explain_indices)}
    global_position = [position[int(i)] for i in global_indices]
    explanations, metadata = [], {}
    for name, weight in weights.items():
        model = bundle["models"][name]
        print(f"Computing additive explanations: {name}", flush=True)
        contrib, bias, X, units, error = component_shap(model, explain)
        global_importance = contrib.iloc[global_position].abs().mean().sort_values(ascending=False)
        global_importance.rename("mean_abs_contribution").to_csv(out/f"shap_global_{name}.csv")
        metadata[name] = {"family": model.family, "weight": weight, "units": units,
                          "max_additivity_error": error, "global_sample_n": len(global_indices),
                          "explains": "uncalibrated component only; not additive to ensemble calibrated probability",
                          "reference": "tree-path-dependent" if model.family != "logistic" else "zero transformed inputs"}
        if name == max(weights, key=weights.get):
            fig, ax = plt.subplots(figsize=(8, 6))
            top = global_importance.head(18).sort_values()
            ax.barh(top.index, top.values, color="#096B68")
            method_label = "Descomposición lineal" if model.family == "logistic" else "SHAP"
            ax.set(xlabel=f"Media |contribución| ({units})", title=f"{method_label} del componente principal: {name}")
            save(fig, figures/"shap_global.png")
        for idx, label in zip(case_indices, case_types):
            pos = position[idx]
            contributions = contrib.iloc[pos]
            top = contributions.abs().nlargest(10).index
            explanations.append({"id": str(df.iloc[idx].id), "case": label, "component": name,
                "component_weight": weight, "target": int(y[idx]), "selected_probability": float(p[idx]),
                "selected_threshold": threshold, "component_raw_probability": float(model.predict_proba(df.iloc[[idx]])[0, 1]),
                "base_value": float(bias[pos]), "sum_contributions": float(contributions.sum()), "units": units,
                "top_contributions": [{"feature": feature, "contribution": float(contributions[feature]),
                    "value": str(X.iloc[pos][feature]) if feature in X else "encoded"} for feature in top]})
    write_json(out/"local_cases.json", explanations)
    write_json(out/"explanation_metadata.json", metadata)
    dominant = max(weights, key=weights.get)
    examples = [c for c in explanations if c["component"] == dominant]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label in zip(axes.flat, ["FP", "FN", "TP", "TN"]):
        c = next(c for c in examples if c["case"] == label)
        vals = list(reversed(c["top_contributions"][:7]))
        ax.barh([v["feature"] for v in vals], [v["contribution"] for v in vals],
                color=["#C05935" if v["contribution"] > 0 else "#096B68" for v in vals])
        ax.axvline(0, color="#555", lw=.7)
        ax.set_title(f"{label}: resultado={c['target']}, riesgo final={c['selected_probability']:.1%}")
        ax.set_xlabel("Contribución a la probabilidad del componente" if c["units"] == "probability" else "Contribución al log-odds del componente")
    fig.suptitle("Casos del modelo congelado: explicaciones del componente principal", y=1.01, fontweight="bold")
    fig.tight_layout()
    save(fig, figures/"casos_locales.png")
    robust = []
    base = metrics(df.target.iloc[chosen], p[chosen], threshold)
    for name, cols in [("income_missing", ["annual_inc", "annual_inc_joint"]),
                       ("FICO_missing", ["fico_range_low", "fico_range_high"]),
                       ("geography_missing", ["addr_state"]),
                       ("DTI_missing", ["dti", "dti_joint"])]:
        for fraction in [.1, .3]:
            raw = df.iloc[chosen].copy()
            affected = rng.choice(len(raw), size=int(len(raw)*fraction), replace=False)
            for col in cols:
                raw.iloc[affected, raw.columns.get_loc(col)] = np.nan
            q = calibrator.predict(selected.predict_proba(raw)[:, 1])
            row = {"scenario": name, "fraction": fraction, **metrics(raw.target, q, threshold)}
            row["AP_delta_vs_intact"] = row["AP"]-base["AP"]
            row["Brier_delta_vs_intact"] = row["Brier"]-base["Brier"]
            robust.append(row)
    pd.DataFrame(robust).drop(columns="confusion_matrix").to_csv(out/"missing_input_stress.csv", index=False)


def diagnostics(data_dir, output):
    data_dir, output = Path(data_dir), Path(output)
    receipt = json.loads((output/"final"/"evaluation_receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "complete"
    lock = json.loads((output/"selection_lock.json").read_text(encoding="utf-8"))
    assert lock["model_sha256"] == file_sha256(output/"models"/"frozen_bundle.joblib")
    bundle = joblib.load(output/"models"/"frozen_bundle.joblib")
    out, figures = output/"analysis", output/"figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(output/"final"/"metrics.csv")
    selection = pd.read_csv(output/"selection_leaderboard.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, frame, title in [(axes[0], selection, "Selección: 2014 H1"),
                             (axes[1], final.query("cohort == 2015 and scope == 'all_resolved'"), "Prueba independiente: 2015")]:
        name_col = "name" if "name" in frame else "model"
        frame = frame.sort_values("AP")
        ax.barh(frame[name_col], frame.AP, color=[COLOR.get(n, "#8B9EAE") for n in frame[name_col]])
        ax.set(xlabel="Average Precision (AP)", title=title)
        ax.axvline(frame.prevalence.iloc[0], color="#555", linestyle="--", label="Prevalencia")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures/"comparacion_modelos.png")
    dev = pd.read_parquet(data_dir/"development.parquet")
    train = dev[dev.issue_date.le("2014-06-30")]
    X_train = FeatureBuilder().fit_transform(train)
    audit = pd.DataFrame({"feature": X_train.columns,
                          "missing_fraction_train": X_train.isna().mean().values,
                          "unique_values_train": X_train.nunique().values,
                          "origin": ["origination_raw" if c in BASE_NUMERIC+BUREAU_NUMERIC+CATEGORICAL else "engineered_origination_only" for c in X_train]})
    audit["uninformative_on_train"] = audit.unique_values_train.le(1)
    assert not FORBIDDEN.intersection(audit.feature)
    audit.to_csv(out/"feature_audit.csv", index=False)
    inventory = []
    for name, model in bundle["models"].items():
        inventory.append({"model": name, "family": model.family,
                          "extended_features": model.extended, "seed": model.seed,
                          "transformed_columns": model.transform(train.iloc[:1]).shape[1],
                          "trees": model.best_iteration_ if model.family in ["catboost", "lightgbm", "xgboost"] else (model.estimator.n_estimators if model.family == "random_forest" else None),
                          "selected_weight": bundle["selected_weights"].get(name, 0)})
    pd.DataFrame(inventory).to_csv(out/"model_inventory.csv", index=False)
    all_segments, all_reliability, all_drift = [], [], []
    cohorts, preds = {}, {}
    for year in [2015, 2016]:
        df = pd.read_parquet(data_dir/f"holdout_{year}.parquet")
        pred = pd.read_parquet(output/"predictions"/f"external_{year}.parquet")
        assert df.id.to_list() == pred.id.to_list()
        cohorts[year], preds[year] = df, pred
        threshold = bundle["thresholds"]["selected"]["f1"]
        all_segments.extend(segments(df, pred.selected.to_numpy(), threshold, year))
        for name in ["baseline19", "selected"]:
            for mode in ["raw", "calibrated"]:
                column = name+"_raw" if mode == "raw" else name
                r = reliability(df.target, pred[column].to_numpy())
                r["cohort"], r["model"], r["mode"] = year, name, mode
                all_reliability.append(r)
        X = FeatureBuilder().transform(df)
        for feature in ["annual_inc", "dti", "fico_midpoint", "loan_amnt", "revol_util", "credit_history_months"]:
            reference = X_train[feature].dropna()
            bins = np.unique(np.r_[-np.inf, reference.quantile(np.linspace(.1, .9, 9)).values, np.inf])
            a = pd.cut(X_train[feature], bins).value_counts(normalize=True, sort=False).to_numpy()
            b = pd.cut(X[feature], bins).value_counts(normalize=True, sort=False).to_numpy()
            a, b = np.maximum(a, 1e-5), np.maximum(b, 1e-5)
            all_drift.append({"cohort": year, "feature": feature, "PSI_nonmissing": float(np.sum((b-a)*np.log(b/a))),
                              "missing_train": float(X_train[feature].isna().mean()), "missing_test": float(X[feature].isna().mean())})
    seg = pd.DataFrame(all_segments).drop(columns="confusion_matrix")
    seg.to_csv(out/"segment_errors.csv", index=False)
    rel = pd.concat(all_reliability, ignore_index=True)
    rel.to_csv(out/"reliability.csv", index=False)
    pd.DataFrame(all_drift).to_csv(out/"distribution_drift.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, year in zip(axes, [2015, 2016]):
        for name in ["baseline19", "selected"]:
            for mode, style in [("raw", ":"), ("calibrated", "-")]:
                r = rel.query("cohort == @year and model == @name and mode == @mode")
                ax.plot(r.mean_prediction, r.observed_rate, style, color=COLOR[name], marker="o", ms=3, label=f"{name}: {mode}")
        ax.plot([0, .75], [0, .75], "--", color="#555", lw=.8)
        ax.set(xlabel="Riesgo medio predicho", ylabel="Incumplimiento observado", title=f"Calibración: {year}", xlim=(0, .75), ylim=(0, .75))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures/"calibracion_temporal.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    temporal = seg.query("dimension == 'issue_month'").copy().sort_values("segment")
    for ax, col, ylabel in [(axes[0], "AP", "Average Precision (AP)"), (axes[1], "mean_predicted_risk", "Proporción")]:
        x = np.arange(len(temporal))
        ax.plot(x, temporal[col], color=COLOR["selected"], marker="o", ms=3, label="Modelo seleccionado")
        if col != "AP":
            ax.plot(x, temporal.prevalence, color=COLOR["baseline19"], marker="s", ms=3, label="Tasa observada")
        ax.axvline(11.5, color="#777", linestyle="--", lw=.8)
        ax.set_xticks(x[::3], temporal.segment.iloc[::3], rotation=35)
        ax.set(ylabel=ylabel, title="Estabilidad por mes de emisión")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures/"estabilidad_temporal.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dimension in zip(axes, ["FICO", "income"]):
        frame = seg.query("cohort == 2015 and dimension == @dimension and low_support == False").sort_values("segment")
        ypos = np.arange(len(frame))
        ax.barh(ypos, frame.FNR, color="#C05935", alpha=.85)
        ax.errorbar(frame.FNR, ypos, xerr=np.vstack([frame.FNR-frame.FNR_low, frame.FNR_high-frame.FNR]), fmt="none", ecolor="#333", capsize=3)
        ax.set_yticks(ypos, frame.segment)
        ax.set(xlabel="Falsos negativos / incumplimientos (IC Wilson 95%)", title=f"Errores por {dimension} — 2015", xlim=(0, 1))
    fig.tight_layout()
    save(fig, figures/"errores_segmentos.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df, pred = cohorts[2015], preds[2015]
    for name in ["baseline19", "selected"]:
        precision, recall, _ = precision_recall_curve(df.target, pred[name])
        fpr, tpr, _ = roc_curve(df.target, pred[name])
        axes[0].plot(recall, precision, color=COLOR[name], label=name)
        axes[1].plot(fpr, tpr, color=COLOR[name], label=name)
    axes[0].axhline(df.target.mean(), color="#777", linestyle="--")
    axes[1].plot([0, 1], [0, 1], "--", color="#777")
    axes[0].set(xlabel="Recall", ylabel="Precisión", title="Prueba 2015: precisión-recall")
    axes[1].set(xlabel="Tasa de falsos positivos", ylabel="Recall", title="Prueba 2015: ROC")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    save(fig, figures/"curvas_pr_roc.png")
    interpret(bundle, cohorts[2015], preds[2015], out, figures)
    write_json(out/"analysis_scope.json", {"frozen_model_sha256": lock["model_sha256"],
               "model_selection_changed": False, "permutation_test_sample_n": 10000,
               "SHAP_global_sample_n": 1200, "global_seed": 42,
               "segment_support_rule": "low_support if n<300 or positive/negative count<30",
               "fairness_limit": "No protected attributes or rejected applicants; segment disparities do not establish fairness",
               "maturity_limit": "Read final/maturity_coverage.csv before interpreting 2015 60m or 2016 scores"})
    print("Post-freeze diagnostics completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        diagnostics(args.data_dir, args.output)
