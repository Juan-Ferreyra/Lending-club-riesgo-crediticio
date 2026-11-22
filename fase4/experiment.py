"""Fase 4: comparación de familias adicionales y técnicas del curso.

Alcance (fijado en experiments/protocol_v4.json antes de ejecutar):
- Préstamos a 36 meses con desenlace terminal, emitidos 2012-01 a 2014-12.
- Entrenamiento: 2012-01-01 a 2014-09-30. Selección/diagnóstico: Q4 2014.
- No se leen ni se puntúan préstamos de 2015 o 2016.
- Todos los candidatos usan las mismas variables de originación (src.features.FeatureBuilder)
  y las mismas filas de validación. La referencia interna es LightGBM con parámetros fijos.

Uso:
    python -m fase4.experiment --source data/accepted_2007_to_2018Q4.csv.gz --output outputs/research_v4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

from src.features import CATEGORICAL, IDENTIFIERS, RAW_PREDICTORS, FeatureBuilder
from src.utils import write_json, utc_now

SEED = 42
TARGET_STATUSES = ("Fully Paid", "Charged Off")
TRAIN_END = pd.Timestamp("2014-09-30")
SELECT_START, SELECT_END = pd.Timestamp("2014-10-01"), pd.Timestamp("2014-12-31")
GEO = ["addr_state"]
SMALL_N = 60_000  # presupuesto de filas para familias lentas (como RF/SVM en V2)


# ----------------------------------------------------------------------------- datos
def load_development(source: str | Path) -> pd.DataFrame:
    cols = list(dict.fromkeys(IDENTIFIERS + RAW_PREDICTORS))
    parts = []
    for chunk in pd.read_csv(source, usecols=cols, dtype=str, chunksize=200_000):
        chunk = chunk[chunk.loan_status.isin(TARGET_STATUSES) & chunk.term.str.contains("36", na=False)]
        date = pd.to_datetime(chunk.issue_d, format="%b-%Y", errors="coerce")
        keep = (date >= "2012-01-01") & (date <= SELECT_END)
        chunk = chunk[keep].copy()
        chunk["issue_date"] = date[keep]
        parts.append(chunk)
    data = pd.concat(parts, ignore_index=True)
    data["target"] = data.loan_status.eq("Charged Off").astype(int)
    assert data.issue_date.max() <= SELECT_END, "No debe leerse ningún préstamo posterior a 2014"
    return data


# ----------------------------------------------------------------------------- pipelines
def _split_columns(X):
    cats = [c for c in CATEGORICAL if c in X]
    return cats, [c for c in X if c not in cats]


def tree_preprocessor(cats, nums):
    return ColumnTransformer([
        ("num", "passthrough", nums),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cats),
    ], verbose_feature_names_out=False)


def linear_preprocessor(cats, nums, quantile=False):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    num_steps.append(("scale", QuantileTransformer(output_distribution="normal", random_state=SEED,
                                                   subsample=100_000) if quantile else StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cats),
    ], verbose_feature_names_out=False)


def candidates(cats, nums, empty=()):
    """Cada entrada: (nombre, familia_curso, pipeline, filas_max, notas)."""
    nums_ok = [c for c in nums if c not in set(empty)]
    cats_nogeo = [c for c in cats if c not in GEO]
    from lightgbm import LGBMClassifier
    lgb_params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                      reg_lambda=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=SEED, n_jobs=6, verbosity=-1, deterministic=True, force_col_wise=True)
    out = [
        ("lightgbm_referencia", "6.6.5 Boosting", Pipeline([
            ("prep", tree_preprocessor(cats, nums)), ("model", LGBMClassifier(**lgb_params))]), None,
         "Referencia interna con parámetros fijos; mismas variables que todos"),
        ("lightgbm_sin_geografia", "6.10.1 Sesgo algorítmico", Pipeline([
            ("prep", tree_preprocessor(cats_nogeo, nums)), ("model", LGBMClassifier(**lgb_params))]), None,
         "Igual que la referencia pero sin addr_state (proxy geográfico)"),
        ("hist_gradient_boosting", "6.6.5 Boosting (sklearn)", Pipeline([
            ("prep", tree_preprocessor(cats, nums_ok)),
            ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                                     l2_regularization=5.0, min_samples_leaf=200,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     random_state=SEED))]), None,
         "Early stopping con 10% aleatorio interno; categorías como códigos ordinales "
         "(la opción categórica nativa falló en la primera ejecución); se excluyen columnas 100% faltantes en entrenamiento, que HistGB no admite"),
        ("logistica_l2", "6.6.1 Regresión logística", Pipeline([
            ("prep", linear_preprocessor(cats, nums)),
            ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))]), None,
         "Baseline lineal (regularización L2)"),
        ("logistica_l1", "6.3.1 Regularización L1", Pipeline([
            ("prep", linear_preprocessor(cats, nums)),
            ("model", LogisticRegression(penalty="l1", C=0.01, solver="saga", max_iter=300, tol=1e-3,
                                         random_state=SEED))]), SMALL_N,
         "Selección de variables por L1; se reportan coeficientes no nulos"),
        ("logistica_cuantiles", "6.4.3 Transformaciones", Pipeline([
            ("prep", linear_preprocessor(cats, nums, quantile=True)),
            ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))]), None,
         "Feature map por cuantiles normales (robusto a colas pesadas)"),
        ("pca_logistica", "6.8.1 PCA", Pipeline([
            ("prep", linear_preprocessor(cats, nums)), ("pca", PCA(n_components=0.95, random_state=SEED)),
            ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))]), None,
         "PCA conservando 95% de varianza antes de la logística"),
        ("naive_bayes", "6.5.4 Naive Bayes", Pipeline([
            ("prep", linear_preprocessor(cats, nums, quantile=True)), ("model", GaussianNB())]), None,
         "Gaussiano sobre variables transformadas a normal"),
        ("arbol_decision", "6.6.3 Árboles de decisión", Pipeline([
            ("prep", tree_preprocessor(cats, nums)),
            ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=500, random_state=SEED))]), None,
         "Árbol poco profundo e interpretable"),
        ("extra_trees", "6.6.5 Bagging", Pipeline([
            ("prep", linear_preprocessor(cats, nums)),
            ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.5,
                                           n_jobs=6, random_state=SEED))]), SMALL_N,
         f"Submuestra de {SMALL_N} filas por costo"),
        ("mlp", "6.7 Redes neuronales", Pipeline([
            ("prep", linear_preprocessor(cats, nums, quantile=True)),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512,
                                    learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                    n_iter_no_change=5, random_state=SEED))]), None,
         "Feed-forward 64-32, regularización L2 (alpha) y early stopping"),
    ]
    return out


# ----------------------------------------------------------------------------- métricas
def summarize(y, p):
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(0.2 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return {"AP": float(average_precision_score(y, p)), "ROC_AUC": float(roc_auc_score(y, p)),
            "Brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "recall_at_top20": float(y[top].sum() / y.sum())}


def paired_bootstrap_ap(y, p, q, reps=300, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p, q = map(np.asarray, (y, p, q))
    deltas = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(average_precision_score(y[idx], p[idx]) - average_precision_score(y[idx], q[idx]))
    return {"delta": float(average_precision_score(y, p) - average_precision_score(y, q)),
            "ci95_low": float(np.quantile(deltas, 0.025)), "ci95_high": float(np.quantile(deltas, 0.975)),
            "reps": reps, "kind": "row"}


def segment_errors(frame, score_col, threshold):
    rows = []
    inc = pd.to_numeric(frame.annual_inc, errors="coerce")
    segs = {
        "home_ownership": frame.home_ownership.fillna("NA"),
        "income_quintile": pd.qcut(inc.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str),
        "verification_status": frame.verification_status.fillna("NA"),
    }
    flag = frame[score_col] >= threshold
    for dim, values in segs.items():
        for seg in sorted(values.unique()):
            m = values.eq(seg)
            y, f = frame.target[m], flag[m]
            pos, neg = (y == 1).sum(), (y == 0).sum()
            rows.append({"model": score_col, "dimension": dim, "segment": seg, "n": int(m.sum()),
                         "prevalence": float(y.mean()), "flag_rate": float(f.mean()),
                         "TPR": float((f & (y == 1)).sum() / max(pos, 1)),
                         "FPR": float((f & (y == 0)).sum() / max(neg, 1))})
    return rows


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/accepted_2007_to_2018Q4.csv.gz")
    ap.add_argument("--output", default="outputs/research_v4")
    ap.add_argument("--only", nargs="*", help="Ejecutar solo estos candidatos")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_development(args.source)
    train = data[data.issue_date <= TRAIN_END].reset_index(drop=True)
    valid = data[(data.issue_date >= SELECT_START)].reset_index(drop=True)
    fb = FeatureBuilder(extended=True).fit(train)
    Xtr, Xva = fb.transform(train), fb.transform(valid)
    cats, nums = _split_columns(Xtr)
    empty = [c for c in nums if Xtr[c].isna().all()]
    write_json(out / "data_scope.json", {
        "created_utc": utc_now(), "train_n": len(train), "train_positive_rate": float(train.target.mean()),
        "selection_n": len(valid), "selection_positive_rate": float(valid.target.mean()),
        "train_range": [str(train.issue_date.min().date()), str(train.issue_date.max().date())],
        "selection_range": [str(valid.issue_date.min().date()), str(valid.issue_date.max().date())],
        "n_features": Xtr.shape[1], "categorical": cats, "all_missing_in_train": empty, "load_seconds": round(time.time() - t0, 1),
        "opened_2015_2016": False})
    print(f"datos: train={len(train)} valid={len(valid)} feats={Xtr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    small_idx = np.sort(rng.choice(len(train), size=min(SMALL_N, len(train)), replace=False))
    preds = pd.DataFrame({"id": valid.id, "target": valid.target})
    results, extra = [], {}
    for name, topic, pipe, max_rows, note in candidates(cats, nums, empty):
        if args.only and name not in args.only:
            continue
        idx = small_idx if max_rows else np.arange(len(train))
        Xfit = Xtr.iloc[idx]
        if name == "lightgbm_sin_geografia":
            Xfit, Xeval = Xfit.drop(columns=GEO), Xva.drop(columns=GEO)
        else:
            Xeval = Xva
        start = time.time()
        try:
            pipe.fit(Xfit, train.target.iloc[idx])
            p = pipe.predict_proba(Xeval)[:, 1]
            status = "OK"
        except Exception as exc:  # se registra como resultado negativo
            print(f"  {name}: FALLÓ {exc!r}", flush=True)
            results.append({"model": name, "topic": topic, "status": f"FAILED: {exc!r}"[:200]})
            continue
        secs = time.time() - start
        preds[name] = p
        row = {"model": name, "topic": topic, "status": status, "train_n": len(idx),
               "selection_n": len(valid), "seconds": round(secs, 1), "notes": note, **summarize(valid.target, p)}
        model = pipe.named_steps["model"]
        if name == "logistica_l1":
            coef = model.coef_.ravel()
            names = pipe.named_steps["prep"].get_feature_names_out()
            row["nonzero_coefficients"] = int((coef != 0).sum())
            row["total_coefficients"] = int(len(coef))
            extra["l1_coefficients"] = sorted(
                [{"feature": f, "coef": float(c)} for f, c in zip(names, coef) if c != 0],
                key=lambda d: -abs(d["coef"]))
        if name == "pca_logistica":
            row["pca_components"] = int(pipe.named_steps["pca"].n_components_)
        if name == "arbol_decision":
            row["tree_leaves"] = int(model.get_n_leaves())
        results.append(row)
        print(f"  {name}: AP={row['AP']:.4f} AUC={row['ROC_AUC']:.4f} ({secs:.0f}s)", flush=True)
        pd.DataFrame(results).to_csv(out / "leaderboard_q4.csv", index=False)

    board = pd.DataFrame(results)
    ok = board[board.status.eq("OK")].sort_values("AP", ascending=False)
    board.to_csv(out / "leaderboard_q4.csv", index=False)
    if "lightgbm_referencia" in preds:
        ref = preds["lightgbm_referencia"]
        comps = {m: paired_bootstrap_ap(valid.target, preds[m], ref) for m in ok.model if m != "lightgbm_referencia"}
        write_json(out / "paired_vs_reference.json", comps)
        thr = float(np.quantile(ref, 0.8))
        seg = segment_errors(valid.assign(lightgbm_referencia=ref), "lightgbm_referencia", thr)
        if "lightgbm_sin_geografia" in preds:
            g = preds["lightgbm_sin_geografia"]
            seg += segment_errors(valid.assign(lightgbm_sin_geografia=g), "lightgbm_sin_geografia",
                                  float(np.quantile(g, 0.8)))
        pd.DataFrame(seg).to_csv(out / "fairness_segments_top20.csv", index=False)
    if extra:
        write_json(out / "l1_selected_features.json", extra)

    # Curva de aprendizaje (6.3.2) con la referencia
    if not args.only or "learning_curve" in args.only:
        from lightgbm import LGBMClassifier
        lc = []
        for n in [15_000, 30_000, 60_000, 120_000, len(train)]:
            sub = np.sort(rng.choice(len(train), size=n, replace=False)) if n < len(train) else np.arange(len(train))
            pipe = Pipeline([("prep", tree_preprocessor(cats, nums)), ("model", LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200, reg_lambda=5,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=6,
                verbosity=-1, deterministic=True, force_col_wise=True))])
            pipe.fit(Xtr.iloc[sub], train.target.iloc[sub])
            ptr = pipe.predict_proba(Xtr.iloc[sub[:30000]])[:, 1]
            pva = pipe.predict_proba(Xva)[:, 1]
            lc.append({"train_n": n, "train_AP": float(average_precision_score(train.target.iloc[sub[:30000]], ptr)),
                       "selection_AP": float(average_precision_score(valid.target, pva))})
            print(f"  curva n={n}: AP={lc[-1]['selection_AP']:.4f}", flush=True)
        pd.DataFrame(lc).to_csv(out / "learning_curve.csv", index=False)

    preds.to_parquet(out / "predictions_q4.parquet", index=False) if False else None  # predicciones fuera de Git
    write_json(out / "run_complete.json", {"finished_utc": utc_now(), "total_seconds": round(time.time() - t0, 1)})
    print("FIN", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
