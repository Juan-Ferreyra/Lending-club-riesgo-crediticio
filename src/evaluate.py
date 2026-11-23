"""Metrics, fixed decision policies, calibration and paired uncertainty."""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             log_loss, precision_recall_curve, confusion_matrix, roc_curve)


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    assert len(y) == len(p) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    tp = int(((p >= threshold) & (y == 1)).sum())
    fp = int(((p >= threshold) & (y == 0)).sum())
    fn = int(((p < threshold) & (y == 1)).sum())
    tn = int(((p < threshold) & (y == 0)).sum())
    precision = tp / max(tp+fp, 1)
    recall = tp / max(tp+fn, 1)
    two_classes = len(np.unique(y)) == 2
    fpr, tpr, _ = roc_curve(y, p) if two_classes else ([0], [0], None)
    k = max(1, int(np.ceil(len(y)*0.2)))
    top = np.argsort(-p, kind="stable")[:k]
    return {"n": len(y), "positive_n": int(y.sum()), "prevalence": float(y.mean()),
            "AP": float(average_precision_score(y, p)) if two_classes else None,
            "ROC_AUC": float(roc_auc_score(y, p)) if two_classes else None,
            "Brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "precision": precision, "recall": recall, "F1": 2*precision*recall/max(precision+recall, 1e-15),
            "threshold": float(threshold), "flag_rate": float((p >= threshold).mean()),
            "FPR": fp/max(fp+tn, 1), "FNR": fn/max(fn+tp, 1), "KS": float(np.max(np.asarray(tpr)-np.asarray(fpr))),
            "recall_at_top20": float(y[top].sum()/max(y.sum(), 1)),
            "precision_at_top20": float(y[top].mean()),
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def choose_thresholds(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    best = int(np.argmax(f1))
    eligible = np.flatnonzero(recall[:-1] >= 0.60)
    # Highest threshold that still meets the declared recall on development.
    recall_idx = eligible[-1] if len(eligible) else 0
    return {"f1": float(thresholds[best]), "recall60": float(thresholds[recall_idx]),
            "capacity20": float(np.quantile(p, 0.8)), "default": 0.5}


class ProbabilityCalibrator:
    def __init__(self, method="identity"):
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1000, max_iter=1000).fit(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        elif self.method != "identity":
            raise ValueError(self.method)
        return self

    def predict(self, p):
        p = np.asarray(p)
        if self.method == "sigmoid":
            p = self.model.predict_proba(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            p = self.model.predict(p)
        return np.clip(p, 1e-6, 1-1e-6)


class WeightedRankMetrics:
    """Exact weighted AP and ROC-AUC with tied prediction scores."""
    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.order = np.argsort(-self.p, kind="stable")
        ordered = self.p[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]

    def compute(self, weights):
        weights = np.asarray(weights, dtype=float)
        w = weights[self.order]
        ys = self.y[self.order]
        tp = np.cumsum(w*ys)[self.ends]
        fp = np.cumsum(w*(1-ys))[self.ends]
        if tp[-1] == 0 or fp[-1] == 0:
            return np.array([np.nan]*3)
        ap = np.sum(np.diff(np.r_[0, tp])/tp[-1] * np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0))
        auc = np.trapezoid(np.r_[0, tp/tp[-1]], np.r_[0, fp/fp[-1]])
        brier = np.average((self.y-self.p)**2, weights=weights)
        return np.array([ap, auc, brier])


def paired_bootstrap(y, p, baseline, months, reps=300, seed=42):
    rng = np.random.default_rng(seed)
    a, b = WeightedRankMetrics(y, p), WeightedRankMetrics(y, baseline)
    _, groups = np.unique(np.asarray(months).astype(str), return_inverse=True)
    n, blocks = len(y), int(groups.max()+1)
    result = {}
    for kind in ["row", "issue_month_block"]:
        samples = []
        for _ in range(reps):
            if kind == "row":
                weights = np.bincount(rng.integers(0, n, size=n), minlength=n)
            else:
                weights = np.bincount(rng.integers(0, blocks, size=blocks), minlength=blocks)[groups]
            ma, mb = a.compute(weights), b.compute(weights)
            samples.append(np.r_[ma, ma[0]-mb[0]])
        samples = np.asarray(samples)
        result[kind] = {name: {"low": float(np.nanquantile(samples[:, i], .025)),
                              "high": float(np.nanquantile(samples[:, i], .975))}
                        for i, name in enumerate(["AP", "ROC_AUC", "Brier", "paired_AP_delta_vs_baseline"])}
        result[kind]["repetitions"] = reps
    return result
