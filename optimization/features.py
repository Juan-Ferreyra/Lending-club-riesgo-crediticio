"""Train-only origination features for the third optimization experiment.

The baseline is the unchanged V2 stable transformer. Financial extensions use
only application and origination bureau fields. Employer and ZIP encodings use
unlabelled training frequencies, never outcomes or validation frequencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from research.models import AuditFeatures
from src.features import FORBIDDEN


MISSING = "__MISSING__"
RARE = "__RARE__"
UNSEEN = "__UNSEEN__"
CONTEXT_RAW = ("emp_title", "zip_code")


def _numeric(raw, name):
    """Nonnegative monetary balances/counts; invalid values remain missing."""
    values = raw[name] if name in raw else pd.Series(np.nan, index=raw.index)
    values = pd.to_numeric(values, errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.ge(0))


def _ratio(numerator, denominator):
    return numerator.div(denominator.where(denominator.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )


def _context_values(raw, name):
    values = raw[name] if name in raw else pd.Series(pd.NA, index=raw.index)
    values = values.astype("string").str.strip().str.lower()
    if name == "emp_title":
        values = values.str.replace(r"[^\w\s]", " ", regex=True)
        values = values.str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        # Preserve only the published coarse three-digit area, never a full ZIP.
        values = values.str.extract(r"^(\d{3})(?:xx|\d{2})?$", expand=False)
    return values.replace("", pd.NA).fillna(MISSING).astype(str)


def financial_features(raw):
    """Interactions unavailable to a purely additive borrower score.

    Ratios describe balances, available credit and account mix at origination.
    ``principal_burden_plus_dti`` is a principal-only proxy, not a contractual
    installment: it deliberately excludes Lending Club's assigned interest rate.
    Missing/invalid inputs propagate; no arbitrary upper truncation is applied.
    """
    names = [
        "loan_amnt", "annual_inc", "dti", "revol_bal", "total_rev_hi_lim",
        "bc_open_to_buy", "tot_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort",
        "total_bc_limit", "num_actv_rev_tl", "num_op_rev_tl",
        "num_rev_tl_bal_gt_0", "num_rev_accts", "num_actv_bc_tl", "num_bc_sats",
        "acc_open_past_24mths", "total_acc", "inq_last_6mths", "open_acc",
        "num_accts_ever_120_pd",
    ]
    n = {name: _numeric(raw, name) for name in names}
    income = n["annual_inc"].where(n["annual_inc"].gt(0))
    available_revolving = (n["total_rev_hi_lim"] - n["revol_bal"]).clip(lower=0)
    free_income = income * (1 - n["dti"] / 100)
    mortgage_balance = n["tot_cur_bal"] - n["total_bal_ex_mort"]
    bankcard_balance = n["total_bc_limit"] - n["bc_open_to_buy"]
    term_raw = raw["term"] if "term" in raw else pd.Series(pd.NA, index=raw.index)
    term = pd.to_numeric(term_raw.astype("string").str.extract(r"(\d+)", expand=False), errors="coerce")
    issue_raw = raw["issue_d"] if "issue_d" in raw else pd.Series(pd.NA, index=raw.index)
    earliest_raw = raw["earliest_cr_line"] if "earliest_cr_line" in raw else pd.Series(pd.NA, index=raw.index)
    issue = pd.to_datetime(issue_raw, format="%b-%Y", errors="coerce")
    earliest = pd.to_datetime(earliest_raw, format="%b-%Y", errors="coerce")
    history = (issue.dt.year - earliest.dt.year) * 12 + issue.dt.month - earliest.dt.month
    history = history.where(history.gt(0))
    values = {
        "credit_capacity_to_income": _ratio(n["total_rev_hi_lim"], income),
        "available_revolving_to_income": _ratio(available_revolving, income),
        "loan_to_revolving_limit": _ratio(n["loan_amnt"], n["total_rev_hi_lim"]),
        "loan_to_available_revolving": _ratio(n["loan_amnt"], available_revolving),
        "loan_to_available_bankcard": _ratio(n["loan_amnt"], n["bc_open_to_buy"]),
        "balance_to_total_credit_limit": _ratio(n["tot_cur_bal"], n["tot_hi_cred_lim"]),
        "nonmortgage_balance_share": _ratio(n["total_bal_ex_mort"], n["tot_cur_bal"]),
        "bankcard_limit_share": _ratio(n["total_bc_limit"], n["total_rev_hi_lim"]),
        "active_revolving_share": _ratio(n["num_actv_rev_tl"], n["num_op_rev_tl"]),
        "revolving_with_balance_share": _ratio(n["num_rev_tl_bal_gt_0"], n["num_rev_accts"]),
        "active_bankcard_share": _ratio(n["num_actv_bc_tl"], n["num_bc_sats"]),
        "recent_account_share": _ratio(n["acc_open_past_24mths"], n["total_acc"]),
        "inquiries_per_open_account": _ratio(n["inq_last_6mths"], n["open_acc"]),
        "severe_delinquency_share": _ratio(n["num_accts_ever_120_pd"], n["total_acc"]),
        "accounts_per_history_year": _ratio(n["total_acc"] * 12, history),
        "loan_to_annual_free_income": _ratio(n["loan_amnt"], free_income),
        "principal_burden_plus_dti": _ratio(_ratio(n["loan_amnt"], term), income / 12) + n["dti"] / 100,
        "revolving_balance_per_active_account": _ratio(n["revol_bal"], n["num_actv_rev_tl"]),
        "bankcard_balance_to_income": _ratio(bankcard_balance.where(bankcard_balance.ge(0)), income),
        "mortgage_balance_to_income": _ratio(mortgage_balance.where(mortgage_balance.ge(0)), income),
    }
    return pd.DataFrame({"v3_" + key: value for key, value in values.items()}, index=raw.index).astype(float)


class V3Features(BaseEstimator, TransformerMixin):
    """V2 baseline plus prespecified feature groups and fitted preparation.

    ``mode`` selects baseline, financial, context or combined. ``preparation``
    selects native missing values, train-quantile clipping, or training-median
    imputation with training-observed missingness indicators. Models can consume
    ``cats`` and ``nums`` to configure their own encoders/scalers. Engineered
    columns with <=1 distinct value or >95% missing in training are discarded.
    """

    def __init__(self, mode="baseline", preparation="native", min_category_count=50,
                 max_categories=1500):
        self.mode = mode
        self.preparation = preparation
        self.min_category_count = min_category_count
        self.max_categories = max_categories

    def fit(self, raw, y=None):
        if self.mode not in {"baseline", "financial", "context", "combined"}:
            raise ValueError("Unknown feature mode: " + str(self.mode))
        if self.preparation not in {"native", "clip", "median"}:
            raise ValueError("Unknown feature preparation: " + str(self.preparation))
        if self.min_category_count < 1 or self.max_categories < 1:
            raise ValueError("Category limits must be positive")
        self.base_ = AuditFeatures("stable").fit(raw)
        self.context_counts_ = {}
        self.context_vocabulary_ = {}
        if self.mode in {"context", "combined"}:
            for name in CONTEXT_RAW:
                normalized = _context_values(raw, name)
                counts = normalized[normalized.ne(MISSING)].value_counts()
                # Stable lexical tie-break makes the cap deterministic.
                counts = counts.rename_axis("value").reset_index(name="count").sort_values(
                    ["count", "value"], ascending=[False, True]
                ).set_index("value")["count"]
                self.context_counts_[name] = counts.to_dict()
                eligible = counts[counts.ge(self.min_category_count)].head(self.max_categories)
                self.context_vocabulary_[name] = set(eligible.index)
        X = self._unprepared(raw)
        extra = [name for name in X if name not in self.base_.columns]
        keep_extra = [name for name in extra if X[name].nunique() > 1 and X[name].isna().mean() <= .95]
        self.retained_columns_ = list(self.base_.columns) + keep_extra
        self.cats = list(self.base_.cats) + [name for name in keep_extra if name.endswith("_category")]
        self.numeric_columns_ = [name for name in self.retained_columns_ if name not in self.cats]
        numeric = X[self.numeric_columns_]
        self.lo_ = numeric.quantile(.005)
        self.hi_ = numeric.quantile(.995)
        self.medians_ = numeric.median().fillna(0)
        self.indicators_ = [name for name in self.numeric_columns_ if numeric[name].isna().any()]
        indicators = [name + "__missing" for name in self.indicators_] if self.preparation == "median" else []
        self.columns = self.retained_columns_ + indicators
        self.nums = self.numeric_columns_ + indicators
        self.feature_names_out_ = np.asarray(self.columns, dtype=object)
        if FORBIDDEN.intersection(self.columns):
            raise AssertionError("Forbidden outcome/pricing field entered the feature contract")
        return self

    def _unprepared(self, raw):
        pieces = [self.base_.transform(raw)]
        if self.mode in {"financial", "combined"}:
            pieces.append(financial_features(raw))
        if self.mode in {"context", "combined"}:
            context = {}
            for name in CONTEXT_RAW:
                normalized = _context_values(raw, name)
                counts = normalized.map(self.context_counts_[name]).fillna(0)
                vocab = self.context_vocabulary_[name]
                category = normalized.where(normalized.isin(vocab), RARE)
                category = category.where(counts.gt(0), UNSEEN)
                category = category.where(normalized.ne(MISSING), MISSING)
                context["v3_" + name + "_category"] = category.astype(str)
                context["v3_" + name + "_log_frequency"] = np.log1p(counts).astype(float)
            pieces.append(pd.DataFrame(context, index=raw.index))
        return pd.concat(pieces, axis=1)

    def transform(self, raw):
        check_is_fitted(self, "feature_names_out_")
        X = self._unprepared(raw)[self.retained_columns_].copy()
        if self.preparation == "clip":
            X[self.numeric_columns_] = X[self.numeric_columns_].clip(self.lo_, self.hi_, axis=1)
        elif self.preparation == "median":
            flags = {name + "__missing": X[name].isna().astype(float) for name in self.indicators_}
            X[self.numeric_columns_] = X[self.numeric_columns_].fillna(self.medians_)
            X = pd.concat([X, pd.DataFrame(flags, index=raw.index)], axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
