"""Explicit origination-only feature contract and deterministic transformations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc",
]
CATEGORICAL = ["term", "emp_length", "home_ownership", "verification_status",
               "purpose", "addr_state", "application_type"]
BUREAU_NUMERIC = [
    "mths_since_last_delinq", "mths_since_last_record",
    "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "annual_inc_joint", "dti_joint", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths",
    "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit",
]
RAW_PREDICTORS = BASE_NUMERIC + CATEGORICAL + BUREAU_NUMERIC + ["earliest_cr_line", "issue_d"]
IDENTIFIERS = ["id", "loan_status"]
FORBIDDEN = {"target", "loan_status", "id", "member_id", "grade", "sub_grade", "int_rate",
             "installment", "funded_amnt", "funded_amnt_inv", "initial_list_status",
             "out_prncp", "out_prncp_inv", "total_pymnt", "recoveries", "last_pymnt_d",
             "last_fico_range_low", "last_fico_range_high", "last_credit_pull_d"}


def safe_ratio(a, b):
    return a.div(b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """No fitted statistics; the following model fits imputers/encoders on train only."""

    def __init__(self, extended=True):
        self.extended = extended

    def fit(self, X, y=None):
        result = self.transform(X)
        self.feature_names_out_ = result.columns.to_numpy()
        assert not FORBIDDEN.intersection(result.columns)
        return self

    def transform(self, X):
        numeric = BASE_NUMERIC + (BUREAU_NUMERIC if self.extended else [])
        result = pd.DataFrame(index=X.index)
        for col in numeric:
            value = X[col] if col in X else pd.Series(np.nan, index=X.index)
            result[col] = pd.to_numeric(value, errors="coerce").astype(float)
        # Fixed domain checks, not distribution-derived winsorization.
        for col in ["annual_inc", "annual_inc_joint"]:
            if col in result:
                result[col] = result[col].where(result[col] > 0)
        for col in ["dti", "dti_joint", "revol_util", "loan_amnt"]:
            if col in result:
                result[col] = result[col].where(result[col] >= 0)
        for col in ["fico_range_low", "fico_range_high"]:
            result[col] = result[col].where(result[col].between(300, 850))
        for col in CATEGORICAL:
            value = X[col] if col in X else pd.Series(np.nan, index=X.index)
            result[col] = value.fillna("__MISSING__").astype(str).str.strip().replace("", "__MISSING__")
        if self.extended:
            result["fico_midpoint"] = (result.fico_range_low + result.fico_range_high) / 2
            term = result["term"].str.extract(r"(\d+)", expand=False)
            result["term_months"] = pd.to_numeric(term, errors="coerce")
            employment = result.emp_length.str.extract(r"(\d+)", expand=False)
            result["employment_years"] = pd.to_numeric(employment, errors="coerce").astype(float)
            result.loc[result.emp_length.eq("< 1 year"), "employment_years"] = 0.5
            issue = pd.to_datetime(X.get("issue_d"), format="%b-%Y", errors="coerce")
            earliest = pd.to_datetime(X.get("earliest_cr_line"), format="%b-%Y", errors="coerce")
            history = (issue.dt.year-earliest.dt.year)*12 + issue.dt.month-earliest.dt.month
            result["credit_history_months"] = history.where(history >= 0)
            for col in ["annual_inc", "revol_bal", "loan_amnt", "tot_cur_bal", "tot_coll_amt"]:
                result[f"log1p_{col}"] = np.log1p(result[col].where(result[col] >= 0))
            result["loan_to_income"] = safe_ratio(result.loan_amnt, result.annual_inc)
            result["revolving_debt_to_income"] = safe_ratio(result.revol_bal, result.annual_inc)
            result["open_to_total_accounts"] = safe_ratio(result.open_acc, result.total_acc)
            result["revolving_balance_to_limit"] = safe_ratio(result.revol_bal, result.total_rev_hi_lim)
            result["loan_to_monthly_income"] = result.loan_to_income * 12
            result["non_mortgage_debt_to_income"] = safe_ratio(result.total_bal_ex_mort, result.annual_inc)
            result["missing_bureau_count"] = result[BUREAU_NUMERIC].isna().sum(axis=1).astype(float)
        return result.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
