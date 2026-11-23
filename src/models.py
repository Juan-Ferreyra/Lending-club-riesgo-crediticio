"""Comparable learners with train-only preprocessing and CPU limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features import FeatureBuilder, CATEGORICAL


class LoanModel:
    def __init__(self, family, params=None, extended=True, seed=42, threads=3):
        self.family, self.params = family, dict(params or {})
        self.extended, self.seed, self.threads = extended, seed, threads

    def fit(self, raw, y, validation=None):
        self.features = FeatureBuilder(extended=self.extended)
        X = self.features.fit_transform(raw)
        V = self.features.transform(validation[0]) if validation is not None else None
        cats = [c for c in CATEGORICAL if c in X]
        nums = [c for c in X if c not in cats]
        self.feature_names = list(X.columns)
        self.best_iteration_ = None
        if self.family == "catboost":
            from catboost import CatBoostClassifier
            self.preprocessor = None
            defaults = dict(iterations=1000, depth=6, learning_rate=0.045, l2_leaf_reg=8,
                            loss_function="Logloss", eval_metric="AUC", random_seed=self.seed,
                            thread_count=self.threads, verbose=False, allow_writing_files=False,
                            boosting_type="Plain", max_ctr_complexity=1)
            defaults.update(self.params)
            self.estimator = CatBoostClassifier(**defaults)
            kwargs = dict(cat_features=cats)
            if validation is not None:
                kwargs.update(eval_set=(V, validation[1]), early_stopping_rounds=70)
            self.estimator.fit(X, y, **kwargs)
            self.best_iteration_ = self.estimator.tree_count_
        elif self.family == "xgboost":
            from xgboost import XGBClassifier
            self.preprocessor = ColumnTransformer([
                ("numeric", "passthrough", nums),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ], verbose_feature_names_out=False)
            Z = self.preprocessor.fit_transform(X).astype(np.float32)
            self.categorical_indices_ = list(range(len(nums), len(nums)+len(cats)))
            defaults = dict(n_estimators=1400, max_depth=4, learning_rate=0.04,
                            min_child_weight=20, reg_lambda=10, subsample=0.8,
                            colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
                            tree_method="hist", n_jobs=self.threads, random_state=self.seed,
                            enable_categorical=True,
                            feature_types=["q"]*len(nums)+["c"]*len(cats))
            defaults.update(self.params)
            if validation is not None:
                defaults["early_stopping_rounds"] = 70
            self.estimator = XGBClassifier(**defaults)
            kwargs = {}
            if validation is not None:
                Z_valid = self.preprocessor.transform(V).astype(np.float32)
                for j in self.categorical_indices_:
                    Z_valid[Z_valid[:, j] < 0, j] = np.nan
                kwargs.update(eval_set=[(Z_valid, validation[1])], verbose=False)
            self.estimator.fit(Z, y, **kwargs)
            self.best_iteration_ = self.estimator.best_iteration+1 if validation is not None else defaults["n_estimators"]
        elif self.family == "lightgbm":
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation
            self.preprocessor = ColumnTransformer([
                ("numeric", "passthrough", nums),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ], verbose_feature_names_out=False)
            Z = self.preprocessor.fit_transform(X).astype("float32")
            defaults = dict(n_estimators=1200, learning_rate=0.035, num_leaves=31,
                            min_child_samples=100, reg_lambda=5, verbosity=-1,
                            random_state=self.seed, n_jobs=self.threads, deterministic=True,
                            force_col_wise=True)
            defaults.update(self.params)
            self.estimator = LGBMClassifier(**defaults)
            kwargs = dict(categorical_feature=list(range(len(nums), len(nums)+len(cats))))
            if validation is not None:
                kwargs.update(eval_set=[(self.preprocessor.transform(V).astype("float32"), validation[1])],
                              eval_metric="average_precision",
                              callbacks=[early_stopping(70, first_metric_only=True, verbose=False), log_evaluation(0)])
                self.estimator.set_params(metric="average_precision")
            self.estimator.fit(Z, y, **kwargs)
            self.best_iteration_ = self.estimator.best_iteration_ or defaults["n_estimators"]
        else:
            numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
            if self.family == "logistic":
                numeric_steps.append(("scale", StandardScaler()))
            self.preprocessor = ColumnTransformer([
                ("numeric", Pipeline(numeric_steps), nums),
                ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32), cats),
            ], sparse_threshold=0.0 if self.family == "random_forest" else 0.7, verbose_feature_names_out=False)
            Z = self.preprocessor.fit_transform(X)
            if self.family == "random_forest":
                Z = Z.astype(np.float32)
            if self.family == "logistic":
                defaults = dict(C=0.1, class_weight=None, max_iter=1500, solver="lbfgs", random_state=self.seed)
                defaults.update(self.params)
                self.estimator = LogisticRegression(**defaults)
            elif self.family == "random_forest":
                defaults = dict(n_estimators=250, min_samples_leaf=25, max_features=0.65,
                                max_samples=0.8, n_jobs=self.threads, random_state=self.seed)
                defaults.update(self.params)
                self.estimator = RandomForestClassifier(**defaults)
            else:
                raise ValueError(self.family)
            kwargs = {}
            self.estimator.fit(Z, y, **kwargs)
        return self

    def transform(self, raw):
        X = self.features.transform(raw)
        if self.preprocessor is None:
            return X
        Z = self.preprocessor.transform(X)
        if self.family == "xgboost":
            Z = Z.astype(np.float32)
            for j in self.categorical_indices_:
                Z[Z[:, j] < 0, j] = np.nan
        return Z

    def predict_proba(self, raw):
        return self.estimator.predict_proba(self.transform(raw))


class LoanEnsemble:
    def __init__(self, models, weights):
        self.models, self.weights = models, np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

    def predict_proba(self, raw):
        p = sum(w * m.predict_proba(raw)[:, 1] for m, w in zip(self.models, self.weights))
        return np.column_stack([1-p, p])
