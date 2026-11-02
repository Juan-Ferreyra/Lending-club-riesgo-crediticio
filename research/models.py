"""Train-only data recipes and comparable learners, including three SVM strategies."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from src.features import FeatureBuilder, CATEGORICAL, BUREAU_NUMERIC


RECIPES = [
    {'name': 'legacy_all_history', 'mode': 'legacy', 'term': 'all', 'start': None},
    {'name': 'stable_all_history', 'mode': 'stable', 'term': 'all', 'start': None},
    {'name': 'median_all_history', 'mode': 'median', 'term': 'all', 'start': None},
    {'name': 'clip_all_history', 'mode': 'clip', 'term': 'all', 'start': None},
    {'name': 'stable_36_history', 'mode': 'stable', 'term': '36 months', 'start': None},
    {'name': 'stable_recent', 'mode': 'stable', 'term': 'all', 'start': '2012-01-01'},
    {'name': 'stable_36_recent', 'mode': 'stable', 'term': '36 months', 'start': '2012-01-01'},
    {'name': 'median_36_recent', 'mode': 'median', 'term': '36 months', 'start': '2012-01-01'},
    {'name': 'clip_36_recent', 'mode': 'clip', 'term': '36 months', 'start': '2012-01-01'},
]


class AuditFeatures:
    def __init__(self, mode='stable', pricing=False):
        self.mode, self.pricing = mode, pricing

    def raw_transform(self, raw):
        X = FeatureBuilder().transform(raw)
        if self.mode != 'legacy':
            # These are domain constraints, not arbitrary upper caps on plausible debt.
            for c in ['open_acc', 'total_acc', 'delinq_2yrs', 'pub_rec', 'inq_last_6mths',
                      'num_accts_ever_120_pd', 'num_tl_30dpd', 'num_tl_90g_dpd_24m']:
                X[c] = X[c].where(X[c]>=0)
            for c in ['pct_tl_nvr_dlq', 'percent_bc_gt_75']:
                X[c] = X[c].where(X[c].between(0, 100))
            wrong = X.fico_range_low.gt(X.fico_range_high)
            X.loc[wrong, ['fico_range_low', 'fico_range_high', 'fico_midpoint']] = np.nan
            X = X.drop(columns=['loan_to_monthly_income', 'term_months'])
        if self.pricing:
            for c in ['grade', 'sub_grade']:
                X[c] = raw[c].fillna('__MISSING__').astype(str)
            X['int_rate'] = pd.to_numeric(raw.int_rate, errors='coerce')
        return X

    def fit(self, raw):
        X = self.raw_transform(raw)
        self.bureau = [c for c in BUREAU_NUMERIC if X[c].notna().mean()>=.05 and X[c].nunique()>1]
        if self.mode != 'legacy':
            X['missing_bureau_count'] = X[self.bureau].isna().sum(axis=1).astype(float)
            self.columns = X.columns[(X.nunique()>1)&(X.isna().mean()<=.95)].tolist()
        else:
            self.columns = list(X)
        self.cats = [c for c in CATEGORICAL+['grade', 'sub_grade'] if c in self.columns]
        self.nums = [c for c in self.columns if c not in self.cats]
        self.lo = X[self.nums].quantile(.005)
        self.hi = X[self.nums].quantile(.995)
        self.medians = X[self.nums].median().fillna(0)
        self.indicators = [c for c in self.nums if X[c].isna().any()]
        return self

    def transform(self, raw):
        X = self.raw_transform(raw)
        if self.mode != 'legacy':
            X['missing_bureau_count'] = X[self.bureau].isna().sum(axis=1).astype(float)
        X = X[self.columns].copy()
        if self.mode == 'clip':
            X[self.nums] = X[self.nums].clip(self.lo, self.hi, axis=1)
        if self.mode == 'median':
            indicators = {c+'__missing': X[c].isna().astype(float) for c in self.indicators}
            X[self.nums] = X[self.nums].fillna(self.medians)
            X = pd.concat([X, pd.DataFrame(indicators, index=X.index)], axis=1)
        return X


class ResearchModel:
    def __init__(self, family, params, recipe, seed=42, pricing=False):
        self.family, self.params, self.recipe = family, dict(params), dict(recipe)
        self.seed, self.pricing = seed, pricing

    def fit(self, raw, y, validation=None):
        self.features = AuditFeatures(self.recipe['mode'], self.pricing).fit(raw)
        X = self.features.transform(raw)
        cats = self.features.cats
        nums = [c for c in X if c not in cats]
        self.best_iteration = None
        self.convergence_warnings = []
        params = dict(self.params)
        if self.family == 'catboost':
            from catboost import CatBoostClassifier
            defaults = dict(loss_function='Logloss', iterations=740, depth=7, learning_rate=.02736,
                            l2_leaf_reg=27.33, thread_count=3, random_seed=self.seed, verbose=False,
                            allow_writing_files=False, boosting_type='Plain', max_ctr_complexity=1)
            defaults.update(params)
            self.preprocessor = None
            self.estimator = CatBoostClassifier(**defaults)
            kwargs = dict(cat_features=cats)
            if validation is not None:
                kwargs.update(eval_set=(self.features.transform(validation[0]), validation[1]), early_stopping_rounds=60)
            self.estimator.fit(X, y, **kwargs)
            self.best_iteration = self.estimator.tree_count_
        elif self.family in ['lightgbm', 'xgboost']:
            self.preprocessor = ColumnTransformer([
                ('num', 'passthrough', nums),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cats),
            ], verbose_feature_names_out=False)
            Z = self.preprocessor.fit_transform(X).astype('float32')
            self.cat_indices = list(range(len(nums), len(nums)+len(cats)))
            V = self.encode(validation[0]) if validation is not None else None
            if self.family == 'lightgbm':
                from lightgbm import LGBMClassifier, early_stopping
                defaults = dict(n_estimators=143, num_leaves=28, learning_rate=.0508, min_child_samples=152,
                                verbosity=-1, n_jobs=3, random_state=self.seed, deterministic=True, force_col_wise=True)
                defaults.update(params)
                self.estimator = LGBMClassifier(**defaults)
                kwargs = dict(categorical_feature=self.cat_indices)
                if validation is not None:
                    self.estimator.set_params(metric='average_precision')
                    kwargs.update(eval_set=[(V, validation[1])], eval_metric='average_precision',
                                  callbacks=[early_stopping(60, first_metric_only=True, verbose=False)])
                self.estimator.fit(Z, y, **kwargs)
                self.best_iteration = self.estimator.best_iteration_ or defaults['n_estimators']
            else:
                from xgboost import XGBClassifier
                defaults = dict(n_estimators=326, tree_method='hist', objective='binary:logistic', eval_metric='aucpr',
                                n_jobs=3, random_state=self.seed, enable_categorical=True,
                                feature_types=['q']*len(nums)+['c']*len(cats))
                defaults.update(params)
                if validation is not None:
                    defaults['early_stopping_rounds'] = 60
                self.estimator = XGBClassifier(**defaults)
                self.estimator.fit(Z, y, **({'eval_set': [(V, validation[1])], 'verbose': False} if validation is not None else {}))
                self.best_iteration = self.estimator.best_iteration+1 if validation is not None else defaults['n_estimators']
        else:
            numeric = [('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
            if self.family != 'random_forest':
                numeric.append(('scale', StandardScaler()))
            self.preprocessor = ColumnTransformer([
                ('num', Pipeline(numeric), nums),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50, dtype=np.float32), cats),
            ], sparse_threshold=0 if self.family in ['random_forest', 'svm_rbf', 'svm_nystroem'] else .8)
            Z = self.preprocessor.fit_transform(X).astype('float32')
            if self.family == 'logistic':
                defaults = dict(max_iter=2500, solver='lbfgs', random_state=self.seed)
                defaults.update(params); self.estimator = LogisticRegression(**defaults)
            elif self.family == 'random_forest':
                defaults = dict(n_estimators=250, n_jobs=3, random_state=self.seed)
                defaults.update(params); self.estimator = RandomForestClassifier(**defaults)
            elif self.family == 'svm_rbf':
                defaults = dict(kernel='rbf', cache_size=512, max_iter=20000, tol=.002, random_state=self.seed)
                defaults.update(params); self.estimator = SVC(**defaults)
            elif self.family == 'svm_nystroem':
                components = params.pop('n_components', 300)
                gamma = params.pop('gamma', .01)
                self.kernel_map = Nystroem(kernel='rbf', gamma=gamma, n_components=min(components, len(raw)), random_state=self.seed, n_jobs=3)
                Z = self.kernel_map.fit_transform(Z)
                defaults = dict(dual='auto', max_iter=12000, tol=.0005, random_state=self.seed)
                defaults.update(params); self.estimator = LinearSVC(**defaults)
            else:
                defaults = dict(dual='auto', max_iter=12000, tol=.0005, random_state=self.seed)
                defaults.update(params); self.estimator = LinearSVC(**defaults)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                self.estimator.fit(Z, y)
            self.convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        return self

    def encode(self, raw):
        X = self.features.transform(raw)
        if self.preprocessor is None:
            return X
        Z = self.preprocessor.transform(X).astype('float32')
        if self.family == 'xgboost':
            for j in self.cat_indices:
                Z[Z[:, j]<0, j] = np.nan
        if self.family == 'svm_nystroem':
            Z = self.kernel_map.transform(Z)
        return Z

    def predict_score(self, raw):
        Z = self.encode(raw)
        if self.family.startswith('svm'):
            return self.estimator.decision_function(Z)
        return logit(np.clip(self.estimator.predict_proba(Z)[:, 1], 1e-7, 1-1e-7))


class ScoreCalibrator:
    def __init__(self, method):
        self.method = method

    def fit(self, scores, y):
        if self.method == 'sigmoid':
            self.model = LogisticRegression(C=1000, max_iter=1500).fit(np.asarray(scores).reshape(-1, 1), y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds='clip').fit(scores, y)
        return self

    def predict(self, scores):
        if self.method == 'sigmoid':
            p = self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
        else:
            p = self.model.predict(scores)
        return np.clip(p, 1e-6, 1-1e-6)
