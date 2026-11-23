"""Algorithm-specific preparation and exact-AP early stopping for experiment V3.

Training weights are supplied only to the training loss. The validation metric
uses the unchanged, unweighted validation population for every learner.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

from optimization.features import V3Features


class ExactAveragePrecision:
    """Unweighted sklearn AP, including ties, even with class-weighted loss.

    CatBoost can supply class-derived evaluation weights. Ignoring ``weight``
    prevents class weighting from silently changing the validation population.
    """

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        result = average_precision_score(
            np.asarray(target), np.asarray(approxes[0])
        )
        return float(result), 1.0

    def get_final_error(self, error, weight):
        return error


class SignedLogColumns(BaseEstimator, TransformerMixin):
    """Compress selected monetary magnitudes; leave rates and scores unchanged."""

    def __init__(self, indices=()):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = np.asarray(X, dtype=np.float64).copy()
        if self.indices:
            values = result[:, self.indices]
            result[:, self.indices] = np.sign(values) * np.log1p(np.abs(values))
        return result


def monetary_log_indices(names):
    """An explicit semantic rule, independent of outcomes or validation values."""
    monetary = {
        'loan_amnt', 'annual_inc', 'annual_inc_joint', 'revol_bal', 'tot_coll_amt',
        'tot_cur_bal', 'total_bal_il', 'max_bal_bc', 'total_rev_hi_lim', 'avg_cur_bal',
        'bc_open_to_buy', 'delinq_amnt', 'tot_hi_cred_lim', 'total_bal_ex_mort',
        'total_bc_limit', 'total_il_high_credit_limit',
    }
    return tuple(i for i, name in enumerate(names) if name in monetary)


class V3Model:
    """A fitted feature contract, algorithm preparation, and binary estimator.

    ``predict_score`` returns an unrounded raw margin. Calibration is deliberately
    separate so that a probability clip or isotonic ties cannot silently change AP.
    ``log_clip`` clips using training quantiles, then log-transforms monetary
    predictors before scaling in linear and kernel strategies.
    """

    def __init__(self, family, params, feature_mode='baseline', preparation='native', seed=42):
        self.family = family
        self.params = dict(params)
        self.feature_mode = feature_mode
        self.preparation = preparation
        self.seed = seed

    def fit(self, raw, y, validation=None, sample_weight=None):
        if self.preparation not in {'native', 'clip', 'median', 'log_clip'}:
            raise ValueError(f'Unknown preparation: {self.preparation}')
        if self.family not in {
            'catboost', 'lightgbm', 'xgboost', 'logistic', 'svm_linear', 'svm_nystroem'
        }:
            raise ValueError(f'Unknown family: {self.family}')
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if (sample_weight.shape != (len(raw),) or not np.isfinite(sample_weight).all()
                    or (sample_weight < 0).any() or sample_weight.sum() <= 0):
                raise ValueError('Training weights must be finite, nonnegative, and match the rows.')
        self.features = V3Features(
            mode=self.feature_mode,
            preparation='clip' if self.preparation == 'log_clip' else self.preparation,
        ).fit(raw)
        X = self.features.transform(raw)
        cats = self.features.cats
        nums = [c for c in X if c not in cats]
        self.best_iteration = None
        self.convergence_warnings = []
        self.early_stopping_metric = 'sklearn_average_precision' if validation is not None else None
        params = dict(self.params)
        patience = params.pop('early_stopping_rounds', 60)

        if self.family == 'catboost':
            from catboost import CatBoostClassifier

            defaults = dict(
                loss_function='Logloss', iterations=1177, depth=7, learning_rate=.0220557845,
                l2_leaf_reg=54.1299336, thread_count=3, random_seed=self.seed, verbose=False,
                allow_writing_files=False, boosting_type='Plain', max_ctr_complexity=1,
            )
            defaults.update(params)
            if validation is not None:
                defaults['eval_metric'] = ExactAveragePrecision()
            self.preprocessor = None
            self.estimator = CatBoostClassifier(**defaults)
            kwargs = dict(cat_features=cats, sample_weight=sample_weight)
            if validation is not None:
                kwargs.update(
                    eval_set=(self.features.transform(validation[0]), validation[1]),
                    early_stopping_rounds=patience, use_best_model=True,
                )
            with warnings.catch_warnings():
                # Python/sklearn evaluation is intentional; CatBoost may try numba.
                warnings.filterwarnings('ignore', message='Failed to optimize method.*', category=UserWarning)
                warnings.filterwarnings(
                    'ignore', message='Failed to import numba for optimizing custom metrics and objectives',
                    category=UserWarning,
                )
                self.estimator.fit(X, y, **kwargs)
            self.best_iteration = self.estimator.tree_count_

        elif self.family in {'lightgbm', 'xgboost'}:
            self.preprocessor = ColumnTransformer([
                ('num', 'passthrough', nums),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cats),
            ], verbose_feature_names_out=False)
            Z = self.preprocessor.fit_transform(X).astype('float32')
            self.cat_indices = list(range(len(nums), len(nums) + len(cats)))
            V = self.encode(validation[0]) if validation is not None else None
            if self.family == 'lightgbm':
                from lightgbm import LGBMClassifier, early_stopping

                defaults = dict(
                    n_estimators=1200, num_leaves=28, learning_rate=.03, min_child_samples=152,
                    verbosity=-1, n_jobs=3, random_state=self.seed, deterministic=True,
                    force_col_wise=True, metric='average_precision',
                )
                defaults.update(params)
                defaults['metric'] = 'average_precision'
                self.estimator = LGBMClassifier(**defaults)
                kwargs = dict(categorical_feature=self.cat_indices, sample_weight=sample_weight)
                if validation is not None:
                    kwargs.update(
                        eval_set=[(V, validation[1])], eval_metric='average_precision',
                        callbacks=[early_stopping(patience, first_metric_only=True, verbose=False)],
                    )
                self.estimator.fit(Z, y, **kwargs)
                self.best_iteration = self.estimator.best_iteration_ or defaults['n_estimators']
            else:
                from xgboost import XGBClassifier
                from xgboost.callback import EarlyStopping

                defaults = dict(
                    n_estimators=1200, tree_method='hist', objective='binary:logistic',
                    eval_metric=average_precision_score, n_jobs=3, random_state=self.seed,
                    enable_categorical=True, feature_types=['q'] * len(nums) + ['c'] * len(cats),
                )
                defaults.update(params)
                defaults['eval_metric'] = average_precision_score
                if validation is not None:
                    # A callable is otherwise assumed to be a loss to minimize.
                    defaults['callbacks'] = [EarlyStopping(
                        rounds=patience, metric_name='average_precision_score',
                        data_name='validation_0', maximize=True, save_best=True,
                    )]
                self.estimator = XGBClassifier(**defaults)
                self.estimator.fit(
                    Z, y, sample_weight=sample_weight,
                    **({'eval_set': [(V, validation[1])], 'verbose': False}
                       if validation is not None else {}),
                )
                self.best_iteration = (self.estimator.best_iteration + 1
                                       if validation is not None else defaults['n_estimators'])

        else:
            numeric = [('impute', SimpleImputer(
                strategy='median', add_indicator=True, keep_empty_features=True
            ))]
            if self.preparation == 'log_clip':
                numeric.append(('log_money', SignedLogColumns(monetary_log_indices(nums))))
            numeric.append(('scale', StandardScaler()))
            self.preprocessor = ColumnTransformer([
                ('num', Pipeline(numeric), nums),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50, dtype=np.float32), cats),
            ], sparse_threshold=1.0)
            Z = self.preprocessor.fit_transform(X).astype('float32')
            if self.family == 'logistic':
                defaults = dict(max_iter=2500, solver='lbfgs', random_state=self.seed)
                defaults.update(params)
                self.estimator = LogisticRegression(**defaults)
            else:
                if self.family == 'svm_nystroem':
                    components = params.pop('n_components', 600)
                    gamma = params.pop('gamma', .01)
                    self.kernel_map = Nystroem(
                        kernel='rbf', gamma=gamma, n_components=min(components, len(raw)),
                        random_state=self.seed, n_jobs=3,
                    )
                    Z = self.kernel_map.fit_transform(Z).astype('float32')
                defaults = dict(dual='auto', max_iter=15000, tol=.0005, random_state=self.seed)
                defaults.update(params)
                self.estimator = LinearSVC(**defaults)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                self.estimator.fit(Z, y, sample_weight=sample_weight)
            self.convergence_warnings = [
                str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
            ]
        return self

    def encode(self, raw):
        X = self.features.transform(raw)
        if self.preprocessor is None:
            return X
        Z = self.preprocessor.transform(X).astype('float32')
        if self.family == 'xgboost':
            for j in self.cat_indices:
                Z[Z[:, j] < 0, j] = np.nan
        if self.family == 'svm_nystroem':
            Z = self.kernel_map.transform(Z).astype('float32')
        return Z

    def predict_score(self, raw):
        Z = self.encode(raw)
        if self.family == 'catboost':
            score = self.estimator.predict(Z, prediction_type='RawFormulaVal')
        elif self.family == 'lightgbm':
            score = self.estimator.predict(Z, raw_score=True, num_iteration=self.best_iteration)
        elif self.family == 'xgboost':
            score = self.estimator.predict(Z, output_margin=True)
        else:
            score = self.estimator.decision_function(Z)
        return np.asarray(score, dtype=np.float64)
