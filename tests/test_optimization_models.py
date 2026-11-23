"""Metric direction, training weights, and algorithm adapter checks for V3."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from optimization.models import ExactAveragePrecision, V3Model


def model_frame(n=360):
    rng = np.random.default_rng(123)
    fico = rng.integers(640, 800, size=n)
    loan = rng.uniform(1000, 35000, size=n)
    raw = pd.DataFrame({
        'loan_amnt': loan, 'annual_inc': rng.uniform(25000, 150000, size=n),
        'dti': rng.uniform(0, 40, size=n), 'fico_range_low': fico,
        'fico_range_high': fico + 4, 'issue_d': ['Jan-2013'] * n,
        'earliest_cr_line': ['Jan-2000'] * n, 'term': ['36 months'] * n,
        'emp_length': ['5 years'] * n,
        'purpose': np.where(np.arange(n) % 2, 'credit_card', 'debt_consolidation'),
        'open_acc': rng.integers(1, 30, size=n), 'total_acc': rng.integers(30, 50, size=n),
        'revol_bal': rng.uniform(100, 50000, size=n), 'revol_util': rng.uniform(0, 100, size=n),
    })
    y = ((fico < 710) | (loan > 28000)).astype(int)
    y[np.arange(n) % 11 == 0] = 1 - y[np.arange(n) % 11 == 0]
    return raw, y


class OptimizationModelTests(unittest.TestCase):
    def test_exact_ap_ignores_loss_weights_and_matches_tied_sklearn_definition(self):
        y = np.array([1, 0, 1, 0, 1])
        score = np.array([3., 3., 1., 0., 0.])
        weight = np.array([1., 2., .5, 1., 3.])
        metric = ExactAveragePrecision()
        result, denominator = metric.evaluate([score], y, weight)
        self.assertEqual(result, average_precision_score(y, score))
        self.assertNotEqual(result, average_precision_score(y, score, sample_weight=weight))
        self.assertEqual(metric.get_final_error(result, denominator), result)
        self.assertTrue(metric.is_max_optimal())

    def test_boosters_maximize_exact_validation_ap(self):
        raw, y = model_frame()
        train, validation = raw.iloc[:260], raw.iloc[260:]
        configs = {
            'catboost': {'iterations': 45, 'depth': 3, 'learning_rate': .15, 'thread_count': 1,
                        'class_weights': [1., 4.]},
            'lightgbm': {'n_estimators': 45, 'num_leaves': 7, 'min_child_samples': 10,
                         'learning_rate': .15, 'n_jobs': 1},
            'xgboost': {'n_estimators': 45, 'max_depth': 3, 'learning_rate': .15, 'n_jobs': 1},
        }
        for family, params in configs.items():
            with self.subTest(family=family):
                model = V3Model(family, {**params, 'early_stopping_rounds': 8}).fit(
                    train, y[:260], validation=(validation, y[260:]),
                    sample_weight=np.linspace(.5, 1.5, len(train)),
                )
                score = model.predict_score(validation)
                self.assertTrue(np.isfinite(score).all())
                self.assertGreater(average_precision_score(y[260:], score), .65)
                if family == 'catboost':
                    log = model.estimator.get_evals_result()['validation']['ExactAveragePrecision']
                elif family == 'lightgbm':
                    log = model.estimator.evals_result_['valid_0']['average_precision']
                else:
                    log = model.estimator.evals_result()['validation_0']['average_precision_score']
                self.assertEqual(model.best_iteration, int(np.argmax(log)) + 1)
                self.assertAlmostEqual(average_precision_score(y[260:], score), max(log), places=5)

    def test_linear_preparation_and_training_weights_have_effect(self):
        raw, y = model_frame()
        plain = V3Model('logistic', {'C': .1}, preparation='log_clip').fit(raw, y)
        weighted = V3Model('logistic', {'C': .1}, preparation='log_clip').fit(
            raw, y, sample_weight=np.where(y == 1, 4., 1.),
        )
        self.assertTrue(np.isfinite(weighted.predict_score(raw)).all())
        self.assertGreater(np.median(weighted.predict_score(raw) - plain.predict_score(raw)), .2)
        for family, params in [('svm_linear', {'C': .1}),
                               ('svm_nystroem', {'C': .1, 'gamma': .01, 'n_components': 30})]:
            model = V3Model(family, params, preparation='log_clip').fit(raw, y)
            self.assertTrue(np.isfinite(model.predict_score(raw)).all())
            self.assertEqual(len(model.predict_score(raw)), len(raw))

    def test_invalid_training_weights_are_rejected(self):
        raw, y = model_frame(20)
        for weight in [np.ones(19), np.full(20, np.nan), np.full(20, -1.), np.zeros(20)]:
            with self.assertRaises(ValueError):
                V3Model('logistic', {}).fit(raw, y, sample_weight=weight)


if __name__ == '__main__':
    unittest.main()
