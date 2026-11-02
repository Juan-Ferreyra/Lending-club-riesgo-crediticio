"""Feature contract and training-only statistics for optimization round three."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from optimization.features import V3Features, financial_features
from research.models import AuditFeatures
from src.features import FORBIDDEN


class OptimizationFeatureTests(unittest.TestCase):
    def frame(self):
        n = 12
        return pd.DataFrame({
            'loan_amnt': np.linspace(1000, 12000, n),
            'annual_inc': np.linspace(20000, 100000, n),
            'dti': np.linspace(1, 30, n),
            'fico_range_low': np.arange(n) * 5 + 650,
            'fico_range_high': np.arange(n) * 5 + 654,
            'issue_d': ['Jan-2013'] * n, 'earliest_cr_line': ['Jan-2000'] * n,
            'term': ['36 months'] * n, 'emp_length': ['5 years'] * n,
            'open_acc': np.arange(n) + 1, 'total_acc': np.arange(n) + 12,
            'revol_bal': np.arange(n) * 500., 'revol_util': np.linspace(0, 100, n),
            'total_rev_hi_lim': np.linspace(6000, 24000, n),
            'emp_title': ['Teacher', 'teacher', 'TEACHER', 'Nurse', 'nurse',
                          'Engineer', 'Other', None, 'other', 'other', 'teacher', 'nurse'],
            'zip_code': ['123xx'] * 5 + ['456xx'] * 4 + ['789xx'] * 3,
        })

    def test_baseline_matches_current_catboost_features(self):
        raw = self.frame()
        expected = AuditFeatures('stable').fit(raw).transform(raw)
        actual = V3Features().fit_transform(raw)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(clone(V3Features(mode='combined')).mode, 'combined')

    def test_future_context_cannot_refit_categories_or_frequencies(self):
        raw = self.frame()
        features = V3Features('context', min_category_count=2).fit(raw)
        counts = {k: dict(v) for k, v in features.context_counts_.items()}
        future = raw.copy()
        future['emp_title'] = ['Astronaut'] * 10 + ['Teacher', None]
        transformed = features.transform(future)
        self.assertEqual(features.context_counts_, counts)
        self.assertEqual(transformed.loc[0, 'v3_emp_title_category'], '__UNSEEN__')
        self.assertEqual(transformed.loc[0, 'v3_emp_title_log_frequency'], 0)
        self.assertAlmostEqual(transformed.loc[10, 'v3_emp_title_log_frequency'], np.log1p(4))
        self.assertEqual(transformed.loc[11, 'v3_emp_title_category'], '__MISSING__')
        self.assertEqual(features.transform(raw).loc[5, 'v3_emp_title_category'], '__RARE__')

    def test_later_available_financial_fields_cannot_enter_schema(self):
        raw = self.frame()
        features = V3Features('financial').fit(raw)
        expected = features.transform(raw)
        future = raw.copy()
        future['tot_cur_bal'] = np.arange(12) * 10000
        future['tot_hi_cred_lim'] = np.arange(12) * 20000
        pd.testing.assert_frame_equal(expected, features.transform(future))

    def test_outcomes_pricing_identifiers_do_not_change_features(self):
        raw = self.frame()
        features = V3Features('combined', min_category_count=2).fit(raw)
        before = features.transform(raw)
        contaminated = raw.copy()
        for name in FORBIDDEN:
            contaminated[name] = 'Charged Off' if name == 'loan_status' else 999999
        for name in ['total_rec_prncp', 'total_rec_int', 'settlement_amount', 'hardship_flag']:
            contaminated[name] = 999999
        pd.testing.assert_frame_equal(before, features.transform(contaminated))
        self.assertFalse(FORBIDDEN.intersection(features.get_feature_names_out()))

    def test_invalid_denominators_do_not_create_infinite_or_negative_ratios(self):
        raw = self.frame()
        raw.loc[0, 'annual_inc'] = 0
        raw.loc[1, 'annual_inc'] = -1
        raw.loc[2, 'total_rev_hi_lim'] = 0
        raw.loc[3, 'dti'] = 150
        raw.loc[4, 'open_acc'] = -1
        raw['inq_last_6mths'] = 2
        X = financial_features(raw)
        self.assertFalse(np.isinf(X.to_numpy()).any())
        self.assertTrue(X.loc[[0, 1], 'v3_credit_capacity_to_income'].isna().all())
        self.assertTrue(np.isnan(X.loc[2, 'v3_loan_to_revolving_limit']))
        self.assertTrue(np.isnan(X.loc[3, 'v3_loan_to_annual_free_income']))
        self.assertTrue(np.isnan(X.loc[4, 'v3_inquiries_per_open_account']))
        self.assertTrue((X.fillna(0) >= 0).all().all())

    def test_clip_and_median_are_fitted_only_on_training(self):
        raw = self.frame()
        raw.loc[1, 'annual_inc'] = np.nan
        future = raw.copy()
        future['annual_inc'] = 1e12
        future.loc[1, 'annual_inc'] = np.nan
        clipped = V3Features('financial', preparation='clip').fit(raw)
        bounds = clipped.hi_.copy()
        out = clipped.transform(future)
        self.assertLessEqual(out.annual_inc.max(), bounds.annual_inc)
        pd.testing.assert_series_equal(bounds, clipped.hi_)
        med = V3Features('combined', preparation='median', min_category_count=2).fit(raw)
        out = med.transform(future)
        self.assertEqual(out.loc[1, 'annual_inc'], raw.annual_inc.median())
        self.assertEqual(out.loc[1, 'annual_inc__missing'], 1)
        self.assertTrue(np.isfinite(out[med.nums].to_numpy()).all())
        self.assertEqual(list(out.columns), list(med.get_feature_names_out()))


if __name__ == '__main__':
    unittest.main()
