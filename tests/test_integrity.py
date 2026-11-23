"""Focused checks for leakage, tie-correct metrics and temporal boundaries."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from src.features import FeatureBuilder, FORBIDDEN
from src.evaluate import WeightedRankMetrics, ProbabilityCalibrator, choose_thresholds
from src.train import subset
from src.final_test import evaluate_frozen
from src.prepare import file_sha256


class IntegrityTests(unittest.TestCase):
    def test_external_outcomes_cannot_open_without_a_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.final_test.pd.read_parquet") as read:
                with self.assertRaises(FileNotFoundError):
                    evaluate_frozen(folder, folder)
                read.assert_not_called()

    def test_changed_prediction_code_rejected_before_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"models").mkdir()
            (root/"models"/"frozen_bundle.joblib").write_bytes(b"not loaded")
            (root/"manifest.json").write_text("{}", encoding="utf-8")
            (root/"selection.json").write_text("{}", encoding="utf-8")
            lock = {"model_sha256": file_sha256(root/"models"/"frozen_bundle.joblib"),
                    "protocol_sha256": file_sha256("experiments/protocol.json"),
                    "data_manifest_sha256": file_sha256(root/"manifest.json"),
                    "selection_sha256": file_sha256(root/"selection.json"),
                    "code_sha256": {"src/features.py": "0"*64}}
            (root/"selection_lock.json").write_text(json.dumps(lock), encoding="utf-8")
            with patch("src.final_test.pd.read_parquet") as read:
                with self.assertRaisesRegex(AssertionError, "code changed"):
                    evaluate_frozen(root, root)
                read.assert_not_called()

    def frame(self):
        return pd.DataFrame({"loan_amnt": [10000., 20000., 0], "annual_inc": [50000., 0., 60000.],
            "fico_range_low": [680., 700., 800.], "fico_range_high": [684., 704., 804.],
            "issue_d": ["Jan-2010", "Jan-2014", "Dec-2014"],
            "earliest_cr_line": ["Jan-2000", "Jan-2020", "Dec-2004"],
            "term": [" 36 months", " 60 months", " 36 months"],
            "emp_length": ["< 1 year", "10+ years", None]})

    def test_outcomes_cannot_change_predictors(self):
        raw = self.frame()
        first = FeatureBuilder().fit_transform(raw)
        raw["loan_status"] = ["Charged Off"]*3
        for col in FORBIDDEN:
            if col != "loan_status":
                raw[col] = 999999
        second = FeatureBuilder().fit_transform(raw)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(FORBIDDEN.intersection(second.columns))

    def test_domain_and_history_checks(self):
        result = FeatureBuilder().fit_transform(self.frame())
        self.assertEqual(result.loc[0, "credit_history_months"], 120)
        self.assertTrue(np.isnan(result.loc[1, "credit_history_months"]))
        self.assertTrue(np.isnan(result.loc[1, "loan_to_income"]))
        self.assertEqual(result.loc[0, "employment_years"], .5)
        numeric = result.select_dtypes(include="number").to_numpy()
        self.assertFalse(np.isinf(numeric).any())

    def test_small_batch_without_missing_employment(self):
        # pandas 3 forbids assigning 0.5 into an inferred int employment column.
        raw = self.frame().iloc[:2].copy()
        out = FeatureBuilder().fit_transform(raw)
        self.assertEqual(out.loc[0, "employment_years"], .5)
        out = FeatureBuilder().fit_transform(raw.iloc[[1]])
        self.assertEqual(out.iloc[0].employment_years, 10.)

    def test_weighted_ap_and_auc_handle_ties(self):
        y = np.array([1, 0, 1, 1, 0, 0, 1])
        p = np.array([.2, .2, .4, .8, .8, .9, .1])
        w = np.array([1, 2, 0, 4, 1, 3, 2])
        actual = WeightedRankMetrics(y, p).compute(w)
        expected = [average_precision_score(y, p, sample_weight=w), roc_auc_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_constant_scores_equal_prevalence_and_chance_auc(self):
        y = np.array([0, 0, 1, 0, 1])
        p = np.full(5, .25)
        values = WeightedRankMetrics(y, p).compute(np.ones(5))
        self.assertAlmostEqual(values[0], y.mean())
        self.assertAlmostEqual(values[1], .5)

    def test_calibrators_and_thresholds_stay_valid(self):
        p = np.linspace(.01, .99, 100)
        y = ((np.arange(100)%7) < (p*7)).astype(int)
        for method in ["identity", "sigmoid", "isotonic"]:
            c = ProbabilityCalibrator(method).fit(p, y)
            q = c.predict(p)
            self.assertTrue(((q > 0) & (q < 1)).all())
        t = choose_thresholds(y, p)
        self.assertGreaterEqual(float((p[y == 1] >= t["recall60"]).mean()), .6)

    def test_date_boundaries_are_disjoint(self):
        df = pd.DataFrame({"issue_date": pd.to_datetime(["2013-12-01", "2014-01-01", "2014-06-01", "2014-07-01", "2014-09-01", "2014-10-01", "2014-12-01", "2015-01-01"])})
        partitions = [subset(df, end="2013-12-31"), subset(df, "2014-01-01", "2014-06-30"),
                      subset(df, "2014-07-01", "2014-09-30"), subset(df, "2014-10-01", "2014-12-31")]
        for i,a in enumerate(partitions):
            for b in partitions[i+1:]:
                self.assertTrue(set(a.index).isdisjoint(b.index))
                self.assertLess(a.issue_date.max(), b.issue_date.min())


if __name__ == "__main__":
    unittest.main()
