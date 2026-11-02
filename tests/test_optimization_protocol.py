"""Checks for temporal separation, rank preservation, and the executed search budget."""
from types import SimpleNamespace
import unittest

import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from optimization.data import train_rows, valid_rows, recency_weights
from optimization.ensemble import PositiveSigmoid, ensemble_score
from optimization.evaluation import december_split
from optimization.experiments import remaining_trials


class OptimizationProtocolTests(unittest.TestCase):
    def history(self):
        dates = pd.date_range('2012-01-01', '2014-12-01', freq='MS').repeat(12)
        n = len(dates)
        return pd.DataFrame({
            'id': [f'loan_{i:04}' for i in range(n)],
            'issue_date': dates,
            'term': np.where(np.arange(n) % 3 == 0, ' 60 months', ' 36 months'),
            'target': (np.arange(n) % 5 == 0).astype(int),
        })

    def test_temporal_cohorts_remain_disjoint_and_comparable_under_training_recipes(self):
        raw = self.history()
        valid = valid_rows(raw, '2014-01-01', '2014-06-30', cap=30)
        repeated = valid_rows(raw, '2014-01-01', '2014-06-30', cap=30)
        self.assertEqual(valid.id.tolist(), repeated.id.tolist())
        self.assertTrue(valid.term.str.strip().eq('36 months').all())
        for start in ['2012-01-01', '2013-01-01']:
            train = train_rows(raw, '2013-12-31', start, cap=50)
            self.assertTrue(train.issue_date.between(start, '2013-12-31').all())
            self.assertTrue(train.term.str.strip().eq('36 months').all())
            self.assertTrue(set(train.id).isdisjoint(valid.id))
            self.assertLess(train.issue_date.max(), valid.issue_date.min())

    def test_recency_changes_training_influence_without_changing_validation_population(self):
        train = train_rows(self.history(), '2013-12-31')
        weights = recency_weights(train, '2013-12-31', half_life=12)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertLess(weights[0], weights[-1])
        older = train.issue_date.eq(pd.Timestamp('2012-01-01')).to_numpy()
        newer = train.issue_date.eq(pd.Timestamp('2013-01-01')).to_numpy()
        self.assertAlmostEqual(weights[newer].mean() / weights[older].mean(),
                               2 ** (366 / 30.4375 / 12), places=10)
        self.assertIsNone(recency_weights(train, '2013-12-31', half_life=0))

    def test_december_calibration_policy_split_is_id_based_disjoint_and_target_blind(self):
        raw = valid_rows(self.history(), '2014-12-01', '2014-12-31')
        cal, policy = december_split(raw)
        self.assertGreater(len(cal), 0)
        self.assertGreater(len(policy), 0)
        self.assertTrue(set(cal.id).isdisjoint(policy.id))
        self.assertEqual(set(cal.id) | set(policy.id), set(raw.id))
        altered = raw.sample(frac=1, random_state=2026).copy()
        altered['target'] = 1 - altered.target
        cal_again, policy_again = december_split(altered)
        self.assertEqual(set(cal.id), set(cal_again.id))
        self.assertEqual(set(policy.id), set(policy_again.id))
        self.assertTrue(cal.issue_date.eq(pd.Timestamp('2014-12-01')).all())
        self.assertTrue(policy.issue_date.eq(pd.Timestamp('2014-12-01')).all())

    def test_positive_sigmoid_keeps_ap_and_tied_score_groups(self):
        scores = np.repeat(np.linspace(-3, 3, 15), 4)
        labels = (np.arange(60) % 4 < np.repeat(np.arange(15) / 4, 4)).astype(int)
        calibrator = PositiveSigmoid().fit(scores, labels)
        probabilities = calibrator.predict(scores)
        self.assertGreater(calibrator.model.coef_[0, 0], 0)
        self.assertAlmostEqual(average_precision_score(labels, scores),
                               average_precision_score(labels, probabilities), places=14)
        self.assertEqual(len(np.unique(scores)), len(np.unique(probabilities)))
        self.assertTrue((np.diff(probabilities) >= 0).all())
        with self.assertRaises(ValueError):
            PositiveSigmoid().fit(np.arange(20), (np.arange(20) < 10).astype(int))

    def test_ensemble_matches_named_members_independent_of_dict_or_row_order(self):
        score_a = np.array([-2., -.5, .8, 2.])
        score_b = np.array([1., -.3, .1, -1.])
        scores = {'unused': np.zeros(4), 'model_b': score_b, 'model_a': score_a}
        members = {'model_a': .25, 'model_b': .75}
        calibration = SimpleNamespace(predict=lambda x: expit(.4 * np.asarray(x) - .1))
        actual = ensemble_score(scores, members, {'model_b': calibration})
        expected = .25 * expit(score_a) + .75 * expit(.4 * score_b - .1)
        np.testing.assert_allclose(expit(actual), expected, rtol=1e-13)
        order = np.array([2, 0, 3, 1])
        permuted = {name: values[order] for name, values in reversed(list(scores.items()))}
        np.testing.assert_allclose(
            ensemble_score(permuted, dict(reversed(list(members.items()))), {'model_b': calibration}),
            actual[order], rtol=1e-13,
        )

    def test_waiting_trials_do_not_consume_executed_search_budget(self):
        states = optuna.trial.TrialState
        study = SimpleNamespace(trials=[SimpleNamespace(state=states.WAITING) for _ in range(3)])
        self.assertEqual(remaining_trials(study, 4), 4)
        for state in [states.COMPLETE, states.PRUNED, states.FAIL]:
            study.trials.append(SimpleNamespace(state=state))
        self.assertEqual(remaining_trials(study, 4), 1)
        self.assertEqual(remaining_trials(study, 2), 0)


if __name__ == '__main__':
    unittest.main()
