"""Regression checks for train-only data recipes in the second experiment."""
import unittest
import numpy as np
import pandas as pd
from research.models import AuditFeatures, ScoreCalibrator


class ResearchRecipeTests(unittest.TestCase):
    def frame(self):
        n=12
        return pd.DataFrame({'loan_amnt':np.linspace(1000,12000,n),
            'annual_inc':np.linspace(20000,100000,n),'dti':np.linspace(1,30,n),
            'fico_range_low':np.arange(n)*5+650,'fico_range_high':np.arange(n)*5+654,
            'issue_d':['Jan-2013']*n,'earliest_cr_line':['Jan-2000']*n,
            'term':['36 months']*n,'emp_length':['5 years']*n,
            'open_acc':np.arange(n)+1,'total_acc':np.arange(n)+12,
            'revol_bal':np.arange(n)*500.,'revol_util':np.linspace(0,100,n)})

    def test_unavailable_future_bureau_cannot_change_stable_features(self):
        raw=self.frame()
        transformer=AuditFeatures('stable').fit(raw)
        before=transformer.transform(raw)
        changed=raw.copy()
        changed['annual_inc_joint']=500000.
        changed['open_acc_6m']=99.
        changed['inq_last_12m']=42.
        pd.testing.assert_frame_equal(before,transformer.transform(changed))

    def test_clipping_bounds_fit_on_training_only(self):
        raw=self.frame()
        transformer=AuditFeatures('clip').fit(raw)
        bound=transformer.hi.copy()
        changed=raw.copy();changed['annual_inc']=1e10
        result=transformer.transform(changed)
        pd.testing.assert_series_equal(bound,transformer.hi)
        self.assertLessEqual(result.annual_inc.max(),bound.annual_inc)

    def test_outcomes_and_recoveries_do_not_enter_independent_recipe(self):
        raw=self.frame()
        transformer=AuditFeatures('stable').fit(raw)
        before=transformer.transform(raw)
        raw['loan_status']='Charged Off';raw['target']=1
        raw['recoveries']=1e9;raw['grade']='A';raw['int_rate']=1.
        pd.testing.assert_frame_equal(before,transformer.transform(raw))

    def test_margin_calibration_produces_finite_bounded_probabilities(self):
        score=np.linspace(-4,4,60)
        y=((np.arange(60)%5)<np.linspace(0,5,60)).astype(int)
        for method in ['sigmoid','isotonic']:
            calibration=ScoreCalibrator(method).fit(score,y)
            p=calibration.predict(np.array([-100.,0.,100.]))
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p>0)&(p<1)).all())


if __name__=='__main__':unittest.main()
