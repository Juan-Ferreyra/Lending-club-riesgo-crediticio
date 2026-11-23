"""Serializable, development-fitted mappings and fixed probability ensembles."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


class PositiveSigmoid:
    """A strictly increasing sigmoid in exact arithmetic; no probability clipping."""

    def fit(self, score, y):
        values = np.asarray(score, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Calibration scores must be a finite one-dimensional vector')
        self.model = LogisticRegression(C=1000, max_iter=1500).fit(values.reshape(-1, 1), y)
        if not np.array_equal(self.model.classes_, [0, 1]):
            raise ValueError('Calibration requires both binary outcomes, encoded 0 and 1')
        if not np.isfinite(self.model.coef_).all() or self.model.coef_[0, 0] <= 0:
            raise ValueError('Nonpositive sigmoid slope: ranking cannot be preserved')
        self.slope_ = float(self.model.coef_[0, 0])
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def predict(self, score):
        values = np.asarray(score, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Calibration scores must be finite')
        return expit(self.slope_ * values + self.intercept_)

    def to_dict(self):
        return {'slope': self.slope_, 'intercept': self.intercept_, 'method': 'positive_sigmoid'}

    @classmethod
    def from_dict(cls, parameters):
        result = cls()
        result.slope_ = float(parameters['slope'])
        result.intercept_ = float(parameters['intercept'])
        if not np.isfinite([result.slope_, result.intercept_]).all() or result.slope_ <= 0:
            raise ValueError('Stored sigmoid must have a finite positive slope and finite intercept')
        # Retain the familiar fitted-estimator interface for diagnostics and inspection.
        result.model = LogisticRegression(C=1000, max_iter=1500)
        result.model.classes_ = np.array([0, 1])
        result.model.coef_ = np.array([[result.slope_]])
        result.model.intercept_ = np.array([result.intercept_])
        result.model.n_features_in_ = 1
        return result


def ensemble_score(scores, members, mappings):
    """Apply frozen nonnegative weights; SVM probability scales come from earlier CV."""
    if not members:
        raise ValueError('An ensemble requires at least one member')
    weights = np.asarray(list(members.values()), dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Ensemble weights must be nonnegative and sum to one')
    probability = None
    for name, weight in members.items():
        values = np.asarray(scores[name], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f'Invalid ensemble member scores: {name}')
        p = mappings[name].predict(values) if name in mappings else expit(values)
        if probability is None:
            probability = np.zeros(len(p), dtype=float)
        if len(p) != len(probability):
            raise ValueError('Ensemble member rows must be aligned')
        probability += weight * p
    # Numerical safety for conversion back to margin; raw component scores remain saved.
    return logit(np.clip(probability, 1e-12, 1 - 1e-12))
