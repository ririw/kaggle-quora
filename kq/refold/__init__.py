import os

import luigi
from sklearn import model_selection
from sklearn.base import ClassifierMixin, BaseEstimator

from kq import core

refold_max_folds = 6


class BaseTargetBuilder:
    def __init__(self, *parts, add_rf_cache=True):
        if add_rf_cache:
            self.parts = ['rf_cache'] + list(parts)
        else:
            self.parts = list(parts)

    def get(self):
        return os.path.join(*self.parts)

    def __add__(self, part):
        return BaseTargetBuilder(*(self.parts + [part]), add_rf_cache=False)


class AutoExitingGBMLike(BaseEstimator):
    def __init__(self, cls, additional_fit_args=None):
        self.additional_fit_args = {} if additional_fit_args is None else additional_fit_args
        self.cls = cls

    def fit(self, X, y):
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=0.05)
        self.cls.fit(X_tr, y_tr,
                     sample_weight=core.weight_from(y_tr),
                     eval_set=[(X_te, y_te)],
                     early_stopping_rounds=50,
                     **self.additional_fit_args)

    def predict_proba(self, X):
        return self.cls.predict_proba(X)

    # def get_params(self, deep=True):
    #    res = {'additional_fit_args': self.additional_fit_args,
    #           'underlying': self.cls.get_params(deep=deep)}
    #    return res
    #
    # def set_params(self, params):
    #    self.additional_fit_args = params['additional_fit_args']
    #    self.cls.set_params(params['underlying'])

    @property
    def feature_importances_(self):
        return self.cls.feature_importances_

    def __repr__(self):
        return 'AutoExitingGBMLike({:s})'.format(repr(self.cls))
