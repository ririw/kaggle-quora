import hyperopt
import luigi
import numpy as np
import pandas
import sklearn.linear_model
from plumbum import colors
from sklearn import pipeline, preprocessing, ensemble, model_selection, feature_selection
from lightgbm.sklearn import LGBMClassifier

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper, fold_max
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder, rf_ab, AutoExitingGBMLike
from kq.refold.argpassing_rfe import ArgpassingRFE

__all__ = ['ABSklearn', 'AllAB']


class ABSklearn(FoldDependent):
    resources = {'cpu': 8, 'mem': 2}

    def make_path(self, fname):
        raise NotImplementedError

    def _load(self, name, as_df):
        path = self.make_path(name + '.npz')
        if as_df:
            return pandas.DataFrame({self.__class__.__name__: np.load(path)['data']})
        else:
            return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_ab.ABDataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def make_cls(self):
        raise NotImplementedError('Implement Sklearn compatible classifier')

    def run(self):
        self.output().makedirs()
        ab_data = rf_ab.ABDataset()

        X = ab_data.load('train', self.fold)
        y = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate

        cls = self.make_cls()
        cls.fit(X, y)

        X_val = ab_data.load('valid', self.fold)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = ab_data.load('test', None)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score


class ABLinear(ABSklearn):
    resources = {'cpu': 2, 'mem': 4}

    def make_cls(self):
        return pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2)),
            ('lin', sklearn.linear_model.LogisticRegression(
                C=self.C,
                class_weight=core.dictweights,
                solver='sag',
                max_iter=1000))
        ])

    C = hyper_helper.LuigiTuneableHyperparam(
        prior=hyperopt.hp.randint('ABLinear.C', 6),
        transform=lambda v: 10 ** ((v - 2) / 2),
        default=4,
        disable=False)

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_lin',
            'C_{:f}'.format(self.C),
            str(self.fold)
        )
        return (base_path + fname).get()


class AB_XTC(ABSklearn):
    min_items = hyper_helper.TuneableHyperparam(
        'AB_XTC.min_items',
        prior=hyperopt.hp.randint('AB_XTC.min_items', 9),
        transform=lambda v: 2 ** v,
        default=4,
        disable=False)
    poly_fetures = hyper_helper.TuneableHyperparam(
        'AB_XTC.poly_fetures',
        prior=hyperopt.hp.randint('AB_XTC.poly_fetures', 2),
        transform=lambda v: v + 1,
        default=1,
        disable=False)

    def make_cls(self):
        inner_cls = sklearn.ensemble.ExtraTreesClassifier(
                n_estimators=512, n_jobs=-1,
                verbose=1,
                class_weight=core.dictweights,
                min_samples_leaf=self.min_items.get())
        return pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2, include_bias=False)),
            ('xtc', inner_cls)
        ])

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_xtc',
            'min_items_{:f}_pf_{:d}'.format(self.min_items.get(), self.poly_fetures.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


class AB_LGB(ABSklearn):
    learning_rate = hyper_helper.TuneableHyperparam(
        'AB_LGB.learning_rate',
        prior=hyperopt.hp.normal('AB_LGB.learning_rate', 0, 0.25),
        default=0.0932165701272348,
        transform=np.abs,
        disable=False)

    min_child_samples = hyper_helper.TuneableHyperparam(
        'AB_LGB.min_child_samples',
        prior=hyperopt.hp.randint('AB_LGB.min_child_samples', 6),
        default=5,
        transform=lambda v: 2 ** v,
        disable=False)

    def make_cls(self):
        cls = LGBMClassifier(
            n_estimators=2048,
            learning_rate=self.learning_rate.get(),
            min_child_samples=self.min_child_samples.get(),
            subsample=0.75,
        )

        return pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2, include_bias=False)),
            ('lgb', AutoExitingGBMLike(cls, additional_fit_args={'verbose': False}))
        ])

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_lgbm',
            'lr_{:f}_mc_{:d}'.format(self.learning_rate.get(), self.min_child_samples.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


from xgboost.sklearn import XGBClassifier


class AB_XGB(ABSklearn):
    max_depth = hyper_helper.TuneableHyperparam(
        'AB_XGB.max_depth',
        prior=hyperopt.hp.randint('AB_XGB.max_depth', 11),
        default=9,
        transform=lambda v: v + 1,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'AB_XGB.learning_rate',
        prior=hyperopt.hp.normal('AB_XTC.learning_rate', 0, 0.25),
        default=0.48585291338722997,
        transform=np.abs,
        disable=False)

    def make_cls(self):
        cls = XGBClassifier(
                n_estimators=2048,
                learning_rate=self.learning_rate.get(),
                max_depth=self.max_depth.get(),
                subsample=0.75)
        return pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2, include_bias=False)),
            ('lgb', AutoExitingGBMLike(cls))
        ])

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_xgb',
            'lr_{:f}_md_{:d}'.format(self.learning_rate.get(), self.max_depth.get()),
            str(self.fold)
        )
        return (base_path + fname).get()
