import hyperopt
import luigi
import numpy as np
import sklearn.linear_model
from plumbum import colors
from sklearn import pipeline, preprocessing, ensemble, model_selection
from lightgbm.sklearn import LGBMClassifier

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper, fold_max
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder, rf_ab

__all__ = ['ABSklearn', 'AllAB']


class ABSklearn(FoldDependent):
    resources = {'cpu': 8, 'mem': 2}

    def make_path(self, fname):
        raise NotImplementedError

    def _load(self, name):
        path = self.make_path(name + '.npz')
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

        X = ab_data.load('train', self.fold, as_np=False)
        y = rf_dataset.Dataset().load('train', self.fold, as_np=False).is_duplicate

        cls = self.fit(X, y)

        X_val = ab_data.load('valid', self.fold, as_np=False)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_np=False).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = ab_data.load('test', None, as_np=False)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score

    def fit(self, X, y):
        cls = self.make_cls()
        cls.fit(X, y)


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
        transform=lambda v: 2**v,
        default=10,
        disable=False)

    def make_cls(self):
        return sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=500, n_jobs=-1,
            class_weight=core.dictweights,
            min_samples_leaf=self.min_items.get())

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_xtc',
            'min_items_{:f}'.format(self.min_items.get()),
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

    num_leaves = hyper_helper.TuneableHyperparam(
        'AB_LGB.num_leaves',
        prior=hyperopt.hp.randint('AB_LGB.num_leaves', 6),
        default=0,
        transform=lambda v: 2 ** (v + 3),
        disable=False)

    def make_cls(self):
        return LGBMClassifier(
            n_estimators=4096,
            learning_rate=self.learning_rate.get(),
            num_leaves=self.num_leaves.get(),
            subsample=0.75
        )

    def fit(self, X, y):
        cls = self.make_cls()
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=0.05)
        cls.fit(X_tr, y_tr, sample_weight=core.weight_from(y_tr), eval_set=[(X_te, y_te)],
                early_stopping_rounds=25)
        return cls

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_lgbm',
            'lr_{:f}_nl_{:d}'.format(self.learning_rate.get(), self.num_leaves.get()),
            str(self.fold)
        )
        return (base_path + fname).get()

from xgboost.sklearn import XGBClassifier

class AB_XGB(ABSklearn):
    max_depth = hyper_helper.TuneableHyperparam(
        'AB_XGB.max_depth',
        prior=hyperopt.hp.randint('AB_XGB.max_depth', 11),
        default=9,
        transform=lambda v: v+1,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'AB_XGB.learning_rate',
        prior=hyperopt.hp.normal('AB_XTC.learning_rate', 0, 0.25),
        default=0.48585291338722997,
        transform=np.abs,
        disable=False)

    def make_cls(self):
        return XGBClassifier(
            n_estimators=4096,
            learning_rate=self.learning_rate.get(),
            max_depth=self.max_depth.get(),
            subsample=0.75
        )

    def fit(self, X, y):
        cls = self.make_cls()
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=0.05)
        cls.fit(X_tr, y_tr, sample_weight=core.weight_from(y_tr), eval_set=[(X_te, y_te)],
                early_stopping_rounds=25)
        return cls

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_xgb',
            'lr_{:f}_md_{:d}'.format(self.learning_rate.get(), self.max_depth.get()),
            str(self.fold)
        )
        return (base_path + fname).get()
