import hyperopt
import luigi
import numpy as np
import pandas
from lightgbm.sklearn import LGBMClassifier
from plumbum import colors
from sklearn import model_selection, feature_selection, ensemble, linear_model

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder, AutoExitingGBMLike

__all__ = ['WC_XGB', 'WC_LGB']


class WCSklearn(FoldDependent):
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
        yield rf_word_count_features.WordCountMatrix()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def make_cls(self):
        raise NotImplementedError('Implement Sklearn compatible classifier')

    def run(self):
        self.output().makedirs()
        wc_data = rf_word_count_features.WordCountMatrix()

        X = wc_data.load('train', self.fold).astype(np.float32)
        y = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate

        cls = self.make_cls()
        cls.fit(X, y)

        X_val = wc_data.load('valid', self.fold).astype(np.float32)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = wc_data.load('test', None).astype(np.float32)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score


class WC_LGB(WCSklearn):
    learning_rate = hyper_helper.TuneableHyperparam(
        'WC_LGB.learning_rate',
        prior=hyperopt.hp.normal('WC_LGB.learning_rate', 0, 0.25),
        default=0.32639409757569904,
        transform=np.abs,
        disable=False)

    min_child_samples = hyper_helper.TuneableHyperparam(
        'WC_LGB.min_child_samples',
        prior=hyperopt.hp.randint('WC_LGB.min_child_samples', 6),
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
        return AutoExitingGBMLike(cls)

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_lgbm',
            'lr_{:f}_mc_{:d}'.format(self.learning_rate.get(), self.min_child_samples.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


from xgboost.sklearn import XGBClassifier


class WC_XGB(WCSklearn):
    max_depth = hyper_helper.TuneableHyperparam(
        'WC_XGB.max_depth',
        prior=hyperopt.hp.randint('WC_XGB.max_depth', 11),
        default=9,
        transform=lambda v: v + 1,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'WC_XGB.learning_rate',
        prior=hyperopt.hp.normal('WC_XGB.learning_rate', 0, 0.25),
        default=0.48585291338722997,
        transform=np.abs,
        disable=False)

    def make_cls(self):
        return AutoExitingGBMLike(XGBClassifier(
            n_estimators=4096,
            learning_rate=self.learning_rate.get(),
            max_depth=self.max_depth.get(),
            subsample=0.75
        ))

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_xgb',
            'lr_{:f}_md_{:d}'.format(self.learning_rate.get(), self.max_depth.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


class WC_XTC(WCSklearn):
    min_leaf_samples = hyper_helper.TuneableHyperparam(
        name='WordCountXTC_min_leaf_samples',
        prior=hyperopt.hp.randint('WordCountXTC_min_leaf_samples', 20),
        default=2,
        transform=lambda v: (v + 1) * 5
    )

    def make_cls(self):
        return ensemble.ExtraTreesClassifier(
            n_jobs=-1,
            n_estimators=500,
            min_samples_leaf=self.min_leaf_samples.get()
        )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_xtc',
            'ls_{:d}'.format(self.min_leaf_samples.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


class WC_Logit(WCSklearn):
    def make_cls(self):
        return linear_model.LogisticRegression(solver='sag', C=100)

    resources = {'cpu': 2, 'mem': 4}

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_logit',
            'lr_{:f}_md_{:d}'.format(self.learning_rate.get(), self.max_depth.get()),
            str(self.fold)
        )
        return (base_path + fname).get()
