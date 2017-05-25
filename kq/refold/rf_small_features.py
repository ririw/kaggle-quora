import hyperopt
import pandas
import nose.tools
import numpy as np
from sklearn import pipeline, preprocessing, linear_model, ensemble
import lightgbm.sklearn

from kq import core
from kq.feat_abhishek import FoldIndependent, hyper_helper
from kq.refold import rf_dataset, rf_decomposition, rf_distances, rf_vectorspaces, BaseTargetBuilder, AutoExitingGBMLike
from kq.refold.rf_sklearn import RF_SKLearn

__all__ = ['SmallFeatureLogit', 'SmallFeatureXTC', 'SmallFeatureLGB']


class SmallFeaturesTask(FoldIndependent):
    def _load_test(self, as_df):
        Xs = []
        for r in self.requires():
            x = r.load_all('test', as_df)
            Xs.append(x)
        if as_df:
            return pandas.concat(Xs, 1)
        else:
            return np.concatenate(Xs, 1)

    def _load(self, as_df):
        Xs = []
        for r in self.requires():
            x = r.load_all('train', as_df)
            nose.tools.assert_equal(len(x.shape), 2, repr(r))
            Xs.append(x)

        folds = rf_dataset.Dataset().load_dataset_folds()
        if as_df:
            return pandas.concat(Xs, 1), folds
        else:
            return np.concatenate(Xs, 1), folds

    def requires(self):
        yield rf_decomposition.AllDecompositions()
        yield rf_distances.RFDistanceCalculator()
        yield rf_vectorspaces.VectorSpaceTask(include_space=False)

    def complete(self):
        for req in self.requires():
            if not req.complete():
                return False
        return True


class SmallFeatureLogit(RF_SKLearn):
    resources = {'cpu': 2, 'mem': 2}

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    def make_cls(self):
        return pipeline.Pipeline([
            ('norm', preprocessing.Normalizer()),
            ('lin', linear_model.LogisticRegression(
                C=100,
                class_weight=core.dictweights,
                solver='sag',
                max_iter=1000))
        ])

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_small_feat',
            'lin',
            str(self.fold)
        )
        return (base_path + fname).get()


class SmallFeatureXTC(RF_SKLearn):
    resources = {'cpu': 7}

    min_items = hyper_helper.TuneableHyperparam(
        'SmallFeatureXTC.min_items',
        prior=hyperopt.hp.randint('SmallFeatureXTC.min_items', 9),
        transform=lambda v: 2 ** v,
        default=2,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    def make_cls(self):
        return ensemble.ExtraTreesClassifier(
            n_estimators=500,
            n_jobs=-1,
            min_samples_leaf=self.min_items.get(),
            class_weight=core.dictweights)

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_small_feat',
            'xtc',
            'mi_{:d}'.format(self.min_items.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


class SmallFeatureLGB(RF_SKLearn):
    resources = {'cpu': 7}

    min_items = hyper_helper.TuneableHyperparam(
        'SmallFeatureLGB.min_items',
        prior=hyperopt.hp.randint('SmallFeatureLGB.min_items', 9),
        transform=lambda v: 2 ** v,
        default=2,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'SmallFeatureLGB.learning_rate',
        prior=hyperopt.hp.uniform('SmallFeatureLGB.learning_rate', 0, 0.4),
        default=0.10293737153579499,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    def make_cls(self):
        return AutoExitingGBMLike(
            lightgbm.sklearn.LGBMClassifier(
                n_estimators=2048,
                num_leaves=1024,
                min_child_samples=self.min_items.get(),
                learning_rate=self.learning_rate.get()
            )
        )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_small_feat',
            'lgb',
            'mi_{:d}_lr_{:f}'.format(self.min_items.get(), self.learning_rate.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


from xgboost.sklearn import XGBClassifier


class SmallFeatureXGB(RF_SKLearn):
    resources = {'cpu': 7}

    max_depth = hyper_helper.TuneableHyperparam(
        'SmallFeatureXGB.max_depth',
        prior=hyperopt.hp.randint('SmallFeatureXGB.max_depth', 11),
        default=8,
        transform=lambda v: v + 1,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'SmallFeatureXGB.learning_rate',
        prior=hyperopt.hp.normal('SmallFeatureXGB.learning_rate', 0, 0.25),
        default=-0.26776551679694416,
        transform=np.abs,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    def make_cls(self):
        return AutoExitingGBMLike(
            XGBClassifier(
                n_estimators=2048,
                learning_rate=self.learning_rate.get(),
                max_depth=self.max_depth.get(),
                subsample=0.75)
        )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_small_feat',
            'xgb',
            'md_{:d}_lr_{:f}'.format(self.max_depth.get(), self.learning_rate.get()),
            str(self.fold)
        )
        return (base_path + fname).get()
