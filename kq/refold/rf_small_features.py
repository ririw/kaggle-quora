import hyperopt
import pandas
import numpy as np
from sklearn import pipeline, preprocessing, linear_model, ensemble

from kq import core
from kq.feat_abhishek import FoldIndependent, hyper_helper
from kq.refold import rf_dataset, rf_decomposition, rf_distances, rf_vectorspaces, BaseTargetBuilder
from kq.refold.rf_sklearn import RF_SKLearn

__all__ = ['SmallFeatureLogit', 'SmallFeatureXTC']


class SmallFeaturesTask(FoldIndependent):
    def _load_test(self, as_df):
        Xs = [r.load_all('test', as_df) for r in self.requires()]
        if as_df:
            return pandas.concat(Xs, 1)
        else:
            return np.concatenate(Xs, 1)

    def _load(self, as_df):
        Xs = [r.load_all('train', as_df) for r in self.requires()]
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
    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    resources = {'cpu': 2, 'mem': 4}

    def make_cls(self):
        return pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2)),
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
    min_items = hyper_helper.TuneableHyperparam(
        'SmallFeatureXTC.min_items',
        prior=hyperopt.hp.randint('SmallFeatureXTC.min_items', 9),
        transform=lambda v: 2 ** v,
        default=2,
        disable=False)
    poly_fetures = hyper_helper.TuneableHyperparam(
        'SmallFeatureXTC.poly_fetures',
        prior=hyperopt.hp.randint('SmallFeatureXTC.poly_fetures', 2),
        transform=lambda v: v + 1,
        default=1,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

    resources = {'cpu': 2, 'mem': 4}

    def make_cls(self):
        return pipeline.Pipeline([
            ('poly', preprocessing.PolynomialFeatures(self.poly_fetures.get())),
            ('xtc', ensemble.ExtraTreesClassifier(
                n_estimators=1000,
                n_jobs=-1,
                min_samples_leaf=self.min_items.get(),
                class_weight=core.dictweights
            ))
        ])

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_small_feat',
            'xtc',
            'pf_{:d}_mi_{:d}'.format(self.poly_fetures.get(), self.min_items.get()),
            str(self.fold)
        )
        return (base_path + fname).get()
