import hyperopt
import luigi
import pandas
import nose.tools
import numpy as np
import scipy.sparse as sp
from plumbum import colors
from sklearn import pipeline, preprocessing, linear_model, ensemble, neighbors
import lightgbm.sklearn

from kq import core
from kq.feat_abhishek import FoldIndependent, hyper_helper
from kq.refold import rf_dataset, rf_decomposition, rf_distances, rf_vectorspaces, BaseTargetBuilder, \
    AutoExitingGBMLike, rf_magic_features, rf_word_count_features, rf_word_count_distances, rf_leaky, rf_pos_distances
from kq.refold.rf_sklearn import RF_SKLearn

__all__ = ['AllFeatureXTC', 'AllFeatureLGB']


class AllFeaturesTask(FoldIndependent):
    def _load_test(self, as_df):
        assert not as_df, 'Pandas mode not supported'
        Xs = []
        n_vs = None
        for r in self.requires():
            x = r.load_all('test', False)
            if isinstance(x, np.ndarray):
                x[np.isnan(x)] = 999
                x = np.clip(x, -1000, 1000)
            if n_vs is None:
                n_vs = x.shape[0]
            else:
                nose.tools.assert_equal(n_vs, x.shape[0], repr(r))
            Xs.append(x)
        res = sp.hstack(Xs)
        return res.tocsr()

    def _load(self, as_df):
        assert not as_df, 'Pandas mode not supported'
        Xs = []
        n_vs = None
        for r in self.requires():
            x = r.load_all('train', False)
            nose.tools.assert_equal(len(x.shape), 2, repr(r))
            if isinstance(x, np.ndarray):
                x[np.isnan(x)] = 999
                x = np.clip(x, -1000, 1000)
            if n_vs is None:
                n_vs = x.shape[0]
            else:
                nose.tools.assert_equal(n_vs, x.shape[0], repr(r))
            Xs.append(x)

        folds = rf_dataset.Dataset().load_dataset_folds()
        res = sp.hstack(Xs)
        return res.tocsr(), folds

    def requires(self):
        yield rf_decomposition.AllDecompositions()
        yield rf_word_count_distances.WordCountDistances()
        yield rf_distances.RFDistanceCalculator()
        yield rf_vectorspaces.VectorSpaceTask(include_space=False)
        yield rf_magic_features.QuestionFrequency()
        yield rf_magic_features.NeighbourhoodFeature()
        yield rf_magic_features.QuestionOrderFeature()
        yield rf_leaky.RF_LeakyXGB_Dataset()
        yield rf_pos_distances.RF_POS_Distance()
        yield rf_word_count_features.WordCountMatrix()

    def complete(self):
        for req in self.requires():
            if not req.complete():
                return False
        return True


class AllFeatureLGB(RF_SKLearn):
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
        default=0.02,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return AllFeaturesTask()

    def make_cls(self):
        return AutoExitingGBMLike(
            lightgbm.sklearn.LGBMClassifier(
                n_estimators=1024,
                num_leaves=1024,
                min_child_samples=self.min_items.get(),
                learning_rate=self.learning_rate.get()
            )
        )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_all_feat',
            'lgb',
            'mi_{:d}_lr_{:f}'.format(self.min_items.get(), self.learning_rate.get()),
            str(self.fold)
        )
        return (base_path + fname).get()


from xgboost.sklearn import XGBClassifier


class AllFeatureXGB(RF_SKLearn):
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
        default=.05,
        transform=np.abs,
        disable=False)

    def xdataset(self) -> FoldIndependent:
        return AllFeaturesTask()

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
            'rf_all_feat',
            'xgb',
            'md_{:d}_lr_{:f}'.format(self.max_depth.get(), self.learning_rate.get()),
            str(self.fold)
        )
        return (base_path + fname).get()

