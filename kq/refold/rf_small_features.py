import hyperopt
import pandas
import nose.tools
import numpy as np
from plumbum import colors
from sklearn import pipeline, preprocessing, linear_model, ensemble
import lightgbm.sklearn

from kq import core
from kq.feat_abhishek import FoldIndependent, hyper_helper
from kq.refold import rf_dataset, rf_decomposition, rf_distances, rf_vectorspaces, BaseTargetBuilder, \
    AutoExitingGBMLike, rf_magic_features, rf_word_count_features, rf_word_count_distances, rf_leaky, rf_pos_distances
from kq.refold.rf_sklearn import RF_SKLearn

__all__ = ['SmallFeatureLogit', 'SmallFeatureXTC', 'SmallFeatureLGB']


class SmallFeaturesTask(FoldIndependent):
    def _load_test(self, as_df):
        Xs = []
        n_vs = None
        for r in self.requires():
            x = r.load_all('test', as_df)
            if n_vs is None:
                n_vs = x.shape[0]
            else:
                nose.tools.assert_equal(n_vs, x.shape[0], repr(r))
            if as_df:
                x.columns = [r.__class__.__name__ + '_' + c for c in x.columns]
            Xs.append(x)
        if as_df:
            return pandas.concat(Xs, 1).reset_index(drop=True).fillna(999).clip(-1000, 1000)
        else:
            r = np.concatenate(Xs, 1)
            r[np.isnan(r)] = 999
            return np.clip(r, -1000, 1000)

    def _load(self, as_df):
        Xs = []
        n_vs = None
        for r in self.requires():
            x = r.load_all('train', as_df)
            nose.tools.assert_equal(len(x.shape), 2, repr(r))
            if n_vs is None:
                n_vs = x.shape[0]
            else:
                nose.tools.assert_equal(n_vs, x.shape[0], repr(r))
            if as_df:
                nose.tools.assert_is_instance(x, pandas.DataFrame, repr(r))
                x.columns = [r.__class__.__name__ + '_' + c for c in x.columns]
                x = x.fillna(999).clip(-1000, 1000)
            Xs.append(x)

        folds = rf_dataset.Dataset().load_dataset_folds()
        if as_df:
            res = pandas.concat(Xs, 1).reset_index(drop=True)
            return res, folds
        else:
            res = np.concatenate(Xs, 1)
            res[np.isnan(res)] = 999
            return np.clip(res, -1000, 1000), folds

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
            n_estimators=512,
            n_jobs=-1,
            min_samples_leaf=self.min_items.get(),
            class_weight=core.dictweights)

    def post_fit(self, cls):
        xs = SmallFeaturesTask().load('train', 0, as_df=True)
        series = pandas.Series(cls.feature_importances_, index=xs.columns)
        print(colors.yellow | str(series))

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
        default=0.02,
        disable=False)

    def post_fit(self, cls):
        xs = SmallFeaturesTask().load('train', 0, as_df=True)
        series = pandas.Series(cls.feature_importances_, index=xs.columns).sort_values(ascending=False)[:20]
        print(colors.yellow | str(series))

    def xdataset(self) -> FoldIndependent:
        return SmallFeaturesTask()

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
        default=6,
        transform=lambda v: v + 1,
        disable=False)

    learning_rate = hyper_helper.TuneableHyperparam(
        'SmallFeatureXGB.learning_rate',
        prior=hyperopt.hp.normal('SmallFeatureXGB.learning_rate', 0, 0.25),
        default=.02,
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
