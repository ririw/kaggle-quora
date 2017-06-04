import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hyperopt
import seaborn as sns
import luigi
import nose.tools
import numpy as np
import os
import pandas
from plumbum import colors
from sklearn import linear_model, preprocessing, pipeline, svm, feature_selection
from xgboost import XGBClassifier

import keras
import keras.wrappers.scikit_learn
from kq import core
from kq.feat_abhishek import HyperTuneable, fold_max
from kq.feat_abhishek.hyper_helper import TuneableHyperparam
from kq.refold import rf_dataset, BaseTargetBuilder, rf_ab_sklearn, rf_wc_sklearn, \
    rf_small_features, rf_keras, rf_naive_bayes, rf_leaky, AutoExitingGBMLike, rf_all_features


class Stacker(luigi.Task, HyperTuneable):
    npoly = TuneableHyperparam(
        "Stacker_npoly", hyperopt.hp.randint('Stacker_npoly', 3), 2, transform=lambda x: x + 1)

    def requires(self):
        yield rf_dataset.Dataset()
        xs = []
        for fold in range(fold_max):
            for cls in self.classifiers(fold):
                xs.append(cls)

        for v in sorted(xs, key=lambda c: c.__class__.__name__):
            yield v

    def classifiers(self, fold):
        return [
            rf_wc_sklearn.WC_LGB(fold=fold),
            rf_wc_sklearn.WC_XGB(fold=fold),
            rf_wc_sklearn.WC_XTC(fold=fold),
            rf_wc_sklearn.WC_Logit(fold=fold),
            rf_ab_sklearn.AB_Logit(fold=fold),
            rf_ab_sklearn.AB_XTC(fold=fold),
            rf_ab_sklearn.AB_LGB(fold=fold),
            rf_ab_sklearn.AB_XGB(fold=fold),
            rf_small_features.SmallFeatureXTC(fold=fold),
            rf_small_features.SmallFeatureLogit(fold=fold),
            rf_small_features.SmallFeatureLGB(fold=fold),
            rf_small_features.SmallFeatureXGB(fold=fold),
            rf_small_features.SmallFeaturesKeras(fold=fold),
            rf_keras.SiameseModel(fold=fold),
            rf_keras.ReaderModel(fold=fold),
            rf_keras.ConcurrentReaderModel(fold=fold),
            rf_keras.SiameseNoDistModel(fold=fold),
            rf_keras.ReaderNoDistModel(fold=fold),
            rf_keras.ConcurrentNoDistReaderModel(fold=fold),
            rf_naive_bayes.RF_NaiveBayes(fold=fold),
            rf_leaky.RFLeakingModel_XGB(fold=fold),
            rf_leaky.RFLeakingModel_LGB(fold=fold),
            rf_all_features.AllFeatureXGB(fold=fold),
            rf_all_features.AllFeatureLGB(fold=fold),
        ]

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_stacker',
        )
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('stacked_pred.csv.gz'))

    def fold_x(self, fold, dataset):
        xs = []
        x_len = None
        for c in self.classifiers(fold):
            x = c.load(dataset)
            if x_len is None:
                x_len = x.shape[0]
            nose.tools.assert_equal(x_len, x.shape[0], 'Shape mismatch for ' + repr(c))
            xs.append(x)
        res = np.vstack(xs).T
        return pandas.DataFrame(res, columns=[c.__class__.__name__ for c in self.classifiers(fold)])

    def simple_nn(self):
        n_inputs = len(self.classifiers(0))
        m = keras.models.Sequential()
        m.add(keras.layers.Dense(n_inputs, input_shape=[n_inputs]))
        m.add(keras.layers.PReLU())
        m.add(keras.layers.Dropout(0.5))
        m.add(keras.layers.Dense(n_inputs * 2))
        m.add(keras.layers.PReLU())
        m.add(keras.layers.Dropout(0.5))
        m.add(keras.layers.Dense(1, activation='sigmoid'))
        m.compile('adam', 'binary_crossentropy')

        return m

    @property
    def score(self):
        self.output().makedirs()
        train_Xs = []
        train_ys = []
        for fold in range(1, fold_max):
            y = rf_dataset.Dataset().load('valid', fold, as_df=True).is_duplicate.values.squeeze()
            x = self.fold_x(fold, 'valid')
            nose.tools.assert_equal(x.shape[0], y.shape[0])
            train_Xs.append(x)
            train_ys.append(y)
        sns.clustermap(pandas.concat(train_Xs, 0).corr())
        plt.yticks(rotation=90)
        plt.savefig('./corr.png')
        plt.close()
        train_X = pandas.concat(train_Xs, 0).values
        train_y = np.concatenate(train_ys, 0).squeeze()

        cls = AutoExitingGBMLike(XGBClassifier(
            n_estimators=1024,
            learning_rate=0.05,
            max_depth=8,
            gamma=1,
            subsample=0.5
        ), additional_fit_args={'verbose': False})

        #cls = AutoExitingGBMLike(lightgbm.sklearn.LGBMClassifier(
        #    n_estimators=1024,
        #    learning_rate=0.01,
        #    subsample=0.5,
        #    num_leaves=2048
        #), additional_fit_args={'verbose': False})
        #cls = pipeline.Pipeline([
        #    ('poly', preprocessing.PolynomialFeatures(2)),
        #    ('anova', feature_selection.SelectPercentile(feature_selection.f_classif)),
        #    ('lin', linear_model.LogisticRegression(C=1, class_weight=core.dictweights))
        #])
        #cls = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.simple_nn)

        cls.fit(train_X, train_y)
        if hasattr(cls, 'feature_importances_'):
            ds_names = [repr(d) for d in self.classifiers(0)]
            print(colors.yellow | str(pandas.Series(cls.feature_importances_, index=ds_names).sort_values()))

        test_x = self.fold_x(0, 'valid').values
        test_y = rf_dataset.Dataset().load('valid', 0, as_df=True).is_duplicate.values.squeeze()

        score = core.score_data(test_y, cls.predict_proba(test_x)[:, 1])
        return score, cls

    def run(self):
        # for c in self.classifiers(0):
        #    print(repr(c), c.load('test', 0).shape)
        score, cls = self.score

        print(colors.green | colors.bold | (Stacker.__name__ + '::' + str(score)))

        preds = []
        for fold in range(fold_max):
            test_X = self.fold_x(fold, 'test').values
            test_pred = cls.predict_proba(test_X)[:, 1]
            preds.append(test_pred)
        predmat = np.vstack(preds).mean(0)

        index = pandas.Index(np.arange(test_X.shape[0]), name='test_id')
        pred = pandas.Series(predmat, index=index, name='is_duplicate').to_frame()
        with gzip.open(self.make_path('stacked_pred.csv.gz.tmp'), 'wt') as f:
            pred.to_csv(f)
        os.rename(self.make_path('stacked_pred.csv.gz.tmp'), self.output().path)
        return score

