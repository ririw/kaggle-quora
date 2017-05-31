import gzip

import hyperopt
import luigi
import nose.tools
import numpy as np
import os
import pandas
from plumbum import colors
from sklearn import linear_model, preprocessing

from kq import core
from kq.feat_abhishek import HyperTuneable, fold_max
from kq.feat_abhishek.hyper_helper import TuneableHyperparam
from kq.refold import rf_dataset, BaseTargetBuilder, rf_ab_sklearn, rf_wc_sklearn, \
    rf_small_features, rf_keras, rf_naive_bayes, rf_leaky


class Stacker(luigi.Task, HyperTuneable):
    npoly = TuneableHyperparam(
        "Stacker_npoly", hyperopt.hp.randint('Stacker_npoly', 3), 1, transform=lambda x: x+1)

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
            #rf_keras.SiameseModel(fold=fold),
            #rf_keras.ReaderModel(fold=fold),
            rf_naive_bayes.RF_NaiveBayes(fold=fold),
            rf_leaky.RFLeakingModel_XGB(fold=fold),
            rf_leaky.RFLeakingModel_LGB(fold=fold),
        ]

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_stacker',
            'np_{:d}'.format(self.npoly.get()),
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

    def score(self):
        self.output().makedirs()
        poly = preprocessing.PolynomialFeatures(self.npoly.get())
        train_Xs = []
        train_ys = []
        for fold in range(1, fold_max):
            y = rf_dataset.Dataset().load('valid', fold, as_df=True).is_duplicate.values.squeeze()
            x = self.fold_x(fold, 'valid')
            nose.tools.assert_equal(x.shape[0], y.shape[0])
            train_Xs.append(x)
            train_ys.append(y)

        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(train_Xs[0], train_ys[0])
        print(pandas.Series(cls.coef_[0], index=train_Xs[0].columns))

        train_X = poly.fit_transform(pandas.concat(train_Xs, 0).values)
        train_y = np.concatenate(train_ys, 0).squeeze()

        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(train_X, train_y)

        test_x = poly.transform(self.fold_x(0, 'valid'))
        test_y = rf_dataset.Dataset().load('valid', 0, as_df=True).is_duplicate.values.squeeze()

        score = core.score_data(test_y, cls.predict_proba(test_x))
        return score, poly, cls

    def run(self):
        #for c in self.classifiers(0):
        #    print(repr(c), c.load('test', 0).shape)
        score, poly, cls = self.score()

        print(colors.green | colors.bold | (Stacker.__name__ + '::' + str(score)))

        preds = []
        for fold in range(fold_max):
            test_X = poly.transform(self.fold_x(fold, 'test'))
            test_pred = cls.predict_proba(test_X)[:, 1]
            preds.append(test_pred)
        predmat = np.vstack(preds).mean(0)

        index = pandas.Index(np.arange(test_X.shape[0]), name='test_id')
        pred = pandas.Series(predmat, index=index, name='is_duplicate').to_frame()
        with gzip.open(self.make_path('stacked_pred.csv.gz.tmp'), 'wt') as f:
            pred.to_csv(f)
        os.rename(self.make_path('stacked_pred.csv.gz.tmp'), self.make_path('stacked_pred.csv.gz'))
        return score
