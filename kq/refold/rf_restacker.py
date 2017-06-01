import gzip
import os

import lightgbm
import luigi
import nose.tools
import numpy as np
import pandas
from plumbum import colors
from sklearn import linear_model, ensemble, pipeline, preprocessing
from xgboost.sklearn import XGBClassifier

from kq.core import score_data, dictweights
from kq.feat_abhishek import fold_max
from kq.refold import rf_dataset, BaseTargetBuilder, rf_wc_sklearn, \
    AutoExitingGBMLike, rf_ab_sklearn, rf_small_features, rf_naive_bayes, rf_leaky, rf_keras


class ReStacker(luigi.Task):
    def requires(self):
        yield rf_dataset.Dataset()
        xs = []
        for fold in range(fold_max):
            for cls in self.datasets(fold):
                xs.append(cls)

        for v in sorted(xs, key=lambda c: c.__class__.__name__):
            yield v

    @staticmethod
    def datasets(fold):
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
            rf_keras.SiameseModel(fold=fold),
            rf_keras.ReaderModel(fold=fold),
            rf_naive_bayes.RF_NaiveBayes(fold=fold),
            rf_leaky.RFLeakingModel_XGB(fold=fold),
            rf_leaky.RFLeakingModel_LGB(fold=fold),
        ]

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_restacker')
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('stacked_pred.csv.gz'))

    def fold_x(self, fold, dataset):
        xs = []
        x_len = None
        for c in self.datasets(fold):
            x = c.load(dataset)
            if x_len is None:
                x_len = x.shape[0]
            nose.tools.assert_equal(x_len, x.shape[0], 'Shape mismatch for ' + repr(c))
            xs.append(x)
        res = np.vstack(xs).T
        return pandas.DataFrame(res, columns=[c.__class__.__name__ for c in self.datasets(fold)])

    def fold_y(self, fold, dataset):
        y = rf_dataset.Dataset().load(dataset, fold, as_df=True).is_duplicate.values.squeeze()
        return y

    @staticmethod
    def classifiers():
        yield AutoExitingGBMLike(XGBClassifier(
            n_estimators=1024,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.75
        ), additional_fit_args={'verbose': False})
        yield AutoExitingGBMLike(lightgbm.LGBMClassifier(
            num_leaves=1024,
            subsample=0.75,
            n_estimators=1024,
            learning_rate=0.05),
            additional_fit_args={'verbose': False})
        yield ensemble.ExtraTreesClassifier(n_estimators=512,
                                            min_samples_leaf=5,
                                            n_jobs=-1)

        #yield svm.SVC()

    def run(self):
        self.output().makedirs()
        fold_ord = np.random.permutation(fold_max)
        merge_fold = fold_ord[0]
        test_fold = fold_ord[1]
        stack_folds = fold_ord[2:]

        stack_Xs = [self.fold_x(f, 'valid') for f in stack_folds]
        stack_ys = [self.fold_y(f, 'valid') for f in stack_folds]
        stack_X = pandas.concat(stack_Xs, 0)
        stack_y = np.concatenate(stack_ys, 0)
        merge_X = self.fold_x(merge_fold, 'valid')
        merge_y = self.fold_y(merge_fold, 'valid')
        test_X = self.fold_x(test_fold, 'valid')
        test_y = self.fold_y(test_fold, 'valid')

        classifiers = list(self.classifiers())
        merge_preds = []
        test_preds = []
        ds_names = [repr(d) for d in self.datasets(0)]
        for cls in classifiers:
            print(colors.blue | colors.bold | "Training {:s}".format(repr(cls)))
            cls.fit(stack_X, stack_y)
            print(colors.yellow | str(pandas.Series(cls.feature_importances_, index=ds_names).sort_values()))
            test_pred = cls.predict_proba(test_X)[:, 1]
            merge_pred = cls.predict_proba(merge_X)[:, 1]
            score = score_data(test_y, test_pred)
            print(colors.yellow | 'Score: {:f}'.format(score))

            merge_preds.append(merge_pred)
            test_preds.append(test_pred)

        merge_pred = np.vstack(merge_preds).T
        test_pred = np.vstack(test_preds).T

        #merge_cls = AutoExitingGBMLike(XGBClassifier(
        #    n_estimators=1024,
        #    learning_rate=0.05,
        #    max_depth=6,
        #    subsample=0.75
        #), additional_fit_args={'verbose': False})
        merge_cls = pipeline.Pipeline([
            ('poly', preprocessing.PolynomialFeatures(3)),
            ('cls',  linear_model.LogisticRegression())
        ])

        merge_cls.fit(merge_pred, merge_y)

        test_score = score_data(test_y, merge_cls.predict_proba(test_pred)[:, 1])
        print(colors.green | 'Final score: {:f}'.format(test_score))

        fold_preds = []
        for fold in range(fold_max):
            fold_X = self.fold_x(fold, 'test')
            fold_merge_X = np.zeros([fold_X.shape[0], len(classifiers)])
            for ix, cls in enumerate(classifiers):
                fold_merge_X[:, ix] = cls.predict_proba(fold_X)[:, 1]
            fold_preds.append(merge_cls.predict_proba(fold_merge_X)[:, 1])

        predmat = np.vstack(fold_preds).mean(0)

        index = pandas.Index(np.arange(fold_X.shape[0]), name='test_id')
        print(predmat.shape)
        print(index.shape)
        pred = pandas.Series(predmat, index=index, name='is_duplicate').to_frame()
        with gzip.open(self.make_path('stacked_pred.csv.gz.tmp'), 'wt') as f:
            pred.to_csv(f)
        os.rename(self.make_path('stacked_pred.csv.gz.tmp'), self.make_path('stacked_pred.csv.gz'))
        return test_score
