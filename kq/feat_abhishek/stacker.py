import gzip

import numpy as np
import luigi
import pandas
import os
from plumbum import colors
from sklearn import linear_model
import nose.tools

from kq import core
from kq.feat_abhishek import xgb
from kq.feat_abhishek import xval_dataset, xtc, logit, lightgbt


class Stacker(luigi.Task):
    def requires(self):
        yield xval_dataset.BaseDataset()
        for fold in range(9):
            for cls in self.classifiers(fold):
                yield cls

    def classifiers(self, fold):
        return [
            xtc.XTCClassifier(fold=fold),
            logit.LogitClassifier(fold=fold),
            lightgbt.LightGBMClassifier(fold=fold)
        ]

    def output(self):
        return luigi.LocalTarget('cache/abhishek/stacked_pred.csv.gz')

    def fold_x(self, fold, dataset):
        xtc_x = xtc.XTCClassifier(fold=fold).load(dataset)
        logit_x = logit.LogitClassifier(fold=fold).load(dataset)

        return np.concatenate([xtc_x, logit_x], 1)

    def run(self):
        self.output().makedirs()
        train_Xs = []
        train_ys = []
        for fold in range(1, 9):
            y = xval_dataset.BaseDataset().load('valid', fold).squeeze()
            x = self.fold_x(fold, 'valid')
            nose.tools.assert_equal(x.shape[0], y.shape[0])
            train_Xs.append(x)
            train_ys.append(y)
        train_X = np.concatenate(train_Xs, 0)
        train_y = np.concatenate(train_ys, 0).squeeze()
        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(train_X, train_y)

        test_x = self.fold_x(0, 'valid')
        test_y = xval_dataset.BaseDataset().load('valid', 0).squeeze()

        score = core.score_data(test_y, cls.predict_proba(test_x))

        print(colors.green | colors.bold | (Stacker.__name__ + '::' + str(score)))

        test_Xs = []
        for fold in range(9):
            x = self.fold_x(fold, 'test')
            test_Xs.append(x)
        test_X = np.concatenate(test_Xs, 0)
        test_pred = cls.predict_proba(test_X)

        index = pandas.Index(np.arange(test_X.shape[0]), name='test_id')
        pred = pandas.Series(test_pred[:, 1], index=index, name='is_duplicate').to_frame()
        with gzip.open('cache/abhishek/stacked_pred.csv.gz.tmp', 'wt') as f:
            pred.to_csv(f)
        os.rename('cache/abhishek/stacked_pred.csv.gz.tmp', 'cache/abhishek/stacked_pred.csv.gz')

