import numpy as np
import luigi
from plumbum import colors
from sklearn import linear_model

from kq import core
from kq.feat_abhishek import xval_dataset, xtc, logit


class Stacker(luigi.Task):
    def requires(self):
        yield xval_dataset.BaseDataset()
        for fold in range(9):
            yield xtc.XTCClassifier(fold=fold)
            yield logit.LogitClassifier(fold=fold)

    def output(self):
        return luigi.LocalTarget('cache/abhishek/stacked_pred.csv.gz')

    def fold_x(self, fold, dataset):
        xtc_x = xtc.XTCClassifier(fold=fold).load(dataset)
        logit_x = logit.LogitClassifier(fold=fold).load(dataset)

        return np.concatenate([xtc_x, logit_x], 1)

    def run(self):
        train_Xs = []
        train_ys = []
        for fold in range(1, 9):
            y = xval_dataset.BaseDataset().load('valid', fold)
            x = self.fold_x(fold, 'valid')
            train_Xs.append(x)
            train_ys.append(y)
        train_X = np.concatenate(train_Xs, 0)
        train_y = np.concatenate(train_ys, 0)

        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(train_X, train_y)

        test_x = self.fold_x(0)
        test_y = xval_dataset.BaseDataset().load('valid', 0)

        score = core.score_data(test_y, cls.predict_proba(test_x))

        print(colors.green | colors.bold | (Stacker.__name__ + str(score)))

        test_Xs = []
        for fold in range(9):
            x = self.fold_x(fold, 'test')
            test_Xs.append(x)
        test_X = np.concatenate(test_Xs, 0)
        test_pred = cls.predict(test_X)