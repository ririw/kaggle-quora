import luigi
import pandas
from plumbum import colors
from sklearn import linear_model
import numpy as np

from kq import core
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['LogitClassifier']


class LogitClassifier(FoldDependent):
    def _load(self, name):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/logit/{:d}/{:s}.npy'.format(self.fold, name)
        return pandas.DataFrame({'LogitClassifier': np.load(fn)})

    def output(self):
        return luigi.LocalTarget('cache/abhishek/logit/{:d}/done'.format(self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold, as_np=False)
        y = xval_dataset.BaseDataset().load('train', self.fold)
        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(X, y)

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        y = xval_dataset.BaseDataset().load('valid', self.fold)
        y_pred = cls.predict_proba(validX)[:, 1]

        print(colors.green | colors.bold | str(core.score_data(y, y_pred)))

        np.save('cache/abhishek/logit/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict(trainX)
        np.save('cache/abhishek/logit/{:d}/test.npy'.format(self.fold), pred)

        with self.output().open('w'):
            pass