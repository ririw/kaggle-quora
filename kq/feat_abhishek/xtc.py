import luigi
import numpy as np
from plumbum import colors
from sklearn import ensemble

from kq import core
from kq.feat_abhishek.hyper_helper import TuneableIntHyperparam
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['XTCClassifier']


class XTCClassifier(FoldDependent):
    n_estimators = TuneableIntHyperparam(default=500)

    def load(self, name):
        assert name in {'test', 'valid'}
        return np.load('cache/abhishek/xtc/{:d}/{:s}.npy'.format(self.fold, name))

    def output(self):
        return luigi.LocalTarget('cache/abhishek/xtc/{:d}/done'.format(self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold).values.astype(float)
        y = xval_dataset.BaseDataset().load('train', self.fold)
        cls = ensemble.ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            class_weight=core.dictweights)
        cls.fit(X, y)

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold).values.astype(float)
        y = xval_dataset.BaseDataset().load('valid', self.fold)
        y_pred = cls.predict_proba(validX)[:, 1]
        print(colors.green | colors.bold | str(core.score_data(y, y_pred)))

        np.save('cache/abhishek/logit/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict(trainX)
        np.save('cache/abhishek/logit/{:d}/test.npy'.format(self.fold), pred)
