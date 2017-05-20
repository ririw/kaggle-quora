import luigi
import numpy as np
import pandas
from plumbum import colors
from sklearn import ensemble

from kq import core
from kq.feat_abhishek.hyper_helper import TuneableIntHyperparam
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['XTCClassifier']


class XTCClassifier(FoldDependent):
    resources = {'cpu': 7}

    def _load(self, name):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/xtc/{:d}/{:s}.npy'.format(self.fold, name)
        return pandas.Series(np.load(fn), name='XTC').to_frame()

    def output(self):
        return luigi.LocalTarget('cache/abhishek/xtc/{:d}/done'.format(self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold)
        y = xval_dataset.BaseDataset().load('train', self.fold).squeeze()
        cls = ensemble.ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=10,
            n_jobs=-1,
            class_weight=core.dictweights)
        cls.fit(X, y)

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        y = xval_dataset.BaseDataset().load('valid', self.fold).squeeze()
        y_pred = cls.predict_proba(validX)[:, 1]
        scorestr = "{:s} = {:f}".format(repr(self), core.score_data(y, y_pred))
        print(colors.green | colors.bold | scorestr)

        np.save('cache/abhishek/xtc/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict_proba(trainX)[:, 1]
        np.save('cache/abhishek/xtc/{:d}/test.npy'.format(self.fold), pred)

        with self.output().open('w') as f:
            cols = abhishek_feats.AbhishekFeatures().load('valid', self.fold, as_np=False).columns
            v = pandas.Series(cls.feature_importances_, index=cols).sort_values()
            v.to_csv(f)
            f.write("\n")
            f.write("\n")
            f.write(scorestr)
            f.write("\n")