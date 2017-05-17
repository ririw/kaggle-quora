import luigi
import numpy as np
import pandas
from plumbum import colors
import xgboost.sklearn

from kq import core
from kq.feat_abhishek.hyper_helper import TuneableIntHyperparam
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['XGBClassifier']


class XGBClassifier(FoldDependent):
    def _load(self, name):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/xgb/{:d}/{:s}.npy'.format(self.fold, name)
        return pandas.DataFrame({'XGBClassifier': np.load(fn)})

    def output(self):
        return luigi.LocalTarget('cache/abhishek/xgb/{:d}/done'.format(self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold)
        y = xval_dataset.BaseDataset().load('train', self.fold)
        cls = xgboost.sklearn.XGBClassifier(max_depth=7, nthread=4)
        cls.fit(X, y)

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        y = xval_dataset.BaseDataset().load('valid', self.fold)
        y_pred = cls.predict_proba(validX)[:, 1]
        print(colors.green | colors.bold | str(core.score_data(y, y_pred)))

        np.save('cache/abhishek/xgb/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict(trainX)
        np.save('cache/abhishek/xgb/{:d}/test.npy'.format(self.fold), pred)

        with self.output().open('w'):
            pass