import luigi
import numpy as np
import pandas
from plumbum import colors
from lightgbm import sklearn as lgbsklearn
from sklearn import model_selection

from kq import core
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['LightGBMClassifier']


class LightGBMClassifier(FoldDependent):
    def _load(self, name):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/lgbm/{:d}/{:s}.npy'.format(self.fold, name)
        return pandas.Series(np.load(fn), name='LightGBM').to_frame()

    def output(self):
        return luigi.LocalTarget('cache/abhishek/lgbm/{:d}/done'.format(self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold)
        y = xval_dataset.BaseDataset().load('train', self.fold).squeeze()
        cls = lgbsklearn.LGBMClassifier(num_leaves=512, n_estimators=500, is_unbalance=True)
        X_tr, X_va, y_tr, y_va = model_selection.train_test_split(X, y, test_size=0.05)
        cls.fit(X_tr, y_tr, eval_set=(X_va, y_va))

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        y = xval_dataset.BaseDataset().load('valid', self.fold).squeeze()
        y_pred = cls.predict_proba(validX)[:, 1]
        print(colors.green | colors.bold | str(core.score_data(y, y_pred)))

        np.save('cache/abhishek/lgbm/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict(trainX)
        np.save('cache/abhishek/lgbm/{:d}/test.npy'.format(self.fold), pred)

        with self.output().open('w') as f:
            cols = abhishek_feats.AbhishekFeatures().load('valid', self.fold, as_np=False).columns
            v = pandas.Series(cls.feature_importances_, index=cols).sort_values()
            v.to_csv(f)