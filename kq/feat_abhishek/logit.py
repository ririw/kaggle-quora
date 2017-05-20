import luigi
import pandas
from plumbum import colors
from sklearn import linear_model, pipeline, preprocessing
import numpy as np

from kq import core
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['LogitClassifier']


class LogitClassifier(FoldDependent):
    resources = {'cpu': 1}

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

        preproc = pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2))
        ])

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold, as_np=False)
        X = preproc.fit_transform(X)
        y = xval_dataset.BaseDataset().load('train', self.fold).squeeze()
        cls = linear_model.LogisticRegression(class_weight=core.dictweights)
        cls.fit(X, y)

        print('Validating')
        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        validX = preproc.transform(validX)
        y = xval_dataset.BaseDataset().load('valid', self.fold).squeeze()
        y_pred = cls.predict_proba(validX)[:, 1]

        scorestr = "{:s} = {:f}".format(repr(self), core.score_data(y, y_pred))
        print(colors.green | colors.bold | scorestr)

        np.save('cache/abhishek/logit/{:d}/valid.npy'.format(self.fold), y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        trainX = preproc.transform(trainX)
        pred = cls.predict_proba(trainX)[:, 1]
        np.save('cache/abhishek/logit/{:d}/test.npy'.format(self.fold), pred)

        with self.output().open('w') as f:
            f.write(scorestr)
            f.write("\n")
