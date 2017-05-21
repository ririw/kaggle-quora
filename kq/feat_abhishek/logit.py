import hyperopt
import luigi
import pandas
from plumbum import colors
from sklearn import linear_model, pipeline, preprocessing
import numpy as np

from kq import core
from kq.feat_abhishek import hyper_helper, HyperTuneable
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['LogitClassifier']


class LogitClassifier(FoldDependent, HyperTuneable):
    def score(self):
        assert self.complete()
        return self.train()[0]

    resources = {'cpu': 1}
    C = hyper_helper.TuneableHyperparam(
        "LogitClassifier_C",
        prior=hyperopt.hp.normal('LogitClassifier_C', 0, 10),
        default=56.9600392248474,
        transform=np.abs)
    npoly = hyper_helper.TuneableHyperparam(
        "LogitClassifier_npoly",
        prior=hyperopt.hp.randint('LogitClassifier_npoly', 3),
        default=1,
        transform=lambda v: v+1)

    def _load(self, name):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/logit/{:f}/{:d}/{:s}.npy'.format(self.C.get(), self.fold, name)
        return pandas.DataFrame({'LogitClassifier': np.load(fn)})

    def output(self):
        return luigi.LocalTarget('cache/abhishek/logit/{:f}/{:d}/done'.format(self.C.get(), self.fold))

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def train(self):
        self.output().makedirs()
        preproc = pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(self.npoly.get()))
        ])

        X = abhishek_feats.AbhishekFeatures().load('train', self.fold, as_np=False)
        X = preproc.fit_transform(X)
        y = xval_dataset.BaseDataset().load('train', self.fold).squeeze()
        cls = linear_model.LogisticRegression(
            C=self.C.get(),
            solver='sag',
            class_weight=core.dictweights)
        cls.fit(X, y)

        print('Validating')
        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        validX = preproc.transform(validX)
        y = xval_dataset.BaseDataset().load('valid', self.fold).squeeze()
        y_pred = cls.predict_proba(validX)[:, 1]

        score = core.score_data(y, y_pred)
        np.save('cache/abhishek/logit/{:f}/{:d}/valid.npy'.format(self.C.get(), self.fold), y_pred)

        return score, cls, preproc


    def run(self):
        self.output().makedirs()

        score, cls, preproc = self.train()
        scorestr = "{:s} = {:f}".format(repr(self), score)
        print(colors.green | colors.bold | scorestr)


        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        trainX = preproc.transform(trainX)
        pred = cls.predict_proba(trainX)[:, 1]
        np.save('cache/abhishek/logit/{:f}/{:d}/test.npy'.format(self.C.get(), self.fold), pred)

        with self.output().open('w') as f:
            f.write(scorestr)
            f.write("\n")
#
        return score