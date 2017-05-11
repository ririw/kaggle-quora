from collections import OrderedDict

import luigi
import numpy as np
import pandas
from plumbum import colors
from sklearn import metrics, linear_model, model_selection, preprocessing, pipeline, svm
from tqdm import tqdm

from kq import core, xtc, keras
from kq.dataset import Dataset
from kq.distances import AllDistances
from kq.keras import KaggleKeras
from kq.lightgbm import GBMClassifier
from kq.vw import VWClassifier
from kq.word_nb import NaiveBayesClassifier
from kq.xgb import XGBlassifier

import torch
import torch.nn.functional
import torchsample.modules
import torchsample.callbacks
import torchsample.metrics

class SimpleLogit(torch.nn.Module):
    def __init__(self, input_dimen):
        super().__init__()
        self.tf = torch.nn.Linear(input_dimen, 1)

    def forward(self, X):
        return torch.nn.Sigmoid()(self.tf(X))

class TorchLogit:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        if False:
            self.model = SimpleLogit(X.shape[1]).cuda()
            opt = torch.optim.Adam(self.model.parameters())
            bar = tqdm(range(10000))
            for _ in bar:
                opt.zero_grad()
                batch = np.random.choice(X.shape[0], size=128)
                Xs = torch.autograd.Variable(torch.from_numpy(X[batch].astype(np.float32)), requires_grad=True).cuda()
                ys = torch.autograd.Variable(torch.from_numpy(y[batch].astype(np.float32))).cuda()
                pred = self.model(Xs)
                loss = torch.nn.BCELoss()(pred, ys)
                loss.backward()
                bar.set_description(str(loss.data.numpy()[0]))
                opt.step()
        self.model = SimpleLogit(X.shape[1])
        trainer = torchsample.modules.ModuleTrainer(self.model)
        callbacks = [torchsample.callbacks.EarlyStopping(patience=10),
                     torchsample.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
        metrics = [torchsample.metrics.BinaryAccuracy()]
        trainer.compile(loss='binary_cross_entropy', optimizer='adam')
        trainer.fit(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)),
                    nb_epoch=20, batch_size=128, verbose=1)

    def predict_proba(self, X):
        pass


class Stacks(luigi.Task):
    def requires(self):
        yield XGBlassifier()
        yield GBMClassifier(dataset_kind='simple')
        yield GBMClassifier(dataset_kind='complex')
        #yield GBMClassifier(dataset_kind='words')
        yield VWClassifier()
        yield NaiveBayesClassifier()
        yield xtc.XTCComplexClassifier()
        yield xtc.XTCSimpleClassifier()
        yield keras.KerasLSTMModel()
        yield keras.ReadReadLSTM()
        #yield KaggleKeras()

    def output(self):
        return luigi.LocalTarget('cache/stacked_pred.csv.gz')

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return self.output().exists()

    def run(self):
        data = OrderedDict()
        shapes = {}
        for r in self.requires():
            x = r.load().squeeze()
            data[r.task_id] = x
            shapes[r.task_id] = x.shape[1] if len(x.shape) == 2 else 1

        data = pandas.DataFrame(data)[list(data.keys())]
        alldist = AllDistances().load()[1]
        dist_pd = pandas.DataFrame(alldist, columns=['alldist_%d' % i for i in range(alldist.shape[1])])

        data = pandas.concat([data, dist_pd], 1)
        data.to_csv('cache/R.csv')

        #data.drop(['alldist_1', 'alldist_4', 'alldist_5', 'alldist_6'], 1)
        data['is_duplicate'] = Dataset().load()[1].is_duplicate
        X = data.drop('is_duplicate', 1).values
        print(X.max(), X.min(), np.isnan(X).sum())
        y = data.is_duplicate.values
        np.savetxt('cache/Ry.csv', data.is_duplicate, header='is_duplicate', delimiter=',')

        weights = core.weights[y]
        scores = []
        cls = linear_model.LogisticRegression(C=10)
        cls.fit(X, y)
        print(pandas.Series(cls.coef_[0], data.drop('is_duplicate', 1).columns))

        polytransform = preprocessing.PolynomialFeatures(2)
        scaletransform = preprocessing.Normalizer()
        transform = pipeline.Pipeline([('scale', scaletransform), ('poly', polytransform)])

        for train_index, test_index in model_selection.KFold(n_splits=10).split(X, y):
            cls = linear_model.LogisticRegression(C=10)
            #cls = TorchLogit()
            X_train, X_test = X[train_index], X[test_index]
            X_train = transform.fit_transform(X_train)
            X_test = transform.transform(X_test)

            y_train, y_test = y[train_index], y[test_index]
            w_train, w_test = weights[train_index], weights[test_index]
            cls.fit(X_train.copy(), y_train.copy())#, sample_weight=w_train)
            pred = cls.predict_proba(X_test)
            score = metrics.log_loss(y_test, pred, sample_weight=w_test)
            print(score)
            scores.append(score)
        print(colors.yellow | '!----++++++----!')
        print(colors.yellow | colors.bold | '|' + str(np.mean(scores)) + '|')
        print(colors.yellow | '¡----++++++----¡')

        X = transform.transform(X)
        cls.fit(X, y, sample_weight=weights)

        data = OrderedDict()
        for r in self.requires():
            x = r.load_test().squeeze()
            data[r.task_id] = x
            assert shapes[r.task_id] == x.shape[1] if len(x.shape) == 2 else 1,\
                "Shape: {} did not match expected {}" % (x.shape, shapes[r.task_id])
            #print(r.__class__.__name__, '\t', x.shape, type(x))
        data = pandas.DataFrame.from_dict(data)
        alldist = AllDistances().load_named('test')
        dist_pd = pandas.DataFrame(alldist, columns=['alldist_%d' % i for i in range(alldist.shape[1])])
        data = pandas.concat([data, dist_pd], 1)#.drop(['alldist_1', 'alldist_4', 'alldist_5', 'alldist_6'], 1)

        X = data.values
        X = transform.transform(X)
        index = pandas.Index(np.arange(X.shape[0]), name='test_id')
        pred = pandas.Series(cls.predict_proba(X)[:, 1], index=index, name='is_duplicate').to_frame()
        print(colors.green | str(pred.head()))

        with self.output().open('w') as f:
            pred.to_csv(f, compression='gzip')
