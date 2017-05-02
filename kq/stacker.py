from collections import OrderedDict

import luigi
import numpy as np
import pandas
from plumbum import colors
from sklearn import metrics, linear_model, model_selection, preprocessing

from kq import core, xtc
from kq.dataset import Dataset
from kq.distances import AllDistances
from kq.keras import KaggleKeras
from kq.lightgbm import GBMClassifier
from kq.vw import VWClassifier
from kq.word_nb import NaiveBayesClassifier
from kq.xgb import XGBlassifier


class Stacks(luigi.Task):
    def requires(self):
        yield XGBlassifier()
        yield GBMClassifier(dataset_kind='simple')
        yield GBMClassifier(dataset_kind='complex')
        yield GBMClassifier(dataset_kind='words')
        yield VWClassifier()
        yield NaiveBayesClassifier()
        yield xtc.XTCComplexClassifier()
        yield xtc.XTCSimpleClassifier()
        #yield KerasLSTMModel()
        yield KaggleKeras()

    def output(self):
        return luigi.LocalTarget('cache/stacked_pred.csv')

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
        data['is_duplicate'] = Dataset().load()[1].is_duplicate
        X = data.drop('is_duplicate', 1).values
        y = data.is_duplicate.values

        weights = core.weights[y]
        scores = []
        cls = linear_model.LogisticRegression(C=10)

        cls.fit(X, y)
        print(pandas.Series(cls.coef_[0], data.drop('is_duplicate', 1).columns))

        polytransform = preprocessing.PolynomialFeatures(2)
        for train_index, test_index in model_selection.KFold(n_splits=10).split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            X_train = polytransform.fit_transform(X_train)
            X_test = polytransform.transform(X_test)

            y_train, y_test = y[train_index], y[test_index]
            w_train, w_test = weights[train_index], weights[test_index]
            cls.fit(X_train, y_train, sample_weight=w_train)
            score = metrics.log_loss(y_test, cls.predict_proba(X_test), sample_weight=w_test)
            print(score)
            scores.append(score)
        print(colors.yellow | '!----++++++----!')
        print(colors.yellow | colors.bold | '|' + str(np.mean(scores)) + '|')
        print(colors.yellow | '¡----++++++----¡')

        X = polytransform.transform(X)
        cls.fit(X, y, sample_weight=weights)

        data = OrderedDict()
        for r in self.requires():
            x = r.load_test().squeeze()
            data[r.task_id] = x
            assert shapes[r.task_id] == x.shape[1] if len(x.shape) == 2 else 1,\
                "Shape: {} did not match expected {}" % (x.shape, shapes[r.task_id])
            #print(r.__class__.__name__, '\t', x.shape, type(x))
        data = pandas.DataFrame.from_dict(data)[list(data.keys())]
        alldist = AllDistances().load_named('test')
        dist_pd = pandas.DataFrame(alldist, columns=['alldist_%d' % i for i in range(alldist.shape[1])])
        data = pandas.concat([data, dist_pd], 1)

        X = data.values
        X = polytransform.transform(X)
        index = pandas.Index(np.arange(X.shape[0]), name='test_id')
        pred = pandas.Series(cls.predict_proba(X)[:, 1], index=index, name='is_duplicate').to_frame()
        print(colors.green | str(pred.head()))

        with self.output().open('w') as f:
            pred.to_csv(f)
