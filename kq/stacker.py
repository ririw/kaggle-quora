import luigi
import pandas
import numpy as np
from plumbum import colors

from kq import core
from kq.xtc import XTCClassifier
from kq.dataset import Dataset
from kq.lightgbm import XGBlassifier, GBMClassifier
from kq.vw import VWClassifier
from kq.word_nb import NaiveBayesClassifier

from sklearn import metrics, linear_model, model_selection, svm, preprocessing


class Stacks(luigi.Task):
    def requires(self):
        yield XGBlassifier()
        yield GBMClassifier()
        yield VWClassifier()
        yield NaiveBayesClassifier()
        yield XTCClassifier()

    def output(self):
        return luigi.LocalTarget('cache/stacked_pred.csv')

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return self.output().exists()

    def run(self):
        data = {}
        for r in self.requires():
            x = r.load().squeeze()
            data[r.__class__.__name__] = x
        data = pandas.DataFrame(data)
        data['is_duplicate'] = Dataset().load()[1].is_duplicate
        X = data.drop('is_duplicate', 1).values
        y = data.is_duplicate.values

        weights = core.weights[y]
        scores = []
        cls = linear_model.LogisticRegression()

        polytransform = preprocessing.PolynomialFeatures(3)
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

        data = {}
        for r in self.requires():
            x = r.load_test().squeeze()
            data[r.__class__.__name__] = x
            #print(r.__class__.__name__, x.shape)
        data = pandas.DataFrame(data)
        X = data.values
        X = polytransform.transform(X)
        index = pandas.Index(np.arange(X.shape[0]), name='test_id')
        pred = pandas.Series(cls.predict_proba(X)[:, 1], index=index, name='is_duplicate').to_frame()
        print(colors.green | str(pred.head()))

        with self.output().open('w') as f:
            pred.to_csv(f)
