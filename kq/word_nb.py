import os

import luigi
import pandas
from plumbum import colors

from kq import core
from kq.dataset import Dataset
from kq.shared_words import WordVectors
from sklearn import naive_bayes, metrics


class NaiveBayesClassifier(luigi.Task):
    def requires(self):
        yield WordVectors()
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('cache/nb/predictions.csv.gz')

    def run(self):
        self.output().makedirs()
        train_X, merge_X, valid_X = WordVectors().load()
        train, merge, valid = Dataset().load()
        cls = naive_bayes.BernoulliNB()
        cls.fit(train_X, train.is_duplicate)

        preds = cls.predict_proba(valid_X)[:, 1]
        weights = core.weights[valid.is_duplicate.values]
        loss = metrics.log_loss(valid.is_duplicate, preds, sample_weight=weights)
        print(colors.green | str(loss))

        merge_pred = cls.predict_proba(merge_X)[:, 1]
        pandas.Series(merge_pred).to_csv('cache/nb/merge_predictions.csv')

        del train, merge, valid
        del train_X, merge_X, valid_X

        test = WordVectors().load_test()
        pred = cls.predict_proba(test)[:, 1]
        pandas.Series(pred).to_csv('cache/nb/predictions_tmp.csv')
        os.rename('cache/nb/predictions_tmp.csv', 'cache/nb/predictions.csv')