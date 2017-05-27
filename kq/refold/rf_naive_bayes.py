import os

import luigi
import pandas
from plumbum import colors
from sklearn import naive_bayes

from kq.core import score_data
from kq.feat_abhishek import FoldDependent, fold_max
from kq.refold import BaseTargetBuilder, rf_word_count_features, rf_dataset
import numpy as np

class RF_NaiveBayes(FoldDependent):
    resources = {'cpu': 1}
    def _load(self, name, as_df):
        res = np.load(self.make_path('done.npz'))[name]
        if as_df:
            res = pandas.Series(res, name=repr(self))
        return res

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_naive_bayes',
            str(self.fold)
        )
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def requires(self):
        yield rf_word_count_features.WordCountMatrix()
        yield rf_dataset.Dataset()

    def run(self):
        self.output().makedirs()
        m1, m2 = rf_word_count_features.WordCountMatrix().load_raw_vectors('train')
        folds = (rf_dataset.Dataset().load_dataset_folds() + self.fold) % fold_max
        train_X = np.abs(m1[folds != 0] - m2[folds != 0])
        train_y = rf_dataset.Dataset().load('train', fold=self.fold, as_df=True).is_duplicate.values
        cls = naive_bayes.MultinomialNB()
        cls.fit(train_X, train_y)

        valid_X = np.abs(m1[folds == 0] - m2[folds == 0])
        valid_y = rf_dataset.Dataset().load('valid', fold=self.fold, as_df=True).is_duplicate.values
        valid_pred = cls.predict_proba(valid_X)[:, 1]

        score = score_data(valid_y, valid_pred)

        print(colors.green | "Score for {:s}: {:f}".format(repr(self), score))

        t1, t2 = rf_word_count_features.WordCountMatrix().load_raw_vectors('test')
        test_X = np.abs(t1 - t2)
        test_pred = cls.predict_proba(test_X)[:, 1]
        np.savez_compressed(self.make_path('done_tmp.npz'), valid=valid_pred, test=test_pred)
        os.rename(self.make_path('done_tmp.npz'), self.make_path('done.npz'))
        return score