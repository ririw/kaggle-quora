import luigi
import pandas
import numpy as np
import os
import mmh3
from . import FoldIndependent


__all__ = ['BaseDataset']


class BaseDataset(FoldIndependent):
    def _load_test(self):
        raise Exception('Cannot load testing target feature.')

    def _load(self):
        v = pandas.Series(np.load('cache/abhishek/dataset/train_data.npy'), name='is_duplicate')
        f = np.load('cache/abhishek/dataset/folds.npy')

        return v, f

    def load_dataset_folds(self):
        return np.load('cache/abhishek/dataset/folds.npy')

    def output(self):
        return luigi.LocalTarget('cache/abhishek/dataset/done')

    def run(self):
        self.output().makedirs()
        data_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/train.csv')
        test_data_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/test.csv')
        kaggle_train_data = pandas.read_csv(data_loc).drop('id', 1)
        kaggle_test_data = pandas.read_csv(test_data_loc)

        q1_fold = kaggle_train_data.qid1.apply(lambda qid: mmh3.hash(str(qid)) % 3)
        q2_fold = kaggle_train_data.qid2.apply(lambda qid: mmh3.hash(str(qid)) % 3 * 3)

        fold_n = (q1_fold + q2_fold).values
        v = kaggle_train_data.is_duplicate.values
        tv = np.ones(kaggle_test_data.shape[0]) * -1

        np.save('cache/abhishek/dataset/train_data.npy', v)
        np.save('cache/abhishek/dataset/test_data.npy', tv)
        np.save('cache/abhishek/dataset/folds.npy', fold_n)

        with self.output().open('w'):
            pass

