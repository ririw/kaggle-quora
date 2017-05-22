import os

import luigi
import pandas

from . import FoldIndependent, xval_dataset

__all__ = ['AbhishekFeatures']


class AbhishekFeatures(FoldIndependent, luigi.Task):
    def requires(self):
        yield xval_dataset.BaseDataset()

    def output(self):
        return luigi.LocalTarget('cache/abhishek/dataset/done')

    def run(self):
        self.output().makedirs()
        train_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/train_features.csv')
        train_data = pandas.read_csv(train_loc, encoding="ISO-8859-1").drop(['question1', 'question2'], 1)
        train_data = train_data.fillna(0).clip(-100000, 1000000)

        train_data.to_pickle('cache/abhishek/dataset/train.pkl')

        test_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/test_features.csv')
        test_data = pandas.read_csv(test_loc, encoding="ISO-8859-1").drop(['question1', 'question2'], 1)
        test_data = test_data.fillna(0).clip(-100000, 1000000)
        test_data.to_pickle('cache/abhishek/dataset/test.pkl')

        with self.output().open('w'):
            pass

    def _load_test(self, as_df):
        if as_df:
            return pandas.read_pickle('cache/abhishek/dataset/test.pkl')
        else:
            return pandas.read_pickle('cache/abhishek/dataset/test.pkl').values

    def _load(self, as_df):
        data = pandas.read_pickle('cache/abhishek/dataset/train.pkl')
        if not as_df:
            data = data.values
        folds = xval_dataset.BaseDataset().load_dataset_folds()
        return data, folds
