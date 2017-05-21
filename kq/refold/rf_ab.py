import os

import luigi
import pandas

from kq.feat_abhishek import FoldIndependent
from kq.refold import rf_dataset, BaseTargetBuilder

__all__ = ['ABDataset']


class ABDataset(FoldIndependent):
    resources = {'cpu': 1}

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab',
        )
        return (base_path + fname).get()

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        train_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/train_features.csv')
        train_data = pandas.read_csv(train_loc, encoding="ISO-8859-1").drop(['question1', 'question2'], 1)
        train_data = train_data.fillna(0).clip(-100000, 1000000)

        train_data.to_pickle(self.make_path('train.pkl'))

        test_loc = os.path.expanduser('~/Datasets/Kaggle-Quora/test_features.csv')
        test_data = pandas.read_csv(test_loc, encoding="ISO-8859-1").drop(['question1', 'question2'], 1)
        test_data = test_data.fillna(0).clip(-100000, 1000000)
        test_data.to_pickle(self.make_path('test.pkl'))

        with self.output().open('w'):
            pass

    def _load_test(self):
        return pandas.read_pickle(self.make_path('test.pkl'))

    def _load(self):
        data = pandas.read_pickle(self.make_path('train.pkl'))
        folds = rf_dataset.Dataset().load_dataset_folds()
        return data, folds
