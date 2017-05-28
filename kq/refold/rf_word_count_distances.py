import luigi
import numpy as np
import os
import pandas
import scipy.spatial.distance as sp_dist
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
from kq.refold import BaseTargetBuilder, rf_dataset, rf_word_count_features

__all__ = ['WordCountDistances']

_FAR = np.ones(4)

_NEAR = np.zeros(4)


def distances(v1, v2):
    if v1.sum() == 0 or v2.sum() == 0:
        if v1.sum() == v2.sum():
            return _NEAR
        else:
            return _FAR
    v1 = v1.toarray()
    v2 = v2.toarray()

    b1 = v1 > 0
    b2 = v2 > 0
    return np.asarray([
        sp_dist.cosine(v1, v2),
        sp_dist.dice(b1, b2),
        sp_dist.hamming(b1, b2),
        sp_dist.kulsinski(b1, b2)
    ])


def distances_from_mats(m1, m2):
    res = []
    for row in tqdm(range(m1.shape[0]), desc='Computing word count based distances'):
        res.append(distances(m1[row], m2[row]))
    return np.asarray(res)


class WordCountDistances(FoldIndependent):
    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_word_count_distances')
        return (base_path + fname).get()

    def _load(self, as_df):
        res = np.load(self.output().path)['train_distances']
        if as_df:
            res = pandas.DataFrame(res, columns=['cosine', 'dice', 'hamming', 'kulsinski'])
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def _load_test(self, as_df):
        res = np.load(self.output().path)['test_distances']
        if as_df:
            res = pandas.DataFrame(res, columns=['cosine', 'dice', 'hamming', 'kulsinski'])
        return res

    def requires(self):
        yield rf_word_count_features.WordCountMatrix()
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def run(self):
        self.output().makedirs()
        train_q1, train_q2 = rf_word_count_features.WordCountMatrix().load_raw_vectors('train')
        train_distances = distances_from_mats(train_q1, train_q2)
        test_q1, test_q2 = rf_word_count_features.WordCountMatrix().load_raw_vectors('test')
        test_distances = distances_from_mats(test_q1, test_q2)

        np.savez_compressed(self.make_path('done_tmp.npz'), train_distances=train_distances,
                            test_distances=test_distances)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)
