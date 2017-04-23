import tempfile
import os

import distance
import luigi
import numpy as np
from tqdm import tqdm

from kq import dataset


class DistanceBase(luigi.Task):
    def dist_fn(self, xs, ys):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/distance-%s.npz' % self.name)

    def run(self):
        train, merge, valid = dataset.Dataset().load()
        train_dists = []
        for _, row in tqdm(train.iterrows(), desc='Train distance %s' % self.name, total=train.shape[0]):
            train_dists.append(self.dist_fn(row.question1_tokens, row.question2_tokens))
        train_dists = np.asarray(train_dists)

        merge_dists = []
        for _, row in tqdm(merge.iterrows(), desc='Merge distance %s' % self.name, total=merge.shape[0]):
            merge_dists.append(self.dist_fn(row.question1_tokens, row.question2_tokens))
        merge_dists = np.asarray(merge_dists)

        valid_dists = []
        for _, row in tqdm(valid.iterrows(), desc='Valid distance %s' % self.name, total=valid.shape[0]):
            valid_dists.append(self.dist_fn(row.question1_tokens, row.question2_tokens))
        valid_dists = np.asarray(valid_dists)

        del train, valid, merge

        test = dataset.Dataset().load_test()
        test_dists = []
        for _, row in tqdm(test.iterrows(), desc='Test distance %s' % self.name, total=test.shape[0]):
            test_dists.append(self.dist_fn(row.question1_tokens, row.question2_tokens))
        test_dists = np.asarray(test_dists)

        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            np.savez(tf, train_dists=train_dists, valid_dists=valid_dists,
                     test_dists=test_dists, merge_dists=merge_dists)
            os.rename(tf.name, self.output().path)
        except Exception as e:
            os.remove(tf)
            raise

    def load(self):
        assert self.complete()
        f = np.load(self.output().path)
        return f['train_dists'], f['merge_dists'], f['valid_dists']

    def load_test(self):
        assert self.complete()
        f = np.load(self.output().path)
        return f['test_dists']


class JaccardDistance(DistanceBase):
    def dist_fn(self, xs, ys):
        try:
            return distance.jaccard(xs, ys)
        except ZeroDivisionError:
            return 1

    @property
    def name(self):
        return 'jaccard_distance'


class LevenshteinDistance1(DistanceBase):
    def dist_fn(self, xs, ys):
        return distance.nlevenshtein(xs, ys, method=1)

    @property
    def name(self):
        return 'levenshtein1'


class LevenshteinDistance2(DistanceBase):
    def dist_fn(self, xs, ys):
        return distance.nlevenshtein(xs, ys, method=2)

    @property
    def name(self):
        return 'levenshtein2'


class SorensenDistance(DistanceBase):
    def dist_fn(self, xs, ys):
        try:
            return distance.sorensen(xs, ys)
        except ZeroDivisionError:
            return 1

    @property
    def name(self):
        return 'sorensen'

class WordsDistance(DistanceBase):
    def dist_fn(self, xs, ys):
        return abs(len(xs) - len(ys))

    @property
    def name(self):
        return 'word_dist'

class CharsDistance(DistanceBase):
    def dist_fn(self, xs, ys):
        return abs(sum((len(x) for x in xs)) - sum((len(y) for y in ys)))

    @property
    def name(self):
        return 'char_dist'

class AllDistances:
    @staticmethod
    def requires():
        yield JaccardDistance()
        yield LevenshteinDistance1()
        yield LevenshteinDistance2()
        yield SorensenDistance()
        yield WordsDistance()
        yield CharsDistance()

    def load(self):
        all_train, all_merge, all_valid = [], [], []
        for req in self.requires():
            assert req.complete()
            train, merge, valid = req.load()
            all_train.append(train)
            all_merge.append(merge)
            all_valid.append(valid)

        all_train = np.vstack(all_train).T
        all_merge = np.vstack(all_merge).T
        all_valid = np.vstack(all_valid).T

        return all_train, all_merge, all_valid

    def load_test(self):
        all_test = []
        for req in self.requires():
            assert req.complete()
            test = req.load_test()
            all_test.append(test)

        return np.vstack(all_test).T