import os

import dask.dataframe
import distance
import gensim
import luigi
import numpy as np
from tqdm import tqdm

from kq import dataset
from kq.utils import w2v_file


class DistanceSubTask(luigi.Task):
    resources = {'cpu': 1}
    dataset = luigi.Parameter()

    def dist_fn(self, xs, ys):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/functional-distance/%s-%s.npy' % (self.name, self.dataset))

    def run(self):
        self.output().makedirs()
        data = dataset.Dataset().load_named(self.dataset)
        dists = np.zeros(data.shape[0])
        i = 0
        for _, row in tqdm(data.iterrows(),
                           desc='%s distance %s' % (self.dataset, self.name),
                           total=data.shape[0]):
            dists[i] = self.dist_fn(row.question1_tokens, row.question2_tokens)
            i += 1
        np.save('cache/functional-distance/%s-%s.tmp.npy' % (self.name, self.dataset), dists)
        os.rename('cache/functional-distance/%s-%s.tmp.npy' % (self.name, self.dataset),
                  'cache/functional-distance/%s-%s.npy' % (self.name, self.dataset))


class DistancesGroup(luigi.Task):
    SubTask = None

    def requires(self):
        yield self.SubTask(dataset='train')
        yield self.SubTask(dataset='valid')
        yield self.SubTask(dataset='merge')
        yield self.SubTask(dataset='test')

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True

    def run(self):
        pass

    def load_named(self, name):
        ds = self.SubTask(dataset='train').name
        res = np.clip(
            np.nan_to_num(np.load('cache/functional-distance/%s-%s.npy' % (ds, name))),
            -1000, 1000)
        return res


class JaccardDistance(DistancesGroup):
    class JaccardDistanceSubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            try:
                return distance.jaccard(xs, ys)
            except ZeroDivisionError:
                return 1

        @property
        def name(self):
            return 'jaccard'

    SubTask = JaccardDistanceSubTask


class LevenshteinDistance1(DistancesGroup):
    class LevenshteinDistance1SubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            return distance.nlevenshtein(xs, ys, method=1)

        @property
        def name(self):
            return 'levenshtein1'

    SubTask = LevenshteinDistance1SubTask


class LevenshteinDistance2(DistancesGroup):
    class LevenshteinDistance2SubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            return distance.nlevenshtein(xs, ys, method=2)

        @property
        def name(self):
            return 'levenshtein2'

    SubTask = LevenshteinDistance2SubTask


class SorensenDistance(DistancesGroup):
    class SorensenDistanceSubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            try:
                return distance.sorensen(xs, ys)
            except ZeroDivisionError:
                return 1

        @property
        def name(self):
            return 'sorensen'

    SubTask = SorensenDistanceSubTask


class WordsDistance(DistancesGroup):
    class WordsDistanceSubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            return abs(len(xs) - len(ys))

        @property
        def name(self):
            return 'word_dist'

    SubTask = WordsDistanceSubTask


class CharsDistance(DistancesGroup):
    class CharsDistanceSubTask(DistanceSubTask):
        def dist_fn(self, xs, ys):
            return abs(sum((len(x) for x in xs)) - sum((len(y) for y in ys)))

        @property
        def name(self):
            return 'char_dist'

    SubTask = CharsDistanceSubTask



class WordMoverDistance(DistancesGroup):
    class WordMoverDistanceSubTask(DistanceSubTask):
        resources = {'cpu': 2, 'mem': 2}

        def dist_fn(self, xs, ys):
            assert False, "Should never be called."

        def run(self):
            self.output().makedirs()
            kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
            data = dataset.Dataset().load_named(self.dataset)
            dists = np.zeros(data.shape[0])
            i = 0

            #def wmd(row):
            #    return kvecs.wmdistance(row.question1_raw, row.question2_raw)
            #dists = dask.dataframe.from_pandas(data, npartitions=16, sort=False).apply(wmd).compute().values

            for q1, q2 in tqdm(zip(data.question1_raw, data.question2_raw),
                               total=data.question1_raw.shape[0],
                               desc='Computing %s WMD' % self.dataset):
                dists[i] = kvecs.wmdistance(q1, q2)
                i += 1

            np.save('cache/functional-distance/%s-%s.tmp.npy' % (self.name, self.dataset), dists)
            os.rename('cache/functional-distance/%s-%s.tmp.npy' % (self.name, self.dataset),
                      'cache/functional-distance/%s-%s.npy' % (self.name, self.dataset))

        @property
        def name(self):
            return 'wmd_dist'

    SubTask = WordMoverDistanceSubTask


class AllDistances(luigi.Task):
    def requires(self):
        yield JaccardDistance()
        yield LevenshteinDistance1()
        yield LevenshteinDistance2()
        yield SorensenDistance()
        yield WordsDistance()
        yield CharsDistance()
        yield WordMoverDistance()

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True

    def run(self):
        pass

    def load(self):
        all_train, all_merge, all_valid = [], [], []
        for req in self.requires():
            assert req.complete()
            train = req.load_named('train')
            merge = req.load_named('merge')
            valid = req.load_named('valid')
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
            test = req.load_named('test')
            all_test.append(test)

        return np.vstack(all_test).T

    def load_named(self, name):
        assert name in {'train', 'valid', 'merge', 'test'}
        assert self.complete()
        res = []
        for r in self.requires():
            assert r.complete()
            res.append(r.load_named(name))
        return np.vstack(res).T
