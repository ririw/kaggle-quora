import luigi
import numpy as np
import tqdm
import os
from scipy import spatial

from kq import wordmat_distance


class QuestionVectorTask(luigi.Task):
    resources = {'cpu': 1}
    dataset = luigi.Parameter()

    def requires(self):
        #yield wordmat_distance.WeightedSentenceVecs()
        yield wordmat_distance.SentenceVecs()

    def output(self):
        return luigi.LocalTarget('./cache/question_distance/%s.npy' % self.dataset)

    def merge_vecs(self, v1, v2):
        distances = [
            spatial.distance.euclidean,
            spatial.distance.sqeuclidean,
            spatial.distance.cityblock,
            spatial.distance.cosine,
            spatial.distance.correlation,
            spatial.distance.chebyshev,
            spatial.distance.canberra,
            spatial.distance.braycurtis]
        total_work = v1.shape[0] * len(distances)
        bar = tqdm.tqdm(desc='Question vector: %s' % self.dataset, total=total_work)
        distance_vecs = []
        for d in distances:
            dists = []
            for a, b in zip(v1, v2):
                dists.append(d(a, b))
                bar.update()

        stds = np.std(v1 - v2, 1)
        distance_vecs.append(stds)
        distance_mat = np.asarray(distance_vecs).T

        return distance_mat
        #return np.concatenate([diffs, distance_mat], 1)

    def run(self):
        self.output().makedirs()
        tqdm.tqdm.pandas(tqdm.tqdm)

        #vecs1, vecs2 = wordmat_distance.WeightedSentenceVecs().load(dataset)
        #dists_a = self.merge_vecs(vecs1, vecs2)
        vecs1, vecs2 = wordmat_distance.SentenceVecs().load(self.dataset)
        dists = self.merge_vecs(vecs1, vecs2)

        #dists = np.concatenate([dists_a, dists_b], 0)

        np.save('cache/question_distance/%s_tmp.npy' % self.dataset, dists)
        os.rename('cache/question_distance/%s_tmp.npy' % self.dataset, self.output().path)

class QuestionVector(luigi.Task):
    def requires(self):
        yield QuestionVectorTask(dataset='train')
        yield QuestionVectorTask(dataset='test')
        yield QuestionVectorTask(dataset='merge')
        yield QuestionVectorTask(dataset='valid')

    def complete(self):
        return (QuestionVectorTask(dataset='train').complete() and
                QuestionVectorTask(dataset='test').complete() and
                QuestionVectorTask(dataset='merge').complete() and
                QuestionVectorTask(dataset='valid').complete())

    def run(self):
        pass

    def load_named(self, name):
        assert self.complete()
        return np.load('cache/question_distance/%s.npy' % name, mmap_mode='r')
