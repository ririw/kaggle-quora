import luigi
import numpy as np
import tqdm
from scipy import spatial

from kq import wordmat_distance


class QuestionVector(luigi.Task):
    def requires(self):
        yield wordmat_distance.WeightedSentenceVecs()
        yield wordmat_distance.SentenceVecs()

    def output(self):
        return luigi.LocalTarget('./cache/question_distance/done')

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
        distance_vecs = [[d(a, b) for a, b in zip(v1, v2)] for d in distances]
        distance_mat = np.asarray(distance_vecs).T

        return distance_mat
        #diffs = np.abs(v1 - v2)
        #return np.concatenate([diffs, distance_mat], 1)

    def run(self):
        self.output().makedirs()
        tqdm.tqdm.pandas(tqdm.tqdm)

        for dataset in {'train', 'merge', 'test', 'valid'}:
            vecs1, vecs2 = wordmat_distance.WeightedSentenceVecs().load(dataset)
            dists_a = self.merge_vecs(vecs1, vecs2)
            vecs1, vecs2 = wordmat_distance.SentenceVecs().load(dataset)
            dists_b = self.merge_vecs(vecs1, vecs2)

            dists = np.concatenate([dists_a, dists_b], 0)

            np.save('cache/question_distance/%s.npy' % dataset, dists)
        with self.output().open('w'):
            pass

    def load_named(self, name):
        assert self.complete()
        return np.load('cache/question_distance/%s.npy' % name, mmap_mode='r')
