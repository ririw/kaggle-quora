import tempfile
import os

import distance
import gensim
import joblib
import luigi
import spacy
import numpy as np
from scipy import spatial
from tqdm import tqdm

from kq.dataset import Dataset
from kq.utils import w2v_file

class SharedEntity(luigi.Task):
    resources = {'cpu': 1}

    task_name = luigi.Parameter()
    def requires(self):
        return Dataset()

    def extract_ents(self, words):
        doc = self.nlp(words)
        entities = {ent.text: ent.vector for ent in doc.ents}
        flat = ' '.join([doc.text for doc in doc.ents])
        return entities, flat

    distances = [
        spatial.distance.euclidean,
        spatial.distance.sqeuclidean,
        spatial.distance.cityblock,
        spatial.distance.cosine,
        spatial.distance.correlation,
        spatial.distance.chebyshev,
        spatial.distance.canberra,
        spatial.distance.braycurtis]

    def entitiy_distances(self, q1, q2):
        e1, flat1 = self.extract_ents(q1)
        e2, flat2 = self.extract_ents(q2)

        if len(e1) == 0 or len(e2) == 0:
            return np.ones(len(self.distances) + 3)

        mean_vec1 = np.mean(list(e1.values()))
        mean_vec2 = np.mean(list(e2.values()))
        wmd = self.kvecs.wmdistance(flat1, flat2)

        distances = [d(mean_vec1, mean_vec2) for d in self.distances]
        jaccard_ents = distance.jaccard(e1.keys(), e2.keys())
        stddiff = np.std(mean_vec1 - mean_vec2)

        res = np.asarray(distances + [jaccard_ents, wmd, stddiff])
        return res

    def output(self):
        return luigi.LocalTarget('cache/distances/%s.npy' % self.task_name)

    def run(self):
        self.nlp = spacy.load('en')
        self.kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)

        self.output().makedirs()
        data = Dataset().load_named(self.task_name)
        dists = []
        for _, r in tqdm(data.iterrows(), total=data.shape[0],
                         desc='Processing shared entites: ' + self.task_name):
            dists.append(self.entitiy_distances(r.question1_raw, r.question2_raw))
        dists = np.clip(dists, -1000, 1000)
        np.save('cache/distances/%s_tmp.npy' % self.task_name, np.vstack(dists))
        os.rename('cache/distances/%s_tmp.npy' % self.task_name, 'cache/distances/%s.npy' % self.task_name)


class SharedEntities(luigi.Task):
    def requires(self):
        yield SharedEntity(task_name='test')
        yield SharedEntity(task_name='train')
        yield SharedEntity(task_name='merge')
        yield SharedEntity(task_name='valid')

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True

    def run(self):
        pass

    def load_named(self, name):
        assert self.complete()
        r = np.load('cache/distances/%s.npy' % name, mmap_mode='r')
        return np.clip(r, -1000, 1000)