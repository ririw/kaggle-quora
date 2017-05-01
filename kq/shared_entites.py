import tempfile
import os

import distance
import gensim
import luigi
import spacy
import numpy as np
from scipy import spatial
from tqdm import tqdm

from kq.dataset import Dataset
from kq.utils import w2v_file


class SharedEntities(luigi.Task):
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
            return np.ones(len(self.distances) + 1)

        mean_vec1 = np.mean(list(e1.values()))
        mean_vec2 = np.mean(list(e2.values()))
        wmd = self.kvecs.wmdistance(flat1, flat2)

        distances = [d(mean_vec1, mean_vec2) for d in self.distances]
        jaccard_ents = distance.jaccard(e1.keys(), e2.keys())


        return np.asarray(distances + [jaccard_ents, wmd])

    def output(self):
        return luigi.LocalTarget('cache/entities.npz')

    def run(self):
        self.nlp = spacy.load('en')
        self.kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
        train, merge, valid = Dataset().load()
        test = Dataset().load_test()

        train_distances = []
        merge_distances = []
        valid_distances = []
        test_distances = []

        for _, r in tqdm(train.iterrows(), total=train.shape[0], desc='Processing: train'):
            train_distances.append(self.entitiy_distances(r.question1_raw, r.question2_raw))
        for _, r in tqdm(merge.iterrows(), total=merge.shape[0], desc='Processing: merge'):
            merge_distances.append(self.entitiy_distances(r.question1_raw, r.question2_raw))
        for _, r in tqdm(valid.iterrows(), total=valid.shape[0], desc='Processing: valid'):
            valid_distances.append(self.entitiy_distances(r.question1_raw, r.question2_raw))
        for _, r in tqdm(test.iterrows(), total=test.shape[0], desc='Processing: test'):
            test_distances.append(self.entitiy_distances(r.question1_raw, r.question2_raw))

        train_distances = np.vstack(train_distances)
        merge_distances = np.vstack(merge_distances)
        valid_distances = np.vstack(valid_distances)
        test_distances = np.vstack(test_distances)

        self.output().makedirs()
        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            np.savez(
                tf,
                train_distances=train_distances,
                merge_distances=merge_distances,
                valid_distances=valid_distances,
                test_distances=test_distances,
            )
            os.rename(tf.name, self.output().path)
        except:
            os.remove(tf)

    def load_named(self, name):
        assert self.complete()
        r = np.load(self.output().path, mmap_mode='r')
        return r['%s_distances' % name]