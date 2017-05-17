import luigi
import pandas

from kq import distances, lda_feature
from kq import question_freq
from kq import question_vectors
from kq import sentiments
from kq import shared_entites
from kq import wordmat_distance


class FeatureCollection(luigi.Task):
    def requires(self):
        yield question_vectors.QuestionVector()
        yield distances.AllDistances()
        yield shared_entites.SharedEntities()
        yield wordmat_distance.WordMatDistance()
        yield sentiments.SentimentTasks()
        yield question_freq.QuestionFrequencyFeature()
        yield lda_feature.LDADecompositionFeatureVectors()
        yield lda_feature.NMFDecompositionFeatureVectors()

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True

    def load_named(self, name):
        cols = {}
        for r in self.requires():
            v = r.load_named(name)
            if isinstance(v, pandas.DataFrame):
                v = v.values
            if len(v.shape) == 1:
                cols[r.__class__.__name__] = v
            else:
                for col in range(v.shape[1]):
                    cols['{}_{}'.format(r.__class__.__name__, col)] = v[:, col]
        return pandas.DataFrame(cols).fillna(-1000).clip(-999, 999)
