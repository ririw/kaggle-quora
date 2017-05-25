import multiprocessing

import distance
import gensim
import luigi
import pandas
from nltk.stem import snowball
from nltk.tokenize import treebank
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
from kq.refold import BaseTargetBuilder, rf_dataset
from kq.utils import w2v_file

__all__ = ['RFDistanceCalculator']


_independent_transformers = {}
def register_indep_transform(name):
    def inner(fn):
        global _independent_transformers
        _independent_transformers[name] = fn
        return fn
    return inner


@register_indep_transform('jaccard_letter')
def jaccard_letter_distance(q1, q2, t1, t2):
    return distance.jaccard(q1, q2)


@register_indep_transform('jaccard_tok')
def jaccard_tok_distance(q1, q2, t1, t2):
    return distance.jaccard(t1, t2)


@register_indep_transform('lev1_letter')
def lev1_letter_distance(q1, q2, t1, t2):
    return distance.nlevenshtein(q1, q2, method=1)


@register_indep_transform('lev1_tok')
def lev1_tok_distance(q1, q2, t1, t2):
    return distance.nlevenshtein(t1, t2, method=1)


@register_indep_transform('lev2_letter')
def lev2_letter_distance(q1, q2, t1, t2):
    return distance.nlevenshtein(q1, q2, method=2)


@register_indep_transform('lev2_tok')
def lev2_tok_distance(q1, q2, t1, t2):
    return distance.nlevenshtein(t1, t2, method=2)


@register_indep_transform('sor_letter')
def sor_letter_distance(q1, q2, t1, t2):
    return distance.sorensen(q1, q2)


@register_indep_transform('sor_tok')
def sor_tok_distance(q1, q2, t1, t2):
    return distance.sorensen(t1, t2)


@register_indep_transform('len_letter')
def len_letter_distance(q1, q2, t1, t2):
    return abs(len(q1) - len(q2))


@register_indep_transform('len_tok')
def len_tok_distance(q1, q2, t1, t2):
    return abs(len(t1) - len(t2))


class WordMoverDistance:
    def __init__(self):
        self.kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)

    def __call__(self, q1, q2, t1, t2):
        self.kvecs.wmdistance(q1, q2)


_train_loc = BaseTargetBuilder('rf_distance', 'train.msg').get()
_test_loc = BaseTargetBuilder('rf_distance', 'test.msg').get()


def transform(item):
    q1, q2, t1, t2 = item
    res = {}
    for name, transform in _independent_transformers.items():
        res[name] = transform(q1, q2, t1, t2)
    return res


class RFDistanceCalculator(FoldIndependent):
    def requires(self):
        yield rf_dataset.Dataset()

    def _load_test(self, as_df):
        features = pandas.read_msgpack(_test_loc).fillna(9999).clip(-10000, 10000)
        if as_df:
            return features
        else:
            return features.values

    def _load(self, as_df):
        folds = rf_dataset.Dataset().load_dataset_folds()
        features = pandas.read_msgpack(_train_loc).fillna(9999).clip(-10000, 10000)
        if as_df:
            features = features.values
        return features, folds

    def tokenize(self, q):
        tokens = self.tokenzier.tokenize(q)
        stems = [self.stemmer.stem(w) for w in tokens]
        return stems

    def output(self):
        return luigi.LocalTarget(BaseTargetBuilder('rf_distance', 'done').get())

    def run(self):
        global _independent_transformers

        self.tokenzier = treebank.TreebankWordTokenizer()
        self.stemmer = snowball.SnowballStemmer('english')

        train_data = rf_dataset.Dataset().load_all('train', as_df=True)[['question1_clean', 'question2_clean']]
        test_data = rf_dataset.Dataset().load_all('test', as_df=True)[['question1_clean', 'question2_clean']]

        all_data = pandas.concat([train_data, test_data], 0)
        all_q1 = list(all_data['question1_clean'])
        all_t1 = list(tqdm(multiprocessing.Pool().imap(self.tokenize, all_q1, chunksize=10000),
                           total=len(all_q1), desc='Tokenizing: 1'))

        all_q2 = list(all_data['question2_clean'])
        all_t2 = list(tqdm(multiprocessing.Pool().imap(self.tokenize, all_q2, chunksize=10000),
                           total=len(all_q2), desc='Tokenizing: 2'))


        all_indep_dists = list(tqdm(
            multiprocessing.Pool().imap(transform, zip(all_q1, all_q2, all_t1, all_t2), chunksize=10000),
            total=len(all_q1),
            desc='Computing distances'
        ))
        all_df = pandas.DataFrame(all_indep_dists)

        print('Loading...')
        dependent_transformers = {'word_mover': WordMoverDistance()}
        print('Finished!')

        for name, fn in dependent_transformers.items():
            dist = [fn(q1, q2, t1, t2) for q1, q2, t1, t2 in
                    tqdm(zip(all_q1, all_q2, all_t1, all_t2), total=len(all_q1), desc=name)]
            all_df[name] = dist

        self.output().makedirs()
        train_dists = all_df.iloc[:train_data.shape[0]]
        test_dists = all_df.iloc[train_data.shape[0]:]
        train_dists.to_msgpack(_train_loc)
        test_dists.to_msgpack(_test_loc)

        with self.output().open('w'):
            pass
