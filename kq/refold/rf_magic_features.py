import multiprocessing

import luigi
import mmh3

import networkx
import pandas
import numpy as np
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
from kq.refold import BaseTargetBuilder, rf_dataset

__all__ = ['QuestionFrequency', 'NeighbourhoodFeature', 'QuestionOrderFeature']


class QuestionFrequency(FoldIndependent):
    def _load_test(self, as_df):
        res = pandas.read_msgpack(self.make_path('test.msg'))
        if not as_df:
            res = res.values
        return res

    def _load(self, as_df):
        res = pandas.read_msgpack(self.make_path('train.msg'))
        if not as_df:
            res = res.values
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        train_data = rf_dataset.Dataset().load_all('train', as_df=True)
        test_data = rf_dataset.Dataset().load_all('test', as_df=True)

        all_questions = pandas.concat([
            train_data.question1_clean,
            train_data.question2_clean,
            test_data.question1_clean,
            test_data.question2_clean,
        ])
        question_freq = all_questions.value_counts().to_dict()

        train_feat = pandas.DataFrame({
            'freq1': train_data.question1_clean.map(question_freq),
            'freq2': train_data.question2_clean.map(question_freq)
        })
        train_feat['freq_diff'] = np.abs(train_feat.freq1 - train_feat.freq2)
        test_feat = pandas.DataFrame({
            'freq1': test_data.question1_clean.map(question_freq),
            'freq2': test_data.question2_clean.map(question_freq)
        })
        test_feat['freq_diff'] = np.abs(test_feat.freq1 - test_feat.freq2)

        train_feat.to_msgpack(self.make_path('train.msg'))
        test_feat.to_msgpack(self.make_path('test.msg'))

        with self.output().open('w'):
            pass

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_magic',
            'question_freq'
        )
        return (base_path + fname).get()


class NeighbourhoodFeature(FoldIndependent):
    def _load_test(self, as_df):
        res = np.load(self.make_path('test.npz'))['data']
        if as_df:
            res = pandas.DataFrame(res, columns=['l1_neighbours', 'graph_neighbours', 'l2_neighbours'])
        return res

    def _load(self, as_df):
        res = np.load(self.make_path('train.npz'))['data']
        if as_df:
            res = pandas.DataFrame(res, columns=['l1_neighbours', 'graph_neighbours', 'l2_neighbours'])
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_magic',
            'neighbours'
        )
        return (base_path + fname).get()

    @staticmethod
    def hash_question(q):
        return mmh3.hash(q)

    def hash_series(self, qs):
        return [self.hash_question(q) for q in qs]

    def rehash(self, h1, h2):
        G = self.hashgraph
        n1 = G.neighbors(h1)
        n2 = G.neighbors(h2)
        n1_2 = networkx.ego_graph(G, h1)
        n2_2 = networkx.ego_graph(G, h2)

        return [
            len(set(n1).intersection(set(n2))),
            len(list(networkx.common_neighbors(G, h1, h2))),
            len(set(n1_2).intersection(set(n2_2))),
        ]

    def run(self):
        self.output().makedirs()
        train_data = rf_dataset.Dataset().load_all('train', as_df=True)
        test_data = rf_dataset.Dataset().load_all('test', as_df=True)

        all_questions = (list(train_data.question1_clean) +
                         list(test_data.question1_clean) +
                         list(train_data.question2_clean) +
                         list(test_data.question2_clean))

        allq = multiprocessing.Pool().imap(self.hash_question, all_questions, chunksize=10000)
        hashes = list(tqdm(allq, total=len(all_questions)))
        train_size = train_data.shape[0]
        test_size = test_data.shape[0]
        q1s = hashes[:(train_size + test_size)]
        q2s = hashes[(train_size + test_size):]
        hashes = {
            'train_q1': q1s[:train_size],
            'train_q2': q2s[:train_size],
            'test_q1': q1s[train_size:],
            'test_q2': q2s[train_size:],
        }
        self.hashgraph = networkx.Graph()
        self.hashgraph.add_edges_from(zip(hashes['train_q1'], hashes['train_q2']))
        self.hashgraph.add_edges_from(zip(hashes['test_q1'], hashes['test_q2']))

        train_iter = tqdm(zip(hashes['train_q1'], hashes['train_q2']), total=train_size, desc='train_feat')
        train_feat = [self.rehash(h1, h2) for h1, h2 in train_iter]
        np.savez_compressed(self.make_path('train.npz'), data=np.asarray(train_feat))

        test_iter = tqdm(zip(hashes['test_q1'], hashes['test_q2']), total=test_size, desc='test_feat')
        test_feat = [self.rehash(h1, h2) for h1, h2 in test_iter]
        np.savez_compressed(self.make_path('test.npz'), data=np.asarray(test_feat))

        with self.output().open('w'):
            pass

class QuestionOrderFeature(FoldIndependent):
    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_magic',
            'ordering'
        )
        return (base_path + fname).get()

    def _load_test(self, as_df):
        res = np.load(self.make_path('test.npz'))['data']
        if as_df:
            res = pandas.DataFrame(res, columns=['qprob'])
        return res

    def _load(self, as_df):
        res = np.load(self.make_path('train.npz'))['data']
        if as_df:
            res = pandas.DataFrame(res, columns=['qprob'])
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def run(self):
        self.output().makedirs()
        train = rf_dataset.Dataset().load_all('train', as_df=True)
        test = rf_dataset.Dataset().load_all('test', as_df=True)

        true_qid = {q: id for q, id in zip(train.question1, train.qid1)}
        true_qid.update({q: id for q, id in zip(train.question2, train.qid2)})
        train_max_id = max(true_qid.values())
        step_size = 507000 / 2345806
        current_id = train_max_id + step_size

        for q1, q2 in tqdm(zip(test.question1, test.question2), total=test.shape[0]):
            if q1 not in true_qid:
                true_qid[q1] = current_id
                current_id += step_size
            if q2 not in true_qid:
                true_qid[q2] = current_id
                current_id += step_size

        train_feature = [min(true_qid[q1], true_qid[q2]) for q1, q2
                         in tqdm(zip(train.question1, train.question2), total=train.shape[0])]

        test_feature = [min(true_qid[q1], true_qid[q2]) for q1, q2
                         in tqdm(zip(test.question1, test.question2), total=test.shape[0])]

        np.savez_compressed(self.make_path('train.npz'), data=np.asarray(train_feature)[:, None])
        np.savez_compressed(self.make_path('test.npz'), data=np.asarray(test_feature)[:, None])
        with self.output().open('w'):
            pass