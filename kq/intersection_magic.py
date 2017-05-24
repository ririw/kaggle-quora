import multiprocessing

import logging

import coloredlogs
import joblib
import luigi
import numpy as np
import nose.tools
import mmh3

import networkx
import networkx.algorithms.connectivity
import sys

from tqdm import tqdm

from kq.dataset import Dataset


class QuestionIdMagicFeature(luigi.Task):
    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('cache/question_id_magic/done')

    def load_named(self, name):
        assert self.complete()
        return np.load('cache/question_id_magic/{:s}.npz'.format(name))


    def run(self):
        coloredlogs.install(level=logging.INFO)
        logger = logging.getLogger(repr(self))
        self.output().makedirs()
        logger.info('Started, loading data')
        train_data = Dataset().load_named('train')[['qid1', 'qid2']]
        merge_data = Dataset().load_named('merge')[['qid1', 'qid2']]
        valid_data = Dataset().load_named('valid')[['qid1', 'qid2']]
        test_data = Dataset().load_named('test')[['qid1', 'qid2']]

        logger.info('processing data')


class IntersectionMagicFeature(luigi.Task):
    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('cache/intersection_magic/done')

    def load_named(self, name):
        assert self.complete()
        return np.load('cache/intersection_magic/{:s}.npz'.format(name))['data']

    def hash_question(self, q):
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
        coloredlogs.install(level=logging.INFO)
        logger = logging.getLogger(repr(self))
        self.output().makedirs()
        logger.info('Started, loading data')
        train_data = Dataset().load_named('train')[['question1_raw', 'question2_raw']]
        merge_data = Dataset().load_named('merge')[['question1_raw', 'question2_raw']]
        valid_data = Dataset().load_named('valid')[['question1_raw', 'question2_raw']]
        test_data = Dataset().load_named('test')[['question1_raw', 'question2_raw']]

        logger.info('processing data')
        hashes = joblib.Parallel(n_jobs=-1)(map(joblib.delayed(self.hash_series), [
            train_data.question1_raw,
            train_data.question2_raw,
            merge_data.question1_raw,
            merge_data.question2_raw,
            valid_data.question1_raw,
            valid_data.question2_raw,
            test_data.question1_raw,
            test_data.question2_raw,
        ]))

        logger.info('building graph')
        self.hashgraph = networkx.Graph()

        self.hashgraph.add_edges_from(zip(hashes[0], hashes[1]))
        self.hashgraph.add_edges_from(zip(hashes[2], hashes[3]))
        self.hashgraph.add_edges_from(zip(hashes[4], hashes[5]))
        self.hashgraph.add_edges_from(zip(hashes[6], hashes[7]))

        logger.info('rehashing...')

        assert(len(hashes[0]) == train_data.shape[0])
        assert(len(hashes[2]) == merge_data.shape[0])
        assert(len(hashes[4]) == valid_data.shape[0])
        assert(len(hashes[6]) == test_data.shape[0])

        train_iter = tqdm(zip(hashes[0], hashes[1]), total=len(hashes[0]), desc='train_feat')
        train_feat = [self.rehash(h1, h2) for h1, h2 in train_iter]
        np.savez_compressed('cache/intersection_magic/train.npz', data=np.asarray(train_feat))

        merge_iter = tqdm(zip(hashes[2], hashes[3]), total=len(hashes[2]), desc='merge_feat')
        merge_feat = [self.rehash(h1, h2) for h1, h2 in merge_iter]
        np.savez_compressed('cache/intersection_magic/merge.npz', data=np.asarray(merge_feat))

        valid_iter = tqdm(zip(hashes[4], hashes[5]), total=len(hashes[4]), desc='valid_feat')
        valid_feat = [self.rehash(h1, h2) for h1, h2 in valid_iter]
        np.savez_compressed('cache/intersection_magic/valid.npz', data=np.asarray(valid_feat))

        test_iter =  tqdm(zip(hashes[6], hashes[7]), total=len(hashes[6]), desc='test_feat')
        test_feat =  [self.rehash(h1, h2) for h1, h2 in test_iter]
        np.savez_compressed('cache/intersection_magic/test.npz', data=np.asarray(test_feat))

        logger.info('saving data')

        with self.output().open('w'):
            pass