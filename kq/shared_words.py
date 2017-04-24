import logging
import os
import tempfile
from collections import Counter
from zipfile import ZipFile

import luigi
import numpy as np
import pandas
import spacy.en
from scipy import sparse, io, spatial
from sklearn import neighbors
import tqdm

from kq import core, dataset


def vectorize_sent(sent, word_ix_map, max_ix):
    vec = np.zeros(max_ix + 1)
    for tok in sent:
        if tok in word_ix_map:
            vec[word_ix_map[tok]] += 1
    return vec


class Vocab(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/vocab.msg')

    def run(self):
        train_data, _, _ = dataset.Dataset().load()
        vocab_count = Counter()
        for sent in tqdm.tqdm(train_data.question1_tokens,
                              desc='Counting questions one',
                              total=train_data.shape[0]):
            for tok in sent:
                vocab_count[tok] += 1

        for sent in tqdm.tqdm(train_data.question1_tokens,
                              desc='Counting questions two',
                              total=train_data.shape[0]):
            for tok in sent:
                vocab_count[tok] += 1

        vocab_counts = pandas.Series(vocab_count)
        self.output().makedirs()
        vocab_counts.to_msgpack(self.output().path)

    def load_vocab(self, min_occurances=10):
        assert self.complete()
        vocab_counts = pandas.read_msgpack('./cache/vocab.msg')
        admissible_vocab = vocab_counts[vocab_counts > min_occurances].copy()
        admissible_vocab.index = admissible_vocab.index.rename('word')
        admissible_vocab = admissible_vocab.to_frame('count').sort_values('count', ascending=False)
        admissible_vocab['word_id'] = np.arange(admissible_vocab.shape[0]) + 1
        return admissible_vocab


class WordVectors(luigi.Task):
    def requires(self):
        yield dataset.Dataset()
        yield Vocab()

    def output(self):
        return luigi.LocalTarget('./cache/word_vectors.zip')

    def run(self):
        train, merge, valid = dataset.Dataset().load()
        vocab_frame = Vocab().load_vocab()
        train_mat = self.vectorize_mat(train, vocab_frame, ' -- train data')
        merge_mat = self.vectorize_mat(merge, vocab_frame, ' -- merge data')
        valid_mat = self.vectorize_mat(valid, vocab_frame, ' -- valid data')
        self.output().makedirs()
        res_tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            with ZipFile(res_tf, mode='w') as zf:
                with zf.open('train', 'w') as f:
                    io.mmwrite(f, train_mat)
                with zf.open('merge', 'w') as f:
                    io.mmwrite(f, merge_mat)
                with zf.open('valid', 'w') as f:
                    io.mmwrite(f, valid_mat)
                del train, train_mat, valid, valid_mat, merge, merge_mat
                test = dataset.Dataset().load_test()
                test_mat = self.vectorize_mat(test, vocab_frame, ' -- test data')
                with zf.open('test', 'w') as f:
                    io.mmwrite(f, test_mat)
            os.rename(res_tf.name, self.output().path)
        except Exception:
            logging.warning('Deleting temporary file, due to error')
            os.remove(res_tf.name)
            raise

    def vectorize_mat(self, data, vocab_frame, vec_note=''):
        vocab_dict = vocab_frame.word_id.to_dict()
        vocab_max_id = vocab_frame.word_id.max()
        mat_size = (vocab_max_id + 1)

        vs = []
        rs = []
        cs = []

        for ix, (s1, s2) in tqdm.tqdm(
                enumerate(zip(data.question1_tokens, data.question2_tokens)), total=data.shape[0],
                desc='Vectorizing matrix' + vec_note):
            vec = np.zeros(mat_size*2)
            v1 = vectorize_sent(s1, vocab_dict, vocab_max_id)
            v2 = vectorize_sent(s2, vocab_dict, vocab_max_id)

            assert (max(np.max(v1), np.max(v2)) < 255)
            vec[:mat_size] = v1 * v2
            vec[mat_size:] = v1 + v2
            nonzero_ix = np.nonzero(vec)[0]
            nonzero_vs = vec[nonzero_ix]
            vs.append(nonzero_vs)
            cs.append(nonzero_ix)
            rs.append(np.ones_like(nonzero_ix) * ix)
        vs = np.concatenate(vs)
        cs = np.concatenate(cs)
        rs = np.concatenate(rs)
        res = sparse.coo_matrix((vs, (rs, cs)), shape=[data.shape[0], mat_size * 4])
        return res.tocsr()

    def load(self):
        with ZipFile(self.output().path, 'r') as zf:
            with zf.open('train', 'r') as f:
                train = io.mmread(f).tocsr()
            with zf.open('merge', 'r') as f:
                merge = io.mmread(f).tocsr()
            with zf.open('valid', 'r') as f:
                valid = io.mmread(f).tocsr()

        return train, merge, valid

    def load_test(self):
        with ZipFile(self.output().path, 'r') as zf:
            with zf.open('test', 'r') as f:
                return io.mmread(f).tocsr()


class SharedWordCount(luigi.Task, core.MergableFeatures):
    def valid_feature(self):
        return self.load()[1]

    def test_feature(self):
        return self.load()[2]

    def train_feature(self):
        return self.load()[0]

    def requires(self):
        return [Vocab(), dataset.Dataset()]

    def output(self):
        return luigi.LocalTarget('./cache/shared_word_count.npz')

    def run(self):
        train, valid = dataset.Dataset().load()
        vocab_frame = Vocab().load_vocab()
        vocab_dict = vocab_frame.word_id.to_dict()
        vocab_max_id = vocab_frame.word_id.max()

        shared_train_vec = self.calculate_shared_vec(train, vocab_dict, vocab_max_id)
        shared_valid_vec = self.calculate_shared_vec(valid, vocab_dict, vocab_max_id)
        test = dataset.Dataset().load_test()
        shared_test_vec = self.calculate_shared_vec(test, vocab_dict, vocab_max_id)

        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        try:
            np.savez(tmpfile,
                     shared_valid_vec=shared_valid_vec,
                     shared_train_vec=shared_train_vec,
                     shared_test_vec=shared_test_vec)
            tmpfile.close()

            self.output().makedirs()
            os.rename(tmpfile.name, self.output().path)
        except Exception:
            logging.warning('Deleting temporary file, due to error')
            os.remove(tmpfile.name)
            raise

    @staticmethod
    def calculate_shared_vec(data, vocab_dict, vocab_max_id):
        shared_vec = []
        for s1, s2 in tqdm.tqdm(zip(data.question1_tokens, data.question2_tokens), total=data.shape[0]):
            v1 = vectorize_sent(s1, vocab_dict, vocab_max_id)
            v2 = vectorize_sent(s2, vocab_dict, vocab_max_id)
            shared_vec.append(((v1 * v2) > 0).sum())

        return np.asarray(shared_vec)

    def load(self):
        assert self.complete()
        data = np.load(self.output().path, mmap_mode='r')
        return pandas.Series(data['shared_train_vec'], name='shared_word_count'), \
               pandas.Series(data['shared_valid_vec'], name='shared_word_count'), \
               pandas.Series(data['shared_test_vec'], name='shared_word_count'),


def save_sparse_csr(filename, array):
    np.savez_compressed(filename,
                        data=array.data,
                        indices=array.indices,
                        indptr=array.indptr,
                        shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((
        loader['data'],
        loader['indices'],
        loader['indptr']), shape=loader['shape'])


class QuestionVector(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/question_vec.npz')

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
        diffs = np.abs(v1 - v2)
        return np.concatenate([diffs, distance_mat], 1)


    def run(self):
        English = spacy.en.English()
        tqdm.tqdm.pandas(tqdm.tqdm)
        import coloredlogs
        coloredlogs.install(level=logging.INFO)

        train, merge, valid = dataset.Dataset().load()
        logging.info('Vectorizing: train/q1')
        train_vecs1 = np.vstack(train.question1_clean.progress_apply(lambda q: English(q).vector).values)
        logging.info('Vectorizing: train/q2')
        train_vecs2 = np.vstack(train.question2_clean.progress_apply(lambda q: English(q).vector).values)
        train_vecs = self.merge_vecs(train_vecs1, train_vecs2)
        del train, train_vecs1, train_vecs2

        logging.info('Vectorizing: merge/q1')
        merge_vecs1 = np.vstack(merge.question1_clean.progress_apply(lambda q: English(q).vector).values)
        logging.info('Vectorizing: merge/q2')
        merge_vecs2 = np.vstack(merge.question2_clean.progress_apply(lambda q: English(q).vector).values)
        merge_vecs = self.merge_vecs(merge_vecs1, merge_vecs2)
        del merge, merge_vecs1, merge_vecs2

        logging.info('Vectorizing: valid/q1')
        valid_vecs1 = np.vstack(valid.question1_clean.progress_apply(lambda q: English(q).vector).values)
        logging.info('Vectorizing: valid/q2')
        valid_vecs2 = np.vstack(valid.question2_clean.progress_apply(lambda q: English(q).vector).values)
        valid_vecs = self.merge_vecs(valid_vecs1, valid_vecs2)
        del valid, valid_vecs1, valid_vecs2

        test = dataset.Dataset().load_test()
        logging.info('Vectorizing: test/q1')
        test_vecs1 = np.vstack(test.question1_clean.progress_apply(lambda q: English(q).vector).values)
        logging.info('Vectorizing: test/q2')
        test_vecs2 = np.vstack(test.question2_clean.progress_apply(lambda q: English(q).vector).values)
        test_vecs = self.merge_vecs(test_vecs1, test_vecs2)
        del test, test_vecs1, test_vecs2

        tmpfile = tempfile.mktemp()
        with open(tmpfile, 'wb') as f:
            np.savez(f, train_vecs=train_vecs, merge_vecs=merge_vecs,
                     valid_vecs=valid_vecs, test_vecs=test_vecs)
        self.output().makedirs()
        os.rename(tmpfile, self.output().path)

    def load(self):
        assert self.complete()
        data = np.load(self.output().path, mmap_mode='r')
        return data['train_vecs'], data['merge_vecs'], data['valid_vecs']

    def load_test(self):
        assert self.complete()
        data = np.load(self.output().path, mmap_mode='r')
        return data['test_vecs']
