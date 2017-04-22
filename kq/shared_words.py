import logging
import os
import tempfile
from collections import Counter
from zipfile import ZipFile

import luigi
import numpy as np
import pandas
import spacy.en
from scipy import sparse, io
from sklearn import neighbors
import tqdm

from kq import core, dataset

En = spacy.en.English()


def extract_words(sent):
    res = []
    for tok in En(sent):
        if not tok.is_alpha or tok.is_stop:
            continue
        res.append(tok.lemma_)
    return res


def vectorize_sent(sent, word_ix_map, max_ix):
    vec = np.zeros(max_ix + 1)
    for tok in extract_words(sent):
        if tok in word_ix_map:
            vec[word_ix_map[tok]] += 1
    return vec


class Vocab(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/vocab.msg')

    def run(self):
        train_data, _ = dataset.Dataset().load()
        vocab_count = Counter()
        for sent in tqdm.tqdm(train_data.question1,
                              desc='Counting questions one',
                              total=train_data.shape[0]):
            for tok in extract_words(sent):
                vocab_count[tok] += 1

        for sent in tqdm.tqdm(train_data.question2,
                              desc='Counting questions two',
                              total=train_data.shape[0]):
            for tok in extract_words(sent):
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
        train, valid = dataset.Dataset().load()
        vocab_frame = Vocab().load_vocab()
        train_mat = self.vectorize_mat(train, vocab_frame, ' -- train data')
        valid_mat = self.vectorize_mat(valid, vocab_frame, ' -- valid data')
        self.output().makedirs()
        res_tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            with ZipFile(res_tf, mode='w') as zf:
                with zf.open('train', 'w') as f:
                    io.mmwrite(f, train_mat)
                with zf.open('valid', 'w') as f:
                    io.mmwrite(f, valid_mat)
                del train, train_mat, valid, valid_mat
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
                enumerate(zip(data.question1, data.question2)), total=data.shape[0],
                desc='Vectorizing matrix' + vec_note):
            vec = np.zeros(mat_size * 4)
            v1 = vectorize_sent(s1, vocab_dict, vocab_max_id)
            v2 = vectorize_sent(s2, vocab_dict, vocab_max_id)

            assert (max(np.max(v1), np.max(v2)) < 255)
            vec[:mat_size] = v1 * v2
            vec[mat_size:mat_size*2] = v1 + v2
            vec[mat_size*2:mat_size*3] = v1
            vec[mat_size*3:] = v2
            nonzero_ix = np.nonzero(vec)[0]
            nonzero_vs = vec[nonzero_ix]
            vs.append(nonzero_vs)
            cs.append(nonzero_ix)
            rs.append(np.ones_like(nonzero_ix) * ix)
        vs = np.concatenate(vs)
        cs = np.concatenate(cs)
        rs = np.concatenate(rs)
        res = sparse.coo_matrix((vs, (rs, cs)), shape=[data.shape[0], mat_size * 2])
        return res.tocsr()

    def load(self):
        with ZipFile(self.output().path, 'r') as zf:
            with zf.open('train', 'r') as f:
                train = io.mmread(f).tocsr()
            with zf.open('valid', 'r') as f:
                valid = io.mmread(f).tocsr()

        return train, valid

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
        for s1, s2 in tqdm.tqdm(zip(data.question1, data.question2), total=data.shape[0]):
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


class SmearedWordVectors(luigi.Task):
    def requires(self):
        yield Vocab()
        yield WordVectors()

    def output(self):
        return luigi.LocalTarget('./cache/smeared/done')

    def load(self):
        assert self.complete()
        train = load_sparse_csr('./cache/smeared/train.npz')
        valid = load_sparse_csr('./cache/smeared/valid.npz')
        return train, valid

    def load_test(self):
        assert self.complete()
        return load_sparse_csr('./cache/smeared/test.npz')

    def run(self):
        self.output().makedirs()
        try:
            mat = load_sparse_csr('cache/smeared/mat.npz')
        except FileNotFoundError:
            mat = self.word_matrix()
            save_sparse_csr('cache/smeared/mat.npz', mat)
        train_vecs, valid_vecs = WordVectors().load()
        half_width = train_vecs.shape[1] / 2
        print('Smearing train')
        print(train_vecs.shape)
        print(train_vecs[:, :half_width].shape)
        print(mat.shape)
        smear_train_a = train_vecs[:, :half_width] * mat
        smear_train_b = train_vecs[:, half_width:] * mat
        smear_train = sparse.hstack([smear_train_a, smear_train_b]).tocsr()
        print('Writing train')
        save_sparse_csr('./cache/smeared/train.npz', smear_train)
        del smear_train_b, smear_train_a, smear_train, train_vecs

        print('Smearing valid')
        smear_valid_a = valid_vecs[:, :half_width] * mat
        smear_valid_b = valid_vecs[:, half_width:] * mat
        smear_valid = sparse.hstack([smear_valid_a, smear_valid_b]).tocsr()
        print('Writing valid')
        save_sparse_csr('./cache/smeared/valid.npz', smear_valid)
        del smear_valid, smear_valid_a, smear_valid_b, valid_vecs

        print('Smearing test')
        test_vecs = WordVectors().load_test()
        smear_test_a = test_vecs[:, :half_width] * mat
        smear_test_b = test_vecs[:, half_width:] * mat
        smear_test = sparse.hstack([smear_test_a, smear_test_b]).tocsr()
        print('Finished smearing test, I\'ll bet it took ages, but now we\'re writing it out.')
        save_sparse_csr('./cache/smeared/test.npz', smear_test)
        with self.output().open('w'):
            # Boop!
            pass

    def word_matrix(self):
        print('Making smear matrix...')
        vocab = Vocab().load_vocab()
        # return sparse.csr_matrix(np.eye(vocab.shape[0]+1, vocab.shape[0]+1))
        ix_vocab = vocab.reset_index().set_index('word_id', drop=True)
        glove = GloveInterface()

        word_vecs = []
        unknown_ix = set()
        j = 0
        for word, ix in tqdm.tqdm(zip(vocab.index, vocab.word_id), desc='computing vectors', total=vocab.shape[0]):
            assert j == ix - 1, '%d -- %d' % (j, ix)
            j += 1
            if word in glove.words:
                word_vecs.append(glove[word])
            else:
                word_vecs.append(np.zeros(50))
                unknown_ix.add(ix)

        X = np.vstack(word_vecs)

        print('Computing kd tree')
        tree = neighbors.KDTree(X)
        print('Done!')
        smear_matrix = np.zeros([X.shape[0] + 1, X.shape[0] + 1])
        for word_tree_ix in tqdm.tqdm(range(X.shape[0])):
            word_vocab_ix = word_tree_ix - 1
            if word_vocab_ix not in unknown_ix:
                closest = tree.query(X[word_tree_ix:word_tree_ix + 1], k=3, return_distance=False)[0]
                if np.random.uniform() < 0.001:
                    print(ix_vocab.ix[closest + 1])
                smear_matrix[word_vocab_ix, closest + 1] = 0.2
            smear_matrix[word_vocab_ix, word_vocab_ix] = 1
        return sparse.csr_matrix(smear_matrix)


class GloveInterface:
    def __init__(self):
        words = {}
        with open('/Users/richardweiss/Datasets/glove.6B.50d.txt') as f:
            for line in f:
                items = line.split(' ')
                word = items[0]
                nums = np.array([float(v) for v in items[1:]])
                assert len(nums) == 50
                words[word] = nums
        self.words = words

    def __getitem__(self, item):
        return self.words[item]

    def get(self, item, default=None):
        return self.words.get(item, default)


class QuestionVector(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/question_vec.npz')

    def run(self):
        tqdm.tqdm.pandas(tqdm.tqdm)
        import coloredlogs
        coloredlogs.install(level=logging.INFO)

        train, valid = dataset.Dataset().load()
        logging.info('Vectorizing: train/q1')
        train_vecs1 = np.vstack(train.question1.progress_apply(lambda q: En(q).vector).values)
        logging.info('Vectorizing: train/q2')
        train_vecs2 = np.vstack(train.question2.progress_apply(lambda q: En(q).vector).values)
        train_vecs = np.concatenate([train_vecs1, train_vecs2], 1)
        del train, train_vecs1, train_vecs2

        logging.info('Vectorizing: valid/q1')
        valid_vecs1 = np.vstack(valid.question1.progress_apply(lambda q: En(q).vector).values)
        logging.info('Vectorizing: valid/q2')
        valid_vecs2 = np.vstack(valid.question2.progress_apply(lambda q: En(q).vector).values)
        valid_vecs = np.concatenate([valid_vecs1, valid_vecs2], 1)
        del valid, valid_vecs1, valid_vecs2

        test = dataset.Dataset().load_test()
        test_vecs1 = np.vstack(test.question1.progress_apply(lambda q: En(q).vector).values)
        test_vecs2 = np.vstack(test.question2.progress_apply(lambda q: En(q).vector).values)
        test_vecs = np.concatenate([test_vecs1, test_vecs2], 1)
        del test, test_vecs1, test_vecs2

        tmpfile = tempfile.mktemp()
        with open(tmpfile, 'wb') as f:
            np.savez(f, train_vecs=train_vecs, valid_vecs=valid_vecs, test_vecs=test_vecs)
        self.output().makedirs()
        os.rename(tmpfile, self.output().path)

    def load(self):
        assert self.complete()
        data = np.load(self.output().path, mmap_mode='r')
        return data['train_vecs'], data['valid_vecs']

    def load_test(self):
        assert self.complete()
        data = np.load(self.output().path, mmap_mode='r')
        return data['test_vecs']
