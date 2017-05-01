import luigi
import pandas
import spacy
from tqdm import tqdm

from kq import count_matrix, tfidf_matrix, dataset, vocab
import numpy as np
import scipy.sparse as sp
from sklearn import decomposition

class WordMatDistance(luigi.Task):
    def requires(self):
        yield count_matrix.CountFeature()
        yield tfidf_matrix.TFIDFFeature()

    def output(self):
        return luigi.LocalTarget('cache/word_mat_distances/done')

    def shared_mean_transform(self, m1, m2):
        return np.mean(np.sqrt(m1.multiply(m2)), 1)

    def sparse_svm_transform(self, m1, m2, transformer):
        stackmat = sp.vstack([m1, m2])
        vecs = transformer.transform(stackmat)
        u1 = vecs[:m1.shape[0]]
        u2 = vecs[m1.shape[0]:]
        return np.concatenate([u1, u2], 1)

    def train_decompositions(self):
        m1, m2 = count_matrix.CountFeature().load_full_mats('train')
        stackmat = sp.vstack([m1, m2])
        self.pca_count = decomposition.TruncatedSVD(6)
        self.pca_count.fit(stackmat)

        m1, m2 = tfidf_matrix.TFIDFFeature().load_full_mats('train')
        stackmat = sp.vstack([m1, m2])
        self.pca_tfidf = decomposition.TruncatedSVD(6)
        self.pca_tfidf.fit(stackmat)

    def run(self):
        self.output().makedirs()
        self.train_decompositions()
        self.run_ds('train')
        self.run_ds('merge')
        self.run_ds('valid')
        self.run_ds('test')

        with self.output().open('w'):
            pass

    def run_ds(self, dataset):
        cm1, cm2 = count_matrix.CountFeature().load_full_mats(dataset)
        cd = self.shared_mean_transform(cm1, cm2)
        cu = self.sparse_svm_transform(cm1, cm2, self.pca_count)

        tm1, tm2 = tfidf_matrix.TFIDFFeature().load_full_mats(dataset)
        td = self.shared_mean_transform(tm1, tm2)
        tu =  self.sparse_svm_transform(tm1, tm2, self.pca_tfidf)

        print(type(cd), type(cu), type(td), type(tu))

        full_mat = np.concatenate([cd, cu, td, tu], 1)

        np.save('cache/word_mat_distances/%s.npy' % dataset, full_mat)

    def load(self, dataset):
        assert self.complete()
        assert dataset in {'train', 'test', 'merge', 'valid'}
        return np.load('cache/word_mat_distances/%s.npy' % dataset)


class SentenceVecs(luigi.Task):
    def requires(self):
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/question_vectors/done')

    def vectorize_sent(self, sent):
        dat = [(self.English.vocab[tok].vector) for tok in sent]
        return np.mean(np.vstack(dat), 1)

    def vectorize_dataset(self, dataset_name):
        examples = dataset.Dataset().load_named(dataset_name)
        all_examples = pandas.concat([examples.question1_tokens, examples.question2_tokens])
        all_vecs = np.vstack(all_examples.progress_apply(self.vectorize_sent))
        return all_vecs[:all_vecs.shape[0]//2], all_vecs[all_vecs.shape[0]//2:]

    def run(self):
        tqdm.pandas(tqdm)
        self.output().makedirs()
        self.English = spacy.en.English()

        train_data = self.vectorize_dataset('train')
        np.save('cache/question_vectors/train_1.npy', train_data[0])
        np.save('cache/question_vectors/train_2.npy', train_data[1])
        del train_data

        merge_data = self.vectorize_dataset('merge')
        np.save('cache/question_vectors/merge_1.npy', merge_data[0])
        np.save('cache/question_vectors/merge_2.npy', merge_data[1])
        del merge_data

        valid_data = self.vectorize_dataset('valid')
        np.save('cache/question_vectors/valid_1.npy', valid_data[0])
        np.save('cache/question_vectors/valid_2.npy', valid_data[1])
        del valid_data

        test_data = self.vectorize_dataset('test')
        np.save('cache/question_vectors/test_1.npy',  test_data[0])
        np.save('cache/question_vectors/test_2.npy',  test_data[1])

        with self.output().open('w') as f:
            pass

    def load(self, name):
        assert self.complete()
        a = np.load('cache/question_vectors/%s_1.npy' % name, mmap_mode='r')
        b = np.load('cache/question_vectors/%s_2.npy' % name, mmap_mode='r')

        return a, b


class SimpleSentenceDistance(luigi.Task):
    # See: https://openreview.net/pdf?id=SyK00v5xx
    def requires(self):
        yield dataset.Dataset()
        yield vocab.Vocab()

    def output(self):
        return luigi.LocalTarget('cache/simple_vectors/done')

    def vectorize_sent(self, sent):
        dat = [self.English.vocab[tok].vector * 1e-4 / (1e-4 * self.word_probs[tok])
               for tok in sent if tok in self.word_probs]
        if len(dat) == 0:
            return np.zeros(300)

        return np.mean(dat, 0)

    def vectorize_dataset(self, dataset_name):
        examples = dataset.Dataset().load_named(dataset_name)
        all_examples = pandas.concat([examples.question1_tokens, examples.question2_tokens])
        all_vecs = np.vstack(all_examples.progress_apply(self.vectorize_sent))
        return all_vecs[:all_vecs.shape[0]//2], all_vecs[all_vecs.shape[0]//2:]

    def run(self):
        tqdm.pandas(tqdm)
        self.output().makedirs()
        self.English = spacy.en.English()
        v = vocab.Vocab().load_vocab()['count']
        v = v / v.sum()
        self.word_probs = v.to_dict()

        train_vecs = self.vectorize_dataset('train')
        svd_solver = decomposition.TruncatedSVD(n_components=2)
        full_data = np.concatenate(train_vecs, 0)
        #print("vvvvvvvvvvv")
        #print("vvvvvvvvvvv")
        #print("vvvvvvvvvvv")
        #print("vvvvvvvvvvv")
        svd_solver.fit(full_data)
        #print("^^^^^^^^^^^")
        #print("^^^^^^^^^^^")
        #print("^^^^^^^^^^^")
        #print("^^^^^^^^^^^")
        subtract_component = svd_solver.components_[0]

        train_data = [v-subtract_component for v in train_vecs]

        np.save('cache/simple_vectors/train_1.npy', train_data[0])
        np.save('cache/simple_vectors/train_2.npy', train_data[1])
        del train_data

        merge_data = [v-subtract_component for v in self.vectorize_dataset('merge')]
        np.save('cache/simple_vectors/merge_1.npy', merge_data[0])
        np.save('cache/simple_vectors/merge_2.npy', merge_data[1])
        del merge_data

        valid_data = [v-subtract_component for v in self.vectorize_dataset('valid')]
        np.save('cache/simple_vectors/valid_1.npy', valid_data[0])
        np.save('cache/simple_vectors/valid_2.npy', valid_data[1])
        del valid_data

        test_data = [v-subtract_component for v in self.vectorize_dataset('test')]
        np.save('cache/simple_vectors/test_1.npy',  test_data[0])
        np.save('cache/simple_vectors/test_2.npy',  test_data[1])

        with self.output().open('w') as f:
            f.write(','.join(['%f' % v for v in subtract_component]))

    def load(self, name):
        assert self.complete()
        a = np.load('cache/simple_vectors/%s_1.npy' % name, mmap_mode='r')
        b = np.load('cache/simple_vectors/%s_2.npy' % name, mmap_mode='r')

        return a, b