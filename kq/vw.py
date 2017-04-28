import tempfile
import os
from glob import glob

import luigi
import pandas
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from plumbum import local, FG, colors

from kq import dataset, shared_words, distances, shared_entites, core, tfidf_matrix, wordmat_distance


class VWData(luigi.Task):
    data_subset = None  # train, test, merge or valid
    max_size = 100000

    def requires(self):
        yield dataset.Dataset()
        yield tfidf_matrix.TFIDFFeature()
        yield shared_words.QuestionVector()
        yield distances.AllDistances()
        yield shared_entites.SharedEntities()
        yield wordmat_distance.WordMatDistance()

    def complete(self):
        if False and self.data_subset == 'test':
            if not os.path.exists('cache/vw_data/test_0.svm'):
                return False
            test_size = dataset.Dataset().load_test().shape[0]
            target_ixs = self.test_target_indexes(test_size)
            for target_ix in target_ixs:
                if not os.path.exists('cache/vw_data/test_%d.svm' % target_ix):
                    return False
            return True
        else:
            return os.path.exists('cache/vw_data/%s.svm' % self.data_subset)

    def run(self):
        assert self.data_subset in {'train', 'test', 'merge', 'valid'}
        if self.data_subset in {'train', 'valid', 'merge'}:
            ix = {'train': 0, 'merge': 1, 'valid': 2}[self.data_subset]
            vecs = tfidf_matrix.TFIDFFeature.load_dataset(self.data_subset)
            qvecs = shared_words.QuestionVector().load()[ix]
            dvecs = distances.AllDistances().load()[ix]
            evecs = shared_entites.SharedEntities().load()[ix]
            wmvecs = wordmat_distance.WordMatDistance().load(self.data_subset)
            labels = dataset.Dataset().load()[ix].is_duplicate.values
        else:
            vecs = tfidf_matrix.TFIDFFeature.load_dataset('test')
            qvecs = shared_words.QuestionVector().load_test()
            dvecs = distances.AllDistances().load_test()
            evecs = shared_entites.SharedEntities().load_test()
            wmvecs = wordmat_distance.WordMatDistance().load('test')
            labels = np.zeros(qvecs.shape[0], dtype='uint8')

        qvec_offset = 1
        dvec_offset = qvecs.shape[1]
        evec_offset = dvec_offset + dvecs.shape[1]
        wmvec_offset = evec_offset + evecs.shape[1]
        vecs_offset = wmvec_offset + wmvecs.shape[1]

        def write_row(i, f):
            row = vecs[i]
            qvec = qvecs[i]
            dvec = dvecs[i]
            evec = evecs[i]
            wmvec = wmvecs[i]
            label = labels[i] * 2 - 1
            qvec[np.isnan(qvec)] = -1
            dvec[np.isnan(dvec)] = -1
            evec[np.isnan(evec)] = -1
            wmvec[np.isnan(wmvec)] = -1

            qvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(qvec, start=qvec_offset))
            dvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(dvec, start=dvec_offset))
            evec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(evec, start=evec_offset))
            wmvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(wmvec, start=wmvec_offset))
            entries = " ".join(("%d:%.2f" % (ind + vecs_offset, data) for ind, data in zip(row.indices, row.data)))
            f.write("%d %f |Q %s |D %s |E %s |M %s |W %s\n" %
                    (label, core.weights[label], qvec_entries, dvec_entries, evec_entries, wmvec_entries, entries))

        os.makedirs('cache/vw_data', exist_ok=True)
        if False:
            for start_ix in tqdm(self.test_target_indexes(vecs.shape[0])):
                with open('cache/vw_data/test_%d_tmp.svm' % start_ix, 'w') as f:
                    for i in range(start_ix, min(start_ix + self.max_size, vecs.shape[0])):
                        write_row(i, f)
                os.rename('cache/vw_data/test_%d_tmp.svm' % start_ix, 'cache/vw_data/test_%d.svm' % start_ix)
        else:
            with open('cache/vw_data/%s_tmp.svm' % self.data_subset, 'w') as f:
                for i in tqdm(range(qvecs.shape[0]), desc='writing %s data' % self.data_subset):
                    write_row(i, f)
            os.rename('cache/vw_data/%s_tmp.svm' % self.data_subset, 'cache/vw_data/%s.svm' % self.data_subset)

    @staticmethod
    def test_target_indexes(test_size):
        j = 0
        target_ixs = []
        while j < test_size:
            target_ixs.append(j)
            j += VWData.max_size
        return target_ixs


class TrainVWData(VWData):
    data_subset = 'train'


class ValidVWData(VWData):
    data_subset = 'valid'


class MergeVWData(VWData):
    data_subset = 'merge'


class TestVWData(VWData):
    data_subset = 'test'


class VWClassifier(luigi.Task):
    vw_path = luigi.Parameter(default='/usr/local/bin/vw')

    def requires(self):
        yield TrainVWData()
        yield ValidVWData()
        yield MergeVWData()
        yield TestVWData()
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/vw/classifier_pred.csv.gz')

    def train(self):
        self.output().makedirs()
        print(colors.green & colors.bold | "Starting training")
        try:
            os.remove('cache/vw/cache_file')
        except FileNotFoundError:
            pass
        local[self.vw_path]['--binary ' \
                            '-q QE -q QD ' \
                            '-f cache/vw/mdl ' \
                            '--passes 10 --cache_file cache/vw/cache -k ' \
                            '--l2 0.0001 ' \
                            '--l1 0.0001 ' \
                            'cache/vw_data/train.svm'.split(' ')] & FG
        print(colors.green & colors.bold | "Finished training")

    def predict(self, dataset):
        options = '-i cache/vw/mdl -t cache/vw_data/%s.svm -p cache/vw/preds.csv' % dataset
        local[self.vw_path][options.split(' ')] & FG
        print(colors.green & colors.bold | "Finished validation predictions")

        pred = (pandas.read_csv('cache/vw/preds.csv', names=['is_duplicate']) + 1) / 2
        pred.index = pred.index.rename('test_id')
        return pred

    def valid(self):
        pred = self.predict('valid')
        print(colors.green | "prediction sample...")
        print(colors.green | str(pred.head()))
        y = dataset.Dataset().load()[2]
        loss = core.score_data(y.is_duplicate, pred)
        print(colors.green | "Performance: " + str(loss))

        return pred

    def merge(self):
        pred = self.predict('merge')
        pred.to_csv('cache/vw/merge_predictions.csv')

    def test(self):
        local[self.vw_path]['-i cache/vw/mdl -t cache/vw_data/test.svm -p cache/vw/test_preds.csv'.split(' ')] & FG
        print(colors.green & colors.bold | "Finished test predictions")

        pred = (pandas.read_csv('cache/vw/test_preds.csv', names=['is_duplicate']) + 1) / 2
        pred.index = pred.index.rename('test_id')

        return pred

    def run(self):
        self.train()
        self.valid()
        self.merge()

        pred = self.test()

        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            pred.to_csv(tf.name, compression='gzip')
            os.rename(tf.name, self.output().path)
        except:
            os.remove(tf.name)
            raise

    def load(self):
        assert self.complete()
        return pandas.read_csv('cache/vw/merge_predictions.csv', index_col='test_id').values

    def load_test(self):
        assert self.complete()
        return pandas.read_csv('cache/vw/classifier_pred.csv.gz', index_col='test_id').values
