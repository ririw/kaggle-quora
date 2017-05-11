import tempfile
import os
from glob import glob

import luigi
import pandas
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from plumbum import local, FG, colors

from kq import dataset, question_vectors, distances, shared_entites, core, tfidf_matrix, wordmat_distance, question_freq
from kq import feature_collection
from kq import sentiments


class SVMData(luigi.Task):
    resources = {'cpu': 1}
    data_subset = None  # train, test, merge or valid
    max_size = 50000

    def requires(self):
        yield dataset.Dataset()
        yield tfidf_matrix.TFIDFFeature()
        yield feature_collection.FeatureCollection()

    def output(self):
        return luigi.LocalTarget('cache/svm_data/done_%s' % self.data_subset)

    def run(self):
        assert self.data_subset in {'train', 'test', 'merge', 'valid'}
        simple_vecs = feature_collection.FeatureCollection().load(self.data_subset).values
        complex_vecs = tfidf_matrix.TFIDFFeature.load_dataset(self.data_subset)
        labels = dataset.Dataset().load_named(self.data_subset).is_duplicate.values

        def write_row(i, f1, f2, f3):
            simple_vec = simple_vecs[i]
            complex_vec = complex_vecs[i]
            label = labels[i]

            simple_entries = ' '.join(
                '%d:%f' % ix_v for ix_v in enumerate(simple_vec))
            offset = simple_vec.shape[1]
            complex_entries = ' '.join(
                ("%d:%.2f" % (ind + offset, data) for ind, data in zip(complex_vec.indices, complex_vec.data)))

            f1.write(str(label) + ' ')
            f1.write(' '.join(simple_entries))
            f1.write(' '.join(complex_entries))
            f1.write('\n')

            f2.write(str(label) + ' ')
            f2.write(' '.join(simple_entries))
            f2.write('\n')

            f3.write(str(label) + ' ')
            f3.write(' '.join(complex_entries))
            f3.write('\n')

        os.makedirs('cache/svm_data/simple', exist_ok=True)
        os.makedirs('cache/svm_data/complex', exist_ok=True)
        os.makedirs('cache/svm_data/words', exist_ok=True)
        if self.data_subset == 'test':
            for start_ix in tqdm(self.test_target_indexes(labels.shape[0])):
                with open('cache/svm_data/complex/test_%d.svm.tmp' % start_ix, 'w') as f1, \
                     open('cache/svm_data/simple/test_%d.svm.tmp' % start_ix, 'w') as f2, \
                     open('cache/svm_data/words/test_%d.svm.tmp' % start_ix, 'w') as f3:
                    for i in range(start_ix, min(start_ix + self.max_size, labels.shape[0])):
                        write_row(i, f1, f2, f3)
                os.rename('cache/svm_data/simple/test_%d.svm.tmp' % start_ix,
                          'cache/svm_data/simple/test_%d.svm' % start_ix)
                os.rename('cache/svm_data/words/test_%d.svm.tmp' % start_ix,
                          'cache/svm_data/words/test_%d.svm' % start_ix)
                os.rename('cache/svm_data/complex/test_%d.svm.tmp' % start_ix,
                          'cache/svm_data/complex/test_%d.svm' % start_ix)
        else:
            with open('cache/svm_data/complex/%s.svm.tmp' % self.data_subset, 'w') as f1, \
                 open('cache/svm_data/simple/%s.svm.tmp' % self.data_subset, 'w') as f2, \
                 open('cache/svm_data/words/%s.svm.tmp' % self.data_subset, 'w') as f3:
                for i in tqdm(range(labels.shape[0]), desc='writing %s data' % self.data_subset):
                    write_row(i, f1, f2, f3)
            os.rename('cache/svm_data/simple/%s.svm.tmp' % self.data_subset,
                      'cache/svm_data/simple/%s.svm' % self.data_subset)
            os.rename('cache/svm_data/words/%s.svm.tmp' % self.data_subset,
                      'cache/svm_data/words/%s.svm' % self.data_subset)
            os.rename('cache/svm_data/complex/%s.svm.tmp' % self.data_subset,
                      'cache/svm_data/complex/%s.svm' % self.data_subset)
        with self.output().open('w'):
            pass

    @staticmethod
    def test_target_indexes(test_size):
        j = 0
        target_ixs = []
        while j < test_size:
            target_ixs.append(j)
            j += SVMData.max_size
        return target_ixs


class TrainSVMData(SVMData):
    data_subset = 'train'


class ValidSVMData(SVMData):
    data_subset = 'valid'


class MergeSVMData(SVMData):
    data_subset = 'merge'


class TestSVMData(SVMData):
    data_subset = 'test'


class GBMClassifier(luigi.Task):
    resources = {'cpu': 8}

    lightgbm_path = luigi.Parameter(default=os.path.expanduser('~/Downloads/LightGBM/lightgbm'))
    dataset_kind = luigi.Parameter()

    def make_path(self, *rest):
        assert self.dataset_kind in {'simple', 'complex', 'words'}
        parts = ['cache', 'lightgbm', self.dataset_kind] + list(rest)
        return os.path.join(*parts)

    def requires(self):
        yield TrainSVMData()
        yield ValidSVMData()
        yield MergeSVMData()
        yield TestSVMData()
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('classifier_pred.csv.gz'))

    def train(self):
        self.output().makedirs()
        print(colors.green & colors.bold | "Starting training")
        with open(self.make_path('train_gbm_classifier.conf'), 'w') as f:
            conf = self.train_conf.format(data_path='cache/svm_data/%s' % self.dataset_kind,
                                          resulty_path=self.make_path())
            print(conf)
            f.write(conf)
        local[self.lightgbm_path]['config=' + self.make_path('train_gbm_classifier.conf')] & FG
        print(colors.green & colors.bold | "Finished training")

    def pred_simple_target(self, dataset):
        with open(self.make_path('pred.conf'), 'w') as f:
            conf = self.valid_conf.format(data_path='cache/svm_data/%s/%s.svm' % (self.dataset_kind, dataset),
                                          resulty_path=self.make_path())
            print(conf)
            f.write(conf)

        local[self.lightgbm_path]['config=' + self.make_path('pred.conf')] & FG
        pred_loc = self.make_path('preds.csv')
        pred = pandas.read_csv(pred_loc, names=['is_duplicate'])
        pred.index = pred.index.rename('test_id')
        return pred

    def merge(self):
        pred = self.pred_simple_target('merge')
        pred.to_csv(self.make_path('merge_predictions.csv'))

    def valid(self):
        pred = self.pred_simple_target('valid')
        print(colors.green | "prediction sample...")
        print(colors.green | str(pred.head()))
        y = dataset.Dataset().load()[2]
        weights = core.weights[y.is_duplicate.values]
        loss = metrics.log_loss(y.is_duplicate, pred.is_duplicate, sample_weight=weights)
        print(colors.green | "Performance: " + str(loss))

        return pred

    def test(self):
        test_size = dataset.Dataset().load_test().shape[0]
        test_tasks = SVMData.test_target_indexes(test_size)
        print(colors.green & colors.bold | "Predicting test values, this takes a long time...")
        for target_ix in tqdm(test_tasks, desc='Predicting'):
            with open(self.make_path('test_gbmclassifier.conf'), 'w') as f:
                conf = self.test_conf.format(
                    data_path='cache/svm_data/%s' % self.dataset_kind,
                    ix=target_ix,
                    resulty_path=self.make_path()
                )
                print(conf)
                f.write(conf)
            local[self.lightgbm_path]['config='+self.make_path('test_gbmclassifier.conf')] & FG

        preds = []
        for target_ix in tqdm(test_tasks, desc='Reading results file'):
            pred = pandas.read_csv(self.make_path('test_preds_%d.csv' % target_ix), names=['is_duplicate'])
            index = np.arange(target_ix, min(test_size, target_ix + SVMData.max_size))
            pred.index = pandas.Series(index, name='test_id')
            preds.append(pred)
        preds = pandas.concat(preds, 0)
        return preds

    def run(self):
        self.output().makedirs()
        self.train()
        self.merge()
        self.valid()
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
        return pandas.read_csv(self.make_path('merge_predictions.csv'), index_col='test_id').values

    def load_test(self):
        assert self.complete()
        return pandas.read_csv(self.make_path('classifier_pred.csv.gz'), index_col='test_id').values

    train_conf = """
    task = train
    objective = binary
    metric = binary_logloss
    metric_freq = 5
    is_training_metric = true
    early_stopping_round = 10

    data={data_path}/train.svm
    valid_data={data_path}/valid.svm
    output_model={resulty_path}/gbm_model

    boosting_type = gbdt
    learning_rate = 0.05
    num_trees = 1000
    num_leaves = 1500
    scale_pos_weight = 0.46
    min_data_in_leaf = 25
    feature_fraction = 0.75
    bagging_fraction = 0.75
    bagging_freq = 3

    is_sparse = true
    use_two_round_loading = false
    is_save_binary_file = false
    """

    valid_conf = """
    task = predict
    data = {data_path}
    input_model={resulty_path}/gbm_model
    output_result={resulty_path}/preds.csv
    """

    test_conf = """
    task = predict
    data = {data_path}/test_{ix}.svm
    input_model = {resulty_path}/gbm_model
    output_result = {resulty_path}/test_preds_{ix}.csv
    """

