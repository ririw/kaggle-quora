import tempfile
import os
from glob import glob

import luigi
import pandas
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from plumbum import local, FG, colors

from kq import dataset, shared_words, distances


class SVMData(luigi.Task):
    data_subset = None  # train, test, merge or valid
    data_source = shared_words.WordVectors()
    max_size = 100000

    def requires(self):
        yield dataset.Dataset()
        yield shared_words.QuestionVector()
        for req in distances.AllDistances.requires():
            yield req
        yield self.data_source

    def complete(self):
        if self.data_subset == 'test':
            if not os.path.exists('cache/svm_data/test_0.svm'):
                return False
            test_size = dataset.Dataset().load_test().shape[0]
            target_ixs = self.test_target_indexes(test_size)
            for target_ix in target_ixs:
                if not os.path.exists('cache/svm_data/test_%d.svm' % target_ix):
                    return False
            return True
        else:
            return os.path.exists('cache/svm_data/%s.svm' % self.data_subset)
    
    def run(self):
        assert self.data_subset in {'train', 'test', 'merge', 'valid'}
        if self.data_subset == 'train':
            vecs = self.data_source.load()[0]
            qvecs = shared_words.QuestionVector().load()[0]
            dvecs = distances.AllDistances().load()[0]
            labels = dataset.Dataset().load()[0].is_duplicate.values
        elif self.data_subset == 'valid':
            vecs = self.data_source.load()[2]
            qvecs = shared_words.QuestionVector().load()[2]
            dvecs = distances.AllDistances().load()[2]
            labels = dataset.Dataset().load()[2].is_duplicate.values
        elif self.data_subset == 'merge':
            vecs = self.data_source.load()[1]
            qvecs = shared_words.QuestionVector().load()[1]
            dvecs = distances.AllDistances().load()[1]
            labels = dataset.Dataset().load()[1].is_duplicate.values
        else:
            vecs = self.data_source.load_test()
            qvecs = shared_words.QuestionVector().load_test()
            dvecs = distances.AllDistances().load_test()
            labels = np.zeros(qvecs.shape[0], dtype='uint8')

        qvec_offset = qvecs.shape[1]
        dvec_offset = dvecs.shape[1]
        both_offset = qvec_offset + dvec_offset

        def write_row(i, f):
            row = vecs[i]
            qvec = qvecs[i] * 100
            dvec = dvecs[i] * 100
            label = labels[i]
            qvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(qvec))
            dvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(dvec, start=qvec_offset))
            entries = " ".join(("%d:%.2f" % (ind + both_offset, data) for ind, data in zip(row.indices, row.data)))
            f.write("%d %s %s %s\n" % (label, qvec_entries, dvec_entries, entries))

        os.makedirs('cache/svm_data', exist_ok=True)
        if self.data_subset == 'test':
            for start_ix in tqdm(self.test_target_indexes(vecs.shape[0])):
                with open('cache/svm_data/test_%d.svm' % start_ix, 'w') as f:
                    for i in range(start_ix, min(start_ix+self.max_size, vecs.shape[0])):
                        write_row(i, f)
        else:
            with open('cache/svm_data/%s.svm' % self.data_subset, 'w') as f:
                for i in tqdm(range(qvecs.shape[0]), desc='writing %s data' % self.data_subset):
                    write_row(i, f)

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
    data_subset = 'valid'

class TestSVMData(SVMData):
    data_subset = 'test'


class GBMClassifier(luigi.Task):
    lightgbm_path = luigi.Parameter(default='/Users/richardweiss/Downloads/LightGBM/lightgbm')

    def requires(self):
        yield TrainSVMData()
        yield ValidSVMData()
        yield TestSVMData()
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/lightgbm/classifier_pred.csv.gz')

    def train(self):
        self.output().makedirs()
        print(colors.green & colors.bold | "Starting training")
        with open('cache/lightgbm/train_gbmclassifier.conf', 'w') as f:
            f.write(self.train_conf)
        local[self.lightgbm_path]['config=cache/lightgbm/train_gbmclassifier.conf'] & FG
        print(colors.green & colors.bold | "Finished training")

    def valid(self):
        with open('cache/lightgbm/valid_gbmclassifier.conf', 'w') as f:
            f.write(self.valid_conf)
        local[self.lightgbm_path]['config=cache/lightgbm/valid_gbmclassifier.conf'] & FG
        print(colors.green & colors.bold | "Finished validation predictions")
        pred = pandas.read_csv('./cache/lightgbm/valid_preds.csv', names=['is_duplicate'])
        pred.index = pred.index.rename('test_id')

        print(colors.green | "prediction sample...")
        print(colors.green | str(pred.head()))
        y = dataset.Dataset().load()[1]
        print(colors.green | "Performance: " + str(metrics.log_loss(y.is_duplicate, pred.is_duplicate)))

        return pred

    def test(self):
        test_size = dataset.Dataset().load_test().shape[0]
        test_tasks = SVMData.test_target_indexes(test_size)
        print(colors.green & colors.bold | "Predicting test values, this takes a long time...")
        for target_ix in tqdm(test_tasks, desc='Predicting'):
            with open('cache/lightgbm/test_gbmclassifier.conf', 'w') as f:
                f.write(self.test_conf % (target_ix, target_ix))
            local[self.lightgbm_path]['config=cache/lightgbm/test_gbmclassifier.conf'] & FG

        preds = []
        for target_ix in tqdm(test_tasks, desc='Reading results file'):
            pred = pandas.read_csv('./cache/lightgbm/test_preds_%d.csv' % target_ix, names=['is_duplicate'])
            pred.index = pandas.Series(np.arange(target_ix,
                                                 min(test_size, target_ix + SVMData.max_size)), name='test_id')
            preds.append(pred)
        preds = pandas.concat(preds, 0)
        return preds

    def run(self):
        self.train()
        self.valid()
        pred = self.test()

        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            pred.to_csv(tf.name, compression='gzip')
            os.rename(tf.name, self.output().path)
        except:
            os.remove(tf.name)
            raise

    train_conf = """
    task = train
    boosting_type = gbdt
    objective = binary
    metric = binary_logloss
    metric_freq = 5
    is_training_metric = true
    early_stopping_round = 10

    data=cache/svm_data/train.svm
    valid_data=cache/svm_data/valid.svm
    output_model=cache/lightgbm/gbm_model

    learning_rate = 0.1
    num_trees = 1000
    num_leaves = 1023

    bagging_freq = 3
    feature_fraction = 0.8
    bagging_fraction = 0.8

    min_data_in_leaf = 20
    is_enable_sparse = true
    use_two_round_loading = false
    is_save_binary_file = false
    """

    valid_conf = """
    task = predict
    data = cache/svm_data/valid.svm
    input_model=cache/lightgbm/gbm_model
    output_result=cache/lightgbm/valid_preds.csv
    """

    test_conf = """
    task = predict
    data = cache/svm_data/test_%d.svm
    input_model = cache/lightgbm/gbm_model
    output_result = cache/lightgbm/test_preds_%d.csv
    """


class XGBlassifier(luigi.Task):
    xgb_path = luigi.Parameter(default='/Users/richardweiss/Downloads/xgboost/xgboost')

    def requires(self):
        yield TrainSVMData()
        yield ValidSVMData()
        yield TestSVMData()
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/xgb/classifier_pred.csv.gz')

    def train(self):
        self.output().makedirs()
        print(colors.green & colors.bold | "Starting training")
        with open('cache/xgb/train.conf', 'w') as f:
            f.write(self.train_conf)
        local[self.xgb_path]['cache/xgb/train.conf'] & FG
        print(colors.green & colors.bold | "Finished training")

    train_conf = """
    booster = gbtree
    objective = binary:logistic
    
    eta = 0.1
    gamma = 1.0
    min_child_weight = 1
    max_depth = 7
    
    #rate_drop=0.25
    #skip_drop=0.1
    
    subsample=0.8
    
    num_round = 50
    save_period = 0
    data = "cache/svm_data/train.svm"
    eval[test] = "cache/svm_data/valid.svm"
    model_out = "cache/xgb/model"
    nthread=4
    """

    def valid(self):
        with open('cache/xgb/valid.conf', 'w') as f:
            f.write(self.valid_conf)
        local[self.xgb_path]['cache/xgb/valid.conf'] & FG
        print(colors.green & colors.bold | "Finished validation predictions")
        pred = pandas.read_csv('./cache/xgb/valid_preds.csv', names=['is_duplicate'])
        pred.index = pred.index.rename('test_id')

        print(colors.green | "prediction sample...")
        print(colors.green | str(pred.head()))
        y = dataset.Dataset().load()[1]
        print(colors.green | "Performance: " + str(metrics.log_loss(y.is_duplicate, pred.is_duplicate)))

        return pred

    valid_conf = """
    task = pred
    model_in = "cache/xgb/model"
    test:data = "cache/svm_data/valid.svm"
    name_pred = "cache/xgb/valid_preds.csv"
    """

    def test(self):
        test_size = dataset.Dataset().load_test().shape[0]
        test_tasks = SVMData.test_target_indexes(test_size)
        print(colors.green & colors.bold | "Predicting test values, this takes a long time...")
        for target_ix in tqdm(test_tasks, desc='Predicting'):
            with open('cache/xgb/test.conf', 'w') as f:
                f.write(self.test_conf % (target_ix, target_ix))
            local[self.xgb_path]['cache/xgb/test.conf'] & FG

        preds = []
        for target_ix in tqdm(test_tasks, desc='Reading results file'):
            pred = pandas.read_csv('./cache/xgb/test_preds_%d.csv' % target_ix, names=['is_duplicate'])
            pred.index = pandas.Series(
                np.arange(target_ix, min(test_size, target_ix + SVMData.max_size)),
                name='test_id')
            preds.append(pred)
        preds = pandas.concat(preds, 0)
        return preds

    test_conf = """
        task = pred
        model_in = "cache/xgb/model"
        test:data = "cache/svm_data/test_%d.svm"
        name_pred = "cache/xgb/test_preds_%d.csv"
        """

    def run(self):
        self.train()
        self.valid()
        pred = self.test()

        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            pred.to_csv(tf.name, compression='gzip')
            os.rename(tf.name, self.output().path)
        except:
            os.remove(tf.name)
            raise
