import tempfile
import os
from glob import glob

import luigi
import pandas
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from plumbum import local, FG, colors

from kq import dataset, shared_words, distances, shared_entites, core


class SVMData(luigi.Task):
    data_subset = None  # train, test, merge or valid
    data_source = shared_words.WordVectors()
    max_size = 100000

    def requires(self):
        yield dataset.Dataset()
        yield shared_words.QuestionVector()
        for req in distances.AllDistances.requires():
            yield req
        yield shared_entites.SharedEntities()
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
        if self.data_subset in {'train', 'valid', 'merge'}:
            ix = {'train': 0, 'merge': 1, 'valid': 2}[self.data_subset]
            vecs = self.data_source.load()[ix]
            vecs = vecs[:, :vecs.shape[1]//2]
            qvecs = shared_words.QuestionVector().load()[ix]
            dvecs = distances.AllDistances().load()[ix]
            evecs = shared_entites.SharedEntities().load()[ix]
            labels = dataset.Dataset().load()[ix].is_duplicate.values
            print('vecs: ' + str(vecs.shape))
            print('qvecs: ' + str(qvecs.shape))
            print('dvecs: ' + str(dvecs.shape))
            print('evecs: ' + str(evecs.shape))
            print('labels: ' + str(labels.shape))
        else:
            vecs = self.data_source.load_test()
            qvecs = shared_words.QuestionVector().load_test()
            dvecs = distances.AllDistances().load_test()
            evecs = shared_entites.SharedEntities().load_test()
            labels = np.zeros(qvecs.shape[0], dtype='uint8')

        qvec_offset = 1
        dvec_offset = qvecs.shape[1]
        evec_offset = dvec_offset + dvecs.shape[1]
        vecs_offset = evec_offset + evecs.shape[1]

        def write_row(i, f):
            row = vecs[i]
            qvec = np.nan_to_num(qvecs[i])
            dvec = np.nan_to_num(dvecs[i])
            evec = np.nan_to_num(evecs[i])
            label = labels[i]

            qvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(qvec, start=qvec_offset))
            dvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(dvec, start=dvec_offset))
            evec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(evec, start=evec_offset))
            entries = " ".join(("%d:%.2f" % (ind + vecs_offset, data) for ind, data in zip(row.indices, row.data)))
            f.write("%d %s %s %s %s\n" % (label, qvec_entries, dvec_entries, evec_entries, entries))
            #f.write("%d %s %s\n" % (label, qvec_entries, dvec_entries))

        os.makedirs('cache/svm_data', exist_ok=True)
        if self.data_subset == 'test':
            for start_ix in tqdm(self.test_target_indexes(vecs.shape[0])):
                with open('cache/svm_data/test_%d_tmp.svm' % start_ix, 'w') as f:
                    for i in range(start_ix, min(start_ix+self.max_size, vecs.shape[0])):
                        write_row(i, f)
                os.rename('cache/svm_data/test_%d_tmp.svm' % start_ix, 'cache/svm_data/test_%d.svm' % start_ix)
        else:
            with open('cache/svm_data/%s_tmp.svm' % self.data_subset, 'w') as f:
                for i in tqdm(range(qvecs.shape[0]), desc='writing %s data' % self.data_subset):
                    write_row(i, f)
            os.rename('cache/svm_data/%s_tmp.svm' % self.data_subset, 'cache/svm_data/%s.svm' % self.data_subset)

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
    lightgbm_path = luigi.Parameter(default='/Users/richardweiss/Downloads/LightGBM/lightgbm')

    def requires(self):
        yield TrainSVMData()
        yield ValidSVMData()
        yield MergeSVMData()
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

    def pred_simple_target(self, dataset):
        with open('cache/lightgbm/pred.conf', 'w') as f:
            f.write(self.valid_conf % dataset)

        local[self.lightgbm_path]['config=cache/lightgbm/pred.conf'] & FG
        pred = pandas.read_csv('./cache/lightgbm/preds.csv', names=['is_duplicate'])
        pred.index = pred.index.rename('test_id')
        return pred

    def merge(self):
        pred = self.pred_simple_target('valid')
        pred.to_csv('cache/lightgbm/merge_predictions.csv')

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

    learning_rate = 0.01
    num_trees = 1000
    num_leaves = 512
    scale_pos_weight = 0.46
    is_unbalance=true
    lambda_l1=0.00005
    lambda_l2=0.00005

    bagging_freq = 3
    bagging_fraction = 0.8

    is_enable_sparse = true
    use_two_round_loading = false
    is_save_binary_file = false
    """

    valid_conf = """
    task = predict
    data = cache/svm_data/%s.svm
    input_model=cache/lightgbm/gbm_model
    output_result=cache/lightgbm/preds.csv
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
        yield MergeSVMData()
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
    eval_metric=logloss
    
    eta = 0.3
    max_depth = 4
    scale_pos_weight=0.46
    
    subsample=0.8
    
    num_round = 1000
    save_period = 0
    data = "cache/svm_data/train.svm"
    eval[test] = "cache/svm_data/valid.svm"
    model_out = "cache/xgb/model"
    nthread=4
    """

    def pred_simple_target(self, dataset):
        with open('cache/xgb/pred.conf', 'w') as f:
            f.write(self.valid_conf % dataset)

        local[self.xgb_path]['cache/xgb/pred.conf'] & FG
        pred = pandas.read_csv('./cache/xgb/preds.csv', names=['is_duplicate'])
        pred.index = pred.index.rename('test_id')
        return pred


    def valid(self):
        pred = self.pred_simple_target('valid')
        print(colors.green | "prediction sample...")
        print(colors.green | str(pred.head()))
        y = dataset.Dataset().load()[2]
        weights = core.weights[y.is_duplicate.values]
        loss = metrics.log_loss(y.is_duplicate, pred.is_duplicate, sample_weight=weights)
        print(colors.green | "Performance: " + str(loss))

        return pred

    def merge(self):
        pred = self.pred_simple_target('merge')
        pred.to_csv('cache/xgb/merge_predictions.csv')

    valid_conf = """
    task = pred
    model_in = "cache/xgb/model"
    test:data = "cache/svm_data/%s.svm"
    name_pred = "cache/xgb/preds.csv"
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
