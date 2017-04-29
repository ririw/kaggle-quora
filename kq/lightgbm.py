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


class SVMData(luigi.Task):
    data_subset = None  # train, test, merge or valid
    max_size = 100000

    def requires(self):
        yield dataset.Dataset()
        yield shared_words.QuestionVector()
        yield distances.AllDistances()
        yield shared_entites.SharedEntities()
        yield tfidf_matrix.TFIDFFeature()
        yield wordmat_distance.WordMatDistance()

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

        def write_row(i, f1, f2, f3):
            row = vecs[i]
            qvec = np.nan_to_num(qvecs[i])
            dvec = np.nan_to_num(dvecs[i])
            evec = np.nan_to_num(evecs[i])
            wmvec = np.nan_to_num(wmvecs[i])
            label = labels[i]

            qvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(qvec, start=qvec_offset))
            dvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(dvec, start=dvec_offset))
            evec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(evec, start=evec_offset))
            wmvec_entries = ' '.join('%d:%.2f' % ix_v for ix_v in enumerate(wmvec, start=wmvec_offset))
            entries = " ".join(("%d:%.2f" % (ind + vecs_offset, data) for ind, data in zip(row.indices, row.data)))
            f1.write("%d %s %s %s %s %s\n" % (label, qvec_entries, dvec_entries, evec_entries, wmvec_entries, entries))
            f2.write('%d %s %s %s %s\n' % (label, qvec_entries, dvec_entries, evec_entries, wmvec_entries))
            f3.write('%d %s\n' % (label, wmvec_entries))

        os.makedirs('cache/svm_data', exist_ok=True)
        if self.data_subset == 'test':
            for start_ix in tqdm(self.test_target_indexes(vecs.shape[0])):
                with open('cache/svm_data/test_%d_tmp.svm' % start_ix, 'w') as f1, \
                     open('cache/svm_data/test_simple_%d_tmp.svm' % start_ix, 'w') as f2, \
                     open('cache/svm_data/test_words_%d_tmp.svm' % start_ix, 'w') as f3:
                    for i in range(start_ix, min(start_ix + self.max_size, vecs.shape[0])):
                        write_row(i, f1, f2, f3)
                os.rename('cache/svm_data/test_simple_%d_tmp.svm' % start_ix,
                          'cache/svm_data/test_simple_%d.svm' % start_ix)
                os.rename('cache/svm_data/test_words_%d_tmp.svm' % start_ix,
                          'cache/svm_data/test_words_%d.svm' % start_ix)
                os.rename('cache/svm_data/test_%d_tmp.svm' % start_ix, 'cache/svm_data/test_%d.svm' % start_ix)
        else:
            with open('cache/svm_data/%s_tmp.svm' % self.data_subset, 'w') as f1, \
                 open('cache/svm_data/%s_tmp_simple.svm' % self.data_subset, 'w') as f2, \
                 open('cache/svm_data/%s_tmp_words.svm' % self.data_subset, 'w') as f3:
                for i in tqdm(range(qvecs.shape[0]), desc='writing %s data' % self.data_subset):
                    write_row(i, f1, f2)
            os.rename('cache/svm_data/%s_tmp_simple.svm' % self.data_subset,
                      'cache/svm_data/%s_simple.svm' % self.data_subset)
            os.rename('cache/svm_data/%s_tmp_words.svm' % self.data_subset,
                      'cache/svm_data/%s_words.svm' % self.data_subset)
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
    is_simple = luigi.BoolParameter()

    def make_path(self, *rest):
        if self.is_simple:
            parts = ['cache', 'lightgbm', 'simple'] + list(rest)
        else:
            parts = ['cache', 'lightgbm', 'complex'] + list(rest)
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
            f.write(self.train_conf)
        local[self.lightgbm_path]['config=' + self.make_path('train_gbm_classifier.conf')] & FG
        print(colors.green & colors.bold | "Finished training")

    def pred_simple_target(self, dataset):
        with open(self.make_path('pred.conf'), 'w') as f:
            f.write(self.valid_conf % (dataset, self.make_path(), self.make_path()))

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
                f.write(self.test_conf % (target_ix, self.make_path(), target_ix, self.make_path()))
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
    boosting_type = gbdt
    objective = binary
    metric = binary_logloss
    metric_freq = 5
    is_training_metric = true
    early_stopping_round = 10

    data=cache/svm_data/train.svm
    valid_data=cache/svm_data/valid.svm
    output_model=cache/lightgbm/gbm_model

    learning_rate = 0.05
    num_trees = 1000
    num_leaves = 1500
    scale_pos_weight = 0.46
    min_data_in_leaf = 25

    is_enable_sparse = true
    use_two_round_loading = false
    is_save_binary_file = false
    """

    valid_conf = """
    task = predict
    data = cache/svm_data/%s.svm
    input_model=%s/gbm_model
    output_result=%s/preds.csv
    """

    test_conf = """
    task = predict
    data = cache/svm_data/test_%d.svm
    input_model = %s/gbm_model
    output_result = %s/test_preds_%d.csv
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
    #eval_metric=logloss
    
    eta = 0.1
    max_depth = 7
    scale_pos_weight=0.46
    early_stop_round = 10
    
    num_round = 250
    save_period = 0
    data = "cache/svm_data/train_simple.svm"
    eval[test] = "cache/svm_data/valid_simple.svm"
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
    test:data = "cache/svm_data/%s_simple.svm"
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
        test:data = "cache/svm_data/test_simple_%d.svm"
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

    def load(self):
        assert self.complete()
        res = pandas.read_csv('cache/xgb/merge_predictions.csv', index_col='test_id')
        return res

    def load_test(self):
        assert self.complete()
        return pandas.read_csv('cache/xgb/classifier_pred.csv.gz', index_col='test_id').values
