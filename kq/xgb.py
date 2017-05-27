"""
XGB experiments log

 - Simple dataset + small depth vs Simple dataset + deep depth
"""

import os
import tempfile

import luigi
import numpy as np
import pandas
from plumbum import local, FG, colors
from sklearn import metrics
from tqdm import tqdm

from kq import dataset, core, lightgbm


class XGBlassifier(luigi.Task):
    resources = {'cpu': 7}

    xgb_path = luigi.Parameter(default=os.path.expanduser('~/Downloads/xgboost/xgboost'))

    def requires(self):
        yield lightgbm.TrainSVMData()
        yield lightgbm.ValidSVMData()
        yield lightgbm.MergeSVMData()
        yield lightgbm.TestSVMData()
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
    max_depth = 6
    scale_pos_weight=0.46
    early_stop_round = 10

    num_round = 1000
    save_period = 0
    data = "cache/svm_data/simple/train.svm"
    eval[test] = "cache/svm_data/simple/valid.svm"
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
    test:data = "cache/svm_data/simple/%s.svm"
    name_pred = "cache/xgb/preds.csv"
    """

    def test(self):
        test_size = dataset.Dataset().load_test().shape[0]
        test_tasks = lightgbm.SVMData.test_target_indexes(test_size)
        print(colors.green & colors.bold | "Predicting test values, this takes a long time...")
        for target_ix in tqdm(test_tasks, desc='Predicting'):
            with open('cache/xgb/test.conf', 'w') as f:
                f.write(self.test_conf % (target_ix, target_ix))
            local[self.xgb_path]['cache/xgb/test.conf'] & FG

        preds = []
        for target_ix in tqdm(test_tasks, desc='Reading results file'):
            pred = pandas.read_csv('./cache/xgb/test_preds_%d.csv' % target_ix, names=['is_duplicate'])
            pred.index = pandas.Series(
                np.arange(target_ix, min(test_size, target_ix + lightgbm.SVMData.max_size)),
                name='test_id')
            preds.append(pred)
        preds = pandas.concat(preds, 0)
        return preds

    test_conf = """
        task = pred
        model_in = "cache/xgb/model"
        test:data = "cache/svm_data/simple/test_%d.svm"
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
