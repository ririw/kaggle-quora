import hyperopt
import luigi
import numpy as np
import pandas
from plumbum import colors
from xgboost import sklearn as xgbsk
from sklearn import model_selection

from kq import core, utils
from kq.feat_abhishek import hyper_helper
from . import FoldDependent, abhishek_feats, xval_dataset

__all__ = ['XGBoostClassifier']


class XGBoostClassifier(FoldDependent):
    resources = {'cpu': 7}

    max_depth = hyper_helper.TuneableHyperparam(
        name='XGBoostClassifier_max_depth',
        prior=hyperopt.hp.randint('XGBoostClassifier_max_depth', 12),
        default=10,
        transform=lambda x: x+1
    )
    eta = hyper_helper.TuneableHyperparam(
        name='XGBoostClassifier_eta',
        prior=hyperopt.hp.normal('XGBoostClassifier_eta', 0, 0.25),
        default=0.09948116387307111,
        transform=lambda x: np.abs(x)
    )
    n_est = hyper_helper.TuneableHyperparam(
        name='XGBoostClassifier_n_est',
        prior=hyperopt.hp.randint('XGBoostClassifier_n_est', 750),
        default=583,
        transform=lambda x: x + 100
    )

    def _load(self, name, as_df):
        assert name in {'test', 'valid'}
        fn = 'cache/abhishek/xgb/maxdepth_{:d}_eta_{:f}_nest_{:d}/{:d}/{:s}.npy'.format(
            self.max_depth.get(), self.eta.get(), self.n_est.get(), self.fold, name)
        if as_df:
            return pandas.Series(np.load(fn), name='XGBoost').to_frame()
        else:
            return np.load(fn)

    def output(self):
        fn = 'cache/abhishek/xgb/maxdepth_{:d}_eta_{:f}_nest_{:d}/{:d}/done'.format(
            self.max_depth.get(), self.eta.get(), self.n_est.get(), self.fold)

        return luigi.LocalTarget(fn)

    def requires(self):
        yield abhishek_feats.AbhishekFeatures()
        yield xval_dataset.BaseDataset()

    def run(self):
        self.output().makedirs()
        X = abhishek_feats.AbhishekFeatures().load('train', self.fold)
        y = xval_dataset.BaseDataset().load('train', self.fold).squeeze()
        cls = xgbsk.XGBClassifier(max_depth=self.max_depth.get(),
                                  learning_rate=self.eta.get(),
                                  n_estimators=self.n_est.get())
        X_tr, X_va, y_tr, y_va = model_selection.train_test_split(X, y, test_size=0.05)
        cls.fit(X_tr, y_tr, sample_weight=core.weight_from(y_tr), eval_set=[(X_va, y_va)], early_stopping_rounds=10)

        validX = abhishek_feats.AbhishekFeatures().load('valid', self.fold)
        y = xval_dataset.BaseDataset().load('valid', self.fold).squeeze()
        y_pred = cls.predict_proba(validX)[:, 1]
        score = core.score_data(y, y_pred)
        scorestr = "{:s} = {:f}".format(repr(self), score)
        print(colors.green | colors.bold | scorestr)

        valid_fn = 'cache/abhishek/xgb/maxdepth_{:d}_eta_{:f}_nest_{:d}/{:d}/valid.npy'.format(
            self.max_depth.get(), self.eta.get(), self.n_est.get(), self.fold)

        np.save(valid_fn, y_pred)

        trainX = abhishek_feats.AbhishekFeatures().load('test', None)
        pred = cls.predict_proba(trainX)[:, 1]

        test_fn = 'cache/abhishek/xgb/maxdepth_{:d}_eta_{:f}_nest_{:d}/{:d}/test.npy'.format(
            self.max_depth.get(), self.eta.get(), self.n_est.get(), self.fold)
        np.save(test_fn, pred)

        with self.output().open('w') as f:
            cols = abhishek_feats.AbhishekFeatures().load('valid', self.fold, as_df=True).columns
            v = pandas.Series(cls.feature_importances_, index=cols).sort_values()
            v.to_csv(f)
            f.write("\n\n")
            f.write(scorestr)
            f.write("\n")
        return score