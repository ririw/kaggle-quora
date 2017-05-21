import hyperopt
import luigi
import numpy as np
import sklearn.ensemble
from plumbum import colors

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder

__all__ = ['WordCountXTC']

class WordCountXTC(FoldDependent):
    resources = {'cpu': 7}
    ngram_max = hyper_helper.TuneableHyperparam(
        name='WordCountXTC_ngram_mac',
        prior=hyperopt.hp.randint('WordCountXTC_ngram_mac', 3),
        default=2,
        transform=lambda v: v+1,
        disable=True
    )

    ngram_min_df = hyper_helper.TuneableHyperparam(
        name='WordCountXTC_min_df',
        prior=hyperopt.hp.randint('WordCountXTC_min_df', 10),
        default=6,
        disable=True,
        transform=lambda v: 0.1**(v / 2)
    )

    min_leaf_samples = hyper_helper.TuneableHyperparam(
        name='WordCountXTC_min_leaf_samples',
        prior=hyperopt.hp.randint('WordCountXTC_min_leaf_samples', 20),
        default=2,
        transform=lambda v: (v+1)*5
    )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_xtc',
            'ng_{:d}_mindf_{:f}_ls_{:d}'.format(
                self.ngram_max.get(), self.ngram_min_df.get(), self.min_leaf_samples.get()),
            str(self.fold)
        )
        return (base_path + fname).get()

    def _load(self, name):
        path = self.make_path(name + '.npz')
        return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_word_count_features.WordCountMatrix(
            ngram_max=self.ngram_max.get(), ngram_min_df=self.ngram_min_df.get())

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        wcm = rf_word_count_features.WordCountMatrix(
            ngram_max=self.ngram_max.get(),
            ngram_min_df=self.ngram_min_df.get())
        X = wcm.load('train', self.fold, as_np=False)
        y = rf_dataset.Dataset().load('train', self.fold, as_np=False).is_duplicate

        cls = sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=500,
            verbose=10,
            n_jobs=-1,
            min_samples_leaf=self.min_leaf_samples.get())
        cls.fit(X, y)

        X_val = wcm.load('valid', self.fold, as_np=False)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_np=False).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)
        print('Score: {:s}: {:f}'.format(repr(self), score))

        del X, y, X_val, y_val
        X_test = wcm.load('test', None, as_np=False)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score
