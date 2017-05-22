import hyperopt
import luigi
import numpy as np
import pandas
import sklearn.linear_model
from plumbum import colors

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder

__all__ = ['WordCountLogit']

class WordCountLogit(FoldDependent):
    resources = {'cpu': 2, 'mem': 4}
    ngram_max = hyper_helper.TuneableHyperparam(
        name='WordCountLogit_ngram_mac',
        prior=hyperopt.hp.randint('WordCountLogit_ngram_mac', 2),
        default=1,
        transform=lambda v: v+1
    )
    ngram_min_df = hyper_helper.TuneableHyperparam(
        name='WordCountXTC_min_df',
        prior=hyperopt.hp.randint('WordCountXTC_min_df', 10),
        default=6,
        disable=True,
        transform=lambda v: 0.1**(v / 2)
    )

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_wc_logit',
            'ng_{:d}_mindf_{:f}'.format(self.ngram_max.get(), self.ngram_min_df.get()),
            str(self.fold)
        )
        return (base_path + fname).get()

    def _load(self, name, as_df):
        path = self.make_path(name + '.npz')
        if as_df:
            return pandas.DataFrame({'WordCountLogit': np.load(path)['data']})
        else:
            return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_word_count_features.WordCountMatrix(
            ngram_max=self.ngram_max.get(),
            ngram_min_df=self.ngram_min_df.get())

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        wcm = rf_word_count_features.WordCountMatrix(
            ngram_max=self.ngram_max.get(),
            ngram_min_df=self.ngram_min_df.get())

        X = wcm.load('train', self.fold)
        y = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate

        cls = sklearn.linear_model.LogisticRegression(solver='sag')
        cls.fit(X, y)

        X_val = wcm.load('valid', self.fold)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)
        print('Score: {:s}: {:f}'.format(repr(self), score))

        del X, y, X_val, y_val
        X_test = wcm.load('test', None)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score
