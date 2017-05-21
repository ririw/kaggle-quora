import hyperopt
import luigi
import numpy as np
import sklearn.linear_model
from plumbum import colors

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder


class WordCountLogit(FoldDependent):
    resources = {'cpu': 1}
    ngram_max = hyper_helper.TuneableHyperparam(
        name='WordCountLogit_ngram_mac',
        prior=hyperopt.hp.randint('WordCountLogit_ngram_mac', 3),
        default=2,
        transform=lambda v: v+1
    )
    ngram_min_df = hyper_helper.TuneableHyperparam(
        name='WordCountLogit_min_df',
        prior=hyperopt.hp.uniform('WordCountLogit_min_df', 0, 0.1),
        default=0.001
    )

    def make_path(self, fname):
        print(self.ngram_max.get(), type(self.ngram_max.get()))
        print(self.ngram_min_df.get(), type(self.ngram_min_df.get()))
        base_path = BaseTargetBuilder(
            'rf_wc_logit',
            'ng_{:d}_mindf_{:f}'.format(self.ngram_max.get(), self.ngram_min_df.get()),
            str(self.fold)
        )
        return (base_path + fname).get()

    def _load(self, name):
        path = self.make_path(name + '.npz')
        return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_word_count_features.WordCountMatrix(ngram_max=self.ngram_max.get(), ngram_min_df=self.ngram_min_df.get())

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        X = rf_word_count_features.WordCountMatrix().load('train', self.fold, as_np=False)
        y = rf_dataset.Dataset().load('train', self.fold, as_np=False).is_duplicate

        cls = sklearn.linear_model.LogisticRegression(solver='sag')
        cls.fit(X, y)

        X_val = rf_word_count_features.WordCountMatrix().load('valid', self.fold, as_np=False)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_np=False).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = rf_word_count_features.WordCountMatrix().load('test', None, as_np=False)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score
