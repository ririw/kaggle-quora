import luigi
import numpy as np
import pandas
from sklearn import ensemble, metrics
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from kq import dataset


class SentimentTask(luigi.Task):
    dataset = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget('cache/sentiment/%s.npy' % self.dataset)

    def requires(self):
        return dataset.Dataset()

    def calc_sent(self, dataset, task):
        res = np.zeros([dataset.shape[0], 4])
        for ix, q in tqdm(enumerate(dataset['question1_raw']), total=dataset.shape[0], desc=task + '-q1'):
            sents = self.analyzer.polarity_scores(q)
            res[ix, 0] = sents['neg']
            res[ix, 1] = sents['neu']
            res[ix, 2] = sents['pos']
            res[ix, 3] = sents['compound']
        for ix, q in tqdm(enumerate(dataset['question2_raw']), total=dataset.shape[0], desc=task + '-q2'):
            sents = self.analyzer.polarity_scores(q)
            res[ix, 0] -= sents['neg']
            res[ix, 1] -= sents['neu']
            res[ix, 2] -= sents['pos']
            res[ix, 3] -= sents['compound']
        return res

    def run(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.output().makedirs()

        data = dataset.Dataset().load_named(self.dataset)
        data_sent = self.calc_sent(data, self.dataset)
        np.save('cache/sentiment/{}.npy'.format(self.dataset), data_sent)


class SentimentTasks(luigi.Task):
    def output(self):
        return luigi.LocalTarget('cache/sentiment/done')

    def requires(self):
        yield SentimentTask(dataset='train')
        yield SentimentTask(dataset='test')
        yield SentimentTask(dataset='valid')
        yield SentimentTask(dataset='merge')

    def run(self):
        X = np.load('cache/sentiment/valid.npy')
        y = dataset.Dataset().load_named('valid').is_duplicate.values.astype(int)
        X_test = np.load('cache/sentiment/merge.npy')
        y_test = dataset.Dataset().load_named('merge').is_duplicate.values.astype(int)
        summary_cls = ensemble.ExtraTreesClassifier(n_estimators=200,
                                                    n_jobs=-1)
        summary_cls.fit(X, y)
        perf = summary_cls.predict_proba(X_test)
        importances = pandas.Series(
            summary_cls.feature_importances_,
            index=['neg', 'neu', 'pos', 'compound'])
        score = metrics.log_loss(y_test, perf)

        print(score)
        print(importances)
        with self.output().open('w') as f:
            f.write(str(score))
            f.write('\n')
            f.write(str(importances))

    def load_named(self, name):
        return np.load('cache/sentiment/{}.npy'.format(name), mmap_mode='r')


