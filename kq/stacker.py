import luigi

from kq.lightgbm import XGBlassifier, GBMClassifier
from kq.vw import VWClassifier
from kq.word_nb import NaiveBayesClassifier


class Stacks(luigi.Task):
    def requires(self):
        yield XGBlassifier()
        yield GBMClassifier()
        yield VWClassifier()
        yield NaiveBayesClassifier()