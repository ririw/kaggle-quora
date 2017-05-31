import functools
from collections import Counter
from collections import defaultdict

from plumbum import colors
from sklearn import ensemble, model_selection

import luigi
import numpy as np
import os
import pandas
import pandas as pd
import nose.tools
import xgboost as xgb
import lightgbm.sklearn
from nltk.corpus import stopwords

from kq import core
from kq.core import score_data
from kq.feat_abhishek import FoldIndependent, FoldDependent
from kq.refold import BaseTargetBuilder, rf_dataset


def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1']) * 1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len(
        [x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(
        ''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True)  # 1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True)  # 2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)  # 3

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 8

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True)  # 10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True)  # 11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True)  # 12

    f = functools.partial(char_diff_unique_stop, stops=stops)
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 13

    #     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  # 16

    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)  # 17

    return X


class RF_LeakyXGB_Dataset(FoldIndependent):
    def _load(self, as_df):
        X = pandas.read_msgpack(self.make_path('train.msg'))
        if not as_df:
            X = X.values
        folds = rf_dataset.Dataset().load_dataset_folds()
        return X, folds

    def _load_test(self, as_df):
        X = pandas.read_msgpack(self.make_path('test.msg'))
        if not as_df:
            X = X.values
        return X

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path("done"))

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_leaky', 'dataset')
        return (base_path + fname).get()

    def run(self):
        self.output().makedirs()
        df_train = pd.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/train_features.csv'), encoding="ISO-8859-1")
        X_train_ab = df_train.iloc[:, 2:-1]
        X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
        X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)

        df_train = pd.read_csv('~/Datasets/Kaggle-Quora/train.csv')
        df_train = df_train.fillna(' ')

        df_test = pd.read_csv('~/Datasets/Kaggle-Quora/test.csv')
        ques = pd.concat([df_train[['question1', 'question2']],
                          df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
        q_dict = defaultdict(set)
        for i in range(ques.shape[0]):
            q_dict[ques.question1[i]].add(ques.question2[i])
            q_dict[ques.question2[i]].add(ques.question1[i])

        def q1_freq(row):
            return len(q_dict[row['question1']])

        def q2_freq(row):
            return len(q_dict[row['question2']])

        def q1_q2_intersect(row):
            return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))

        df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
        df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
        df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

        df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
        df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
        df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

        test_leaky = df_test.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]
        del df_test

        train_leaky = df_train.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]

        # explore
        stops = set(stopwords.words("english"))

        df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
        df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

        train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

        words = [x for y in train_qs for x in y]
        counts = Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}

        print('Building Features')
        X_train = build_features(df_train, stops, weights)
        X_train = pd.concat((X_train, X_train_ab, train_leaky), axis=1)

        X_train.to_msgpack(self.make_path('train.msg'))

        print('Building Test Features')
        df_test = pd.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/test_features.csv'), encoding="ISO-8859-1")
        x_test_ab = df_test.iloc[:, 2:-1]
        x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
        x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)

        df_test = pd.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/test.csv'))
        df_test = df_test.fillna(' ')

        df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
        df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

        x_test = build_features(df_test, stops, weights)
        x_test = pd.concat((x_test, x_test_ab, test_leaky), axis=1)

        x_test.to_msgpack(self.make_path('test.msg'))
        with self.output().open('w'):
            pass


class RFLeakingModel_XGB(FoldDependent):
    resources = {'cpu': 7}

    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            res = pandas.Series(res, name=repr(self))
        return res

    def requires(self):
        yield RF_LeakyXGB_Dataset()
        yield rf_dataset.Dataset()

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_leaky', 'xgb_model', str(self.fold))
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def run(self):
        # 0.131986896169
        self.output().makedirs()
        X_train = RF_LeakyXGB_Dataset().load('train', self.fold, as_df=True)
        y_train = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate
        X_valid = RF_LeakyXGB_Dataset().load('valid', self.fold, as_df=True)
        y_valid = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
        del pos_train, neg_train

        #pos_valid = X_valid[y_valid == 1]
        #neg_valid = X_valid[y_valid == 0]
        #X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
        #y_valid = np.array(
        #    [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
        #del pos_valid, neg_valid
        X_tr_tr, X_tr_es, y_tr_tr, y_tr_es = model_selection.train_test_split(X_train, y_train, test_size=0.05)

        d_train = xgb.DMatrix(X_tr_tr, label=y_tr_tr)
        d_es = xgb.DMatrix(X_tr_es, label=y_tr_es)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_es, 'd_es')]

        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.02
        params['max_depth'] = 7
        params['subsample'] = 0.6
        params['base_score'] = 0.2

        #bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
        bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=50)
        p_valid = bst.predict(d_valid)
        print(score_data(y_valid, p_valid, weighted=False))
        X_test = RF_LeakyXGB_Dataset().load('test', None, as_df=True)
        d_test = xgb.DMatrix(X_test)
        p_test = bst.predict(d_test)

        np.savez_compressed(self.make_path('done_tmp.npz'), valid=p_valid, test=p_test)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)


class RFLeakingModel_LGB(FoldDependent): # 0.147723
    resources = {'cpu': 7}

    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            res = pandas.Series(res, name=repr(self))
        return res

    def requires(self):
        yield RF_LeakyXGB_Dataset()
        yield rf_dataset.Dataset()

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_leaky', 'lgb', str(self.fold))
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def run(self):
        self.output().makedirs()
        X_train = RF_LeakyXGB_Dataset().load('train', self.fold, as_df=True)
        y_train = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate
        X_valid = RF_LeakyXGB_Dataset().load('valid', self.fold, as_df=True)
        y_valid = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
        del pos_train, neg_train

        #pos_valid = X_valid[y_valid == 1]
        #neg_valid = X_valid[y_valid == 0]
        #X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
        #y_valid = np.array(
        #    [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
        #del pos_valid, neg_valid

        cls = lightgbm.sklearn.LGBMClassifier(
            #n_estimators=3500,
            n_estimators=512,
            num_leaves=256,
            learning_rate=0.03,
            subsample=0.75
        )
        X_tr_tr, X_tr_es, y_tr_tr, y_tr_es = model_selection.train_test_split(X_train, y_train, test_size=0.05)
        cls.fit(X_tr_tr, y_tr_tr,
                eval_set=[(X_tr_es, y_tr_es)],
                early_stopping_rounds=50)
        valid_pred = cls.predict_proba(X_valid)[:, 1]
        print(colors.green | '{:s} == {:f}'.format(repr(self), score_data(y_valid, valid_pred, weighted=False)))
        print(colors.yellow | str(pandas.Series(cls.feature_importances_, index=X_train.columns).sort_values()))

        X_test = RF_LeakyXGB_Dataset().load('test', None, as_df=True).fillna(-999).clip(-1000, 1000)
        test_pred = cls.predict_proba(X_test)

        np.savez_compressed(self.make_path('done_tmp.npz'), valid=valid_pred, test=test_pred)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)


class RFLeakingModel_XTC(FoldDependent):
    resources = {'cpu': 7}

    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            res = pandas.Series(res, name=repr(self))
        return res

    def requires(self):
        yield RF_LeakyXGB_Dataset()
        yield rf_dataset.Dataset()

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_leaky', 'xtc_model', str(self.fold))
        return (base_path + fname).get()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def run(self):
        self.output().makedirs()
        X_train = RF_LeakyXGB_Dataset().load('train', self.fold, as_df=True).fillna(-999).clip(-1000, 1000)
        y_train = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate
        X_valid = RF_LeakyXGB_Dataset().load('valid', self.fold, as_df=True).fillna(-999).clip(-1000, 1000)
        y_valid = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
        del pos_train, neg_train

        #pos_valid = X_valid[y_valid == 1]
        #neg_valid = X_valid[y_valid == 0]
        #X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
        #y_valid = np.array(
        #    [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
        #del pos_valid, neg_valid

        cls = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=1024)
        cls.fit(X_train.values, y_train.values)

        valid_pred = cls.predict_proba(X_valid)[:, 1]
        print(colors.green | '{:s} == {:f}'.format(repr(self), score_data(y_valid, valid_pred)))
        print(colors.yellow | str(pandas.Series(cls.feature_importances_, index=X_train.columns).sort_values()))
        X_test = RF_LeakyXGB_Dataset().load('test', None, as_df=True).fillna(-999).clip(-1000, 1000)
        test_pred = cls.predict_proba(X_test.values)[:, 1]
        np.savez_compressed(self.make_path('done_tmp.npz'), valid=valid_pred, test=test_pred)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)


"""
    SAMPLING CODE
    # UPDownSampling
    pos_train = X_train[y_train == 1]
    neg_train = X_train[y_train == 0]
    X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
    y_train = np.array(
        [0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = X_valid[y_valid == 1]
    neg_valid = X_valid[y_valid == 0]
    X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array(
        [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid
"""
