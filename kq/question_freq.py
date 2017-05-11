import luigi
import numpy as np
import pandas

from kq import dataset


class QuestionFrequencyFeature(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/question-freq/done')

    def run(self):
        self.output().makedirs()
        train_data = dataset.Dataset().load_named('train')
        merge_data = dataset.Dataset().load_named('merge')
        valid_data = dataset.Dataset().load_named('valid')
        test_data = dataset.Dataset().load_named('test')
        all_questions = pandas.concat([
            train_data.question1_raw, train_data.question2_raw,
            merge_data.question1_raw, merge_data.question2_raw,
            valid_data.question1_raw, valid_data.question2_raw,
            test_data.question1_raw, test_data.question2_raw,
        ])

        question_freq = all_questions.value_counts().to_dict()

        train_feature = pandas.DataFrame({
            'freq1': train_data.question1_raw.apply(question_freq.get),
            'freq2': train_data.question2_raw.apply(question_freq.get)
        })
        train_feature['qfreq_diff'] = np.abs(train_feature.freq1 - train_feature.freq2)
        np.save('cache/question-freq/train.npy', train_feature.values)

        merge_feature = pandas.DataFrame({
            'freq1': merge_data.question1_raw.apply(question_freq.get),
            'freq2': merge_data.question2_raw.apply(question_freq.get)
        })
        merge_feature['qfreq_diff'] = np.abs(merge_feature.freq1 - merge_feature.freq2)
        np.save('cache/question-freq/merge.npy', merge_feature.values)

        valid_feature = pandas.DataFrame({
            'freq1': valid_data.question1_raw.apply(question_freq.get),
            'freq2': valid_data.question2_raw.apply(question_freq.get)
        })
        valid_feature['qfreq_diff'] = np.abs(valid_feature.freq1 - valid_feature.freq2)
        np.save('cache/question-freq/valid.npy', valid_feature.values)

        test_feature = pandas.DataFrame({
            'freq1': test_data.question1_raw.apply(question_freq.get),
            'freq2': test_data.question2_raw.apply(question_freq.get)
        })
        test_feature['qfreq_diff'] = np.abs(test_feature.freq1 - test_feature.freq2)
        np.save('cache/question-freq/test.npy', test_feature.values)

        with self.output().open('w') as f:
            f.write(str(train_feature.groupby(train_data.is_duplicate).qfreq_diff.mean()))

    def load_named(self, name):
        assert self.complete()
        assert name in {'train', 'test', 'merge', 'valid'}
        return np.load('cache/question-freq/%s.npy' % name, mmap_mode='r')
