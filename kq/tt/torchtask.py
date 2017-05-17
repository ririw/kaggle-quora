import os

import luigi
import torchsample as ts
import torchsample.initializers
import torchsample.metrics
import torch
import torch.autograd
from plumbum import colors
import numpy as np

import kq
import kq.core
from kq.feature_collection import FeatureCollection
from kq.keras_kaggle_data import KaggleDataset
from . import to_torchvar


class TorchTask(luigi.Task):
    batch_size = 128
    def requires(self):
        yield KaggleDataset()
        yield FeatureCollection()

    @classmethod
    def makepath(cls, *bits):
        all_bits = ['cache', 'torchtask', cls.__name__] + list(bits)
        return os.path.join(*all_bits)

    def output(self):
        return luigi.LocalTarget(self.makepath('report'))

    def model(self, embedding_mat, vector_input_shape, otherfeature_shape, batch_size):
        raise NotImplementedError

    def run(self):
        self.output().makedirs()
        embedding = KaggleDataset().load_embedding()
        (train_x1, train_x2), train_y = KaggleDataset().load_named('train')
        train_features = FeatureCollection().load_named('train').values.astype(np.float32)

        (valid_x1, valid_x2), valid_y = KaggleDataset().load_named('valid')
        valid_features = FeatureCollection().load_named('valid').values.astype(np.float32)

        model = self.model(embedding, train_x1.shape[1], train_features.shape[1], 128).cuda()
        assert isinstance(model, torch.nn.Module)

        trainer = ts.modules.ModuleTrainer(model)
        callbacks = [ts.callbacks.EarlyStopping(patience=5),
                     ts.callbacks.ModelCheckpoint(
                         self.makepath(),
                         verbose=1,
                         save_best_only=True),
                     ts.callbacks.ReduceLROnPlateau(factor=0.5,
                                                    patience=3,
                                                    verbose=1)]
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        trainer.compile(loss='binary_cross_entropy', optimizer=optimizer, callbacks=callbacks)

        train_stuff = [torch.from_numpy(train_x1).cuda(),
                       torch.from_numpy(train_x2).cuda(),
                       torch.from_numpy(train_features).cuda()]
        valid_stuff = ([torch.from_numpy(valid_x1).cuda(),
                        torch.from_numpy(valid_x2).cuda(),
                        torch.from_numpy(valid_features).cuda()],
                       torch.from_numpy(valid_y.astype(np.float32)).cuda())

        trainer.fit(train_stuff, torch.from_numpy(train_y.astype(np.float32)).cuda(),
                  validation_data=valid_stuff,
                  nb_epoch=200,
                  batch_size=self.batch_size,
                  verbose=1)

        with open(self.makepath('checkpoint.pth.tar'), 'rb') as f:
            model.load_state_dict(torch.load(f))

        preds = trainer.predict(valid_stuff[0]).cpu().data.numpy()
        model_perf = kq.core.score_data(valid_y, preds)
        print(colors.green | str(model_perf))

        del train_stuff, train_x1, train_x2, train_y, train_features
        del valid_stuff, valid_x1, valid_x2, valid_y, valid_features

        (merge_x1, merge_x2), merge_y = KaggleDataset().load_named('merge')
        merge_features = FeatureCollection().load_named('merge').values.astype(np.float32)

        merge_stuff = [torch.from_numpy(merge_x1), torch.from_numpy(merge_x2), torch.from_numpy(merge_features)]
        merge_pred = trainer.predict(merge_stuff).cpu().data.numpy()

        print(colors.yellow | 'Loading test features')
        (test_x1, test_x2), test_y = KaggleDataset().load_named('test', load_only=10000)
        test_features = FeatureCollection().load_named('test').values.astype(np.float32)[:10000]
        print(colors.yellow | colors.bold | 'Done!')

        test_pred = trainer.predict(
            [torch.from_numpy(v) for v in [test_x1, test_x2, test_features]],
            batch_size=self.batch_size,
            verbose=1
        ).cpu().data.numpy()

        np.save(self.makepath('merge.npy'), merge_pred)
        np.save(self.makepath('test.npy'), test_pred)

        with self.output().open('w') as f:
            f.write(str(model_perf))

    def load(self):
        assert self.complete()
        return np.load(self.makepath('merge.npy'))

    def load_test(self):
        assert self.complete()
        return np.load(self.makepath('test.npy'))