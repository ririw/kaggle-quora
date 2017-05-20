from kq.feat_abhishek import hyper_helper
import os

import luigi
import plumbum.cli as cli
import hyperopt

from kq.feat_abhishek import logit

class HyperoptRunner(cli.Application):
    def main(self):
        def objective(args):
            hyper_helper.TuneableHyperparam.set(args)
            task = logit.LogitClassifier(fold=0)
            task.run()

        space = {
            'LogitClassifier_C': hyperopt.hp.uniform('LogitClassifier_C', 0.0001, 10)
        }
        best = hyperopt.fmin(objective, space, hyperopt.tpe.suggest, max_evals=25)
        print(best)

if __name__ == '__main__':
    HyperoptRunner.run()