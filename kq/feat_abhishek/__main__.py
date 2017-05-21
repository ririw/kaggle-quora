import logging
import numpy as np
import importlib
import pprint

from plumbum import colors

from kq.feat_abhishek import hyper_helper, FoldDependent
import os

import luigi
import plumbum.cli as cli
import hyperopt

from kq.feat_abhishek import logit

class HyperoptRunner(cli.Application):
    max_evals = cli.SwitchAttr('-e', argtype=int, help='Max evals', default=10)
    def all_tasks(self, root_task):
        subtasks = []
        for task in root_task.requires():
            subtasks.extend(self.all_tasks(task))
        subtasks.append(root_task)

        return subtasks

    def main(self, task):
        module_name = '.'.join(task.split('.')[:-1])
        cls_name = task.split('.')[-1]
        print('loading: {:s} from {:s}'.format(cls_name, module_name))
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        if issubclass(cls, FoldDependent):
            inst = cls(fold=0)
        else:
            inst = cls()
        space = {}
        for task in self.all_tasks(inst):
            for var in vars(task.__class__):
                v = getattr(task, var)
                if isinstance(v, hyper_helper.TuneableHyperparam):
                    space[v.name] = v.prior

        print(space)

        eval_count = 0

        def objective(args):
            nonlocal eval_count
            hyper_helper.TuneableHyperparam.set(args)
            if issubclass(cls, FoldDependent):
                inst = cls(fold=0)
            else:
                inst = cls()
            # Run each subtask, if it hasn't been finished alread,
            # all the way up to the last task (whic is the inst()
            # task
            for task in self.all_tasks(inst)[:-1]:
                if not task.complete():
                    task.run()
            # Run the inst task and get its score.
            # FIXME: consider what happens when a stacker has already
            # been run, but we change the hyperparams of the logit
            # underneath it. In this case, we'll think stacker has done
            # its work already, because it can't see that its requirements
            # have different params.
            res = inst.run()
            print(inst)
            print(colors.yellow | ('Hyperopt run {:d} of {:d}'.format(eval_count, self.max_evals)))
            print(colors.yellow | ('Hyperopt args: ' + pprint.pformat(args)))
            print(colors.yellow | 'Score: {:f}'.format(res))
            if np.isnan(res):
                logging.warning('Result was NaN, using 1000 instead')
                res = 1000
            return res

        best = hyperopt.fmin(objective, space, hyperopt.tpe.suggest, max_evals=self.max_evals)
        print(best)

if __name__ == '__main__':
    HyperoptRunner.run()