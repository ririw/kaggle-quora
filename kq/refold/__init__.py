import os

import luigi

refold_max_folds = 6

class BaseTargetBuilder:
    def __init__(self, *parts, add_rf_cache=True):
        if add_rf_cache:
            self.parts = ['rf_cache'] + list(parts)
        else:
            self.parts = list(parts)

    def get(self):
        return os.path.join(*self.parts)

    def __add__(self, part):
        return BaseTargetBuilder(*(self.parts + [part]), add_rf_cache=False)