

class TuneableHyperparam:
    opt_args = {}
    def __init__(self, name):
        self.name = name

    def get(self):
        return self.opt_args[self.name]

    @classmethod
    def set(cls, args):
        cls.opt_args = args

