
class TuneableHyperparam:
    opt_args = {}
    def __init__(self, name, prior, default, transform=None):
        self.default = default
        self.name = name
        self.prior = prior
        self.transform = transform

    def get(self):
        res = self.opt_args.get(self.name, self.default)
        if self.transform:
            res = self.transform(res)
        return res

    @classmethod
    def set(cls, args):
        cls.opt_args = args

