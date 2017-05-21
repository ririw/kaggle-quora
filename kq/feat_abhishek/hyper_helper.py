import luigi


class TuneableHyperparam:
    opt_args = {}
    def __init__(self, name, prior, default, transform=None, disable=False):
        self.default = default
        self.name = name
        self.prior = prior
        self.transform = transform
        self.disabled = disable

    def get(self):
        assert self.default is not None, 'No default value'
        res = self.opt_args.get(self.name, self.default)
        if self.transform:
            res = self.transform(res)
        return res

    @classmethod
    def set(cls, args):
        cls.opt_args = args


class LuigiTuneableHyperparam(luigi.Parameter):
    def __init__(self, prior, default, transform=None, disable=False):
        super().__init__(default=default)
        self.prior = prior
        self.transform = transform
        self.disable = disable