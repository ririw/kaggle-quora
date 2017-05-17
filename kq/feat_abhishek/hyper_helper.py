import luigi


class TuneableHyperparam(luigi.Parameter):
    pass


class TuneableFloatHyperparam(luigi.FloatParameter):
    pass


class TuneableIntHyperparam(luigi.FloatParameter):
    pass


