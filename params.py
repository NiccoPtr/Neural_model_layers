# %%

from parameter_manager import ParameterManager

class Parameters(ParameterManager):
    def __init__(self, bg_n=2):
        self.bg_n = bg_n
        super(Parameters, self).__init__()


