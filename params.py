# %%

from parameter_manager import ParameterManager

class Parameters(ParameterManager):
    def __init__(self, N, alpha, threshold, ö, alpha_uo, alpha_ui):
        self.N = N
        self.alpha = alpha
        self.threshold = threshold
        self.ö = ö
        self.alpha_uo = alpha_uo
        self.alpha_ui = alpha_ui
        super(Parameters, self).__init__()
        


