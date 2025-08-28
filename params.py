"""
Created on Sat Aug 16 11:50:44 2025

@author: Utente
"""

from parameter_manager import ParameterManager

class Parameters(ParameterManager):
    def __init__(self, N, alpha, alpha_uo, alpha_ui, baseline, baseline_MGV, baseline_uo, baseline_ui, DLS_GPi_W, STNdl_GPi_W, seed, noise, MGV_noise):
        
        self.N = N
        self.alpha = alpha
        self.alpha_uo = alpha_uo
        self.alpha_ui = alpha_ui
        self.baseline = baseline
        self.baseline_MGV = baseline_MGV
        self.baseline_uo = baseline_uo
        self.baseline_ui = baseline_ui
        self.DLS_GPi_W = DLS_GPi_W
        self.STNdl_GPi_W = STNdl_GPi_W
        self.seed = seed
        self.noise = noise
        self.MGV_noise = MGV_noise
        
        super(Parameters, self).__init__()
        

param_string = "Simulation"
param_file = "prm_file"
        
parameters = Parameters(
                        N = 2,
                        alpha = 0.2,
                        alpha_uo = 0.4,
                        alpha_ui = 0.1,
                        baseline = 0.1,
                        baseline_MGV = 0.4,
                        baseline_uo = 0.0,
                        baseline_ui = 0.0,
                        DLS_GPi_W = 1.0,
                        STNdl_GPi_W = 1.0,
                        seed = 2,
                        noise = 0.0,
                        MGV_noise = 1.0
                        )

parameters.update(param_string)
parameters.save(param_file + ".json", mode="json")