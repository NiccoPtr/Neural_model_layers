"""
Created on Sat Aug 16 11:50:44 2025

@author: Utente
"""

from parameter_manager import ParameterManager

class Parameters(ParameterManager):
    def __init__(self, N, alpha, alpha_uo, alpha_ui, baseline, baseline_uo, baseline_ui, BG_dl_W, seed, noise, Matrices_scalars, DA_values):
        
        self.N = N
        self.alpha = alpha
        self.alpha_uo = alpha_uo
        self.alpha_ui = alpha_ui
        self.baseline = baseline
        self.baseline_uo = baseline_uo
        self.baseline_ui = baseline_ui
        self.BG_dl_W = BG_dl_W
        self.seed = seed
        self.noise = noise
        self.Matrices_scalars = Matrices_scalars
        self.DA_values = DA_values
        
        super(Parameters, self).__init__()
        

param_string = "Simulation"
param_file = "prm_file"

parameters = Parameters(
                        N = 2,
                        alpha = {"BG_dl": 0.1, "MGV": 0.1, "MC": 0.2},
                        alpha_uo = 0.4,
                        alpha_ui = 0.1,
                        baseline = {"DLS": 0.0, "STNdl": 0.0, "GPi": 0.8, "MGV": 0.0, "MC": 0.0},
                        baseline_uo = 0.0,
                        baseline_ui = 0.0,
                        BG_dl_W = {"DLS_GPi_W": 2.4, "STNdl_GPi_W": 1.8},
                        seed = 2,
                        noise = {"BG_dl": 0.0, "MGV": 0.0, "MC": 0.05},
                        Matrices_scalars = {"BGdl_MGV": 1.5, "MGV_MC": 1.5, "MC_MGV": 0.8, "MC_STNdl": 1.6, "MC_DLS": 1.0},
                        DA_values = {"Y": 0.1, "delta": 0.9, "DA": 1}
                        )

parameters.update(param_string)
parameters.save(param_file + ".json", mode="json")