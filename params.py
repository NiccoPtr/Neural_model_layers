"""
Created on Sat Aug 16 11:50:44 2025

@author: Utente
"""

from parameter_manager import ParameterManager

class Parameters(ParameterManager):
    def __init__(self, N, tau, baseline, BG_dl_W, seed, noise, Matrices_scalars, DA_values):
        
        self.N = N
        self.tau = tau
        self.baseline = baseline
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
                        tau = {"BG_dl": 10, "MGV": 10, "MC": 5, "BLA_IC": [10, 10]},
                        baseline = {"DLS": 0.0, "STNdl": 0.0, "GPi": 0.8, "MGV": 0.0, "MC": 0.0, "BLA_IC": [0.0, 0.0]},
                        BG_dl_W = {"DLS_GPi_W": 2.4, "STNdl_GPi_W": 1.8},
                        seed = 2,
                        noise = {"BG_dl": 0.0, "MGV": 0.0, "MC": 0.05, "BLA_IC": 0.0},
                        Matrices_scalars = {"BGdl_MGV": 1.5, "MGV_MC": 1.5, "MC_MGV": 0.8, "MC_STNdl": 1.6, "MC_DLS": 1.0},
                        DA_values = {"Y": 0.1, "delta": 4.9, "DA": 1.0}
                        )

parameters.update(param_string)
parameters.save(param_file + ".json", mode="json")