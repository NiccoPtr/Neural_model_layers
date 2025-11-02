# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:42:49 2025

@author: Utente
"""

"""
Architecture of Model's interconnections
"""

import numpy as np
from Simulation_file import set_DA, set_env
from Layer_types import *


def run_simulation(input_level_1, input_level_2, max_timesteps):
    
    BG_dl, MGV, MC, BLA_IC = set_layers()
    inputs, Ws = set_env(input_level_1, input_level_2, N=2)
    Y, delta, DA = set_DA()
    
    for inp in inputs:
        BG_dl.reset_activity()
        MGV.reset_activity()
        MC.reset_activity()
        BLA_IC.reset_activity()
        
        for epoch in range(max_timesteps):
            output_BGdl = BG_dl.step(np.dot(Ws["inp_BGdl"], ((Y + delta * DA) * inp.copy())),
                                    (np.dot(Ws["MC_STNdl"], MC.output.copy())))
            output_MGV = MGV.step(np.dot(Ws["BGdl_MGV"], output_BGdl.copy()) + np.dot(Ws["MC_MGV"], MC.output.copy()))
            output_MC = MC.step(np.dot(Ws["MGV_MC"], output_MGV.copy()))
           

"""
Defining Layers for Model simulation
"""          

def set_layers(N, tau, baseline, BG_dl_W, seed, noise, Matrices_scalars, DA_values):
    
    rng = np.random.RandomState(seed)
    
    BG_dl = BG_dl_Layer(N["BG_dl"], 
                          tau["BG_dl"], 
                          baseline["DLS"],
                          baseline["STNdl"],
                          baseline["GPi"],
                          BG_dl_W["DLS_GPi_W"], 
                          BG_dl_W["STNdl_GPi_W"],
                          rng,
                          noise["BG_dl"])
    
    MGV = Leaky_units_exc(N["MGV"], 
                          tau["MGV"],
                          baseline["MGV"],
                          rng,
                          noise["MGV"])
    
    MC = Leaky_units_exc(N["MC"], 
                         tau["MC"], 
                         baseline["MC"],
                         rng,
                         noise["MC"])
    
    BLA_IC = BLA_IC_Layer(N["BLA_IC"],
                          tau["BLA_IC"][0],
                          tau["BLA_IC"][1],
                          baseline["BLA_IC"],
                          rng,
                          noise["BLA_IC"],
                          eta_b = 0.1, 
                          tau_t = 10.0, 
                          alpha_t = 0.2, 
                          max_W = 1.0, 
                          theta_da = 0.2)
    
    return BG_dl, MGV, MC, BLA_IC

"""
Parameters for Model
"""

N = {"BG_dl": 2, "MGV": 2, "MC": 2, "BLA_IC": 4},
tau = {"MC": 2000, "PFCd_PC": 2000, "PL": 2000, "MGV": 300, "P": 300, "DM": 300, "BG_dl": 300, "BG_dm": 300, "BG_v": 300, "BLA_IC": [500, 500]},
baseline = {"DLS": 0.0, "STNdl": 0.0, "GPi": 0.8, "MGV": 0.0, "MC": 0.0, "BLA_IC": 0.0},
BG_dl_W = {"DLS_GPi_W": 2.4, "STNdl_GPi_W": 1.8},
seed = 2,
noise = {"BG_dl": 0.0, "MGV": 0.0, "MC": 0.05, "BLA_IC": 0.0},
Matrices_scalars = {"BGdl_MGV": 1.5, "MGV_MC": 1.5, "MC_MGV": 0.8, "MC_STNdl": 1.6, "MC_DLS": 1.0},
DA_values = {"Y": 0.1, "delta": 4.9, "DA": 1.0} 
                        
                        