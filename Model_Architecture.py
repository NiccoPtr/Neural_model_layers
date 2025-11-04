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
Parameters for Model
"""

N = {"BG_dl": 2, "BG_dm": 2, "BG_v": 2, "MGV": 2, "MC": 2, "BLA_IC": 4, "SNpc": 2, "PPN": 1, "LH": 1, "VTA": 1}

tau = {"MC": 2000, "PFCd_PC": 2000, "PL": 2000, "MGV": 300, "P": 300, "DM": 300, "BG_dl": 300, "BG_dm": 300, "BG_v": 300, "BLA_IC": [500, 500], "SNpc": 300, "PPN": [100, 500], "LH": [100, 500], "VTA": 300}

baseline = {"PPN": 0.0,
            "LH": 0.0,
            "VTA": 0.0,
            "BLA_IC": 0.0,
            "SNpc": 0.0,
            "DLS": 0.0, "STNdl": 0.0, "GPi": 0.8, #BG_dl 
            "DMS": 0.0, "STNdm": 0.0, "GPi_SNpr": 0.8, #BG_dm
            "NAc": 0.0, "STNv": 0.0, "SNpr": 0.8, #BG_v
            "MGV": 0.0,
            "MC": 0.0,
            "P": 0.0,
            "PFCd_PC": 0.0,
            "DM": 0.0,
            "PL": 0.0
            }

BG_dl_W = {"DLS_GPi_W": 2.4, "STNdl_GPi_W": 1.8}

BG_dm_W = {"DMS_GPiSNpr_W": 2.4, "STNdm_GPiSNpr_W": 1.8}

BG_v_W = {"NAc_SNpr_W": 2.4, "STNv_SNpr_W": 1.8}

SNpc_W = {"Inh_Layer_1_DA_Layer_1_W": 1.0, "Inh_Layer_2_DA_Layer_2_W": 1.0}

seed = 2

noise = {"BG_dl": 0.0, "BG_dm": 0.0, "BG_v": 0.0, "MGV": 0.0, "MC": 0.05, "BLA_IC": 0.0, "SNpc": 0.0, "PPN": 0.0, "LH": 0.0, "VTA": 0.0}

Matrices_scalars = {"BGdl_MGV": 1.5, "MGV_MC": 1.5, "MC_MGV": 0.8, "MC_STNdl": 1.6, "MC_DLS": 1.0}

DA_values = {"Y": 0.1, "delta": 4.9, "DA": 1.0}


"""
Defining Layers for Model simulation
"""          
    
rng = np.random.RandomState(seed)

PPN = Leaky_onset_units_exc(N["PPN"],
                            tau["PPN"][0],
                            tau["PPN"][1],
                            baseline["PPN"],
                            rng,
                            noise["PPN"])

LH = Leaky_onset_units_exc(N["LH"],
                           tau["LH"][0],
                           tau["LH"][1],
                           baseline["LH"],
                           rng,
                           noise["LH"])

VTA = Leaky_units_exc(N["VTA"],
                     tau["VTA"],
                     baseline["VTA"],
                     rng,
                     noise["VTA"])

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

SNpc = SNpc_Layer(N["SNpc"],
                  tau["SNpc"],
                  baseline["SNpc"],
                  SNpc_W["Inh_Layer_1_DA_Layer_1_W"],
                  SNpc_W["Inh_Layer__DA_Layer_2_W"],
                  rng,
                  noise["SNpc"])

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

BG_dm = BG_dm_Layer(N["BG_dm"], 
                      tau["BG_dm"], 
                      baseline["DMS"],
                      baseline["STNdm"],
                      baseline["GPi_SNpr"],
                      BG_dm_W["DMS_GPiSNpr_W"], 
                      BG_dm_W["STNdm_GPiSNpr_W"],
                      rng,
                      noise["BG_dm"])

P = Leaky_units_exc(N["P"], 
                      tau["P"],
                      baseline["P"],
                      rng,
                      noise["P"])

PFCd_PC = Leaky_units_exc(N["PFCd_PC"], 
                     tau["PFCd_PC"], 
                     baseline["PFCd_PC"],
                     rng,
                     noise["PFCd_PC"])

BG_v = BG_v_Layer(N["BG_v"], 
                      tau["BG_v"], 
                      baseline["NAc"],
                      baseline["STNv"],
                      baseline["SNpr"],
                      BG_v_W["NAc_SNpr_W"], 
                      BG_v_W["STNv_SNpr_W"],
                      rng,
                      noise["BG_v"])

DM = Leaky_units_exc(N["P"], 
                      tau["P"],
                      baseline["P"],
                      rng,
                      noise["P"])

PL = Leaky_units_exc(N["PFCd_PC"], 
                     tau["PFCd_PC"], 
                     baseline["PFCd_PC"],
                     rng,
                     noise["PFCd_PC"]) 

                        