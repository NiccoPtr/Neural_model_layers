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

def delta_Str_learn(eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, max_W_str, W):
    
    delta_W_inp_str = (eta_str *
                       np.maximum(0, DA - theta_DA_str) * 
                       np.outer(
                           np.maximum(0, v_str - theta_str),
                           np.maximum(0, v_inp - theta_inp_str)
                           ) *
                       (max_W_str - W))
    
    return delta_W_inp_str

"""
Model's parameters'
"""
seed = 2

N = {"BG_dl": 2, "BG_dm": 2, "BG_v": 2, "MGV": 2, "MC": 2, "BLA_IC": 4, "SNpc": 2, "PPN": 1, "LH": 1, "VTA": 1, "P": 2, "DM": 2, "PL": 2, "PFCd_PPC": 2}

threshold = {"BG_dl": 0.0, "BG_dm": 0.0, "BG_v": 0.0, "MGV": 0.0, "MC": 0.8, "BLA_IC": 0.0, "SNpc": 1.0, "PPN": 0.0, "LH": 0.0, "VTA": 1.0, "P": 0.0, "DM": 0.0, "PL": 0.0, "PFCd_PPC": 0.0}

tau = {"MC": 6, "PFCd_PPC": 6, "PL": 6, "MGV": 6, "P": 6, "DM": 6, "BG_dl": 6, "BG_dm": 6, "BG_v": 6, "BLA_IC": [10, 10], "SNpc": 6, "PPN": [2, 10], "LH": [2, 10], "VTA": 6}

baseline = {"PPN": 0.0,
            "LH": 0.0,
            "VTA": 0.0,
            "BLA_IC": 0.0,
            "SNpc": 0.0,
            "DLS": 0.0, "STNdl": 0.0, "GPi": 0.8, 
            "DMS": 0.0, "STNdm": 0.0, "GPi_SNpr": 0.8, 
            "NAc": 0.0, "STNv": 0.0, "SNpr": 0.8, 
            "MGV": 0.0,
            "MC": 0.0,
            "P": 0.0,
            "PFCd_PPC": 0.0,
            "DM": 0.0,
            "PL": 0.0
            }

BG_dl_W = {"DLS_GPi_W": 3, "STNdl_GPi_W": 2}

BG_dm_W = {"DMS_GPiSNpr_W": 3, "STNdm_GPiSNpr_W": 2}

BG_v_W = {"NAc_SNpr_W": 3, "STNv_SNpr_W": 2}

SNpc_W = {"SNpci_1_SNpco_1_W": 1.0, "SNpci_2_SNpco_2_W": 1.0}

noise = {"BG_dl": 0.0, "BG_dm": 0.0, "BG_v": 0.0, "MGV": 0.0, "MC": 0.05, "BLA_IC": 0.0, "SNpc": 0.0, "PPN": 0.0, "LH": 0.0, "VTA": 0.0, "P": 0.0, "DM": 0.0, "PL": 0.0, "PFCd_PPC": 0.0}

Matrices_scalars = {"Mani_DLS": 0.0, "Mani_DMS": 0.0, "Mani_BLA_IC": 5,
                    "Food_PPN": 10, "Food_BLA_IC": 5, "Food_LH": 10,
                    "Sat_BLA_IC": 10,
                    "PPN_SNpco": 20,
                    "BLA_IC_NAc": 0.0, "BLA_IC_LH": 5,
                    "LH_VTA": 20,
                    "NAc_SNpci_1": 6, "DMS_SNpci_2": 10,
                    "GPi_MGV": 1.5, "GPi_SNpr_P": 1.5, "SNpr_DM": 1.5,
                    "MGV_MC": 1.2, "P_PFCd_PPC": 1.0, "DM_PL": 1.0,
                    "PL_NAc": 1.0, "PL_STNv": 1.6, "PL_PFCd_PPC": 0.2, 
                    "PFCd_PPC_DMS": 1.0, "PFCd_PPC_STNdm": 1.6, "PFCd_PPC_PL": 1.0, "PFCd_PPC_MC": 1.0, 
                    "MC_MGV": 1.0, "MC_DLS": 1.0, "MC_STNdl": 1.6, "MC_PFCd_PPC": 0.2}

DA_values = {"Y_DLS": 0.2, "Y_DMS": 0.5, "Y_NAc": 0.8, "delta_DLS": 4.0, "delta_DMS": 6.5, "delta_NAc": 1.5}

BLA_Learn = {"eta_b": 0.08, "alpha_t": 10**10, "tau_t": 500, "theta_DA": 0.7, "max_W": 2}

Str_Learn = {"eta_DLS": 0.02, "eta_DMS": 0.02, "eta_NAc": 0.05, 
             "theta_DA_DLS": 0.8, "theta_DA_DMS": 0.8, "theta_DA_NAc": 0.9,
             "theta_DLS": 0.5, "theta_DMS": 0.5, "theta_NAc": 0.9,
             "theta_inp_DLS": 0.5, "theta_inp_DMS": 0.5, "theta_inp_NAc": 0.9,
             "max_W_DLS": 1, "max_W_DMS": 1, "max_W_NAc": 2}

"""
Model's matrices
"""

Ws = {
      "inp_BLA_IC": np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, -1, 0],
                              [0, 0, 0, 1, 0, -1]
                              ]),
      "Mani_DLS": np.ones((N, N)), "Mani_DMS": np.ones((N, N)),
      "Food_PPN": np.array([1, 1]), "Food_LH": np.array([1, 1]),
      "PPN_SNpco": np.array([1]),
      "BLA_IC_NAc": np.array([[0, 0, 1, 1],
                              [0, 0, 1, 1]]), 
      "BLA_IC_LH": np.array([0, 0, 1, 1]),
      "LH_VTA": np.array([1]),
      "NAc_SNpci_1": np.eye(N), "DMS_SNpci_2": np.eye(N),
      "GPi_MGV": np.eye(N), "GPi_SNpr_P": np.eye(N), "SNpr_DM": np.eye(N),
      "MGV_MC": np.eye(N), "P_PFCd_PPC": np.eye(N), "DM_PL": np.eye(N),
      "PL_NAc": np.eye(N), "PL_STNv": np.eye(N), "PL_PFCd_PPC": np.eye(N),
      "PFCd_PPC_DMS": np.eye(N), "PFCd_PPC_STNdm": np.eye(N), "PFCd_PPC_PL": np.eye(N), "PFCd_PPC_MC": np.eye(N),
      "MC_MGV": np.eye(N), "MC_DLS": np.eye(N), "MC_STNdl": np.eye(N), "MC_PFCd_PPC": np.eye(N)
      }

"""
Layers for Model simulation
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
                      BLA_Learn["eta_b"],
                      BLA_Learn["tau_t"],
                      BLA_Learn["alpha_t"],
                      BLA_Learn["theta_DA"],
                      BLA_Learn["max_W"])

SNpc = SNpc_Layer(N["SNpc"],
                  tau["SNpc"],
                  baseline["SNpc"],
                  SNpc_W["SNpci_1_SNpco_1_W"],
                  SNpc_W["SNpci_2_SNpco_2_W"],
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

PFCd_PPC = Leaky_units_exc(N["PFCd_PPC"], 
                     tau["PFCd_PPC"], 
                     baseline["PFCd_PPC"],
                     rng,
                     noise["PFCd_PPC"])

BG_v = BG_v_Layer(N["BG_v"], 
                  tau["BG_v"], 
                  baseline["NAc"],
                  baseline["STNv"],
                  baseline["SNpr"],
                  BG_v_W["NAc_SNpr_W"], 
                  BG_v_W["STNv_SNpr_W"],
                  rng,
                  noise["BG_v"])

DM = Leaky_units_exc(N["DM"], 
                     tau["DM"],
                     baseline["DM"],
                     rng,
                     noise["DM"])

PL = Leaky_units_exc(N["PL"], 
                     tau["PL"], 
                     baseline["PL"],
                     rng,
                     noise["PL"]) 

                        