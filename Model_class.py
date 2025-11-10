# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:38:31 2025

@author: Nicc
"""

from params import parameters
from Layer_types import *
import numpy as np, matplotlib.pyplot as plt


class Model:
    
    def __init__(self, parameters):
        
        rng = np.random.RandomState(parameters.seed)

        self.PPN = Leaky_onset_units_exc(parameters.N["PPN"],
                                    parameters.tau["PPN"][0],
                                    parameters.tau["PPN"][1],
                                    parameters.baseline["PPN"],
                                    rng,
                                    parameters.noise["PPN"])

        self.LH = Leaky_onset_units_exc(parameters.N["LH"],
                                   parameters.tau["LH"][0],
                                   parameters.tau["LH"][1],
                                   parameters.baseline["LH"],
                                   rng,
                                   parameters.noise["LH"])

        self.VTA = Leaky_units_exc(parameters.N["VTA"],
                              parameters.tau["VTA"],
                              parameters.baseline["VTA"],
                              rng,
                              parameters.noise["VTA"])

        self.BLA_IC = BLA_IC_Layer(parameters.N["BLA_IC"],
                              parameters.tau["BLA_IC"][0],
                              parameters.tau["BLA_IC"][1],
                              parameters.baseline["BLA_IC"],
                              rng,
                              parameters.noise["BLA_IC"],
                              parameters.BLA_Learn["eta_b"],
                              parameters.BLA_Learn["tau_t"],
                              parameters.BLA_Learn["alpha_t"],
                              parameters.BLA_Learn["theta_DA"],
                              parameters.BLA_Learn["max_W"])

        self.SNpc = SNpc_Layer(parameters.N["SNpc"],
                          parameters.tau["SNpc"],
                          parameters.baseline["SNpc"],
                          parameters.SNpc_W["SNpci_1_SNpco_1_W"],
                          parameters.SNpc_W["SNpci_2_SNpco_2_W"],
                          rng,
                          parameters.noise["SNpc"])

        self.BG_dl = BG_dl_Layer(parameters.N["BG_dl"], 
                            parameters.tau["BG_dl"], 
                            parameters.baseline["DLS"],
                            parameters.baseline["STNdl"],
                            parameters.baseline["GPi"],
                            parameters.BG_dl_W["DLS_GPi_W"], 
                            parameters.BG_dl_W["STNdl_GPi_W"],
                            rng,
                            parameters.noise["BG_dl"])

        self.MGV = Leaky_units_exc(parameters.N["MGV"], 
                              parameters.tau["MGV"],
                              parameters.baseline["MGV"],
                              rng,
                              parameters.noise["MGV"])

        self.MC = Leaky_units_exc(parameters.N["MC"], 
                             parameters.tau["MC"], 
                             parameters.baseline["MC"],
                             rng,
                             parameters.noise["MC"])

        self.BG_dm = BG_dm_Layer(parameters.N["BG_dm"], 
                            parameters.tau["BG_dm"], 
                            parameters.baseline["DMS"],
                            parameters.baseline["STNdm"],
                            parameters.baseline["GPi_SNpr"],
                            parameters.BG_dm_W["DMS_GPiSNpr_W"], 
                            parameters.BG_dm_W["STNdm_GPiSNpr_W"],
                            rng,
                            parameters.noise["BG_dm"])

        self.P = Leaky_units_exc(parameters.N["P"], 
                            parameters.tau["P"],
                            parameters.baseline["P"],
                            rng,
                            parameters.noise["P"])

        self.PFCd_PPC = Leaky_units_exc(parameters.N["PFCd_PPC"], 
                                   parameters.tau["PFCd_PPC"], 
                                   parameters.baseline["PFCd_PPC"],
                                   rng,
                                   parameters.noise["PFCd_PPC"])

        self.BG_v = BG_v_Layer(parameters.N["BG_v"], 
                          parameters.tau["BG_v"], 
                          parameters.baseline["NAc"],
                          parameters.baseline["STNv"],
                          parameters.baseline["SNpr"],
                          parameters.BG_v_W["NAc_SNpr_W"], 
                          parameters.BG_v_W["STNv_SNpr_W"],
                          rng,
                          parameters.noise["BG_v"])

        self.DM = Leaky_units_exc(parameters.N["DM"], 
                             parameters.tau["DM"],
                             parameters.baseline["DM"],
                             rng,
                             parameters.noise["DM"])

        self.PL = Leaky_units_exc(parameters.N["PL"], 
                             parameters.tau["PL"], 
                             parameters.baseline["PL"],
                             rng,
                             parameters.noise["PL"])
        
        
    def set_env(self, parameters):
        
        self.Ws = {
              "inp_BLA_IC": np.array([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, -1, 0],
                                      [0, 0, 0, 1, 0, -1]
                                      ]),
              "Mani_DLS": np.ones((parameters.N["DLS"], parameters.N["DLS"])), "Mani_DMS": np.ones((parameters.N["DMS"], parameters.N["DMS"])),
              "Food_PPN": np.array([1, 1]), "Food_LH": np.array([1, 1]),
              "PPN_SNpco": np.array([1]),
              "BLA_IC_NAc": np.array([[0, 0, 1, 1],
                                      [0, 0, 1, 1]]), 
              "BLA_IC_LH": np.array([0, 0, 1, 1]),
              "LH_VTA": np.array([1]),
              "NAc_SNpci_1": np.eye(parameters.N["SNpc"]), "DMS_SNpci_2": np.eye(parameters.N["SNpc"]),
              "GPi_MGV": np.eye(parameters.N["MGV"]), "GPi_SNpr_P": np.eye(parameters.N["P"]), "SNpr_DM": np.eye(parameters.N["DM"]),
              "MGV_MC": np.eye(parameters.N["MC"]), "P_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]), "DM_PL": np.eye(parameters.N["PL"]),
              "PL_NAc": np.eye(parameters.N["BG_v"]), "PL_STNv": np.eye(parameters.N["BG_v"]), "PL_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]),
              "PFCd_PPC_DMS": np.eye(parameters.N["BG_dm"]), "PFCd_PPC_STNdm": np.eye(parameters.N["BG_dm"]), "PFCd_PPC_PL": np.eye(parameters.N["PL"]), "PFCd_PPC_MC": np.eye(parameters.N["MC"]),
              "MC_MGV": np.eye(parameters.N["MGV"]), "MC_DLS": np.eye(parameters.N["BG_dl"]), "MC_STNdl": np.eye(parameters.N["BG_dl"]), "MC_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"])
              }
        
        """
        Set input array()
        """
        
        