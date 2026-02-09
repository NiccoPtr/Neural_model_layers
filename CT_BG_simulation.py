# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:03:56 2026

@author: Nicc
"""

from Layer_types import BG_dl_Layer, Leaky_units_exc
import numpy as np, matplotlib.pyplot as plt

class CT_BG():
    
    def __init__(self, parameters, rng):
        
        self.BG_dl = BG_dl_Layer(parameters.N["BG_dl"], 
                            parameters.tau["BG_dl"], 
                            parameters.baseline["DLS"],
                            parameters.baseline["STNdl"],
                            parameters.baseline["GPi"],
                            parameters.BG_dl_W["DLS_GPi_W"], 
                            parameters.BG_dl_W["STNdl_GPi_W"],
                            rng,
                            parameters.noise["BG_dl"],
                            parameters.threshold["BG_dl"])
        
        self.MGV = Leaky_units_exc(parameters.N["MGV"], 
                              parameters.tau["MGV"],
                              parameters.baseline["MGV"],
                              rng,
                              parameters.noise["MGV"],
                              parameters.threshold["MGV"])
        
        self.MC = Leaky_units_exc(parameters.N["MC"], 
                             parameters.tau["MC"], 
                             parameters.baseline["MC"],
                             rng,
                             parameters.noise["MC"],
                             parameters.threshold["MC"])
        
        self.Ws = {"inp_DLS": np.ones(parameters.N["BG_dl"]), 
              "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
              "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
              "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
              "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
              "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"],
              "PFCd_PPC_MC": np.eye(parameters.N["MC"] * 0.0)
              }
        
        self.W_learn_mask = np.ones(parameters.N["BG_dl"])
        
        self.BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
        self.MGV_output_pre = np.zeros(parameters.N["MGV"])
        self.MC_output_pre = np.zeros(parameters.N["MC"])
        
    def reset_activity(self):

        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()
        
    def delta_Str_learn_USV(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W):
        
        DA_term = np.maximum(0, DA - theta_DA_str)[:, None]
        delta_W_inp_str = (eta_str *
                           DA_term * 
                           np.outer(
                               np.maximum(0, v_str - theta_str),
                               np.maximum(0, v_inp - theta_inp_str)
                               ) *
                           (max_W_str - W))
        
        delta_W_inp_str *= mask
        
        return delta_W_inp_str
    
    def learning(self, parameters, da, inp):
        
        delta_W_inp_DLS = self.delta_Str_learn_USV(parameters.Str_Learn["eta_DLS"],
                                           da,
                                           self.BG_dl.output_DLS_pre * -1,
                                           inp,
                                           parameters.Str_Learn["theta_DA_DLS"],
                                           parameters.Str_Learn["theta_DLS"],
                                           parameters.Str_Learn["theta_inp_DLS"],
                                           self.Ws_learn_masks,
                                           parameters.Str_Learn["max_W_DLS"],
                                           self.Ws["inp_DLS"]
                                           )
        
        self.Ws['inp_DLS'] += delta_W_inp_DLS
        
        
    def step(self, parameters, timesteps, inp, da, PFCd_PPC_inp = 0.0):
        
        for _ in range(timesteps):
        
            self.BG_dl.step(np.dot(self.Ws["inp_DLS"], inp),
                       np.dot(self.Ws["MC_DLS"], self.MC_output_pre),
                       np.dot(self.Ws["MC_STNdl"], self.MC_output_pre))
            
            self.MGV.step(np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre +
                     np.dot(self.Ws["MC_MGV"], self.MC_output_pre)))
            
            self.MC.step(np.dot(self.Ws["MGV_MC"], self.MGV_output_pre +
                                np.dot(self.Ws['PFCd_PPC_MC'], PFCd_PPC_inp)))
            
            self.learning(self, parameters, da, inp)
            
            self.BG_dl_output_pre = self.BG_dl.output_BG_dl.copy()
            self.MGV_output_pre = self.MGV.output.copy()
            self.MC_output_pre = self.MC.output.copy()
         
            
    def plotting(self, res):
        
        fig, ax = plt.subplots(1, 1)

        BG_dl = np.array(res["BG_dl_output"]) * -1

        ax.plot(BG_dl[:, 0], label="Unit_1")
        ax.plot(BG_dl[:, 1], label="Unit_2")

        ax.set_title("BG simulation")
        ax.legend()
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Activity level")
        ax.set_ylim(0, 1)

        plt.tight_layout()

        plt.show()