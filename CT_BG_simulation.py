# -*- coding: utf-8 -*-
"""
Created on Mon Feb 9 17:03:56 2026

@author: Nicc
"""

from Layer_types import BG_v2, Leaky_units_exc
import numpy as np

class CT_BG():
    
    def __init__(self, parameters, rng):

        self.parameters = parameters
        
        self.BG_dl = BG_v2(
            self.parameters.N["BG_dl"], 
            self.parameters.tau["BG_dl"], 
            self.parameters.baseline["DLS_1"],
            self.parameters.baseline["DLS_2"],
            self.parameters.baseline["STNdl"],
            self.parameters.baseline["GPi"],
            self.parameters.baseline["GPe"],
            self.parameters.BG_dl_W["DLS_1_GPi_W"], 
            self.parameters.BG_dl_W["DLS_2_GPe_W"],
            self.parameters.BG_dl_W["STNdl_GPi_W"],
            self.parameters.BG_dl_W["STNdl_GPe_W"],
            self.parameters.BG_dl_W["GPe_STNdl_W"],
            self.parameters.BG_dl_W["GPe_GPi_W"],
            rng,
            self.parameters.noise["BG_dl"],
            self.parameters.threshold["BG_dl"]
        )
        
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
        
        self.Ws = {
            "inp_DLS_1": np.ones([parameters.N["BG_dl"], parameters.N["BG_dl"]]) * parameters.Matrices_scalars["Mani_DLS"], 
            "inp_DLS_2": np.ones([parameters.N["BG_dl"], parameters.N["BG_dl"]]) * parameters.Matrices_scalars["Mani_DLS"], 
            "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
            "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
            "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
            "MC_DLS_1": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS_1"],
            "MC_DLS_2": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS_2"],
            "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"],
            "PFCd_PPC_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars['PFCd_PPC_MC']
              }
        
        self.W_learn_mask = np.ones([parameters.N["BG_dl"], parameters.N["BG_dl"]])
        
        self.BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
        self.MGV_output_pre = np.zeros(parameters.N["MGV"])
        self.MC_output_pre = np.zeros(parameters.N["MC"])
        
    def reset_activity(self):

        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()
        
    def update_output_pre(self):
        
        self.BG_dl_output_pre = self.BG_dl.output_BG.copy()
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()
        
    def delta_Str_learn_1(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W):
        
        DA_term = np.maximum(0, DA - theta_DA_str)
        delta_W_inp_str = (eta_str *
                           DA_term * 
                           np.outer(
                               np.maximum(0, v_str - theta_str),
                               np.maximum(0, v_inp - theta_inp_str)
                               ) *
                           (max_W_str - W))
        
        delta_W_inp_str *= mask
        
        return delta_W_inp_str
    
    def delta_Str_learn_2(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W, lambda_xor = 0.1):
        
        DA_term = np.maximum(0, DA - theta_DA_str)

        pre = np.maximum(0, v_inp - theta_inp_str)
        post = np.maximum(0, v_str - theta_str)

        hebb = np.outer(post, pre)

        A = pre[None, :]
        B = post[:, None]

        xor_term = A + B - (2 * A * B)

        delta_W = (
            eta_str
            * DA_term
            * (hebb - lambda_xor * xor_term)
            * (max_W_str - W)
        )

        delta_W *= mask

        return delta_W

    def learning(self, parameters, da, inp):
        
        self.delta_W_inp_DLS_1 = self.delta_Str_learn_1(parameters.Str_Learn["eta_DLS_1"],
                                           da,
                                           self.BG_dl.output_Str1_pre * -1,
                                           inp,
                                           parameters.Str_Learn["theta_DA_DLS_1"],
                                           parameters.Str_Learn["theta_DLS_1"],
                                           parameters.Str_Learn["theta_inp_DLS_1"],
                                           self.W_learn_mask,
                                           parameters.Str_Learn["max_W_DLS"],
                                           self.Ws["inp_DLS_1"]
                                           )
        
        self.Ws['inp_DLS_1'] += self.delta_W_inp_DLS_1

        self.delta_W_inp_DLS_2 = self.delta_Str_learn_2(parameters.Str_Learn["eta_DLS_2"],
                                           da,
                                           self.BG_dl.output_Str2_pre * -1,
                                           inp,
                                           parameters.Str_Learn["theta_DA_DLS_2"],
                                           parameters.Str_Learn["theta_DLS_2"],
                                           parameters.Str_Learn["theta_inp_DLS_2"],
                                           self.W_learn_mask,
                                           parameters.Str_Learn["max_W_DLS"],
                                           self.Ws["inp_DLS_2"]
                                           )
        
        self.Ws['inp_DLS_2'] += self.delta_W_inp_DLS_2
        self.Ws['inp_DLS_2'] = np.maximum(self.Ws['inp_DLS_2'], 0)
        
    def step(self, parameters, inp, da, PFCd_PPC_inp = (0.0, 0.0), learn = True):
        
        self.BG_dl.step(
            (self.parameters.DA_values["Y_DLS_1"] + self.parameters.DA_values["delta_DLS_1"] * da) * np.dot(self.Ws["inp_DLS_1"], inp),
            ((1/(self.parameters.DA_values["Y_DLS_2"] + self.parameters.DA_values["delta_DLS_2"] * da)) * np.dot(self.Ws["inp_DLS_2"], inp)),
            (self.parameters.DA_values["Y_DLS_1"] + self.parameters.DA_values["delta_DLS_1"] * da) * np.dot(self.Ws["MC_DLS_1"], self.MC_output_pre),
            ((1/(self.parameters.DA_values["Y_DLS_2"] + self.parameters.DA_values["delta_DLS_2"] * da)) * np.dot(self.Ws["MC_DLS_2"], self.MC_output_pre)),
            np.dot(self.Ws["MC_STNdl"], self.MC_output_pre)
            )
        
        self.MGV.step(np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre) +
                 np.dot(self.Ws["MC_MGV"], self.MC_output_pre))
        
        self.MC.step(np.dot(self.Ws["MGV_MC"], self.MGV_output_pre) +
                    np.dot(self.Ws['PFCd_PPC_MC'], np.array(PFCd_PPC_inp)))
        
        if learn:
            self.learning(parameters, da, inp)
            
        self.update_output_pre()