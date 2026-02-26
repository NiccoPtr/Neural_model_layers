# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:14:06 2026

@author: Nicc
"""

from Layer_types import BG_v_Layer, BLA_IC_Layer, Leaky_units_exc, Leaky_onset_units_exc
import numpy as np

class CT_BGv_BLA_IC():
    
    def __init__(self, parameters, rng):
        
        
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
                              parameters.noise["VTA"],
                              parameters.threshold["VTA"])
        
        self.BG_v = BG_v_Layer(parameters.N["BG_v"], 
                          parameters.tau["BG_v"], 
                          parameters.baseline["NAc"],
                          parameters.baseline["STNv"],
                          parameters.baseline["SNpr"],
                          parameters.BG_v_W["NAc_SNpr_W"], 
                          parameters.BG_v_W["STNv_SNpr_W"],
                          rng,
                          parameters.noise["BG_v"],
                          parameters.threshold["BG_v"])
        
        self.DM = Leaky_units_exc(parameters.N["DM"], 
                             parameters.tau["DM"],
                             parameters.baseline["DM"],
                             rng,
                             parameters.noise["DM"],
                             parameters.threshold["DM"])

        self.PL = Leaky_units_exc(parameters.N["PL"], 
                             parameters.tau["PL"], 
                             parameters.baseline["PL"],
                             rng,
                             parameters.noise["PL"],
                             parameters.threshold["PL"])
        
        self.Ws = {'inp_BLA_IC': np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 5.0, 0.0, -10.0, 0.0],
                                [0.0, 0.0, 0.0, 5.0, 0.0, -10.0]]),
                   'Food_LH': np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * parameters.Matrices_scalars["Food_LH"],
                   "BLA_IC_LH": np.array([0.0, 0.0, 1.0, 1.0]) * parameters.Matrices_scalars["BLA_IC_LH"],
                   'LH_VTA': np.array([1.0]) * parameters.Matrices_scalars["LH_VTA"],
                   "BLA_IC_NAc": np.array([[0.0, 0.0, 1.0, 1.0],
                                           [0.0, 0.0, 1.0, 1.0]]) * parameters.Matrices_scalars["BLA_IC_NAc"],
                   "SNpr_DM": np.eye(parameters.N["DM"]) * parameters.Matrices_scalars["SNpr_DM"],
                   "DM_PL": np.eye(parameters.N["PL"]) * parameters.Matrices_scalars["DM_PL"],
                   "PL_DM": np.eye(parameters.N["DM"]) * parameters.Matrices_scalars["PL_DM"],
                   "PL_NAc": np.eye(parameters.N["BG_v"]) * parameters.Matrices_scalars["PL_NAc"],
                   "PL_STNv": np.eye(parameters.N["BG_v"]) * parameters.Matrices_scalars["PL_STNv"],
                   "PFCd_PPC_PL": np.eye(parameters.N["PL"]) * parameters.Matrices_scalars["PFCd_PPC_PL"]
                }
        
        self.Ws_learn_mask = np.array([[0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 1.0, 1.0]])
        
        self.LH_output_pre = np.zeros(parameters.N["LH"])
        self.VTA_output_pre = np.zeros(parameters.N["VTA"])
        self.BLA_IC_output_pre = np.zeros(parameters.N["BLA_IC"])
        self.BG_v_output_pre = np.zeros(parameters.N["BG_v"])
        self.NAc_output_pre = np.zeros(parameters.N["BG_v"])
        self.DM_output_pre = np.zeros(parameters.N["DM"])
        self.PL_output_pre = np.zeros(parameters.N["PL"])
        
    def reset_activity(self):
        
        self.LH.reset_activity()
        self.VTA.reset_activity()
        self.BLA_IC.reset_activity()
        self.BG_v.reset_activity()
        self.DM.reset_activity()
        self.PL.reset_activity()
        
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
    
    def learning(self, parameters):
        
        self.BLA_IC.learn(self.VTA_output_pre)
        
        delta_W_BLA_IC_NAc = self.delta_Str_learn_USV(parameters.Str_Learn["eta_NAc"],
                                           self.VTA_output_pre,
                                           self.NAc_output_pre * -1,
                                           self.BLA_IC_output_pre,
                                           parameters.Str_Learn["theta_DA_NAc"],
                                           parameters.Str_Learn["theta_NAc"],
                                           parameters.Str_Learn["theta_inp_NAc"],
                                           self.Ws_learn_mask,
                                           parameters.Str_Learn["max_W_NAc"],
                                           self.Ws["BLA_IC_NAc"]
                                           )
        
        self.Ws["BLA_IC_NAc"] +=  delta_W_BLA_IC_NAc
        
    def step(self, parameters, inp, PFCd_PPC_inp = [0.0, 0.0], learning = True):
        
        self.BLA_IC.step(np.dot(self.Ws["inp_BLA_IC"], inp))
        self.LH.step(np.dot(self.Ws["Food_LH"], inp) + np.dot(self.Ws["BLA_IC_LH"], self.BLA_IC_output_pre))
        self.VTA.step(np.dot(self.Ws["LH_VTA"], self.LH_output_pre))
        self.BG_v.step((parameters.DA_values["Y_NAc"] + (parameters.DA_values["delta_NAc"] * self.VTA_output_pre)) * np.dot(self.Ws["BLA_IC_NAc"], self.BLA_IC_output_pre),
                       np.dot(self.Ws["PL_NAc"], self.PL_output_pre),
                       np.dot(self.Ws["PL_STNv"], self.PL_output_pre)
                       )
        self.DM.step(np.dot(self.Ws["SNpr_DM"], self.BG_v_output_pre) + np.dot(self.Ws["PL_DM"], self.PL_output_pre))
        self.PL.step(np.dot(self.Ws["DM_PL"], self.DM_output_pre) + np.dot(self.Ws["PFCd_PPC_PL"], np.array(PFCd_PPC_inp)))
        
        if learning:
            self.learning(parameters)
            
        self.LH_output_pre = self.LH.output.copy()
        self.VTA_output_pre = self.VTA.output.copy()
        self.BLA_IC_output_pre = self.BLA_IC.output.copy()
        self.BG_v_output_pre = self.BG_v.output_BG_v.copy()
        self.NAc_output_pre = self.BG_v.NAc.output.copy()
        self.DM_output_pre = self.DM.output.copy()
        self.PL_output_pre = self.PL.output.copy()
        