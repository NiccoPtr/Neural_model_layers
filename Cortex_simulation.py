# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:44:25 2026

@author: Nicc
"""

from Layer_types import Leaky_units_exc, Leaky_units_inh
import numpy as np

class Cortex():
    
    def __init__(self, parameters, rng):
        
        self.GPi = Leaky_units_inh(
            parameters.N['BG_dl'],
            parameters.tau['BG_dl'],
            parameters.baseline['GPi'],
            rng,
            parameters.noise['BG_dl'],
            parameters.threshold['BG_dl']
            )
        
        self.MGV = Leaky_units_exc(
            parameters.N["MGV"],
            parameters.tau["MGV"],
            parameters.baseline["MGV"],
            rng,
            parameters.noise["MGV"],
            parameters.threshold["MGV"],
        )

        self.MC = Leaky_units_exc(
            parameters.N["MC"],
            parameters.tau["MC"],
            parameters.baseline["MC"],
            rng,
            parameters.noise["MC"],
            parameters.threshold["MC"],
        )
        
        self.GPi_SNpr = Leaky_units_inh(
            parameters.N['BG_dm'],
            parameters.tau['BG_dm'],
            parameters.baseline['GPi_SNpr'],
            rng,
            parameters.noise['BG_dm'],
            parameters.threshold['BG_dm']
            )
        
        self.P = Leaky_units_exc(
            parameters.N["P"],
            parameters.tau["P"],
            parameters.baseline["P"],
            rng,
            parameters.noise["P"],
            parameters.threshold["P"],
        )

        self.PFCd_PPC = Leaky_units_exc(
            parameters.N["PFCd_PPC"],
            parameters.tau["PFCd_PPC"],
            parameters.baseline["PFCd_PPC"],
            rng,
            parameters.noise["PFCd_PPC"],
            parameters.threshold["PFCd_PPC"],
        )
        
        self.SNpr = Leaky_units_inh(
            parameters.N['BG_v'],
            parameters.tau['BG_v'],
            parameters.baseline['SNpr'],
            rng,
            parameters.noise['BG_v'],
            parameters.threshold['BG_v']
            )
        
        self.DM = Leaky_units_exc(
            parameters.N["DM"],
            parameters.tau["DM"],
            parameters.baseline["DM"],
            rng,
            parameters.noise["DM"],
            parameters.threshold["DM"],
        )

        self.PL = Leaky_units_exc(
            parameters.N["PL"],
            parameters.tau["PL"],
            parameters.baseline["PL"],
            rng,
            parameters.noise["PL"],
            parameters.threshold["PL"],
        )
        
        self.Ws = {
            "inp_i_BG": np.eye(2),
            "inp_e_BG": np.ones((2, 2)),
            "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
            "GPi_SNpr_P": np.eye(parameters.N["P"]) * parameters.Matrices_scalars["GPi_SNpr_P"],
            "SNpr_DM": np.eye(parameters.N["DM"]) * parameters.Matrices_scalars["SNpr_DM"],
            "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
            "P_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars["P_PFCd_PPC"],
            "DM_PL": np.eye(parameters.N["PL"]) * parameters.Matrices_scalars["DM_PL"],
            "PL_DM": np.eye(parameters.N["DM"]) * parameters.Matrices_scalars["PL_DM"],
            "PL_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars["PL_PFCd_PPC"],
            "PFCd_PPC_P": np.eye(parameters.N["P"]) * parameters.Matrices_scalars["PFCd_PPC_P"],
            "PFCd_PPC_PL": np.eye(parameters.N["PL"]) * parameters.Matrices_scalars["PFCd_PPC_PL"],
            "PFCd_PPC_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["PFCd_PPC_MC"],
            "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
            "MC_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars["MC_PFCd_PPC"]
            }
        
        self.GPi_output_pre = np.zeros(parameters.N['BG_dl'])
        self.GPi_SNpr_output_pre = np.zeros(parameters.N['BG_dm'])
        self.SNpr_output_pre = np.zeros(parameters.N['BG_v'])
        self.MGV_output_pre = np.zeros(parameters.N["MGV"])
        self.MC_output_pre = np.zeros(parameters.N["MC"])
        self.P_output_pre = np.zeros(parameters.N["P"])
        self.PFCd_PPC_output_pre = np.zeros(parameters.N["PFCd_PPC"])
        self.DM_output_pre = np.zeros(parameters.N["DM"])
        self.PL_output_pre = np.zeros(parameters.N["PL"])
        
    def reset_activity(self):
        
        self.GPi.reset_activity()
        self.GPi_SNpr.reset_activity()
        self.SNpr.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()
        self.P.reset_activity()
        self.PFCd_PPC.reset_activity()
        self.DM.reset_activity()
        self.PL.reset_activity()
        
    def update_output_pre(self):
        
        self.GPi_output_pre =self.GPi.output.copy()
        self.GPi_SNpr_output_pre = self.GPi_SNpr.output.copy() 
        self.SNpr_output_pre = self.SNpr.output.copy() 
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()
        self.P_output_pre = self.P.output.copy()
        self.PFCd_PPC_output_pre = self.PFCd_PPC.output.copy()
        self.DM_output_pre = self.DM.output.copy()
        self.PL_output_pre = self.PL.output.copy()
        
    def step(self, inp_i, inp_e):
        
        #Basal Ganglia
        self.GPi.step(
            np.dot(self.Ws['inp_i_BG'], inp_i)
            + np.dot(self.Ws['inp_e_BG'], inp_e)
            )
        self.GPi_SNpr.step(
            np.dot(self.Ws['inp_i_BG'], inp_i)
            + np.dot(self.Ws['inp_e_BG'], inp_e)
            )
        self.SNpr.step(
            np.dot(self.Ws['inp_i_BG'], inp_i)
            + np.dot(self.Ws['inp_e_BG'], inp_e)
            )
        
        #Thalamus
        self.DM.step(
            np.dot(self.Ws['SNpr_DM'], self.SNpr_output_pre)
            + np.dot(self.Ws['PL_DM'], self.PL_output_pre)
            )
        self.P.step(
            np.dot(self.Ws['GPi_SNpr_P'], self.GPi_SNpr_output_pre)
            + np.dot(self.Ws['PFCd_PPC_P'], self.PFCd_PPC_output_pre)
            )
        self.MGV.step(
            np.dot(self.Ws['GPi_MGV'], self.GPi_output_pre)
            + np.dot(self.Ws['MC_MGV'], self.MC_output_pre)
            )
        
        #Cortex
        self.PL.step(
            np.dot(self.Ws["DM_PL"], self.DM_output_pre)
            + np.dot(self.Ws["PFCd_PPC_PL"], self.PFCd_PPC_output_pre)
            )
        self.PFCd_PPC.step(
            np.dot(self.Ws["P_PFCd_PPC"], self.P_output_pre)
            + np.dot(self.Ws["PL_PFCd_PPC"], self.PL_output_pre)
            + np.dot(self.Ws["MC_PFCd_PPC"], self.MC_output_pre)
            )
        self.MC.step(
            np.dot(self.Ws["MGV_MC"], self.MGV_output_pre)
            + np.dot(self.Ws["PFCd_PPC_MC"], self.PFCd_PPC_output_pre)
            )
        
        self.update_output_pre()