# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:44:25 2026

@author: Nicc
"""

from Layer_types import Leaky_units_exc, BG_dl_Layer, BG_dm_Layer, BG_v_Layer
import numpy as np

class Cortex():
    
    def __init__(self, parameters, rng, scalar_inp_BLA, scalar_inp):
        
        self.BG_dl = BG_dl_Layer(
            parameters.N["BG_dl"],
            parameters.tau["BG_dl"],
            parameters.baseline["DLS"],
            parameters.baseline["STNdl"],
            parameters.baseline["GPi"],
            parameters.BG_dl_W["DLS_GPi_W"],
            parameters.BG_dl_W["STNdl_GPi_W"],
            rng,
            parameters.noise["BG_dl"],
            parameters.threshold["BG_dl"],
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
        
        self.BG_dm = BG_dm_Layer(
            parameters.N["BG_dm"],
            parameters.tau["BG_dm"],
            parameters.baseline["DMS"],
            parameters.baseline["STNdm"],
            parameters.baseline["GPi_SNpr"],
            parameters.BG_dm_W["DMS_GPiSNpr_W"],
            parameters.BG_dm_W["STNdm_GPiSNpr_W"],
            rng,
            parameters.noise["BG_dm"],
            parameters.threshold["BG_dm"],
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
        
        self.BG_v = BG_v_Layer(
            parameters.N["BG_v"],
            parameters.tau["BG_v"],
            parameters.baseline["NAc"],
            parameters.baseline["STNv"],
            parameters.baseline["SNpr"],
            parameters.BG_v_W["NAc_SNpr_W"],
            parameters.BG_v_W["STNv_SNpr_W"],
            rng,
            parameters.noise["BG_v"],
            parameters.threshold["BG_v"],
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
            "inp_BLA_BG": np.eye(parameters.N["BG_v"]) * scalar_inp_BLA,
            "inp_BG": np.eye(parameters.N["BG_dl"]) * scalar_inp,
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
            "MC_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars["MC_PFCd_PPC"],
            "PL_STNv": np.eye(parameters.N["BG_v"]) * parameters.Matrices_scalars["PL_STNv"],
            "PL_NAc": np.eye(parameters.N["BG_v"]) * parameters.Matrices_scalars["PL_NAc"],
            "PFCd_PPC_STNdm": np.eye(parameters.N["BG_dm"]) * parameters.Matrices_scalars["PFCd_PPC_STNdm"],
            "PFCd_PPC_DMS": np.eye(parameters.N["BG_dm"]) * parameters.Matrices_scalars["PFCd_PPC_DMS"],
            "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
            "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"]
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
        
        self.BG_dl.reset_activity()
        self.BG_dm.reset_activity()
        self.BG_v.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()
        self.P.reset_activity()
        self.PFCd_PPC.reset_activity()
        self.DM.reset_activity()
        self.PL.reset_activity()
        
    def update_output_pre(self):
        
        self.GPi_output_pre =self.BG_dl.GPi.output.copy()
        self.GPi_SNpr_output_pre = self.BG_dm.GPi_SNpr.output.copy()
        self.SNpr_output_pre = self.BG_v.SNpr.output.copy()
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()
        self.P_output_pre = self.P.output.copy()
        self.PFCd_PPC_output_pre = self.PFCd_PPC.output.copy()
        self.DM_output_pre = self.DM.output.copy()
        self.PL_output_pre = self.PL.output.copy()
        
    def step(self, inp_BLA, inp):
        
        #Basal Ganglia
        self.BG_v.step(
            np.dot(self.Ws['inp_BLA_BG'], inp_BLA),
            np.dot(self.Ws['PL_NAc'], self.PL_output_pre),
            np.dot(self.Ws['PL_STNv'], self.PL_output_pre)
            )
        self.BG_dm.step(
            np.dot(self.Ws['inp_BG'], inp),
            np.dot(self.Ws['PFCd_PPC_DMS'], self.PFCd_PPC_output_pre),
            np.dot(self.Ws['PFCd_PPC_STNdm'], self.PFCd_PPC_output_pre)
            )
        self.BG_dl.step(
            np.dot(self.Ws['inp_BG'], inp),
            np.dot(self.Ws['MC_DLS'], self.MC_output_pre),
            np.dot(self.Ws['MC_STNdl'], self.MC_output_pre)
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