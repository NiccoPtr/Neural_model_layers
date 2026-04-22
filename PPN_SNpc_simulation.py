# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:47:56 2026

@author: Nicc
"""

from Layer_types import SNpc_Layer, Leaky_onset_units_exc
import numpy as np

class PPN_SNpc():
    
    def __init__(self, parameters, rng):
        
        self.PPN = Leaky_onset_units_exc(
            parameters.N["PPN"],
            parameters.tau["PPN"][0],
            parameters.tau["PPN"][1],
            parameters.baseline["PPN"],
            rng,
            parameters.noise["PPN"],
        )
        
        self.SNpc = SNpc_Layer(
            parameters.N["SNpc"],
            parameters.tau["SNpc"],
            parameters.baseline["SNpc"],
            parameters.SNpc_W["SNpci_1_SNpco_1_W"],
            parameters.SNpc_W["SNpci_2_SNpco_2_W"],
            rng,
            parameters.noise["SNpc"],
            parameters.threshold["SNpc"],
        )
        
        self.Ws = {
            "Food_PPN": np.array([1.0, 1.0]) * parameters.Matrices_scalars["Food_PPN"],
            "PPN_SNpco": np.array([1.0]) * parameters.Matrices_scalars["PPN_SNpco"],
            "NAc_SNpci_1": np.eye(parameters.N["SNpc"]) * parameters.Matrices_scalars["NAc_SNpci_1"],
            "DMS_SNpci_2": np.eye(parameters.N["SNpc"]) * parameters.Matrices_scalars["DMS_SNpci_2"]
            }
        
        self.PPN_output_pre = np.zeros(parameters.N['PPN'])
        
    def reset_activity(self):
        
        self.PPN.reset_activity()
        self.SNpc.reset_activity()
        
    def update_output_pre(self):
        
        self.PPN_output_pre = self.PPN.output.copy()
        
    def step(self, inp, NAc_inp, DMS_inp):
        
        self.PPN.step(np.dot(self.Ws['Food_PPN'], inp))
        self.SNpc.step(np.dot(self.Ws['NAc_SNpci_1'], NAc_inp),
                       np.dot(self.Ws['DMS_SNpci_2'], DMS_inp),
                       np.dot(self.Ws['PPN_SNpco'], self.PPN_output_pre)
                       )
        
        self.update_output_pre()