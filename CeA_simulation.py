# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:57:56 2026

@author: Nicc
"""

from Layer_types import Leaky_onset_units_exc
import numpy as np

class CeA():
    def __init__(self, parameters, rng):

        self.parameters = parameters
        
        self.CeA = Leaky_onset_units_exc(
            self.parameters.N["CeA"],
            self.parameters.tau["CeA"][0],
            self.parameters.tau["CeA"][1],
            self.parameters.baseline["CeA"],
            rng,
            self.parameters.noise["CeA"],
        )
        
        self.Ws = {
            "Inp_CeA": np.ones([parameters.N["CeA"], parameters.N["CeA"]]) * parameters.Matrices_scalars["Inp_CeA"],
            "Mani_CeA": np.eye(parameters.N["CeA"]) * parameters.Matrices_scalars["Mani_CeA"],
            "CS_CeA": np.eye(parameters.N["CeA"]) * parameters.Matrices_scalars["CS_CeA"],
            "Food_CeA": np.eye(parameters.N["CeA"]) * parameters.Matrices_scalars["Food_CeA"],
            "G_Sat_CeA": np.eye(parameters.N["CeA"]) * parameters.Matrices_scalars["G_Sat_CeA"],
            "BLA_IC_CeA": np.eye(parameters.N["CeA"]) * parameters.Matrices_scalars["BLA_IC_CeA"],
        }

        self.W_learn_mask = np.ones([parameters.N["CeA"], parameters.N["CeA"]])

        self.CeA_output_pre = np.zeros(parameters.N["CeA"])