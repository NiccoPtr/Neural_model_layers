# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:57:56 2026

@author: Nicc
"""

from Layer_types import Leaky_onset_units_exc
import numpy as np

class CeA():
    def __init__(self, parameters, rng, BLA_IC = (1.0, 0.0, 0.0)):

        self.parameters = parameters

        self.BLA_IC = np.array(BLA_IC)
        
        self.CeA = Leaky_onset_units_exc(
            self.parameters.N["CeA"],
            self.parameters.tau["CeA"][0],
            self.parameters.tau["CeA"][1],
            self.parameters.baseline["CeA"],
            rng,
            self.parameters.noise["CeA"],
        )
        
        self.Ws = {
            "Inp_CeA": np.ones(9),
            "BLA_IC_CeA": np.ones(3) * self.parameters.Matrices_scalars["BLA_IC_CeA"],
        }

        self.Ws["Inp_CeA"][0:3] *= self.parameters.Matrices_scalars["CS_CeA"]
        self.Ws["Inp_CeA"][3:5] *= self.parameters.Matrices_scalars["Mani_CeA"]
        self.Ws["Inp_CeA"][5:8] *= self.parameters.Matrices_scalars["Food_CeA"]
        self.Ws["Inp_CeA"][-1] *= self.parameters.Matrices_scalars["G_Sat_CeA"]

        self.W_learn_mask = np.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        self.CeA_output_pre = np.zeros(self.parameters.N["CeA"])

    def reset_activity(self):
        self.CeA.reset_activity()

    def update_output_pre(self):
        self.CeA_output_pre = self.CeA.output.copy()

    def delta_Str_learn(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W):
        
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

    def learning(self, da, inp):

        self.delta_W_CS_CeA = self.delta_Str_learn(self.parameters.Str_Learn["eta_CeA"],
                                           da,
                                           self.CeA.output,
                                           inp,
                                           self.parameters.Str_Learn["theta_DA_CeA"],
                                           self.parameters.Str_Learn["theta_CeA"],
                                           self.parameters.Str_Learn["theta_inp_CeA"],
                                           self.W_learn_mask,
                                           self.parameters.Str_Learn["max_W_CeA"],
                                           self.Ws["Inp_CeA"]
                                           )
        
        self.Ws['Inp_CeA'] += self.delta_W_CS_CeA

    def step(self, da, inp, learn = True):

        # print(f"Input: {inp}")
        # print(f"Inp_CeA_connection: {self.Ws['Inp_CeA']}")
        # print(f"BLA_IC_CeA_connection: {self.Ws['BLA_IC_CeA']}")
        # print(f"Input_CeA_value: {np.dot(self.Ws['Inp_CeA'], inp)}")
        # print(f"BLA_IC_CeA_value: {np.dot(self.Ws['BLA_IC_CeA'], self.BLA_IC)}")
        # print(f"Total_Inp_CeA_value: {np.dot(self.Ws['Inp_CeA'], inp) + np.dot(self.Ws['BLA_IC_CeA'], self.BLA_IC)}")

        self.CeA.step(
            np.dot(self.Ws['Inp_CeA'], inp) + np.dot(self.Ws['BLA_IC_CeA'], self.BLA_IC)
        )

        if learn:
            self.learning(da, inp)

        self.update_output_pre()
