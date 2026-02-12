# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:37:59 2026

@author: Nicc
"""

from Layer_types import BLA_IC_Layer
import numpy as np

class BLA_IC_sm():
    
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
        
        self.W = np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 5.0, 0.0, -10.0, 0.0],
                      [0.0, 0.0, 0.0, 5.0, 0.0, -10.0]])
        
    def reset_activity(self):
        
        self.BLA_IC.reset_activity()
        
    def learning(self, da):
        
        self.BLA_IC.learn(da)
        
    def step(self, inp, da, learning = True):
        
        self.BLA_IC.step(np.dot(self.W, inp))
        
        if learning:
            self.BLA_IC.learn(da)
                
                