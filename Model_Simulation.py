# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

from params import parameters
from Model_class import Model
import numpy as np, matplotlib.pyplot as plt

def Simulation(parameters, inputs, epochs, timesteps):
    
    model = Model(parameters)
    model.set_env(parameters)
    results =  [] 
    
    for epoch in range(epochs):
           
        MC_output = []
        inp = []
        
        for t in range(timesteps):
            model.step(inputs) 
            model.learning(parameters, inputs)
            model.update_output_pre()
            
            MC_output.append(np.round(model.MC.output.copy(), 4))
            inp.append(inputs.copy())
            
        result = {
            "Epoch": epoch,
            "MC_Output": MC_output,
            "Input_income": inp
            }
        
        results.append(result)
    
    return model, results
        
      