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
        inp_t = []
        
        for t in range(timesteps):
            model.step(inputs) 
            model.learning(parameters, inputs)
            model.update_output_pre()
            
            MC_output.append(np.round(model.MC.output.copy(), 4))
            inp_t.append(inputs.copy())
            
        result = {
            "Epoch": epoch,
            "MC_Output": MC_output,
            "Input_timeline": inp_t,
            "Inp": inputs.copy()
            }
        
        results.append(result)
    
    return model, results

def plotting(results, save = False):
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, res in enumerate(results):
        
        axs[i].plot(res["MC_Output"])
        
        axs[i].set_title(f'Input: {res["Inp"]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    # Hide extra unused subplots
    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    if save:
        filename = "Simulation_plot.png"
        plt.savefig(filename, dpi=300)
        
    plt.show()

    
    
        
      