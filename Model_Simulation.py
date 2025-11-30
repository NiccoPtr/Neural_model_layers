# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

from params import Parameters
from Model_class import Model
import numpy as np, matplotlib.pyplot as plt, argparse

def Simulation(parameters, inputs, epochs, timesteps, model = None):
    
    if model is None:
        model = Model(parameters)
        model.set_Ws(parameters)

    results = []
    
    for epoch in range(epochs):
           
        MC_output = []
        inp_t = []
        
        for t in range(timesteps):
            model.step(inputs, learning=True) 
            
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run neural model simulation.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of timesteps per epoch")
    
    parser.add_argument("--inputs", nargs='+', type=float,
                    default=[1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    help="Environmental inputs incoming")
    
    args = parser.parse_args()
    
    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    
    epochs = args.epochs
    timesteps = args.timesteps
    inputs = np.array(args.inputs)
    model, results = Simulation(parameters, inputs, epochs, timesteps)
    
    for res in results:
    	print("Final Motor Cortex output at epoch " + str(res["Epoch"]) + ": " + str(res["MC_Output"][-1]))
    	print("Environmental Input:" + str(res["Inp"]))

    plotting(results)
    
	
	

    

    
    
    
        
      