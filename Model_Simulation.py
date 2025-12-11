# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

from params import Parameters
from Model_class import Model
import numpy as np, matplotlib.pyplot as plt, argparse

def Simulation(parameters, model = None):
    
    if model is None:
        model = Model(parameters)
        
    results = []
    
    for trial in range(parameters.scheduling["trials"]):
        
        model.reset_activity()   
        MC_output = []
        state_t = []
        
        if trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][0]:
            phase = 1
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][1]:
            phase = 2
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][2]:
            phase = 3
            
        else: 
            phase = 4
            
        state = parameters.scheduling["states"][phase - 1]
        
        for t in range(parameters.scheduling["timesteps"]):
            
            model.step(state, learning=True) 
            
            if np.isnan(model.Ws["BLA_IC_NAc"]).any():
                
                print("NaN value detected")
                return model, results
            
            action = model.MC.output.copy()
            
            MC_output.append(np.round(action.copy(), 4))
            state_t.append(state.copy())
            
            if np.any(action >= model.MC.threshold):
                
                winner = np.argmax(action)

                state[2:4] = 0.0
                state[2 + winner] = 1.0
                
            else:
                
                state[2:4] = 0.0
            
        result = {
            "Trial": trial + 1,
            "Phase": phase,
            "MC_Output": MC_output.copy(),
            "States_timeline": state_t.copy()
            }
        
        results.append(result)
    
    return model, results


def plotting(results, save = False):
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, res in enumerate(results):
        
        MC = np.array(res["MC_Output"])
        
        axs[i].plot(MC[:, 0], label = "MC Unit_1")
        axs[i].plot(MC[:, 1], label = "MC Unit_2")
        
        axs[i].set_title(f'Phase: {res["Phase"]}, Input: {res["States_timeline"][0]}')
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
    
def plotting_2(results, save = False):
    
    phases = sorted(set(res["Phase"] for res in results))
    
    rows = 1 + len(phases) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, phase in enumerate(phases):

        phase_results = [r for r in results if r["Phase"] == phase]
        phase_final_res = phase_results[-1]
        
        MC = np.array(phase_final_res["MC_Output"])
        
        axs[i].plot(MC[:, 0], label = "MC Unit_1")
        axs[i].plot(MC[:, 1], label = "MC Unit_2")
        
        axs[i].set_title(f'Phase: {phase_final_res["Phase"]}, Input: {phase_final_res["States_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)
        
    # Hide extra unused subplots
    for j in range(len(phases), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    if save:
        filename = "Simulation_plot.png"
        plt.savefig(filename, dpi=300)
        
    plt.show()
        

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Run neural model simulation.")
    # parser.add_argument("--epochs", type=int, default=10,
    #                     help="Number of training epochs")
    # parser.add_argument("--timesteps", type=int, default=1000,
    #                     help="Number of timesteps per epoch")
    
    # parser.add_argument("--inputs", nargs='+', type=float,
    #                 default=[1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    #                 help="Environmental inputs incoming")
    
    # args = parser.parse_args()
    
    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.scheduling = {
                            "trials": 100,
                            "phases": [0.25, 0.50, 0.75, 1.0],
                            "timesteps": 1000,
                            "states": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                       [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                                       [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                                       ])
                            }
    
    # epochs = args.epochs
    # timesteps = args.timesteps
    # inputs = np.array(args.inputs)
    model, results = Simulation(parameters)
    
    # for res in results:
    # 	print("Final Motor Cortex output at epoch " + str(res["Trial"]) + ": " + str(res["MC_Output"][-1]))
    # 	print("Starting environmental state:" + str(res["States_timeline"][0]))

    plotting_2(results)