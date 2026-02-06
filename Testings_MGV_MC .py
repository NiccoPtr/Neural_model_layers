# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

from params import Parameters
from Layer_types import Leaky_units_exc, BG_dl_Layer
import numpy as np, matplotlib.pyplot as plt

parameters = Parameters()
parameters.load("prm_file.json" ,mode = "json")
parameters.scheduling = {
                        "trials": 10,
                        "phases": [0.25, 0.50, 0.75, 1.0],
                        "timesteps": 1000,
                        "input": np.array([[0.2, 0.2],
                                           [0.2, 0.2],
                                           [0.2, 0.4],
                                           [0.2, 0.4]])
                       }
                                            
rng = np.random.RandomState(parameters.seed)

BG_dl = BG_dl_Layer(parameters.N["BG_dl"], 
                    parameters.tau["BG_dl"], 
                    parameters.baseline["DLS"],
                    parameters.baseline["STNdl"],
                    parameters.baseline["GPi"],
                    parameters.BG_dl_W["DLS_GPi_W"], 
                    parameters.BG_dl_W["STNdl_GPi_W"],
                    rng,
                    parameters.noise["BG_dl"],
                    parameters.threshold["BG_dl"])

MGV = Leaky_units_exc(parameters.N["MGV"], 
                      parameters.tau["MGV"],
                      parameters.baseline["MGV"],
                      rng,
                      parameters.noise["MGV"],
                      parameters.threshold["MGV"])

MC = Leaky_units_exc(parameters.N["MC"], 
                     parameters.tau["MC"], 
                     parameters.baseline["MC"],
                     rng,
                     parameters.noise["MC"],
                     parameters.threshold["MC"])

Ws = {"inp_DMS": np.eye(parameters.N["BG_dl"]),
      "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
      "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
      "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
      "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
      "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"]
      }

BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
MGV_output_pre = np.zeros(parameters.N["MGV"])
MC_output_pre = np.zeros(parameters.N["MC"])

results = []

for trial in range(parameters.scheduling["trials"]):
    
    BG_dl.reset_activity()
    MGV.reset_activity()
    MC.reset_activity()
    BG_dl_output = []
    MGV_output = []
    MC_output = []
    _input_ = []
    
    if trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][0]:
        phase = 1
        
    elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][1]:
        phase = 2
        
    elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][2]:
        phase = 3
        
    else: 
        phase = 4
        
    inp = parameters.scheduling["input"][phase -1]
    
    for t in range(parameters.scheduling["timesteps"]):
    
        BG_dl.step(np.dot(Ws["inp_DMS"], inp),
                   np.dot(Ws["MC_DLS"], MC_output_pre),
                   np.dot(Ws["MC_STNdl"], MC_output_pre))
        
        MGV.step(np.dot(Ws["GPi_MGV"], BG_dl_output_pre +
                 np.dot(Ws["MC_MGV"], MC_output_pre)))
        
        MC.step(np.dot(Ws["MGV_MC"], MGV_output_pre))
        
        BG_dl_output_pre = BG_dl.output_BG_dl.copy()
        
        MGV_output_pre = MGV.output.copy()
        
        MC_output_pre = MC.output.copy()
        
        BG_dl_output.append(np.round(BG_dl.output_BG_dl.copy(), 4))
        MGV_output.append(np.round(MGV.output.copy(), 4))
        MC_output.append(np.round(MC.output.copy(), 4))
        _input_.append(inp.copy())
    
    result = {
        "Trial": trial + 1,
        "Phase": phase,
        "BG_dl_output": BG_dl_output.copy(),
        "MGV_output": MGV_output.copy(),
        "MC_output": MC_output.copy(),
        "Inputs_timeline": _input_.copy()
        }
    
    results.append(result)
    
def plotting(results):
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i,res in enumerate(results):
        
        BG_dl = np.array(res['BG_dl_output']) * -1
        
        axs[i].plot(BG_dl[:, 0], label = 'Unit_1')
        axs[i].plot(BG_dl[:, 1], label = 'Unit_2')
        
        axs[i].set_title(f'BG_dl-Phase: {res["Phase"]}, Input: {res["Inputs_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    plt.show()
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i,res in enumerate(results):
        
        MGV = np.array(res['MGV_output'])
        
        axs[i].plot(MGV[:, 0], label = 'Unit_1')
        axs[i].plot(MGV[:, 1], label = 'Unit_2')
        
        axs[i].set_title(f'MGV-Phase: {res["Phase"]}, Input: {res["Inputs_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    plt.show()
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i,res in enumerate(results):
        
        MC = np.array(res['MC_output'])
        
        axs[i].plot(MC[:, 0], label = 'Unit_1')
        axs[i].plot(MC[:, 1], label = 'Unit_2')
        
        axs[i].set_title(f'MC-Phase: {res["Phase"]}, Input: {res["Inputs_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    plt.show()
    

