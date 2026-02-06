# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 07:43:40 2026

@author: Nicc
"""

from Layer_types import BG_dl_Layer, Leaky_units_exc
import numpy as np, matplotlib.pyplot as plt
from params import Parameters

#BG_dl isolated simulation

parameters = Parameters()
parameters.load("prm_file.json" ,mode = "json")
parameters.scheduling = {
                        "trials": 10,
                        "phases": [0.25, 0.50, 0.75, 1.0],
                        "timesteps": 1000,
                        "input": np.array([[0.2, 0.2],
                                            [0.2, 0.2],
                                            [0.2, 0.4],
                                            [0.2, 0.4]]),
                        "inp_feedbackDLS": np.array([[0.3, 0.3],
                                                     [0.3, 0.3],
                                                     [0.3, 0.3],
                                                     [0.3, 0.3]]),
                        "inp_feedbackSTNdl": np.array([[0.2, 0.2],
                                                       [0.2, 0.2],
                                                       [0.2, 0.2],
                                                       [0.2, 0.2]])
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

Ws = {"Mani_DLS": np.eye(parameters.N["BG_dl"]),
      "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
      "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"]
      }

results = []

for trial in range(parameters.scheduling["trials"]):
    
    BG_dl.reset_activity()
    BG_dl_output = []
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
    inp_fb_DLS = parameters.scheduling["inp_feedbackDLS"][phase -1]
    inp_fb_STNdl = parameters.scheduling["inp_feedbackSTNdl"][phase -1]
    
    for t in range(parameters.scheduling["timesteps"]):
        
        BG_dl.step(np.dot(Ws["Mani_DLS"], inp), 
                   np.dot(Ws["MC_DLS"], inp_fb_DLS), 
                   np.dot(Ws["MC_STNdl"], inp_fb_STNdl)
                   )
        
        BG_dl_output.append(np.round(BG_dl.output_BG_dl.copy(), 4))
        _input_.append(inp.copy())
    
    result = {
        "Trial": trial + 1,
        "Phase": phase,
        "BG_dl_output": BG_dl_output.copy(),
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
        
        axs[i].set_title(f'Phase: {res["Phase"]}, Input: {res["Inputs_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    plt.show()
    