# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:13:28 2026

@author: Nicc
"""

from params import Parameters
from Layer_types import BLA_IC_Layer
import numpy as np, matplotlib.pyplot as plt

parameters = Parameters()
parameters.load("prm_file.json" ,mode = "json")
parameters.scheduling = {
                        "trials": 10,
                        "phases": [0.25, 0.50, 0.75, 1.0],
                        "timesteps": 1000,
                        "input": np.array([[0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
                                           [0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
                                           [0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
                                           [0.2, 0.4, 0.0, 0.0, 0.0, 0.0]]),
                        'DA': np.array([0.0])
                       }

rng = np.random.RandomState(parameters.seed)

BLA_IC = BLA_IC_Layer(parameters.N["BLA_IC"],
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


W = np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 5.0, 0.0, -10.0, 0.0],
              [0.0, 0.0, 0.0, 5.0, 0.0, -10.0]])

results = []

for trial in range(parameters.scheduling['trials']):
    
    BLA_IC.reset_activity()
    BLA_IC_output = []
    _input_ = []
    W_timeline = []
    
    if trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][0]:
        phase = 1
        
    elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][1]:
        phase = 2
        
    elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][2]:
        phase = 3
        
    else: 
        phase = 4
        
    inp = parameters.scheduling["input"][phase -1]
    
    for t in range(parameters.scheduling['timesteps']):
        
        BLA_IC.step(np.dot(W, inp))
        BLA_IC.learn(parameters.scheduling['DA'])
        
        BLA_IC_output.append(np.round(BLA_IC.output.copy(), 4))
        _input_.append(inp.copy())
        W_timeline.append(np.round(BLA_IC.W.copy(), 4))

    result = {'Trial': trial + 1,
              'Phase': phase,
              'BLA_IC_output': BLA_IC_output.copy(),
              'Inputs_timeline': _input_.copy(),
              'W': W_timeline.copy()
              }
    
    results.append(result)

def plotting(results):
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i,res in enumerate(results):
        
        BLA_IC = np.array(res['BLA_IC_output']) 
        
        axs[i].plot(BLA_IC[:, 0], label = 'Unit_1')
        axs[i].plot(BLA_IC[:, 1], label = 'Unit_2')
        axs[i].plot(BLA_IC[:, 2], label = 'Unit_3')
        axs[i].plot(BLA_IC[:, 3], label = 'Unit_4')
        
        axs[i].set_title(f'BLA_IC-Phase: {res["Phase"]}, Input: {res["Inputs_timeline"][0]}')
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
        
        W = np.array(res['W']) 
        n = W.shape[1]
        for z in range(n):       
            for j in range(n): 
                axs[i].plot(W[:, z, j], label=f"{j+1}_{i+1}")
                axs[i].set_title(f'Phase: {res["Phase"]}, Matrix learning')
                axs[i].legend(loc = 'upper right')
                axs[i].set_xlabel('Timestep')
                axs[i].set_ylabel('Weight')
                axs[i].set_ylim(0, parameters.BLA_Learn['max_W'])
                
    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    plt.show()
        
        
