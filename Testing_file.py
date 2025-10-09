# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:21:19 2025

@author: Utente
"""

from Testing_simulation_file import create_input_levels, set_layers, set_env, set_DA, run_simulation
from params import parameters
import numpy as np

deltas = np.linspace(4.9, 4.9, 1)
DAs = np.linspace(0.1, 1, 10)
seeds = np.linspace(1, 40, 40)

results_test = []
for delta in deltas:
    for DA in DAs:
        for seed in seeds:
            BG_dl, MGV, MC = set_layers(parameters, int(seed))
            results = run_simulation((0, 1, 1),
                                 (0.1, 1, 1), 
                                 600,
                                 0.1,
                                 delta,
                                 DA,
                                 BG_dl, 
                                 MGV, 
                                 MC)
            
            results_test.append(results.copy())

list_avg_f_outputs = []

for i, delta in enumerate(deltas):
    avg_final_output = []
    
    for res in results_test:
        if res[0]["delta"] == delta:
            avg_final_output.append(max(res[0]["Final_output"]))
    
    list_avg_f_outputs.append(avg_final_output.copy())

threshold = 0.38
percents = []

for list_ in list_avg_f_outputs:
    group_percent = []
    
    for i in range(0, len(list_), len(seeds)):
        group = list_[i: i+len(seeds)]
        percent = np.sum(np.array(group) > threshold) / len(group) * 100
        group_percent.append(percent)
    percents.append(group_percent)
            
    
            




