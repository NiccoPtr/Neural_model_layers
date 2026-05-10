# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:37:45 2026

@author: Nicc
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Verification tests")
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        help="Range defining seeds to verify; (i.e 20)",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=int,
        help="Verification ID Number",
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    fin_results = []
    
    for seed in range(1, args.seeds + 1):
        
        df = pd.read_csv(f"C:/Users/Nicc/Desktop/CNR_Model/testings/testing_{args.id}/test_seed{seed}/Test_Simulation.csv")
        result = {}
        act_D = []
        act_ND = []
        result["Seed"] = seed
        
        for trial in range(0, (int(((df.iloc[-1]['Trial']))) // 2) + 1):
            
            lim_1 = trial
            lim_2 = trial + (int(((df.iloc[-1]['Trial']))) // 2)
            
            cond_ND = df[
                (df["Trial"] == lim_1)
                ].sort_values("Timestep").copy()
            MC_ND= cond_ND.filter(like="MC_Unit").to_numpy()
            winner_ND = np.argmax(MC_ND[-1])
            
            cond_D = df[
                (df["Trial"] == lim_2)
                ].sort_values("Timestep").copy()
            MC_D = cond_D.filter(like="MC_Unit").to_numpy()
            winner_D = np.argmax(MC_D[-1])
            
            act_ND.append(winner_ND)
            act_D.append(winner_D)
            
        result["ND_action0"] = act_ND.count(0)
        result["ND_action1"] = act_ND.count(1)
        
        result["D_action0"] = act_D.count(0)
        result["D_action1"] = act_D.count(1)
        
        fin_results.append(result)
        
    ND_a0 = [r["ND_action0"] for r in fin_results]
    ND_a1 = [r["ND_action1"] for r in fin_results]
    
    D_a0 = [r["D_action0"] for r in fin_results]
    D_a1 = [r["D_action1"] for r in fin_results]
    
    mean_ND_a0 = np.mean(ND_a0)
    std_ND_a0  = np.std(ND_a0)
    
    mean_ND_a1 = np.mean(ND_a1)
    std_ND_a1  = np.std(ND_a1)
    
    mean_D_a0 = np.mean(D_a0)
    std_D_a0  = np.std(D_a0)
    
    mean_D_a1 = np.mean(D_a1)
    std_D_a1  = np.std(D_a1)
        
    #Plotting results
    
    labels = ["L1", "L2", "L1", "L2"]
    x = np.arange(len(labels))
    bar_width = 0.6

    means = [mean_ND_a0, mean_ND_a1, mean_D_a0, mean_D_a1]
    errors = [std_ND_a0, std_ND_a1, std_D_a0, std_D_a1]
    
    colors = ["lightgray", "lightgray", "gray", "gray"]
    
    fig, ax = plt.subplots(figsize=(7,6))
    
    ax.bar(
    x,
    means,
    yerr=errors,
    capsize=4,
    color=colors,
    edgecolor="black"
)

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add group labels (ND / D)
    ax.text(0.5, -2, "ND", ha='center', va='top', fontsize=11)
    ax.text(2.5, -2, "D", ha='center', va='top', fontsize=11)
    
    ax.set_ylabel(f"Mean number of selections (out of {args.seeds})")
    
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    plt.show()
                        