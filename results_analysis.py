# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:37:45 2026

@author: Nicc
"""

import argparse

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, binomtest

def parse_args():
    parser = argparse.ArgumentParser(description="Verification tests")
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        default=40,
        help="Range defining amount of seeds to verify; (i.e 20)",
    )
    parser.add_argument(
        "-m",
        "--seeds_max",
        type=int,
        default=40,
        help="Final seed in the simulation, end of the loop",
    )
    parser.add_argument(
        "-i",
        "--id",
        default=2,
        help="Verification ID Number",
    )
    parser.add_argument(
        "-w",
        "--save",
        type=str,
        required=True,
        help="'yes' if you want to save the .txt file for paired sample t-test and binomial test",
    )
    parser.add_argument(
        "-p",
        "--save_plot",
        type=str,
        required=True,
        help="'yes' if you want to save the .png file for plotting results",
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    fin_results = []
    
    for seed in range(((args.seeds_max + 1) - args.seeds), args.seeds_max + 1):
        
        df = pd.read_csv(f"C:/Users/Nicc/Desktop/CNR_Model/testings/testing_{str(args.id)}/test_seed{seed}/Test_Simulation.csv")
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
    
    #Paired samples t-test
    
    t_nd, p_nd = ttest_rel(ND_a0, ND_a1)
    t_d, p_d = ttest_rel(D_a0, D_a1)
    
    #Binomial test
    
    total_ND_a0 = sum(ND_a0)
    total_ND_a1 = sum(ND_a1)
    
    total_D_a0 = sum(D_a0)
    total_D_a1 = sum(D_a1)
    
    total_ND = total_ND_a0 + total_ND_a1
    total_D = total_D_a0 + total_D_a1
    
    res_nd = binomtest(
    k=total_ND_a0,      
    n=total_ND,         
    p=0.5,              
    alternative='greater'
    )
    
    res_d = binomtest(
    k=total_D_a0,      
    n=total_D,        
    p=0.5,              
    alternative='greater'
    )
    
    if args.save == 'yes':
        print('Creating .txt results file')
        save_dir = f"results/simulation_ID_{args.id}"
        os.makedirs(save_dir, exist_ok=True)
        output_path = f"results/simulation_ID_{args.id}/analysis_ID_{args.id}.txt"
        
        with open(output_path, "w") as f:
            f.write("=== T-TEST RESULTS ===\n\n")
    
            f.write("ND condition: action0 vs action1\n")
            f.write(f"t = {t_nd:.4f}\n")
            f.write(f"p = {p_nd:.6f}\n\n")
            
            f.write("D condition: action0 vs action1\n")
            f.write(f"t = {t_d:.4f}\n")
            f.write(f"p = {p_d:.6f}\n\n")
            
            f.write("=== BINOMIAL TESTS ===\n\n")
        
            f.write("ND condition\n")
            f.write(f"Action0 selections: {total_ND_a0}/{total_ND}\n")
            f.write(f"p-value: {res_nd.pvalue:.8f}\n\n")
            
            f.write("D condition\n")
            f.write(f"Action0 selections: {total_D_a0}/{total_D}\n")
            f.write(f"p-value: {res_d.pvalue:.8f}")
        
    #Saving analysis results image
        
        stats_table = pd.DataFrame({
            "Test": [
                "ND: A0 vs A1 (paired t-test)",
                "D: A0 vs A1 (paired t-test)",
                "ND: A0 vs chance (binomial)",
                "D: A0 vs chance (binomial)"
            ],
            "Statistic": [
                t_nd,
                t_d,
                None,
                None
            ],
            "p-value": [
                np.round(p_nd, 5),
                np.round(p_d, 5),
                np.round(res_nd.pvalue, 5),
                np.round(res_d.pvalue,5 )
            ]
        })
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        
        table = ax.table(
            cellText=stats_table.values,
            colLabels=stats_table.columns,
            cellLoc="center",
            loc="center"
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        for i, p in enumerate(stats_table["p-value"]):
            if p < 0.05:
                table[(i + 1, 2)].set_text_props(color="red", weight="bold")
                
        save_dir = f"results/simulation_ID_{args.id}"
        os.makedirs(save_dir, exist_ok=True)
        
        table_path = os.path.join(save_dir, f"analysis_ID_{args.id}_stats_table.png")
        
        plt.savefig(table_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
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
    
    if args.save_plot == 'yes':
        save_dir = f"results/simulation_ID_{args.id}"
        os.makedirs(save_dir, exist_ok=True)
        
        plot_path = os.path.join(save_dir, f"analysis_ID_{args.id}_barplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        