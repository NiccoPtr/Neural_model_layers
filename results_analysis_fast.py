# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:07:18 2026

@author: Nicc
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import ttest_rel
from pathlib import Path
from params import Parameters
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Verification tests")
    parser.add_argument(
        "-i",
        "--id",
        default=1,
        help="Verification ID Number",
    )
    parser.add_argument(
        "-p",
        "--save_plot",
        type=str,
        required=True,
        help="'yes' if you want to save the .png file for plotting results, or 'show' to see but not save the image, else 'no'",
    )
    
    return parser.parse_args()

def mean_act(x, thr):
    
    res = x.iloc[900:, :].mean(0)
    res = res.argmax() if sum(res >= thr) == 1 else 2
    
    return res

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

if __name__ == '__main__':
    
    args = parse_args()
    parameters = Parameters()
    if Path(f"C:/Users/Nicc/Desktop/CNR_Model/trainings/training_{str(args.id)}/sim_seed1/prm_file.json").exists():
        parameters.load(f"C:/Users/Nicc/Desktop/CNR_Model/trainings/training_{str(args.id)}/sim_seed1/prm_file.json", mode="json")
        print('Imported parameters succesfully')
    else:
        raise ValueError('Parameters file not found')
        
    files = glob.glob(f"C:/Users/Nicc/Desktop/CNR_Model/testings/testing_{str(args.id)}/test_seed*/Test_Simulation.csv")
    
    thr = parameters.threshold["MC"]
    dfs = [pd.read_csv(f) for f in files]
    for i, df in enumerate(dfs):
        df.loc[:, "Seed"] = i + 1
    
    df = pd.concat(dfs)
    
    df = (
        df.groupby(["Seed", "Phase", "Trial"])[["MC_Unit_0", "MC_Unit_1"]]
        .apply(lambda x: mean_act(x, thr))
        .reset_index(name="Decision")
    )
    
    df.replace(
        {"Decision": {0: "Lever", 1: "Chain", 2: "None"}, "Phase": {1: "ND", 2: "D"}}, inplace=True
    )
    
    
    df = (
        df.groupby(["Seed", "Phase"])["Decision"]
        .apply(
            lambda x: pd.DataFrame(
                dict(Lever=[np.sum(x == "Lever")], Chain=[np.sum(x == "Chain")], NONE=[np.sum(x == "None")])
            )
        )
        .reset_index()
    )
    
    df = pd.melt(
        df,
        id_vars=["Seed", "Phase"],
        value_vars=["Lever", "Chain"],
        value_name="freq",
        var_name="Decision",
    )
    
    stats_df = df.pivot_table(
        index=["Seed", "Phase"],
        columns="Decision",
        values="freq"
    ).dropna().reset_index()
    
    pvals = {}
    
    for phase in ["ND", "D"]:
        sub = stats_df[stats_df["Phase"] == phase]
        
        t, p = ttest_rel(sub["Lever"], sub["Chain"])
        pvals[phase] = p
            
    fig, ax = plt.subplots()
    
    sns.barplot(df, x="Phase", y="freq", hue="Decision", errorbar="sd", order=["ND", "D"])
    
    # -------------------------
    # add significance markers
    # -------------------------
    y_max = df["freq"].max()
    offset = y_max * 0.08
    
    # x positions inside each phase group (Seaborn default hue spacing)
    phase_positions = {"ND": 0, "D": 1}
    
    hue_offset = {"Lever": -0.2, "Chain": 0.2}
    
    for i, phase in enumerate(["ND", "D"]):
        
        star = p_to_stars(pvals[phase])
        
        x1 = phase_positions[phase] + hue_offset["Lever"]
        x2 = phase_positions[phase] + hue_offset["Chain"]
        
        y = y_max + offset * (i + 1)
    
        ax.plot([x1, x2], [y, y], lw=1.5, c="black")
        ax.text((x1 + x2) / 2, y + offset * 0.2, star,
                ha="center", va="bottom")
    
    ax.set_ylim(0, y_max + offset * 4)

    if args.save_plot == 'yes':
        save_dir = f"results/simulation_ID_{args.id}"
        os.makedirs(save_dir, exist_ok=True)
        
        plot_path = os.path.join(save_dir, f"analysis_ID_{args.id}_barplot.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        
    elif args.save_plot == 'show':
        
        plt.show()