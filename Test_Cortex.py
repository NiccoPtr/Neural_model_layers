# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:12:15 2026

@author: Nicc
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

from Cortex_simulation import Cortex
from params import Parameters

plt.ion()


def plotting(res):

    plt.close("all")

    # Isolating single layers
    DM = np.array(res["DM_output"])
    PL = np.array(res["PL_output"])
    P = np.array(res["P_output"])
    PFCd_PPC = np.array(res["PFCd_PPC_output"])
    MGV = np.array(res["MGV_output"])
    MC = np.array(res["MC_output"])
    actions = np.array(res['Action_selection'])

    # Plotting set up
    plots = [
        ("DM", [(DM[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("PL", [(PL[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("P", [(P[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("PFCd_PPC", [(PFCd_PPC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MGV", [(MGV[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MC", [(MC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1))
    ]

    n_rows = len(plots) + 1
    fig = plt.figure(figsize=(14, 2.2 * n_rows))
    gs = GridSpec(n_rows, 2, width_ratios=[1, 6], hspace=0.25)

    shared_ax = None

    for i, (title, lines, ylim) in enumerate(plots):
        title_ax = fig.add_subplot(gs[i, 0])
        ax = fig.add_subplot(gs[i, 1], sharex=shared_ax)

        if shared_ax is None:
            shared_ax = ax

        # Left column: titles only
        title_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
        title_ax.axis("off")

        # Right column: actual plot
        for y, label in lines:
            ax.plot(y, label=label)

        ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=5)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(labelbottom=False)
    #Action selection
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Action selected", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        actions.reshape(-1, 1).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2
    )
    
    ax.set_yticks([])  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Shared x-axis
    ax.set_xlabel("Timestep")

    plt.tight_layout()

    xmin, xmax = shared_ax.get_xlim()
    pad = 0.1 * (xmax - xmin)
    shared_ax.set_xlim(xmin, xmax + pad)

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="BG_dl-MGV-MC loop simulation")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "-p",
        "--inp",
        type=float,
        nargs=2,
        default=[-0.6, -0.6],
        help="Input values (two floats)",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="Number of timesteps",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="plot",
        help="Output mode ('plot', 'save', 'short_save' 'stream')",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps

    parameters = Parameters()
    if Path("prm_file.json").exists():
        parameters.load("prm_file.json", mode="json")
    parameters.seed = args.seed

    rng = np.random.RandomState(parameters.seed)
    C_Model = Cortex(parameters, rng)

    DM_output = []
    PL_output = []
    P_output = []
    PFCd_PPC_output = []
    MGV_output = []
    MC_output = []
    _input_ = []
    actions = []

    C_Model.reset_activity()

    for t in range(timesteps):
    
        C_Model.step(inp)
        
        action = C_Model.MC.output.copy()
        if np.any(action >= C_Model.MC.threshold):
            winner = np.argmax(action) + 1
        else:
            winner = np.array(0)
         
        DM_output.append(C_Model.DM.output.copy())
        PL_output.append(C_Model.PL.output.copy())
        P_output.append(C_Model.P.output.copy())
        PFCd_PPC_output.append(C_Model.PFCd_PPC.output.copy())
        MGV_output.append(C_Model.MGV.output.copy())
        MC_output.append(C_Model.MC.output.copy())
        _input_.append(inp.copy())
        actions.append(winner.copy())

    result = {
        "Seed": np.ones(timesteps) * parameters.seed,
        "Inputs_timeline": _input_,
        "DM_output": DM_output,
        "PL_output": PL_output,
        "P_output": P_output,
        "PFCd_PPC_output": PFCd_PPC_output,
        "MGV_output": MGV_output,
        "MC_output": MC_output,
        'Action_selection': actions
    }
    
    if args.mode == "plot":
        plotting(result)
        print(f"""
              Seed: {args.seed}
              Input: {args.inp}
              Cortex Noise: {parameters.noise['MC']}
              Thalamus Baseline: {parameters.baseline['MGV']}
              """)
        # input("Press Enter to exit")