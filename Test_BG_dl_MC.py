# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

from CT_BG_simulation import CT_BG
from params import Parameters

plt.ion()


def plotting(res):

    plt.close("all")

    # Isolating single layers
    DLS = np.array(res["DLS_output"]) * -1
    STNdl = np.array(res["STNdl_output"])
    BG_dl = np.array(res["BG_dl_output"]) * -1
    MGV = np.array(res["MGV_output"])
    MC = np.array(res["MC_output"])
    actions = np.array(res['Action_selection'])
    W = np.array(res["Weight_timeline"])

    # Plotting set up
    plots = [
        ("DLS", [(DLS[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("STNdl", [(STNdl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPi", [(BG_dl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MGV", [(MGV[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MC", [(MC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1))
    ]

    n_rows = len(plots) + 2
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
    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

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

    # Weight heatmap
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights learning", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])

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
        default=(1.0, 0.0),
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
        help="Output mode ('plot')",
    )
    parser.add_argument(
        "-da",
        "--dopamine",
        type=float,
        nargs=2,
        default=(2.0, 0.0),
        help="Insert dopamine for learnig: float type",
    )
    parser.add_argument(
        "--PFCd_PPC_1", type=float, default=0.2, help="Insert PFCd_PPC_1 input value"
    )
    parser.add_argument(
        "--PFCd_PPC_2", type=float, default=0.2, help="Insert PFCd_PPC_2 input value"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps
    da = np.array(args.dopamine)

    parameters = Parameters()
    if Path("prm_file.json").exists():
        parameters.load("prm_file.json", mode="json")
    parameters.seed = args.seed

    rng = np.random.RandomState(parameters.seed)
    CT_BG_model = CT_BG(parameters, rng)

    DLS_output = []
    STNdl_output = []
    BG_dl_output = []
    MGV_output = []
    MC_output = []
    W_timeline = []
    _input_ = []
    actions = []

    CT_BG_model.reset_activity()

    for t in range(timesteps):
        
        # if t == timesteps*0.15:
        #     da *= 0.0

        CT_BG_model.step(parameters, inp, da, PFCd_PPC_inp=(args.PFCd_PPC_1, args.PFCd_PPC_2), learn=True)
        
        action = CT_BG_model.MC.output.copy()
        if np.any(action >= CT_BG_model.MC.threshold):
            winner = np.argmax(action) + 1
        else:
            winner = np.array(0)
         
        DLS_output.append(CT_BG_model.BG_dl.DLS.output.copy())
        STNdl_output.append(CT_BG_model.BG_dl.STNdl.output.copy())
        BG_dl_output.append(CT_BG_model.BG_dl.output_BG_dl.copy())
        MGV_output.append(CT_BG_model.MGV.output.copy())
        MC_output.append(CT_BG_model.MC.output.copy())
        W_timeline.append(CT_BG_model.Ws["inp_DLS"].copy())
        _input_.append(inp.copy())
        actions.append(winner.copy())

    result = {
        "Seed": np.ones(timesteps) * parameters.seed,
        "Inputs_timeline": _input_,
        'DLS_output': DLS_output,
        'STNdl_output': STNdl_output,
        "BG_dl_output": BG_dl_output,
        "MGV_output": MGV_output,
        "MC_output": MC_output,
        "Weight_timeline": W_timeline,
        'Action_selection': actions
    }
    
    if args.mode == "plot":
        print(f"""
              Seed: {args.seed}
              Input: {args.inp}
              MC Noise: {parameters.noise['MC']}
              GPi Baseline: {parameters.baseline['GPi']}
              MGV Baseline: {parameters.baseline['MGV']}
              PFCd_PPC inp: {(args.PFCd_PPC_1, args.PFCd_PPC_2)}
              """)
        plotting(result)
        plt.show()
