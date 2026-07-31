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


def plotting(res):

    plt.close("all")

    # Isolating single layers
    DLS_1 = np.array(res["DLS_1_output"]) * -1
    DLS_2 = np.array(res["DLS_2_output"]) * -1
    STNdl = np.array(res["STNdl_output"])
    BG_dl = np.array(res["BG_dl_output"]) * -1
    GPe = np.array(res["GPe_output"]) * -1
    MGV = np.array(res["MGV_output"])
    MC = np.array(res["MC_output"])
    DA = np.array(res["DA_timeline"])
    W1 = np.array(res["W1_timeline"])
    W2 = np.array(res["W2_timeline"])

    # Plotting set up
    plots = [
        ("DLS_1", [(DLS_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("DLS_2", [(DLS_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("STNdl", [(STNdl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPi", [(BG_dl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPe", [(GPe[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MGV", [(MGV[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("MC", [(MC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("DA", [(DA[:], f"Unit_{i+1}") for i in range(1)], (-0.1, 1)),
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

    # Weight heatmap
    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "W_1 learning", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W1.reshape(-1, 2 * 2).T,
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

    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "W_2 learning", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W2.reshape(-1, 2 * 2).T,
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
        "-d",
        "--dopamine",
        type=float,
        default=0.0,
        help="Insert dopamine for learnig: float type",
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

    DLS_1_output = []
    DLS_2_output = []
    STNdl_output = []
    BG_dl_output = []
    GPe_output = []
    MGV_output = []
    MC_output = []
    DA = []
    W1_timeline = []
    W2_timeline = []
    _input_ = []

    CT_BG_model.reset_activity()

    winner = None

    for t in range(timesteps):
        
        if t <= 50:
            inp *= 0.0

        else:
            da = np.array(args.dopamine)
            inp = np.array(args.inp)

        if winner and t > 70:
            da = 1.0

            # if inp[winner - 1] == 0:
            #     da = np.array(args.dopamine)

            # elif inp[winner -1] == 1:
            #     da = 1.0

        if t > 90:
            da = np.array(args.dopamine)

        # if t == timesteps//2:
        #     if inp[0] == 1.0:
        #         inp *= 0.0
        #         inp[1] = 1.0
        #     elif inp[1] == 1.0:
        #         inp *= 0.0
        #         inp[0] = 1.0

        CT_BG_model.step(parameters, inp, da, learn=True)

        action = CT_BG_model.MC.output.copy()
        if np.any(action >= CT_BG_model.MC.threshold) and t > 50:
            winner = np.argmax(action) + 1
        else:
            winner = np.array(0)
        
        DLS_1_output.append(CT_BG_model.BG_dl.Str1.output.copy())
        DLS_2_output.append(CT_BG_model.BG_dl.Str2.output.copy())
        STNdl_output.append(CT_BG_model.BG_dl.STN.output.copy())
        BG_dl_output.append(CT_BG_model.BG_dl.output_BG.copy())
        GPe_output.append(CT_BG_model.BG_dl.GPe.output.copy())
        MGV_output.append(CT_BG_model.MGV.output.copy())
        MC_output.append(CT_BG_model.MC.output.copy())
        DA.append(da)
        W1_timeline.append(CT_BG_model.Ws["inp_DLS_1"].copy())
        W2_timeline.append(CT_BG_model.Ws["inp_DLS_2"].copy())
        _input_.append(inp.copy())

    result = {
        "Seed": np.ones(timesteps) * parameters.seed,
        "Inputs_timeline": _input_,
        'DLS_1_output': DLS_1_output,
        'DLS_2_output': DLS_2_output,
        'STNdl_output': STNdl_output,
        "BG_dl_output": BG_dl_output,
        "GPe_output": GPe_output,
        "MGV_output": MGV_output,
        "MC_output": MC_output,
        "DA_timeline": DA,
        "W1_timeline": W1_timeline,
        "W2_timeline": W2_timeline
    }
    
    if args.mode == "plot":
        print(f"""
              Seed: {args.seed}
              Input: {args.inp}
              MC Noise: {parameters.noise['MC']}
              GPi Baseline: {parameters.baseline['GPi']}
              MGV Baseline: {parameters.baseline['MGV']}
              """)
        plotting(result)
        plt.show()