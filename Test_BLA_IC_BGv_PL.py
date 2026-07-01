# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:50:55 2026

@author: Nicc
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from CT_BGv_BLA_IC_simulation import CT_BGv_BLA_IC
from params import Parameters

def plotting(res):

    plt.close("all")

    # Isolating single layers
    BLA_IC = np.array(res["BLA_IC"])
    NAc_1 = -np.array(res["NAc_1"])
    NAc_2 = -np.array(res["NAc_2"])
    BGv = -np.array(res["BGv"])
    DM = np.array(res["DM"])
    PL = np.array(res["PL"])
    W_1 = np.array(res["W_timeline_1"])
    W_2 = np.array(res["W_timeline_2"])
    W_BLA_IC = np.array(res['W_BLA_IC'])
    inp = np.array(res["Inp_timeline"])

    rows, cols = np.ix_([0, 1], [2, 3])
    W_1 = W_1[:, rows, cols]
    W_2 = W_2[:, rows, cols]

    # Plotting set up
    plots = [
        ("BLA_IC", [(BLA_IC[:, i], f"Unit_{i+1}") for i in range(4)], (-0.2, 1.2)),
        ("NAc_1", [(NAc_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("NAc_2", [(NAc_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("BGv", [(BGv[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("DM", [(DM[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("PL", [(PL[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2))
    ]

    n_rows = len(plots) + 4
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
        
    #Input plotting
    title_ax = fig.add_subplot(gs[-4, 0])
    ax = fig.add_subplot(gs[-4, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Input", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        inp.reshape(-1, 6).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(6), ["L", "C", "F_1", "F_2", "S_1", "S_2"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    # BLA_IC learning Weight
    title_ax = fig.add_subplot(gs[-3, 0])
    ax = fig.add_subplot(gs[-3, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "BLA_IC Weight", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_BLA_IC.reshape(-1, 4 * 4).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(16), [f"W_{j}_{i}" for j in range(4) for i in range(4)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # BLA_IC NAc Weight learning
    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "BLA_IC_NAc Weight", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_1.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "BLA_IC_NAc Weight", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_2.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
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
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        help="Input values (two floats)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=20,
        help="Number of trials",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
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
    trials = args.trials
    timesteps = args.timesteps
    seed = args.seed
    
    parameters = Parameters()
    if Path("prm_file.json").exists():
        parameters.load("prm_file.json", mode="json")
    parameters.seed = args.seed

    rng = np.random.RandomState(seed)
    CT_BGv_BLA_IC_model = CT_BGv_BLA_IC(parameters, rng)

    BLA_IC_output = []
    LH_output = []
    VTA_output = []
    NAc_output_1 = []
    NAc_output_2 = []
    BGv_ouput = []
    DM_output = []
    PL_output = []
    W_timeline_1 = []
    W_timeline_2 = []
    W_BLA_IC = []
    inp_timeline = []

    for k in range(trials):
        CT_BGv_BLA_IC_model.reset_activity()
        CT_BGv_BLA_IC_model.update_output_pre()
        inp[2:4] = 0.0
            
        for t in range(timesteps):
            if t < 50:
                inp[0:2] = 0.0
            elif t == 50:
                inp = np.array(args.inp)

            if args.inp[0] == 1.0 and t == timesteps * 0.18:
                inp[2] = 1.0

            elif args.inp[1] == 1.0 and t == timesteps * 0.18:
                inp[3] = 1.0

            CT_BGv_BLA_IC_model.step(parameters, inp)

            BLA_IC_output.append(CT_BGv_BLA_IC_model.BLA_IC.output.copy())
            LH_output.append(CT_BGv_BLA_IC_model.LH.output.copy())
            VTA_output.append(CT_BGv_BLA_IC_model.VTA.output.copy())
            NAc_output_1.append(CT_BGv_BLA_IC_model.BG_v.Str1.output.copy())
            NAc_output_2.append(CT_BGv_BLA_IC_model.BG_v.Str2.output.copy())
            BGv_ouput.append(CT_BGv_BLA_IC_model.BG_v.output_BG.copy())
            DM_output.append(CT_BGv_BLA_IC_model.DM.output.copy())
            PL_output.append(CT_BGv_BLA_IC_model.PL.output.copy())
            W_timeline_1.append(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc_1"].copy())
            W_timeline_2.append(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc_2"].copy())
            W_BLA_IC.append(CT_BGv_BLA_IC_model.BLA_IC.W.copy())
            inp_timeline.append(inp.copy())

            if k == trials - 1:

                result = {
                    "Seed": np.ones(timesteps * trials) * seed,
                    "Inp_timeline": inp_timeline,
                    "W_timeline_1": W_timeline_1,
                    "W_timeline_2": W_timeline_2,
                    'W_BLA_IC': W_BLA_IC,
                    "BLA_IC": BLA_IC_output,
                    "LH": LH_output,
                    "VTA": VTA_output,
                    "NAc_1": NAc_output_1,
                    "NAc_2": NAc_output_2,
                    "BGv": BGv_ouput,
                    "DM": DM_output,
                    "PL": PL_output
                }

    if args.mode == "plot":
        plotting(result)
        plt.show()
