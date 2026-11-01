# -*- coding: utf-8 -*-
"""
Created on Sat Nov 1 11:44:20 2026

@author: Nicc
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

from CeA_simulation import CeA
from params import Parameters

def plotting(res):

    plt.close("all")

    # Isolating single layers
    CeA = np.array(res["CeA_output"])
    BLA_IC = np.array(res["BLA_IC_output"])
    inp_timeline = np.array(res["inp_timeline"])
    da_timeline = np.array(res["DA_timeline"])
    W_inp_CeA_timeline = np.array(res["W_inp_CeA_timeline"])

    # Plotting set up
    plots = [
        ("CeA", [(CeA[:], f"CeA_Unit")], (-0.1, 1)),
        ("BLA_IC", [(BLA_IC[:, i], f"Unit_{i+1}") for i in range(3)], (-0.1, 1)),
        ("DA", [(da_timeline[:], f"VTA")], (-0.1, 1)),
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

    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Input timeline", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        inp_timeline.reshape(-1, 1 * 9).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(9), [f"W_{j}_{i}" for j in range(1) for i in range(9)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Weight heatmap
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "W_inp_CeA learning", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_inp_CeA_timeline.reshape(-1, 1 * 3).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(3), [f"W_{j}_{i}" for j in range(1) for i in range(3)])

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
        nargs=9,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        help="Input values (nine floats)",
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
        help="Insert dopamine for learning: float type",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps
    da = np.array(args.dopamine)

    parameters = Parameters()
    BASE_DIR = Path(__file__).resolve().parent

    prm_file = BASE_DIR / "prm_file.json"

    if prm_file.exists():
        parameters.load(prm_file, mode="json")
        print("Imported parameters successfully")
    else:
        raise ValueError(f"Parameters file not found: {prm_file}")

    rng = np.random.RandomState(parameters.seed)
    CeA_model = CeA(parameters, rng)

    CeA_output = []
    BLA_IC_output = []
    inp_timeline = []
    DA_timeline = []
    W_inp_CeA_timeline = []

    CeA_model.reset_activity()

    for t in range(timesteps):

        if t <= 50:
            inp *= 0.0

        else:
            inp = np.array(args.inp)

        if t < 100:
            da = np.array([parameters.baseline["VTA"]])
        
        elif 100 < t < 120:
            da = np.array([1.0])

        else:
            da = np.array(args.dopamine)

        CeA_model.step(da, inp)

        CeA_output.append(CeA_model.CeA.output.copy())
        BLA_IC_output.append(CeA_model.BLA_IC.copy())
        inp_timeline.append(inp.copy())
        DA_timeline.append(da.copy())
        W_inp_CeA_timeline.append(CeA_model.Ws["Inp_CeA"].copy())

    results = {
        "CeA_output": np.array(CeA_output),
        "BLA_IC_output": np.array(BLA_IC_output),
        "inp_timeline": np.array(inp_timeline),
        "DA_timeline": np.array(DA_timeline),
        "W_inp_CeA_timeline": np.array(W_inp_CeA_timeline),
    }

    if args.mode == "plot":
        print(f"""
              Seed: {args.seed}
              Input: {args.inp}
              """)
        plotting(results)
        plt.show()


