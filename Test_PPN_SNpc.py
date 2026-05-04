# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:05:29 2026

@author: Nicc
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

from PPN_SNpc_simulation import PPN_SNpc
from params import Parameters

plt.ion()


def plotting(res):

    plt.close("all")

    # Isolating single layers
    SNpci_1 = np.array(res['SNpci_1']) * -1
    SNpci_2 = np.array(res['SNpci_2']) * -1
    SNpc_1 = np.array(res['SNpc_output_1'])
    SNpc_2 = np.array(res['SNpc_output_2'])
    PPN = np.array(res['PPN_output'])
    # Plotting set up
    plots = [
        ("SNpci_1", [(SNpci_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("SNpci_2", [(SNpci_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("SNpc_output_1", [(SNpc_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("SNpc_output_2", [(SNpc_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("PPN", [(PPN[:, i], f"Unit_{i+1}") for i in range(1)], (-0.1, 1)),
    ]

    n_rows = len(plots)
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

def parse_args():
    parser = argparse.ArgumentParser(description="BG_dl-MGV-MC loop simulation")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "-f",
        "--food",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
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
    parser.add_argument(
        "--NAc",
        type=int,
        default=(0.0, 0.0),
        help="NAc input",
    )
    parser.add_argument(
        "--DMS",
        type=int,
        default=(0.0, 0.0),
        help="DMS input",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.food)
    timesteps = args.timesteps

    parameters = Parameters()
    if Path("prm_file.json").exists():
        parameters.load("prm_file.json", mode="json")
    parameters.seed = args.seed

    rng = np.random.RandomState(parameters.seed)
    PPN_SNpc_model = PPN_SNpc(parameters, rng)

    PPN_output = []
    SNpci_1 = []
    SNpci_2 = []
    SNpc_output_1 = []
    SNpc_output_2 = []
    _input_ = []

    PPN_SNpc_model.reset_activity()

    for t in range(timesteps):
        
        # if t == timesteps//2:
        #     PPN_SNpc_model.reset_activity()

        PPN_SNpc_model.step(inp, NAc_inp = np.array(args.NAc) * -1, DMS_inp = np.array(args.DMS) * -1)
         
        PPN_output.append(PPN_SNpc_model.PPN.output.copy())
        SNpci_1.append(PPN_SNpc_model.SNpc.output_SNpci_1_pre.copy())
        SNpci_2.append(PPN_SNpc_model.SNpc.output_SNpci_2_pre.copy())
        SNpc_output_1.append(PPN_SNpc_model.SNpc.output_1.copy())
        SNpc_output_2.append(PPN_SNpc_model.SNpc.output_2.copy())
        _input_.append(inp.copy())

    result = {
        "Seed": np.ones(timesteps) * parameters.seed,
        "Inputs_timeline": _input_,
        "PPN_output": PPN_output,
        "SNpci_1": SNpci_1,
        "SNpci_2": SNpci_2,
        "SNpc_output_1": SNpc_output_1,
        "SNpc_output_2": SNpc_output_2
    }
    
    if args.mode == "plot":
        plotting(result)
        print(f"""
              Seed: {args.seed}
              Input: {args.food}
              """)
        # input("Press Enter to exit")
