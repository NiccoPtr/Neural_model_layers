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
        ("BG_dl", [(BG_dl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
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
        default=[1.0, 0.0],
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
        "-da",
        "--dopamine",
        type=float,
        nargs=2,
        default=[0.9, 0.0],
        help="Insert dopamine for learnig: float type",
    )
    parser.add_argument(
        "-nBG_DL",
        "--noise_BG_dl",
        type=float,
        default=0.0,
        help="Insert BG_dl noise in simulation",
    )
    parser.add_argument(
        "-nMGV",
        "--noise_MGV",
        type=float,
        default=0.0,
        help="Insert MGV noise in simulation",
    )
    parser.add_argument(
        "-nMC",
        "--noise_MC",
        type=float,
        default=0.4,
        help="Insert MC noise in simulation",
    )
    parser.add_argument(
        "--MC_MGV_W", type=float, default=2.3, help="Insert MC_MGV matrix strenght"
    )
    parser.add_argument(
        "--MGV_MC_W", type=float, default=1.8, help="Insert MGV_MC matrix strenght"
    )
    parser.add_argument(
        "--GPi_baseline", type=float, default=0.3, help="Insert GPi baseline value"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps
    da = np.array(args.dopamine)

    parameters = Parameters()
    parameters.seed = args.seed
    parameters.noise["BG_dl"] = args.noise_BG_dl
    parameters.noise["MGV"] = args.noise_MGV
    parameters.noise["MC"] = args.noise_MC
    parameters.Matrices_scalars["MC_MGV"] = args.MC_MGV_W
    parameters.Matrices_scalars["MGV_MC"] = args.MGV_MC_W
    parameters.baseline["GPi"] = args.GPi_baseline
    parameters.Str_Learn["eta_DLS"] = 0.001
    parameters.Str_Learn["theta_DLS"] = 0.12
    parameters.Str_Learn["theta_inp_DLS"] = 0.5
    parameters.threshold['MC'] = 0.4
    parameters.tau['MC'] = 6
    parameters.BG_dl_W['DLS_GPi_W'] = 1.8
    parameters.BG_dl_W['STNdl_GPi_W'] = 1.6

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

        CT_BG_model.step(parameters, inp, da)
        
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
        plotting(result)
        input("Press Enter to exit")

    elif args.mode == "stream":
        inp_end = inp.copy()
        W_end = CT_BG_model.Ws["inp_DLS"].copy().flatten()
        mresults = np.hstack((inp_end, W_end))
        print(("{:10.5f} " * len(mresults)).format(*mresults))

    elif args.mode == "save":
        seed_col = ["Seed"]
        input_cols = [f"Input_{i}" for i in range(len(inp.copy()))]
        BG_dl_cols = [f"BG_dl_Unit_{i}" for i in range(CT_BG_model.BG_dl.GPi.N)]
        MGV_cols = [f"MGV_Unit_{i}" for i in range(CT_BG_model.MGV.N)]
        MC_cols = [f"MC_Unit_{i}" for i in range(CT_BG_model.MC.N)]
        W_cols = [
            f"Inp_DLS_W_{x}_{y}"
            for x in range(CT_BG_model.Ws["inp_DLS"].shape[0])
            for y in range(CT_BG_model.Ws["inp_DLS"].shape[1])
        ]
        cols = seed_col + input_cols + BG_dl_cols + MGV_cols + MC_cols + W_cols

        values = [np.asanyarray(result[k]).reshape(timesteps, -1) for k in result.keys()]
        values_conc = np.concatenate(values, axis=1)
        df = pd.DataFrame(values_conc, columns=cols)

        csv_path = "MGV_MC_Testing.csv"

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    elif args.mode == "short_save":
        fin_inp = inp.copy()
        fin_W = CT_BG_model.Ws["inp_DLS"].copy().flatten()

        seed_col = ["Seed"]
        input_cols = [f"Input_{i}" for i in range(len(fin_inp))]
        W_cols = [
            f"Inp_DLS_W_{x}_{y}"
            for x in range(CT_BG_model.Ws["inp_DLS"].shape[0])
            for y in range(CT_BG_model.Ws["inp_DLS"].shape[1])
        ]

        values = np.concatenate([[parameters.seed], fin_inp, fin_W])
        columns = seed_col + input_cols + W_cols

        df = pd.DataFrame([values], columns=columns)

        csv_path = "MGV_MC_short_test.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
