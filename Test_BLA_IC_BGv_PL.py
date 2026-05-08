# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:50:55 2026

@author: Nicc
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from CT_BGv_BLA_IC_simulation import CT_BGv_BLA_IC
from params import Parameters

plt.ion()


def plotting(res):

    plt.close("all")

    # Isolating single layers
    BLA_IC = np.array(res["BLA_IC"])
    LH = np.array(res["LH"])
    VTA = np.array(res["VTA"])
    NAc = -np.array(res["NAc"])
    BGv = -np.array(res["BGv"])
    DM = np.array(res["DM"])
    PL = np.array(res["PL"])
    W = np.array(res["W_timeline"])
    W_BLA_IC = np.array(res['W_BLA_IC_NAc'])
    inp = np.array(res["Inp_timeline"])

    rows, cols = np.ix_([0, 1], [2, 3])
    W = W[:, rows, cols]

    # Plotting set up
    plots = [
        ("LH", [(LH[:], "Unit_1")], (-0.2, 1.2)),
        ("VTA", [(VTA[:], "Unit_1")], (-0.2, 1.2)),
        ("BLA_IC", [(BLA_IC[:, i], f"Unit_{i+1}") for i in range(4)], (-0.2, 1.2)),
        ("NAc", [(NAc[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("BGv", [(BGv[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("DM", [(DM[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2)),
        ("PL", [(PL[:, i], f"Unit_{i+1}") for i in range(2)], (-0.2, 1.2))
    ]

    n_rows = len(plots) + 3
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
    title_ax = fig.add_subplot(gs[-3, 0])
    ax = fig.add_subplot(gs[-3, 1], sharex=shared_ax)

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
    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

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
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "BLA_IC_NAc Weight", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W.reshape(-1, 2 * 2).T,
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
        default=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    parser.add_argument(
        "-nPL",
        "--noise_PL",
        type=float,
        default=0.4,
        help="Insert PL noise in simulation",
    )
    parser.add_argument(
        "--PL_DM_W", type=float, default=2.3, help="Insert PL_DM matrix strenght"
    )
    parser.add_argument(
        "--DM_PL_W", type=float, default=1.8, help="Insert DM_PL matrix strenght"
    )
    parser.add_argument(
        "--SNpr_baseline", type=float, default=0.2, help="Insert SNpr baseline value"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    trials = args.trials
    timesteps = args.timesteps
    seed = args.seed

    parameters = Parameters()

    if args.noise_PL:
        parameters.noise["PL"] = args.noise_PL
    if args.PL_DM_W:
        parameters.Matrices_scalars["PL_DM"] = args.PL_DM_W
    if args.DM_PL_W:
        parameters.Matrices_scalars["DM_PL"] = args.DM_PL_W
    if args.SNpr_baseline:
        parameters.baseline["SNpr"] = args.SNpr_baseline

    rng = np.random.RandomState(seed)
    CT_BGv_BLA_IC_model = CT_BGv_BLA_IC(parameters, rng)

    BLA_IC_output = []
    LH_output = []
    VTA_output = []
    NAc_output = []
    BGv_ouput = []
    DM_output = []
    PL_output = []
    W_timeline = []
    W_BLA_IC_NAc = []
    inp_timeline = []

    for k in range(trials):
        CT_BGv_BLA_IC_model.reset_activity()
        CT_BGv_BLA_IC_model.update_output_pre()
        inp[2:4] = 0.0
        
        if k >= trials*0.7:
            inp[-1] = 1.0
            
        for t in range(timesteps):
            if t < 50:
                inp[0:2] = 0.0
            elif t == 50:
                inp = np.array(args.inp)
                
                if k >= trials*0.7:
                    inp[-1] = 1.0

            if args.inp[0] == 1.0 and t == timesteps * 0.18:
                inp[2] = 1.0

            elif args.inp[1] == 1.0 and t == timesteps * 0.18:
                inp[3] = 1.0

            CT_BGv_BLA_IC_model.step(parameters, inp)

            BLA_IC_output.append(CT_BGv_BLA_IC_model.BLA_IC.output.copy())
            LH_output.append(CT_BGv_BLA_IC_model.LH.output.copy())
            VTA_output.append(CT_BGv_BLA_IC_model.VTA.output.copy())
            NAc_output.append(CT_BGv_BLA_IC_model.BG_v.NAc.output.copy())
            BGv_ouput.append(CT_BGv_BLA_IC_model.BG_v.output_BG_v.copy())
            DM_output.append(CT_BGv_BLA_IC_model.DM.output.copy())
            PL_output.append(CT_BGv_BLA_IC_model.PL.output.copy())
            W_timeline.append(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].copy())
            W_BLA_IC_NAc.append(CT_BGv_BLA_IC_model.BLA_IC.W.copy())
            inp_timeline.append(inp.copy())

            if k == trials - 1:

                result = {
                    "Seed": np.ones(timesteps * trials) * seed,
                    "Inp_timeline": inp_timeline,
                    "W_timeline": W_timeline,
                    'W_BLA_IC_NAc': W_BLA_IC_NAc,
                    "BLA_IC": BLA_IC_output,
                    "LH": LH_output,
                    "VTA": VTA_output,
                    "NAc": NAc_output,
                    "BGv": BGv_ouput,
                    "DM": DM_output,
                    "PL": PL_output
                }

    if args.mode == "plot":
        plotting(result)
        input("Press Enter to exit")

    elif args.mode == "stream":
        inp_end = inp.copy()
        W_end = CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].copy().flatten()
        mresults = np.hstack((inp_end, W_end))
        print(("{:10.5f} " * len(mresults)).format(*mresults))

    elif args.mode == "save":
        seed_col = ["Seed"]
        input_cols = [f"Input_{i}" for i in range(len(inp.copy()))]
        BLA_IC_cols = [f"BLA_IC_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.BLA_IC.N)]
        LH_cols = [f"LH_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.LH.N)]
        VTA_cols = [f"VTA_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.VTA.N)]
        NAc_cols = [f"NAc_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.BG_v.NAc.N)]
        BGv_cols = [f"BGv_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.BG_v.SNpr.N)]
        DM_cols = [f"DM_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.DM.N)]
        PL_cols = [f"PL_Unit_{i}" for i in range(CT_BGv_BLA_IC_model.PL.N)]
        W_cols = [
            f"Inp_DLS_W_{x}_{y}"
            for x in range(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].shape[0])
            for y in range(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].shape[1])
        ]
        cols = (
            seed_col
            + input_cols
            + BLA_IC_cols
            + LH_cols
            + VTA_cols
            + NAc_cols
            + BGv_cols
            + DM_cols
            + PL_cols
            + W_cols
        )

        values = [np.asanyarray(result[k]).reshape(timesteps, -1) for k in result.keys()]
        values_conc = np.concatenate(values, axis=1)
        df = pd.DataFrame(values_conc, columns=cols)

        csv_path = "BLA_IC_BGv_PL_Testing.csv"

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    elif args.mode == "short_save":
        fin_inp = inp.copy()
        fin_W = CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].flatten()

        seed_col = ["Seed"]
        input_cols = [f"Input_{i}" for i in range(len(fin_inp))]
        W_cols = [
            f"Inp_DLS_W_{x}_{y}"
            for x in range(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].shape[0])
            for y in range(CT_BGv_BLA_IC_model.Ws["BLA_IC_NAc"].shape[1])
        ]

        values = np.concatenate([[seed], fin_inp, fin_W])
        columns = seed_col + input_cols + W_cols

        df = pd.DataFrame([values], columns=columns)

        csv_path = "BLA_IC_BGv_PL_short_test.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
