

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

from BGdm_attention import CT_BG
from params import Parameters


def plotting(res):

    plt.close("all")

    # Isolating single layers
    inp = np.array(res["Inputs_timeline"])
    DMS_1 = np.array(res["DMS_1_output"]) * -1
    DMS_2 = np.array(res["DMS_2_output"]) * -1
    STNdm = np.array(res["STNdm_output"])
    BG_dm = np.array(res["BG_dm_output"]) * -1
    GPe = np.array(res["GPe_output"]) * -1
    P = np.array(res["P_output"])
    PFCd_PPC = np.array(res["PFCd_PPC_output"])
    DA = np.array(res["DA_timeline"])
    W1 = np.array(res["W1_timeline"])
    W2 = np.array(res["W2_timeline"])

    # Plotting set up
    plots = [
        ("DMS_1", [(DMS_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("DMS_2", [(DMS_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("STNdm", [(STNdm[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPi", [(BG_dm[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPe", [(GPe[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("P", [(P[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("PFCd_PPC", [(PFCd_PPC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("Inputs", [(inp[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
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
    inp = np.zeros(2)
    timesteps = args.timesteps
    da = np.array(args.dopamine)

    parameters = Parameters()
    if Path("prm_file.json").exists():
        parameters.load("prm_file.json", mode="json")
    parameters.seed = args.seed

    rng = np.random.RandomState(parameters.seed)
    att_model = CT_BG(parameters, rng)

    inp_timeline = []
    DMS_1_output = []
    DMS_2_output = []
    STNdm_output = []
    BG_dm_output = []
    GPe_output = []
    P_output = []
    PFCd_PPC_output = []
    DA = []
    W1_timeline = []
    W2_timeline = []

    att_model.reset_activity()

    winner = None

    for t in range(timesteps):

        if winner is not None:
            inp[winner] = 1.0
            if 50 < t < 100:
                da = 1.0
            else:
                da = np.array(args.dopamine)
        else:
            inp *= 0.0
            da = np.array(args.dopamine)

        att_model.step(parameters, inp, da, learn=True)

        attention = att_model.PFCd_PPC.output.copy()
        if np.any(attention >= att_model.PFCd_PPC.threshold):
            winner = np.argmax(attention)
        else:
            winner = None
        
        inp_timeline.append(inp.copy())
        DMS_1_output.append(att_model.BG_dm.Str1.output.copy())
        DMS_2_output.append(att_model.BG_dm.Str2.output.copy())
        STNdm_output.append(att_model.BG_dm.STN.output.copy())
        BG_dm_output.append(att_model.BG_dm.output_BG.copy())
        GPe_output.append(att_model.BG_dm.GPe.output.copy())
        P_output.append(att_model.P.output.copy())
        PFCd_PPC_output.append(att_model.PFCd_PPC.output.copy())
        DA.append(da)
        W1_timeline.append(att_model.Ws["inp_DMS_1"].copy())
        W2_timeline.append(att_model.Ws["inp_DMS_2"].copy())

    result = {
        "Seed": np.ones(timesteps) * parameters.seed,
        "Inputs_timeline": inp_timeline,
        'DMS_1_output': DMS_1_output,
        'DMS_2_output': DMS_2_output,
        'STNdm_output': STNdm_output,
        "BG_dm_output": BG_dm_output,
        "GPe_output": GPe_output,
        "P_output": P_output,
        "PFCd_PPC_output": PFCd_PPC_output,
        "DA_timeline": DA,
        "W1_timeline": W1_timeline,
        "W2_timeline": W2_timeline
    }
    
    if args.mode == "plot":
        print(f"""
              Seed: {args.seed}
              Input: {inp}
              PFCd_PPC Noise: {parameters.noise['PFCd_PPC']}
              GPi_SNpr Baseline: {parameters.baseline['GPi_SNpr']}
              P Baseline: {parameters.baseline['P']}
              """)
        plotting(result)
        plt.show()