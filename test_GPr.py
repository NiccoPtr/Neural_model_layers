from GPr import GPr
from params import Parameters
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import argparse

def plotting(res):

    plt.close("all")

    DLS_1 = np.array(res["DLS_1"]) * -1
    DLS_2 = np.array(res["DLS_2"]) * -1
    STNdl = np.array(res["STNdl"])
    GPi = np.array(res["GPi"]) * -1
    GPe = np.array(res["GPe"]) * -1
    MGV = np.array(res["MGV"])
    MC = np.array(res["MC"])

    input_ = np.array(res["input"])
    actions = np.array(res["actions"])

    plots = [
        ("DLS_1", [(DLS_1[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("DLS_2", [(DLS_2[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("STNdl", [(STNdl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPi", [(GPi[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
        ("GPe", [(GPe[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1)),
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

    # Input plot
    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Action selected", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        input_.reshape(-1, 1).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2
    )
    
    ax.set_yticks([])  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Action selection
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
        default=1,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "-f",
        "--food",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="Set Food input to Basal Ganglia (e.g; 1.0 0.0)"
    )
    parser.add_argument(
        "-d",
        "--da",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="Set dopaminergic input to Basal Ganglia (e.g; 1.0 0.0)"
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="Define amount of timesteps"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="plot",
        help="Define the mode of operation (e.g., 'plot')"
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    inp = np.array(args.food)
    da = np.array(args.da)
    timesteps = args.timesteps

    parameters = Parameters()
    if Path("C:/Users/Nicc/Desktop/CNR_Model/prm_file.json").exists():
        parameters.load("C:/Users/Nicc/Desktop/CNR_Model/prm_file.json", mode="json")

    else:
        raise ValueError('Parameters file not found')
    
    parameters.seed = args.seed

    rng = np.random.RandomState(parameters.seed)
    C_Th_BG = GPr(parameters)

    DLS_1_output = []
    DLS_2_output = []
    STNdl_output = []
    GPi_output = []
    GPe_output = []
    MGV_output = []
    MC_output = []
    _input_ = []
    actions = []

    C_Th_BG.reset_activity()

    for t in range(timesteps):

        C_Th_BG.step(inp, da)

        action = C_Th_BG.MC.output.copy()
        if np.any(action >= C_Th_BG.MC.threshold):
            winner = np.argmax(action) + 1
        else:
            winner = np.array(0)
        actions.append(winner)

        DLS_1_output.append(C_Th_BG.BG_dl.DLS_1.output.copy())
        DLS_2_output.append(C_Th_BG.BG_dl.DLS_2.output.copy())
        STNdl_output.append(C_Th_BG.BG_dl.STNdl.output.copy())
        GPi_output.append(C_Th_BG.BG_dl.GPi.output.copy())
        GPe_output.append(C_Th_BG.BG_dl.GPe.output.copy())

        MGV_output.append(C_Th_BG.MGV.output.copy())

        MC_output.append(C_Th_BG.MC.output.copy())

        _input_.append(inp.copy())

    result = {
        "DLS_1": np.array(DLS_1_output),
        "DLS_2": np.array(DLS_2_output),
        "STNdl": np.array(STNdl_output),
        "GPi": np.array(GPi_output),
        "GPe": np.array(GPe_output),
        "MGV": np.array(MGV_output),
        "MC": np.array(MC_output),
        "input": np.array(_input_),
        "actions": np.array(actions)
    }

    if args.mode == "plot":
        print(f"""
              Seed: {args.seed}
              Input: {args.food}
              DA: {args.da}
              """)
        plotting(result)
        plt.show()

        



