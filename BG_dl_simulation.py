# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 07:43:40 2026

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from Layer_types import BG_dl_Layer
from params import Parameters


def plotting(res):

    fig, ax = plt.subplots(1, 1)

    BG_dl = np.array(res["BG_dl_output"]) * -1

    ax.plot(BG_dl[:, 0], label="Unit_1")
    ax.plot(BG_dl[:, 1], label="Unit_2")

    ax.set_title("BG simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="BG_dl isolated simulation")
    parser.add_argument(
        "-p",
        "--inp",
        type=float,
        nargs=2,
        default=[0.2, 0.2],
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
        help="Output mode ('plot', 'save', 'stream')",
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0.0,
        help="Insert noise in simulation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps

    parameters = Parameters()
    parameters.load("prm_file.json", mode="json")
    parameters.noise["BG_dl"] = args.noise

    rng = np.random.RandomState(parameters.seed)

    BG_dl = BG_dl_Layer(
        parameters.N["BG_dl"],
        parameters.tau["BG_dl"],
        parameters.baseline["DLS"],
        parameters.baseline["STNdl"],
        parameters.baseline["GPi"],
        parameters.BG_dl_W["DLS_GPi_W"],
        parameters.BG_dl_W["STNdl_GPi_W"],
        rng,
        parameters.noise["BG_dl"],
        parameters.threshold["BG_dl"],
    )

    n = parameters.N["BG_dl"]
    Ws = {
        "Mani_DLS": np.eye(n),
        "MC_DLS": np.eye(n) * parameters.Matrices_scalars["MC_DLS"],
        "MC_STNdl": np.eye(n) * parameters.Matrices_scalars["MC_STNdl"],
    }

    BG_dl_output = []
    input_history = []

    BG_dl.reset_activity()
    for _ in range(timesteps):
        BG_dl.step(
            np.dot(Ws["Mani_DLS"], inp * 0),
            np.dot(Ws["MC_DLS"], inp),
            np.dot(Ws["MC_STNdl"], inp),
        )
        BG_dl_output.append(BG_dl.output_BG_dl.copy())
        input_history.append(inp.copy())

    result = {
        "BG_dl_output": np.vstack(BG_dl_output),
        "Inputs_timeline": np.vstack(input_history),
    }

    if args.mode == "plot":
        plotting(result)
    elif args.mode == "stream":
        mresults = np.hstack([result[key] for key in result.keys()])
        for row in mresults:
            print(("{:5.3f} " * len(row)).format(*row))
