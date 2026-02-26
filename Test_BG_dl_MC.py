# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os

from CT_BG_simulation import CT_BG
from params import Parameters

plt.ion()

def plotting(res):

    plt.close("all")
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

    fig, ax = plt.subplots(1, 1)

    MGV = np.array(res["MGV_output"])

    ax.plot(MGV[:, 0], label="Unit_1")
    ax.plot(MGV[:, 1], label="Unit_2")

    ax.set_title("MGV simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(1, 1)

    MC = np.array(res["MC_output"])

    ax.plot(MC[:, 0], label="Unit_1")
    ax.plot(MC[:, 1], label="Unit_2")

    ax.set_title("MC simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(1, 1)

    W = np.array(res["Weight_timeline"])

    im = ax.imshow(
        W.reshape(-1, 2 * 2).T, interpolation="none", aspect="auto", vmin=0, vmax=1
    )
    
    ax.set_title("Weights learning")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

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
        "-da",
        "--dopamine",
        type=float,
        nargs=2,
        default=[0.8, 0.8],
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
        default=0.24,
        help="Insert MC noise in simulation",
    )
    parser.add_argument(
        "--MC_MGV_W",
        type=float,
        default=3.0,
        help="Insert MC_MGV matrix strenght"
    )
    parser.add_argument(
        "--GPi_baseline",
        type=float,
        default=0.2,
        help="Insert GPi baseline value"
    )
    parser.add_argument(
        "--MGV_MC_W",
        type=float,
        default=1.0,
        help="Insert MGV_MC matrix strenght"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps
    da = np.array(args.dopamine)
    seed = args.seed

    parameters = Parameters()
    parameters.load("prm_file.json", mode="json")
    parameters.noise["BG_dl"] = args.noise_BG_dl
    parameters.noise["MGV"] = args.noise_MGV
    parameters.noise["MC"] = args.noise_MC
    parameters.Matrices_scalars["MC_MGV"] = args.MC_MGV_W
    parameters.Matrices_scalars["MGV_MC"] = args.MGV_MC_W
    parameters.baseline["GPi"] = args.GPi_baseline
    parameters.Str_Learn["eta_DLS"] = 0.2

    rng = np.random.RandomState(seed)
    CT_BG_model = CT_BG(parameters, rng)

    
    BG_dl_output = []
    MGV_output = []
    MC_output = []
    W_timeline = []
    _input_ = []

    CT_BG_model.reset_activity()

    for _ in range(timesteps):

        CT_BG_model.step(parameters, inp, da)
    
        BG_dl_output.append(CT_BG_model.BG_dl.output_BG_dl.copy())
        MGV_output.append(CT_BG_model.MGV.output.copy())
        MC_output.append(CT_BG_model.MC.output.copy())
        W_timeline.append(CT_BG_model.Ws["inp_DLS"].copy())
        _input_.append(inp.copy())
    
    result = {
        'Seed': np.ones(timesteps) * seed,
        "Inputs_timeline": _input_,
        "BG_dl_output": BG_dl_output,
        "MGV_output": MGV_output,
        "MC_output": MC_output,
        "Weight_timeline": W_timeline
    }    

    if args.mode == "plot":
        plotting(result)
        input("Press Enter to exit")
        
    elif args.mode == "stream":
        inp_end = inp.copy()
        W_end = CT_BG_model.Ws['inp_DLS'].copy().flatten()
        mresults = np.hstack((inp_end, W_end))
        print(("{:10.5f} " * len(mresults)).format(*mresults))
        
    elif args.mode == 'save':
        seed_col = ['Seed']
        input_cols = [f"Input_{i}" 
                      for i in range(len(inp.copy()))]
        BG_dl_cols = [f'BG_dl_Unit_{i}'
                     for i in range(CT_BG_model.BG_dl.GPi.N)]
        MGV_cols = [f'MGV_Unit_{i}'
                     for i in range(CT_BG_model.MGV.N)]
        MC_cols = [f'MC_Unit_{i}'
                     for i in range(CT_BG_model.MC.N)]
        W_cols = [f'Inp_DLS_W_{x}_{y}'
                  for x in range(CT_BG_model.Ws['inp_DLS'].shape[0])
                  for y in range(CT_BG_model.Ws['inp_DLS'].shape[1])]
        cols = seed_col + input_cols + BG_dl_cols + MGV_cols + MC_cols + W_cols
        
        values = [np.asanyarray(result[k]).reshape(timesteps, -1)
                 for k in result.keys()]
        values_conc = np.concatenate(values, axis=1)
        df = pd.DataFrame(values_conc, columns=cols)
        
        csv_path = "MGV_MC_Testing.csv"
        
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
            
    elif args.mode == 'short_save':
        fin_inp = inp.copy()
        fin_W = CT_BG_model.Ws['inp_DLS'].copy().flatten()
        
        seed_col = ['Seed']
        input_cols = [f"Input_{i}" for i in range(len(fin_inp))]
        W_cols = [f'Inp_DLS_W_{x}_{y}'
                  for x in range(CT_BG_model.Ws['inp_DLS'].shape[0])
                  for y in range(CT_BG_model.Ws['inp_DLS'].shape[1])]
        
        values = np.concatenate([[seed], fin_inp, fin_W])
        columns = seed_col + input_cols + W_cols 
        
        df = pd.DataFrame([values], columns=columns)
        
        csv_path = "MGV_MC_short_test.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
                                    