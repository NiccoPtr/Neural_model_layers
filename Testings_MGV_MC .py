# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from CT_BG_simulation import CT_BG
from params import Parameters

#refactor: encapsulate an object of type CTBG which manages the used objects

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

    MC = np.array(res["MC_output"]) * -1

    ax.plot(MC[:, 0], label="Unit_1")
    ax.plot(MC[:, 1], label="Unit_2")

    ax.set_title("MC simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="BG_dl-MGV-MC loop simulation")
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
        "-da",
        "--dopamine",
        type=float,
        default=[0.0, 0.0],
        help="Insert dopamine for learnig: float type"
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
        default=0.0,
        help="Insert MC noise in simulation",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    timesteps = args.timesteps
    da = np.array(args.dopamine)

    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.noise['BG_dl'] = args.noise_BG_dl
    parameters.noise['MGV'] = args.noise_MGV
    parameters.noise['MC'] = args.noise_MC
                                                
    rng = np.random.RandomState(parameters.seed)
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
        W_timeline.append(CT_BG_model.Ws['inp_DLS'].copy())
        _input_.append(inp.copy())
    
    result = {
        "BG_dl_output": BG_dl_output.copy(),
        "MGV_output": MGV_output.copy(),
        "MC_output": MC_output.copy(),
        "Weight_timeline": W_timeline,
        "Inputs_timeline": _input_.copy()
        }
    
    if args.mode == "plot":
        plotting(result)
    elif args.mode == "stream":
        mresults = np.hstack([result[key] for key in result.keys()])
        for row in mresults:
            print(("{:5.3f} " * len(row)).format(*row))
