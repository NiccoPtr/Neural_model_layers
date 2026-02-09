# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from Layer_types import BG_dl_Layer, Leaky_units_exc
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

    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.noise['BG_dl'] = args.noise_BG_dl
    parameters.noise['MGV'] = args.noise_MGV
    parameters.noise['MC'] = args.noise_MC
                                                
    rng = np.random.RandomState(parameters.seed)
    
    BG_dl = BG_dl_Layer(parameters.N["BG_dl"], 
                        parameters.tau["BG_dl"], 
                        parameters.baseline["DLS"],
                        parameters.baseline["STNdl"],
                        parameters.baseline["GPi"],
                        parameters.BG_dl_W["DLS_GPi_W"], 
                        parameters.BG_dl_W["STNdl_GPi_W"],
                        rng,
                        parameters.noise["BG_dl"],
                        parameters.threshold["BG_dl"])
    
    MGV = Leaky_units_exc(parameters.N["MGV"], 
                          parameters.tau["MGV"],
                          parameters.baseline["MGV"],
                          rng,
                          parameters.noise["MGV"],
                          parameters.threshold["MGV"])
    
    MC = Leaky_units_exc(parameters.N["MC"], 
                         parameters.tau["MC"], 
                         parameters.baseline["MC"],
                         rng,
                         parameters.noise["MC"],
                         parameters.threshold["MC"])
    
    Ws = {"inp_DMS": np.ones(parameters.N["BG_dl"]), 
          "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
          "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
          "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
          "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
          "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"]
          }
    
    BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
    MGV_output_pre = np.zeros(parameters.N["MGV"])
    MC_output_pre = np.zeros(parameters.N["MC"])
    
    BG_dl_output = []
    MGV_output = []
    MC_output = []
    _input_ = []
        
    BG_dl.reset_activity()
    MGV.reset_activity()
    MC.reset_activity()
    
    for _ in range(timesteps):
    
        BG_dl.step(np.dot(Ws["inp_DMS"], inp),
                   np.dot(Ws["MC_DLS"], MC_output_pre),
                   np.dot(Ws["MC_STNdl"], MC_output_pre))
        
        MGV.step(np.dot(Ws["GPi_MGV"], BG_dl_output_pre +
                 np.dot(Ws["MC_MGV"], MC_output_pre)))
        
        MC.step(np.dot(Ws["MGV_MC"], MGV_output_pre))
        
        BG_dl_output_pre = BG_dl.output_BG_dl.copy()
        
        MGV_output_pre = MGV.output.copy()
        
        MC_output_pre = MC.output.copy()
        
        BG_dl_output.append(BG_dl.output_BG_dl.copy())
        MGV_output.append(MGV.output.copy())
        MC_output.append(MC.output.copy())
        _input_.append(inp.copy())
    
    result = {
        "BG_dl_output": BG_dl_output.copy(),
        "MGV_output": MGV_output.copy(),
        "MC_output": MC_output.copy(),
        "Inputs_timeline": _input_.copy()
        }
    
    if args.mode == "plot":
        plotting(result)
    elif args.mode == "stream":
        mresults = np.hstack([result[key] for key in result.keys()])
        for row in mresults:
            print(("{:5.3f} " * len(row)).format(*row))
