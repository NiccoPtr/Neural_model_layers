# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:50:55 2026

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os

from CT_BGv_BLA_IC_simulation import CT_BGv_BLA_IC
from params import Parameters
plt.ion()

def plotting(res):
    
    plt.close("all")
    fig, ax = plt.subplots(1, 1)

    BLA_IC = np.array(res["BLA_IC"])

    ax.plot(BLA_IC[:, 0], label="Unit_1")
    ax.plot(BLA_IC[:, 1], label="Unit_2")
    ax.plot(BLA_IC[:, 2], label="Unit_3")
    ax.plot(BLA_IC[:, 3], label="Unit_4")

    ax.set_title("BLA_IC simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(1, 1)

    LH = np.array(res["LH"])

    ax.plot(LH[:], label="Unit_1")

    ax.set_title("LH simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(1, 1)

    VTA = np.array(res["VTA"])

    ax.plot(VTA[:], label="Unit_1")

    ax.set_title("VTA simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 1)

    NAc = np.array(res["NAc"]) * -1

    ax.plot(NAc[:, 0], label="Unit_1")
    ax.plot(NAc[:, 1], label="Unit_2")

    ax.set_title("NAc simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    
    BGv = np.array(res["BGv"]) * -1

    ax.plot(BGv[:, 0], label="Unit_1")
    ax.plot(BGv[:, 1], label="Unit_2")

    ax.set_title("BGv simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    
    DM = np.array(res["DM"])

    ax.plot(DM[:, 0], label="Unit_1")
    ax.plot(DM[:, 1], label="Unit_2")

    ax.set_title("DM simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    
    PL = np.array(res["PL"])

    ax.plot(PL[:, 0], label="Unit_1")
    ax.plot(PL[:, 1], label="Unit_2")

    ax.set_title("PL simulation")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(1, 1)

    W = np.array(res["W_timeline"])

    im = ax.imshow(
        W.reshape(-1, 4 * 2).T, interpolation="none", aspect="auto", vmin=0, vmax=1
    )
    
    ax.set_title("Weights learning")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(8), [f"W_{j}_{i}" for j in range(2) for i in range(4)])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    plt.show()
    
    inp = np.array(res["Inp_timeline"])

    ax.plot(inp[:, 0], label="Lever")
    ax.plot(inp[:, 1], label="Chain")
    ax.plot(inp[:, 2], label="Food_1")
    ax.plot(inp[:, 3], label="Food_2")
    ax.plot(inp[:, 4], label="Sat_1")
    ax.plot(inp[:, 5], label="Sat_2")

    ax.set_title("Input timeeline")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activity level")
    ax.set_ylim(0, 1)

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
        nargs=6,
        default=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="Input values (two floats)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=10,
        help="Number of trials",
    )
    parser.add_argument(
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
        "-nPL",
        "--noise_PL",
        type=float,
        default=None,
        help="Insert PL noise in simulation",
    )
    parser.add_argument(
        "--PL_DM_W",
        type=float,
        default=None,
        help="Insert PL_DM matrix strenght"
    )
    parser.add_argument(
        "--DM_PL_W",
        type=float,
        default=None,
        help="Insert DM_PL matrix strenght"
    )
    parser.add_argument(
        "--SNpr_baseline",
        type=float,
        default=None,
        help="Insert SNpr baseline value"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    trials = args.trials
    timesteps = args.timesteps
    seed = args.seed

    parameters = Parameters()
    parameters.load("prm_file.json", mode="json")
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
    inp_timeline = []
    
    for _ in range(trials):
        CT_BGv_BLA_IC_model.reset_activity()
        
        for t in range(timesteps):
            
            if t == timesteps*0.25 or t == timesteps*0.75:
                inp[2] = 1.0
            elif t == timesteps*0.5:
                inp[2] = 0.0
                
            CT_BGv_BLA_IC_model.step(parameters, inp)
            
            BLA_IC_output.append(CT_BGv_BLA_IC_model.BLA_IC.output.copy())
            LH_output.append(CT_BGv_BLA_IC_model.LH.output.copy())
            VTA_output.append(CT_BGv_BLA_IC_model.VTA.output.copy())
            NAc_output.append(CT_BGv_BLA_IC_model.BG_v.NAc.output.copy())
            BGv_ouput.append(CT_BGv_BLA_IC_model.BG_v.output_BG_v.copy())
            DM_output.append(CT_BGv_BLA_IC_model.DM.output.copy())
            PL_output.append(CT_BGv_BLA_IC_model.PL.output.copy())
            W_timeline.append(CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].copy())
            inp_timeline.append(inp.copy())
            
        result = {
            'Seed': np.ones(timesteps) * seed,
            'Inp_timeline': inp_timeline,
            'W_timeline': W_timeline,
            'BLA_IC': BLA_IC_output,
            'LH': LH_output,
            'VTA': VTA_output,
            'NAc': NAc_output,
            'BGv': BGv_ouput,
            'DM': DM_output,
            'PL': PL_output
            }
    
if args.mode == "plot":
    plotting(result)
    input("Press Enter to exit")
    
elif args.mode == "stream":
    inp_end = inp.copy()
    W_end = CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].copy().flatten()
    mresults = np.hstack((inp_end, W_end))
    print(("{:10.5f} " * len(mresults)).format(*mresults))
    
elif args.mode == 'save':
    seed_col = ['Seed']
    input_cols = [f"Input_{i}" 
                  for i in range(len(inp.copy()))]
    BLA_IC_cols = [f'BLA_IC_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.BLA_IC.N)]
    LH_cols = [f'LH_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.LH.N)]
    VTA_cols = [f'VTA_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.VTA.N)]
    NAc_cols = [f'NAc_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.BG_v.NAc.N)]
    BGv_cols = [f'BGv_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.BG_v.SNpr.N)]
    DM_cols = [f'DM_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.DM.N)]
    PL_cols = [f'PL_Unit_{i}'
                 for i in range(CT_BGv_BLA_IC_model.PL.N)]
    W_cols = [f'Inp_DLS_W_{x}_{y}'
              for x in range(CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].shape[0])
              for y in range(CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].shape[1])]
    cols = seed_col + input_cols + BLA_IC_cols + LH_cols + VTA_cols + NAc_cols + BGv_cols + DM_cols + PL_cols + W_cols
    
    values = [np.asanyarray(result[k]).reshape(timesteps, -1)
             for k in result.keys()]
    values_conc = np.concatenate(values, axis=1)
    df = pd.DataFrame(values_conc, columns=cols)
    
    csv_path = "BLA_IC_BGv_PL_Testing.csv"
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
elif args.mode == 'short_save':
    fin_inp = inp.copy()
    fin_W = CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].flatten()
    
    seed_col = ['Seed']
    input_cols = [f"Input_{i}" for i in range(len(fin_inp))]
    W_cols = [f'Inp_DLS_W_{x}_{y}'
              for x in range(CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].shape[0])
              for y in range(CT_BGv_BLA_IC_model.Ws['BLA_IC_NAc'].shape[1])]
    
    values = np.concatenate([[seed], fin_inp, fin_W])
    columns = seed_col + input_cols + W_cols 
    
    df = pd.DataFrame([values], columns=columns)
    
    csv_path = "BLA_IC_BGv_PL_short_test.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    
    
    