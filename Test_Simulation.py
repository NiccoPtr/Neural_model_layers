# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:23:51 2026

@author: Nicc
"""

import argparse
import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from params import Parameters
from scheduling import Scheduling

def parse_args():
    parser = argparse.ArgumentParser(description="BLA_IC simulation")
    parser.add_argument(
        "-i",
        "--id",
        help="ID simulation",
    )
    parser.add_argument(
        "-d",
        "--scheduling",
        type=str,
        help="Filename of the scheduling json",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Input simulation seed for noise",
    )
    parser.add_argument(
        "-l",
        "--lesion",
        type=str,
        default="None",
        help="Report area to lesion (BLA, NAc, DMS, PL)",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    parameters = Parameters()
    BASE_DIR = Path(__file__).resolve().parent
    
    # Full path to the JSON file
    prm_file = BASE_DIR / "prm_file.json"

    if prm_file.exists():
        parameters.load(prm_file, mode="json")
        print("Imported parameters successfully")
    else:
        raise ValueError(f"Parameters file not found: {prm_file}")
        
    parameters.seed = args.seed
    scheduling = Scheduling()
    if args.scheduling is not None:
        scheduling.load(args.scheduling, mode="json")
    parameters.scheduling = scheduling._params_to_dict()

    if len(scheduling.states) != len(scheduling.phases):
        raise ValueError("Input and Phases must have same length")

    model = joblib.load(f'C:/Users/Nicc/Desktop/CNR_Model/trainings/training_{str(args.id)}/sim_seed{int(args.seed)}/Model_{int(args.seed)}.joblib')
    model.parameters = parameters
    
    if args.lesion == "BLA":
        model.BLA_IC.lesion = True
        
    elif args.lesion == "NAc":
        model.BG_v.NAc.lesion = True
        
    elif args.lesion == "DMS":
        model.BG_dm.DMS.lesion = True
        
    elif args.lesion == "PL":
        model.PL.lesion = True

    sched = parameters.scheduling
    timesteps = sched["timesteps"]
    trials = sched["trials"]
    
    results = []

    for trial in range(trials):

        if trial <= (trials*(sched["phases"][0])):
            env = np.array(sched["states"][0])
            phase = 1

        elif trial <= (trials*(sched["phases"][1])):
            env = np.array(sched["states"][1])
            phase = 2

        state = env.copy()
        state[0:4] = 0.0

        model.reset_activity()
        model.update_output_pre()
        MC_output = np.empty((timesteps, model.MC.N), dtype=np.float32)
        PFCd_PPC_output = np.empty((timesteps, model.PFCd_PPC.N), dtype=np.float32)
        PL_output = np.empty((timesteps, model.PL.N), dtype=np.float32)
        state_t = np.empty((timesteps, len(state)), dtype=np.float32)
        DLS_output = np.empty((timesteps, model.BG_dl.Str1.N), dtype=np.float32)
        DMS_output = np.empty((timesteps, model.BG_dm.Str1.N), dtype=np.float32)
        BLA_IC_output = np.empty((timesteps, model.BLA_IC.N), dtype=np.float32)
        NAc_output = np.empty((timesteps, model.BG_v.Str1.N), dtype=np.float32)
        BGv_output = np.empty((timesteps, model.BG_v.GPi_SNpr.N), dtype=np.float32)
        BGdm_output = np.empty((timesteps, model.BG_dm.GPi_SNpr.N), dtype=np.float32)
        BGdl_output = np.empty((timesteps, model.BG_dl.GPi_SNpr.N), dtype=np.float32)
        MGV_output = np.empty((timesteps, model.MGV.N), dtype=np.float32)
        P_output = np.empty((timesteps, model.P.N), dtype=np.float32)
        DM_output = np.empty((timesteps, model.DM.N), dtype=np.float32)
        DA_timeline = np.empty((timesteps, 3), dtype=np.float32)
        W_BLA_IC_NAc = np.empty((timesteps, model.BG_v.Str1.N, model.BLA_IC.N), dtype=np.float32)
        W_Mani_DLS = np.empty((timesteps, model.BG_dl.Str1.N, len(state)), dtype=np.float32)
        W_Mani_DMS = np.empty((timesteps, model.BG_dm.Str1.N, len(state)), dtype=np.float32)
        W_BLA_IC = np.empty((timesteps, model.BLA_IC.N, model.BLA_IC.N), dtype=np.float32)

        MC = model.MC
        PFCd_PPC = model.PFCd_PPC
        PL = model.PL
        NAc = model.BG_v.Str1
        DMS = model.BG_dm.Str1
        DLS = model.BG_dl.Str1
        BLA_IC = model.BLA_IC
        DA_1 = model.SNpc.SNpco_1
        DA_2 = model.SNpc.SNpco_2
        DA_3 = model.VTA
        inp = state.copy()

        for t in range(timesteps):
            
            if t < 50:
                inp = np.zeros_like(state)
                
            elif t >= 50:
                inp = state.copy()

            model.step(inp, learning=False)
            action = MC.output.copy()
            attention = PFCd_PPC.output.copy()
            da = np.array([DA_1.output, DA_2.output, DA_3.output]).squeeze()

            MC_output[t] = action
            PFCd_PPC_output[t] = attention
            PL_output[t] = PL.output
            state_t[t] = state
            DLS_output[t] = DLS.output
            DMS_output[t] = DMS.output
            BLA_IC_output[t] = BLA_IC.output
            NAc_output[t] = NAc.output
            DA_timeline[t] = da
            W_BLA_IC[t] = BLA_IC.W
            W_BLA_IC_NAc[t] = model.Ws["BLA_IC_NAc_1"]
            W_Mani_DLS[t] = model.Ws["Mani_DLS_1"]
            W_Mani_DMS[t] = model.Ws["Mani_DMS_1"]

            if np.any(attention >= PFCd_PPC.threshold):
                attention_winner = np.argmax(attention)

                if env[attention_winner] == 1.0:
                    state[0:2] = 0.0
                    state[attention_winner] = 1.0

            else:
                state[0:2] = 0.0
        
        result = {
            "Seed": np.ones(timesteps) * parameters.seed,
            "Phase": np.ones(timesteps) * phase,
            "Trial": np.ones(timesteps) * trial,
            "Timesteps": np.arange(0, timesteps),
            "States_timeline": state_t.copy(),
            "BLA_IC_output": BLA_IC_output.copy(),
            "NAc_output": NAc_output.copy(),
            "DMS_output": DMS_output.copy(),
            "DLS_output": DLS_output.copy(),
            "MC_output": MC_output.copy(),
            "PFCd_PPC_output": PFCd_PPC_output.copy(),
            "PL_output": PL_output.copy(),
            "DA_timeline": DA_timeline.copy(),
            "W_BLA_IC": W_BLA_IC,
            "W_BLA_IC_NAc": W_BLA_IC_NAc,
            "W_Mani_DLS": W_Mani_DLS,
            "W_Mani_DMS": W_Mani_DMS,
        }
        
        results.append(result)

    print(
        f'Simulation termined: Trials({trials}), Timesteps per-trial({timesteps})'
    )
    
    print(
        "Saving results"
        )
    seed_col = ["Seed"]
    trial_col = ["Trial"]
    timestep_col = ["Timestep"]
    phase_col = ["Phase"]
    state_cols = [f"Input_{i}" for i in range(len(state.copy()))]
    BLA_IC_cols = [f"BLA_IC_Unit_{i}" for i in range(model.BLA_IC.N)]
    NAc_cols = [f"NAc_Unit_{i}" for i in range(model.BG_v.Str1.N)]
    DMS_cols = [f"DMS_Unit_{i}" for i in range(model.BG_dm.Str1.N)]
    DLS_cols = [f"DLS_Unit_{i}" for i in range(model.BG_dl.Str1.N)]
    MC_out_cols = [f"MC_Unit_{i}" for i in range(model.MC.N)]
    PFCd_PPC_out_cols = [f"PFCd_PPC_Unit_{i}" for i in range(model.PFCd_PPC.N)]
    PL_out_cols = [f"PL_Unit_{i}" for i in range(model.PL.N)]
    DA_cols = [f"DA_Unit{i}" for i in range(3)]
    W_cols_1 = [
        f"BLA_IC_W{x}_{y}"
        for x in range(model.BLA_IC.W.shape[0])
        for y in range(model.BLA_IC.W.shape[1])
    ]
    W_cols_2 = [
        f"BLA_IC_NAc_W{x}_{y}"
        for x in range(model.Ws["BLA_IC_NAc_1"].shape[0])
        for y in range(model.Ws["BLA_IC_NAc_1"].shape[1])
    ]
    W_cols_3 = [
        f"Mani_DLS_W{x}_{y}"
        for x in range(model.Ws["Mani_DLS_1"].shape[0])
        for y in range(model.Ws["Mani_DLS_1"].shape[1])
    ]
    W_cols_4 = [
        f"Mani_DMS_W{x}_{y}"
        for x in range(model.Ws["Mani_DMS_1"].shape[0])
        for y in range(model.Ws["Mani_DMS_1"].shape[1])
    ]

    cols = (
        seed_col
        + phase_col
        + trial_col
        + timestep_col
        + state_cols
        + BLA_IC_cols
        + NAc_cols
        + DMS_cols
        + DLS_cols
        + MC_out_cols
        + PFCd_PPC_out_cols
        + PL_out_cols
        + DA_cols
        + W_cols_1
        + W_cols_2
        + W_cols_3
        + W_cols_4
    )
    dfs = []

    for res in results:
        values = [
            np.asanyarray(res[k]).reshape(timesteps, -1)
            for k in res.keys()
        ]
        values_conc = np.concatenate(values, axis=1)
        df_new = pd.DataFrame(values_conc, columns=cols)
        dfs.append(df_new)

    df = pd.concat(dfs, ignore_index=True)
    csv_path = "Test_Simulation.csv"

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
    print(
        f"File {str(csv_path)} saved succesfully"
        )
    