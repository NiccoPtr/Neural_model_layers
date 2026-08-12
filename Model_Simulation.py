# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

import argparse
import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from Model_class import Model
from params import Parameters
from scheduling import Scheduling

def parse_args():
    parser = argparse.ArgumentParser(description="BLA_IC simulation")
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
        default=0,
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
        
    scheduling = Scheduling()
    if args.scheduling is not None:
        scheduling.load(args.scheduling, mode="json")
    parameters.scheduling = scheduling._params_to_dict()
    parameters.seed = args.seed
        
    model = Model(parameters)
    
    if args.lesion == "BLA":
        model.BLA_IC.lesion = True
        
    elif args.lesion == "NAc":
        model.BG_v.Str1.lesion = True
        model.BG_v.Str2.lesion = True
        
    elif args.lesion == "DMS":
        model.BG_dm.Str1.lesion = True
        model.BG_dm.Str2.lesion = True
        
    elif args.lesion == "PL":
        model.PL.lesion = True

    sched = parameters.scheduling
    timesteps = sched["timesteps"]
    trials = sched["trials"]

    results = []

    for trial in range(trials):

        if trial <= (trials*(sched["phases"][0])):
            env = np.array(sched["states"][0])

        elif trial <= (trials*(sched["phases"][1])):
            env = np.array(sched["states"][1])

        state = env.copy() * 0.0

        model.reset_activity()
        model.update_output_pre()
        MC_output = np.empty((timesteps, model.MC.N), dtype=np.float32)
        PFCd_PPC_output = np.empty((timesteps, model.PFCd_PPC.N), dtype=np.float32)
        PL_output = np.empty((timesteps, model.PL.N), dtype=np.float32)
        state_t = np.empty((timesteps, len(state)), dtype=np.float32)
        DLS_output_1 = np.empty((timesteps, model.BG_dl.Str1.N), dtype=np.float32)
        DLS_output_2 = np.empty((timesteps, model.BG_dl.Str2.N), dtype=np.float32)
        DMS_output_1 = np.empty((timesteps, model.BG_dm.Str1.N), dtype=np.float32)
        DMS_output_2 = np.empty((timesteps, model.BG_dm.Str2.N), dtype=np.float32)
        BLA_IC_output = np.empty((timesteps, model.BLA_IC.N), dtype=np.float32)
        NAc_output_1 = np.empty((timesteps, model.BG_v.Str1.N), dtype=np.float32)
        NAc_output_2 = np.empty((timesteps, model.BG_v.Str2.N), dtype=np.float32)
        BGv_output = np.empty((timesteps, model.BG_v.GPi_SNpr.N), dtype=np.float32)
        BGdm_output = np.empty((timesteps, model.BG_dm.GPi_SNpr.N), dtype=np.float32)
        BGdl_output = np.empty((timesteps, model.BG_dl.GPi_SNpr.N), dtype=np.float32)
        MGV_output = np.empty((timesteps, model.MGV.N), dtype=np.float32)
        P_output = np.empty((timesteps, model.P.N), dtype=np.float32)
        DM_output = np.empty((timesteps, model.DM.N), dtype=np.float32)
        DA_timeline = np.empty((timesteps, 3), dtype=np.float32)
        W_BLA_IC_NAc_1 = np.empty((timesteps, model.BG_v.Str1.N, model.BLA_IC.N), dtype=np.float32)
        W_BLA_IC_NAc_2 = np.empty((timesteps, model.BG_v.Str2.N, model.BLA_IC.N), dtype=np.float32)
        W_Mani_DLS_1 = np.empty((timesteps, model.BG_dl.Str1.N, len(state)), dtype=np.float32)
        W_Mani_DLS_2 = np.empty((timesteps, model.BG_dl.Str2.N, len(state)), dtype=np.float32)
        W_Mani_DMS_1 = np.empty((timesteps, model.BG_dm.Str1.N, len(state)), dtype=np.float32)
        W_Mani_DMS_2 = np.empty((timesteps, model.BG_dm.Str2.N, len(state)), dtype=np.float32)
        W_BLA_IC = np.empty((timesteps, model.BLA_IC.N, model.BLA_IC.N), dtype=np.float32)

        MC = model.MC
        PFCd_PPC = model.PFCd_PPC
        PL = model.PL
        NAc_1 = model.BG_v.Str1
        DMS_1 = model.BG_dm.Str1
        DLS_1 = model.BG_dl.Str1
        NAc_2 = model.BG_v.Str2
        DMS_2 = model.BG_dm.Str2
        DLS_2 = model.BG_dl.Str2
        BLA_IC = model.BLA_IC
        DA_1 = model.SNpc.SNpco_1
        DA_2 = model.SNpc.SNpco_2
        DA_3 = model.VTA
        inp = state.copy()

        for t in range(timesteps):

            # if t < 50:
            #     model.SNpc.SNpco_1.baseline = 0.1
            #     model.SNpc.SNpco_2.baseline = 0.1
            #     model.VTA.baseline = 0.1

            # elif t == 50:
            #     model.SNpc.SNpco_1.baseline = parameters.baseline["SNpco"]
            #     model.SNpc.SNpco_2.baseline = parameters.baseline["SNpco"]
            #     model.VTA.baseline = parameters.baseline["VTA"]
            
            if t < 50:
                inp = np.zeros_like(state)
                
            elif t >= 50:
                if np.any(inp[2:4] == 1.0):
                    inp *= 1.0

                else:
                    inp = state.copy()

            model.step(inp)

            action = MC.output.copy()
            attention = PFCd_PPC.output.copy()
            da = np.array([DA_1.output, DA_2.output, DA_3.output]).squeeze()

            MC_output[t] = action
            PFCd_PPC_output[t] = attention
            PL_output[t] = PL.output
            state_t[t] = inp.copy()
            DLS_output_1[t] = DLS_1.output
            DLS_output_2[t] = DLS_2.output
            DMS_output_1[t] = DMS_1.output
            DMS_output_2[t] = DMS_2.output
            BLA_IC_output[t] = BLA_IC.output
            NAc_output_1[t] = NAc_1.output
            NAc_output_2[t] = NAc_2.output
            DA_timeline[t] = da
            W_BLA_IC[t] = BLA_IC.W
            W_BLA_IC_NAc_1[t] = model.Ws["BLA_IC_NAc_1"]
            W_BLA_IC_NAc_2[t] = model.Ws["BLA_IC_NAc_2"]
            W_Mani_DLS_1[t] = model.Ws["Mani_DLS_1"]
            W_Mani_DLS_2[t] = model.Ws["Mani_DLS_2"]
            W_Mani_DMS_1[t] = model.Ws["Mani_DMS_1"]
            W_Mani_DMS_2[t] = model.Ws["Mani_DMS_2"]

            if np.any(attention >= PFCd_PPC.threshold):
                attention_winner = np.argmax(attention)

                if env[attention_winner] == 1.0:
                    state[0:2] = 0.0
                    state[attention_winner] = 1.0

            else:
                state[0:2] = 0.0

            if t >= 100 and np.any(action >= MC.threshold):
                action_winner = np.argmax(action)

                if state[action_winner] == 1.0:
                    state[2:4] = 0.0
                    state[2 + action_winner] = 1.0
        
        result = {
            "Seed": np.ones(timesteps) * parameters.seed,
            "Trial": np.ones(timesteps) * trial,
            "Timesteps": np.arange(0, timesteps),
            "States_timeline": state_t.copy(),
            "BLA_IC_output": BLA_IC_output.copy(),
            "NAc_output_1": NAc_output_1.copy(),
            "NAc_output_2": NAc_output_2.copy(),
            "DMS_output_1": DMS_output_1.copy(),
            "DMS_output_2": DMS_output_2.copy(),
            "DLS_output_1": DLS_output_1.copy(),
            "DLS_output_2": DLS_output_2.copy(),
            "Action": MC_output.copy(),
            "Attention": PFCd_PPC_output.copy(),
            "PL_output": PL_output.copy(),
            "DA_timeline": DA_timeline,
            "W_BLA_IC": W_BLA_IC,
            "W_BLA_IC_NAc_1": W_BLA_IC_NAc_1,
            "W_Mani_DLS_1": W_Mani_DLS_1,
            "W_Mani_DMS_1": W_Mani_DMS_1,
            "W_BLA_IC_NAc_2": W_BLA_IC_NAc_2,
            "W_Mani_DLS_2": W_Mani_DLS_2,
            "W_Mani_DMS_2": W_Mani_DMS_2
        }

        results.append(result)
    
    print(
        f'Simulation termined: Trials({trials}), Timesteps per-trial({timesteps})'
    )

    #Saving results
    print(
        "Saving results"
        )
    seed_col = ["Seed"]
    trial_col = ["Trial"]
    timestep_col = ["Timestep"]
    state_cols = [f"Input_{i}" for i in range(len(state.copy()))]
    BLA_IC_cols = [f"BLA_IC_Unit_{i}" for i in range(model.BLA_IC.N)]
    NAc_1_cols = [f"NAc_1_Unit_{i}" for i in range(model.BG_v.Str1.N)]
    NAc_2_cols = [f"NAc_2_Unit_{i}" for i in range(model.BG_v.Str2.N)]
    DMS_1_cols = [f"DMS_1_Unit_{i}" for i in range(model.BG_dm.Str1.N)]
    DMS_2_cols = [f"DMS_2_Unit_{i}" for i in range(model.BG_dm.Str2.N)]
    DLS_1_cols = [f"DLS_1_Unit_{i}" for i in range(model.BG_dl.Str1.N)]
    DLS_2_cols = [f"DLS_2_Unit_{i}" for i in range(model.BG_dl.Str2.N)]
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
        f"BLA_IC_NAc_1_W{x}_{y}"
        for x in range(model.Ws["BLA_IC_NAc_1"].shape[0])
        for y in range(model.Ws["BLA_IC_NAc_1"].shape[1])
    ]
    W_cols_3 = [
        f"Mani_DLS_1_W{x}_{y}"
        for x in range(model.Ws["Mani_DLS_1"].shape[0])
        for y in range(model.Ws["Mani_DLS_1"].shape[1])
    ]
    W_cols_4 = [
        f"Mani_DMS_1_W{x}_{y}"
        for x in range(model.Ws["Mani_DMS_1"].shape[0])
        for y in range(model.Ws["Mani_DMS_1"].shape[1])
    ]
    W_cols_5 = [
        f"BLA_IC_NAc_2_W{x}_{y}"
        for x in range(model.Ws["BLA_IC_NAc_2"].shape[0])
        for y in range(model.Ws["BLA_IC_NAc_2"].shape[1])
    ]
    W_cols_6 = [
        f"Mani_DLS_2_W{x}_{y}"
        for x in range(model.Ws["Mani_DLS_2"].shape[0])
        for y in range(model.Ws["Mani_DLS_2"].shape[1])
    ]
    W_cols_7 = [
        f"Mani_DMS_2_W{x}_{y}"
        for x in range(model.Ws["Mani_DMS_2"].shape[0])
        for y in range(model.Ws["Mani_DMS_2"].shape[1])
    ]

    cols = (
        seed_col
        + trial_col
        + timestep_col
        + state_cols
        + BLA_IC_cols
        + NAc_1_cols
        + NAc_2_cols
        + DMS_1_cols
        + DMS_2_cols
        + DLS_1_cols
        + DLS_2_cols
        + MC_out_cols
        + PFCd_PPC_out_cols
        + PL_out_cols
        + DA_cols
        + W_cols_1
        + W_cols_2
        + W_cols_3
        + W_cols_4
        + W_cols_5
        + W_cols_6
        + W_cols_7
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
    csv_path = "Model_Simulation.csv"

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
    model_path = f'Model_{int(parameters.seed)}.joblib'
    joblib.dump(model, model_path)
    print(
        f"File {str(csv_path)} saved succesfully"
        )
