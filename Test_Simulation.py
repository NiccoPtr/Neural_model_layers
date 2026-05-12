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
    parameters.seed = args.seed
    if Path("C:/Users/Nicc/Desktop/CNR_Model/prm_file.json").exists():
        parameters.load("C:/Users/Nicc/Desktop/CNR_Model/prm_file.json", mode="json")
        print('Imported parameters succesfully')
    else:
        raise ValueError('Parameters file not found')
        
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
    
    results = []

    for trial in range(parameters.scheduling["trials"]):

        print(f"Running trial {trial + 1}")
        model.reset_activity()
        model.update_output_pre()
        MC_output = []
        PFCd_PPC_output = []
        PL_output = []
        state_t = []
        DLS_output = []
        DMS_output = []
        BLA_IC_output = []
        NAc_output = []
        BGv_output = []
        BGdm_output = []
        BGdl_output = []
        MGV_output = []
        P_output = []
        DM_output = []
        W_BLA_IC_NAc = []
        W_Mani_DLS = []
        W_Mani_DMS = []
        W_BLA_IC = []

        if trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][0]:
            phase = 1

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][1]
        ):
            phase = 2

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][2]
        ):
            phase = 3

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][3]
        ):
            phase = 4

        state = np.asanyarray(parameters.scheduling["states"][phase - 1])

        for t in range(parameters.scheduling["timesteps"]):
            
            if t < 50:
                state[0:2] = 0.0
                
            elif t == 50:
                state = np.asanyarray(parameters.scheduling["states"][phase - 1])

            model.step(state, learning=False)
            action = model.MC.output.copy()

            MC_output.append(action.copy())
            PFCd_PPC_output.append(model.PFCd_PPC.output.copy())
            PL_output.append(model.PL.output.copy())
            state_t.append(state.copy())
            DLS_output.append(model.BG_dl.DLS.output.copy())
            DMS_output.append(model.BG_dm.DMS.output.copy())
            BLA_IC_output.append(model.BLA_IC.output.copy())
            NAc_output.append(model.BG_v.NAc.output.copy())
            BGv_output.append(model.BG_v.SNpr.output.copy())
            BGdm_output.append(model.BG_dm.GPi_SNpr.output.copy())
            BGdl_output.append(model.BG_dl.GPi.output.copy())
            MGV_output.append(model.MGV.output.copy())
            P_output.append(model.P.output.copy())
            DM_output.append(model.DM.output.copy())
            W_BLA_IC.append(model.BLA_IC.W.copy())
            W_BLA_IC_NAc.append(model.Ws["BLA_IC_NAc"].copy())
            W_Mani_DLS.append(model.Ws["Mani_DLS"].copy())
            W_Mani_DMS.append(model.Ws["Mani_DMS"].copy())
        
        result = {
            "Seed": np.ones(parameters.scheduling["timesteps"]) * parameters.seed,
            "Phase": np.ones(parameters.scheduling["timesteps"]) * phase,
            "Trial": np.ones(parameters.scheduling["timesteps"]) * trial,
            "Timesteps": np.arange(0, parameters.scheduling["timesteps"]),
            "States_timeline": state_t.copy(),
            "BLA_IC_output": BLA_IC_output.copy(),
            "NAc_output": NAc_output.copy(),
            "DMS_output": DMS_output.copy(),
            "DLS_output": DLS_output.copy(),
            "MC_output": MC_output.copy(),
            "PFCd_PPC_output": PFCd_PPC_output.copy(),
            "PL_output": PL_output.copy(),
            "W_BLA_IC": W_BLA_IC,
            "W_BLA_IC_NAc": W_BLA_IC_NAc,
            "W_Mani_DLS": W_Mani_DLS,
            "W_Mani_DMS": W_Mani_DMS,
        }

        print(f"End trial {trial + 1}")

        results.append(result)

    print(
        f'Simulation termined: Trials({parameters.scheduling["trials"]}), Timesteps per-trial({parameters.scheduling["timesteps"]})'
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
    NAc_cols = [f"NAc_Unit_{i}" for i in range(model.BG_v.NAc.N)]
    DMS_cols = [f"DMS_Unit_{i}" for i in range(model.BG_dm.DMS.N)]
    DLS_cols = [f"DLS_Unit_{i}" for i in range(model.BG_dl.DLS.N)]
    MC_out_cols = [f"MC_Unit_{i}" for i in range(model.MC.N)]
    PFCd_PPC_out_cols = [f"PFCd_PPC_Unit_{i}" for i in range(model.PFCd_PPC.N)]
    PL_out_cols = [f"PL_Unit_{i}" for i in range(model.PL.N)]
    W_cols_1 = [
        f"BLA_IC_W{x}_{y}"
        for x in range(model.BLA_IC.W.shape[0])
        for y in range(model.BLA_IC.W.shape[1])
    ]
    W_cols_2 = [
        f"BLA_IC_NAc_W{x}_{y}"
        for x in range(model.Ws["BLA_IC_NAc"].shape[0])
        for y in range(model.Ws["BLA_IC_NAc"].shape[1])
    ]
    W_cols_3 = [
        f"Mani_DLS_W{x}_{y}"
        for x in range(model.Ws["Mani_DLS"].shape[0])
        for y in range(model.Ws["Mani_DLS"].shape[1])
    ]
    W_cols_4 = [
        f"Mani_DMS_W{x}_{y}"
        for x in range(model.Ws["Mani_DMS"].shape[0])
        for y in range(model.Ws["Mani_DMS"].shape[1])
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
        + W_cols_1
        + W_cols_2
        + W_cols_3
        + W_cols_4
    )
    dfs = []

    for res in results:
        values = [
            np.asanyarray(res[k]).reshape(parameters.scheduling["timesteps"], -1)
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
    