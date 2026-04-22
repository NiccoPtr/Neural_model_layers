# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:23:51 2026

@author: Nicc
"""

import argparse
import os
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from Model_class import Model
from params import Parameters
from scheduling import Scheduling

def plotting(results, idx):

    plt.close("all")

    res = results[idx]

    # Isolating single layers
    MC = np.array(res["MC_output"])
    PFCd_PPC = np.array(res["PFCd_PPC_output"])
    PL = np.array(res["PL_output"])
    W_BLA_IC = np.array(res["W_BLA_IC"])
    W_BLA_IC_NAc = np.array(res["W_BLA_IC_NAc"])
    W_Mani_DLS = np.array(res["W_Mani_DLS"])
    W_Mani_DMS = np.array(res["W_Mani_DMS"])
    state = np.array(res["States_timeline"])

    rows, cols = np.ix_([0, 1], [2, 3])
    W_BLA_IC_NAc = W_BLA_IC_NAc[:, rows, cols]

    rows, cols = np.ix_([0, 1], [0, 1])
    W_Mani_DLS = W_Mani_DLS[:, rows, cols]
    W_Mani_DMS = W_Mani_DMS[:, rows, cols]

    # Plotting set up
    plots = [
        ("MC", [(MC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("PFCd_PPC", [(PFCd_PPC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("PL", [(PL[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2))
    ]

    n_rows = len(plots) + 5
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
    
    #States plotting
    title_ax = fig.add_subplot(gs[-5, 0])
    ax = fig.add_subplot(gs[-5, 1], sharex=shared_ax)
    
    title_ax.text(0.5, 0.5, "State", ha="center", va="center", fontsize=12)
    title_ax.axis("off")
    
    im = ax.imshow(
        state.T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap='YlOrRd'
    )
    
    ax.set_yticks(np.arange(6), ['Lever',
                                 'Chain',
                                 'Food_1',
                                 'Food_2',
                                 'Sat_1',
                                 'Sat_2']
                  )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Weights plotting -----------------------------
    title_ax = fig.add_subplot(gs[-4, 0])
    ax = fig.add_subplot(gs[-4, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights BLA_IC", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_BLA_IC.reshape(-1, 4 * 4).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(16), [f"W_{j}_{i}" for j in range(4) for i in range(4)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # -----------------------------------------------

    title_ax = fig.add_subplot(gs[-3, 0])
    ax = fig.add_subplot(gs[-3, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights BLA_IC_NAc", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_BLA_IC_NAc.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # -----------------------------------------------

    title_ax = fig.add_subplot(gs[-2, 0])
    ax = fig.add_subplot(gs[-2, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights Mani_DLS", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_Mani_DLS.reshape(-1, 2 * 2).T,
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

    # -----------------------------------------------

    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights Mani_DMS", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_Mani_DMS.reshape(-1, 2 * 2).T,
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

    xmin, xmax = shared_ax.get_xlim()
    pad = 0.1 * (xmax - xmin)
    shared_ax.set_xlim(xmin, xmax + pad)

    plt.show()

def plot_CTBG(results, idx):
    
    plt.close("all")

    res = results[idx]

    # Isolating single layers
    MC = np.array(res["MC_output"])
    PFCd_PPC = np.array(res["PFCd_PPC_output"])
    PL = np.array(res["PL_output"])
    MGV = np.array(res["MGV_output"])
    P = np.array(res["P_output"])
    DM = np.array(res["DM_output"])
    BGdl = np.array(res["BGdl_output"]) * -1
    BGdm = np.array(res["BGdm_output"]) * -1
    BGv = np.array(res["BGv_output"]) * -1
    state = np.array(res["States_timeline"])
    
    # Plotting set up
    plots = [
        ("MC", [(MC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("MGV", [(MGV[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("BGdl", [(BGdl[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("PFCd_PPC", [(PFCd_PPC[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("P", [(P[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("BGdm", [(BGdm[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("PL", [(PL[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("DM", [(DM[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2)),
        ("BGv", [(BGv[:, i], f"Unit_{i+1}") for i in range(2)], (-0.1, 1.2))
       ]

    n_rows = len(plots) + 1
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
    
    #States plotting
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)
    
    title_ax.text(0.5, 0.5, "State", ha="center", va="center", fontsize=12)
    title_ax.axis("off")
    
    im = ax.imshow(
        state.T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap='YlOrRd'
    )
    
    ax.set_yticks(np.arange(6), ['Lever',
                                 'Chain',
                                 'Food_1',
                                 'Food_2',
                                 'Sat_1',
                                 'Sat_2']
                  )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    # Shared x-axis
    ax.set_xlabel("Timestep")
    
    xmin, xmax = shared_ax.get_xlim()
    pad = 0.1 * (xmax - xmin)
    shared_ax.set_xlim(xmin, xmax + pad)
    
    plt.show()

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
        "-m",
        "--mode",
        type=str,
        default="plot",
        help="Output mode ('plot','plot_CTBG, 'save','short_save', 'stream')",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=-1,
        help="Set trial details you want to plot with the plotting function",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    parameters = Parameters()
    parameters.seed = args.seed
    if Path(f"C:/Users/Nicc/Desktop/CNR_Model/sim_seed{int(parameters.seed)}/params.json").exists():
        parameters.load(f"C:/Users/Nicc/Desktop/CNR_Model/sim_seed{int(parameters.seed)}/params.json", mode="json")
        print('Imported parameters succesfully')
    else:
        raise ValueError('Parameters file not found')
        
    scheduling = Scheduling()
    if args.scheduling is not None:
        scheduling.load(args.scheduling, mode="json")
    parameters.scheduling = scheduling._params_to_dict()

    if len(scheduling.states) != len(scheduling.phases):
        raise ValueError("Input and Phases must have same length")

    idx = args.index
    model = joblib.load(f'C:/Users/Nicc/Desktop/CNR_Model/sim_seed{int(parameters.seed)}/Model_{int(parameters.seed)}.joblib')
    model.MC.noise = 0.0
    model.PFCd_PPC.noise = 0.0
    model.PL.noise = 0.0
    # model.Ws['BLA_IC_NAc'] = np.array([[0.0, 0.0, 1.0, 0.0],
    #                                    [0.0, 0.0, 0.0, 1.0]]) * 2.0
    # model.Ws['Mani_DLS'] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    # model.Ws['Mani_DMS'] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    # model.BLA_IC.W = np.ones((4, 4)) * 0.0
    # model.BLA_IC.W[2, 0] = 1.0
    # model.BLA_IC.W[3, 1] = 1.0
    # model.Ws['PFCd_PPC_PL'] *= 0.2
    # model.Ws['PFCd_PPC_MC'] *= 0.2
    # model.Ws['MC_PFCd_PPC'] *= 0.1
    # model.Ws['PL_PFCd_PPC'] *= 0.1
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

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][4]
        ):
            phase = 5

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][5]
        ):
            phase = 6

        elif (
            trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][6]
        ):
            phase = 7
        else:
            phase = 8

        state = np.asanyarray(parameters.scheduling["states"][phase - 1])

        for t in range(parameters.scheduling["timesteps"]):
            
            if t < 50:
                state[0:2] = 0.0
                
            elif t == 50:
                state = np.asanyarray(parameters.scheduling["states"][phase - 1])

            model.step(state)
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
        
        if args.mode == 'plot_CTBG':
            result = {
                "Seed": np.ones(parameters.scheduling["timesteps"]) * parameters.seed,
                "Phase": np.ones(parameters.scheduling["timesteps"]) * phase,
                "Trial": np.ones(parameters.scheduling["timesteps"]) * trial,
                "Timesteps": np.arange(0, parameters.scheduling["timesteps"]),
                "States_timeline": state_t.copy(),
                "PL_output": PL_output.copy(),
                "PFCd_PPC_output": PFCd_PPC_output.copy(),
                "MC_output": MC_output.copy(),
                "BLA_IC_output": BLA_IC_output.copy(),
                "BGv_output": BGv_output.copy(),
                "BGdm_output": BGdm_output.copy(),
                "BGdl_output": BGdl_output.copy(),
                "MGV_output": MGV_output.copy(),
                "P_output": P_output.copy(),
                "DM_output": DM_output.copy(),
                "W_BLA_IC": W_BLA_IC,
                "W_BLA_IC_NAc": W_BLA_IC_NAc,
                "W_Mani_DLS": W_Mani_DLS,
                "W_Mani_DMS": W_Mani_DMS,
            }
    
            print(f"End trial {trial + 1}")
            
        elif args.mode == 'plot' or args.mode == 'save':
            result = {
                "Seed": np.ones(parameters.scheduling["timesteps"]) * parameters.seed,
                "Phase": np.ones(parameters.scheduling["timesteps"]) * phase,
                "Trial": np.ones(parameters.scheduling["timesteps"]) * trial,
                "Timesteps": np.arange(0, parameters.scheduling["timesteps"]),
                "States_timeline": state_t.copy(),
                "BLA_IC_output": BLA_IC_output.copy(),
                "NAc_output": NAc_output.copy(),
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

    if args.mode == "plot":
        plotting(results, idx)
        input("Press Enter to exit")
        
    elif args.mode == "plot_CTBG":
        plot_CTBG(results, idx)
        input("Press Enter to exit")

    elif args.mode == "stream":
        fin_state = state.copy()
        fin_MC_output = model.MC.output.copy()
        mresults = np.hstack((fin_MC_output, fin_state))
        print(("{:10.5f} " * len(mresults)).format(*mresults))

    elif args.mode == "save":
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
        
    elif args.mode == "short_save":
        fin_state = state.copy()
        fin_MC_output = model.MC.output.copy()
        state_cols = [f"Input_{i}" for i in range(len(fin_state))]
        MC_out_cols = [f"MC_Unit_{i}" for i in range(len(fin_MC_output))]

        cols = state_cols + MC_out_cols
        values = np.concatenate([fin_state, fin_MC_output])
        df = pd.DataFrame([values], columns=cols)

        csv_path = "Model_Simulation_short.csv"

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    parameters.save("params.json", mode="json")