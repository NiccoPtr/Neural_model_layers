# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:08:43 2026

@author: Nicc
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from params import Parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        required=True,
        help="Verification ID Number",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file"
    )
    parser.add_argument(
        '-t',
        "--trial",
        type=int,
        help="Define trial to refer for plotting"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    parameters = Parameters()

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "CNR_model_data" / "PIT"

    INST_DIR = DATA_DIR / "instrumental_training"

    param_file = (
        INST_DIR
        / f"instrumental_training_{args.id}"
        / "inst_sim_seed1"
        / "prm_file.json"
    )

    if not param_file.exists():
        param_file = (
            INST_DIR
            / f"instrumental_training_{args.id}"
            / "inst_sim_seed31"
            / "prm_file.json"
        )

    if param_file.exists():
        parameters.load(param_file, mode="json")
        print("Imported parameters successfully")
    else:
        raise FileNotFoundError(
            f"Parameters file not found: {param_file}"
        )

    df = pd.read_csv(args.csv)

    if args.trial not in df['Trial'].values:
        raise ValueError(f"Trial {args.trial} is not present, simulation contains {int(df['Trial'].iloc[-1])} Trials")
            
    #Create a 'df_new' that isolates those rows of interest for every column
    #Take into account Seed and Trial specified by args  
    df_new = df[
        (df["Trial"] == args.trial)
        ].sort_values("Timestep").copy()
    
    timesteps = len(df_new)

    action_threshold = parameters.threshold["MC"]
    attention_threshold = parameters.threshold["PFCd_PPC"]
    
    #Layers
    MC = df_new.filter(like="MC_Unit").to_numpy()
    PFCd_PPC = df_new.filter(like="PFCd_PPC_Unit").to_numpy()
    PL = df_new.filter(like="PL_Unit").to_numpy()
    BLA_IC = df_new.filter(like="BLA_IC_Unit").to_numpy()
    CeA = df_new.filter(like="CeA_Unit").to_numpy() * -1
    NAc = df_new.filter(like="NAc_Unit").to_numpy() * -1
    DMS = df_new.filter(like="DMS_Unit").to_numpy() * -1
    DLS = df_new.filter(like="DLS_Unit").to_numpy() * -1
    DA = df_new.filter(like="DA_Unit").to_numpy()
    
    #State
    state = df_new.filter(like='Input').to_numpy()
    
    plt.close('all')
    labels = [
            'Light_1', 
            'Light_2', 
            'Light_3',
            'Lever', 
            'Chain', 
            'Food_1', 
            'Food_2', 
            'Food_3',
        ]
    
    #Plotting set up
    plots = [
        ('BLA_IC', [(BLA_IC[:, i], labels[i]) for i in range(8)], (-0.1, 1.2), None),
        ('CeA', [(CeA[:], 'CeA_Unit')], (-0.1, 1.2), None),
        ('NAc', [(NAc[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), None),
        ('DMS', [(DMS[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), None),
        ('DLS', [(DLS[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), None),
        ('MC', [(MC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), action_threshold),
        ('PFCd_PPC', [(PFCd_PPC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), attention_threshold),
        ('PL', [(PL[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2), None),
        ("DA", [(DA[:, i], ["SNpco_1", "SNpco_2", "VTA"][i]) for i in range(3)], (-0.1, 1.2), None)
        ]
    
    n_rows = len(plots) + 1
    fig = plt.figure(figsize=(14, 2.2 * n_rows))
    gs = GridSpec(n_rows,
                  2,
                  width_ratios=[0.3, 8],
                  height_ratios=[5.0, 1.2, 1.8, 1.8, 1.8, 2.5, 2.5, 1.8, 2.5, 6.0],
                  hspace=0.25
                  )
    
    shared_ax = None
    
    #Plotting Layers
    for i, (title, lines, ylim, threshold) in enumerate(plots):
        title_ax = fig.add_subplot(gs[i, 0])
        ax = fig.add_subplot(gs[i, 1], sharex=shared_ax)

        if shared_ax is None:
            shared_ax = ax

        # Left column: titles only
        title_ax.text(0.2, 0.5, title, ha="center", va="center", fontsize=15)
        title_ax.axis("off")

        # Right column: actual plot
        for y, label in lines:
            ax.plot(df_new["Timestep"], y, label=label)

            if threshold is not None:
                ax.hlines(
                    y=threshold,
                    xmin=0,
                    xmax=1000,
                    color="black",
                    linestyle=":",
                    linewidth=1.5
                )

            if title != "BLA_IC" and title != "CeA":
                ax.axvline(
                    x=50,
                    color="green",
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.8
                )

        ax.set_ylim(*ylim)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.5))
        else:
            ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.15))

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(labelbottom=False)
        
    #State plotting
    title_ax = fig.add_subplot(gs[-1, 0])
    ax = fig.add_subplot(gs[-1, 1], sharex=shared_ax)
    
    title_ax.text(0.2, 0.5, "State", ha="center", va="center", fontsize=15)
    title_ax.axis("off")
    
    im = ax.imshow(
        state.T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap='YlOrRd'
    )
    
    ax.set_yticks(np.arange(12), [
        'Light_1',
        'Light_2',
        'Light_3',
        'Lever',
        'Chain',
        'Food_1',
        'Food_2',
        'Food_3',
        'G_Sat',
        'Sat_1',
        'Sat_2',
        'Sat_3'
        ],
        fontsize=8
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