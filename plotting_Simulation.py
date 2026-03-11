# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:08:43 2026

@author: Nicc
"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="Model_Simulation.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        '-s',
        "--seed",
        type=int,
        default=0,
        help="Simulation seed to refer, must match used seeds"
    )
    parser.add_argument(
        '-t',
        "--trial",
        type=int,
        default=199,
        help="Define trial to refer for plotting"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.csv)

    if args.seed not in df['Seed'].values:
        raise ValueError(f"Seed {args.seed} not present in Simulation DataFrame")

    if args.trial not in df['Trial'].values:
        raise ValueError(f"Trial {args.trial} is not present in Simulation DataFrame")
        
    #Create a 'df_new' that isolates those rows of interest for every column
    #Take into account Seed and Trial specified by args    
    df_new = df[
        (df["Seed"] == args.seed) &
        (df["Trial"] == args.trial)
        ].sort_values("Timestep").copy()
    
    if df_new.empty:
        
        df_check = df[
            (df["Seed"] == args.seed)
            ].copy()
        
        trials = int(df_check['Trial'].iloc[-1])
        raise ValueError(
            f'Simulation with Seed {args.seed} has {trials} Trials: Trial {args.trial} exceeds excepted values'
        )
    
    timesteps = len(df_new)
    
    #Layers
    MC = df_new.filter(like="MC_Unit").to_numpy()
    PFCd_PPC = df_new.filter(like="PFCd_PPC_Unit").to_numpy()
    PL = df_new.filter(like="PL_Unit").to_numpy()
    state = df_new.filter(like="Input_").to_numpy()
    
    #Matrices
    W_BLA_IC = df_new.filter(like="BLA_IC_W").to_numpy()
    W_BLA_IC_NAc = df_new.filter(like="BLA_IC_NAc_W").to_numpy()
    W_Mani_DLS = df_new.filter(like="Mani_DLS_W").to_numpy()
    W_Mani_DMS = df_new.filter(like="Mani_DMS_W").to_numpy()
    
    #Matrices reshape 
    W_BLA_IC = W_BLA_IC.reshape(timesteps, 4, 4)
    W_BLA_IC_NAc = W_BLA_IC_NAc.reshape(timesteps, 2, 4)
    W_Mani_DLS = W_Mani_DLS.reshape(timesteps, 2, 6)
    W_Mani_DMS = W_Mani_DMS.reshape(timesteps, 2, 6)
    
    #Isolate Matrices' connections of interest
    rows, cols = np.ix_([0, 1], [2, 3])
    W_BLA_IC_NAc = W_BLA_IC_NAc[:, rows, cols]
    
    rows, cols = np.ix_([0, 1], [0, 1])
    W_Mani_DLS = W_Mani_DLS[:, rows, cols]
    W_Mani_DMS = W_Mani_DMS[:, rows, cols]
    
    sns.set_theme(style="whitegrid")
    plt.close('all')
    
    #Plotting set up
    plots = [
        ('MC', [(MC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('PFCd_PPC', [(PFCd_PPC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('PL', [(PL[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('State',[
            (state[:, 0], "Lever"),
            (state[:, 1], "Chain"),
            (state[:, 2], "Food_1"),
            (state[:, 3], "Food_2"),
            (state[:, 4], "Sat_1"),
            (state[:, 5], "Sat_2"),
        ], (-0.1, 1.2))
        ]
    
    n_rows = len(plots) + 4
    fig = plt.figure(figsize=(14, 2.2 * n_rows))
    gs = GridSpec(n_rows, 2, width_ratios=[1, 6], hspace=0.25)
    
    shared_ax = None
    
    #Plotting Layers
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
            ax.plot(df_new["Timestep"], y, label=label)

        ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=5)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(labelbottom=False)
    
    #Weights plotting 
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
        cmap='viridis'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(16), [f"W_{j}_{i}" for j in range(4) for i in range(4)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
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
        cmap='viridis'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
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
        cmap='viridis'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
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
        cmap='viridis'
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