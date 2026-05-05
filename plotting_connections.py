# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:29:06 2026

@author: Nicc
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def parse_args():
    parser = argparse.ArgumentParser()
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

    df = pd.read_csv(args.csv)

    if args.trial not in df['Trial'].values:
        raise ValueError(f"Trial {args.trial} is not present, simulation contains {int(df['Trial'].iloc[-1])} Trials")
            
    #Create a 'df_new' that isolates those rows of interest for every column
    #Take into account Seed and Trial specified by args  
    df_new = df[
        (df["Trial"] == args.trial)
        ].sort_values("Timestep").copy()
    
    timesteps = len(df_new)
    
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
    
    plt.close('all')
    
    #Plotting set up
    
    n_rows = 4
    fig = plt.figure(figsize=(14, 2.2 * n_rows))
    gs = GridSpec(n_rows, 2, width_ratios=[1, 6], hspace=0.25)
    
    shared_ax = None
    
    #Weights plotting 
    title_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights BLA_IC", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_BLA_IC.reshape(-1, 4 * 4).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap='YlOrRd'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(16), [f"W_{j}_{i}" for j in range(4) for i in range(4)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
    title_ax = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[1, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights BLA_IC_NAc", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_BLA_IC_NAc.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap='YlOrRd'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
    title_ax = fig.add_subplot(gs[2, 0])
    ax = fig.add_subplot(gs[2, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights Mani_DLS", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_Mani_DLS.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap='YlOrRd'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    #-----------------------------------------------
    
    title_ax = fig.add_subplot(gs[3, 0])
    ax = fig.add_subplot(gs[3, 1], sharex=shared_ax)

    title_ax.text(0.5, 0.5, "Weights Mani_DMS", ha="center", va="center", fontsize=12)
    title_ax.axis("off")

    im = ax.imshow(
        W_Mani_DMS.reshape(-1, 2 * 2).T,
        interpolation="none",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap='YlOrRd'
    )

    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(4), [f"W_{j}_{i}" for j in range(2) for i in range(2)])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    
    # Shared x-axis
    ax.set_xlabel("Timestep")
    
    plt.show()