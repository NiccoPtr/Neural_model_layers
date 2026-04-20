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
        default="sim_seed0/Model_Simulation.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        '-t',
        "--trial",
        type=int,
        default=149,
        help="Define trial to refer for plotting"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(f'Reading file {str(args.csv)}')
    df = pd.read_csv(args.csv)

    if args.trial not in df['Trial'].values:
        raise ValueError(f"Trial {args.trial} is not present, simulation contains {int(df['Trial'].iloc[-1])} Trials")
            
    #Create a 'df_new' that isolates those rows of interest for every column
    #Take into account Seed and Trial specified by args  
    print(f'Creating DataFrame isolated to Trial {str(args.trial)}')
    df_new = df[
        (df["Trial"] == args.trial)
        ].sort_values("Timestep").copy()
    
    print('Generating plotting image')
    timesteps = len(df_new)
    
    #Layers
    MC = df_new.filter(like="MC_Unit").to_numpy()
    PFCd_PPC = df_new.filter(like="PFCd_PPC_Unit").to_numpy()
    PL = df_new.filter(like="PL_Unit").to_numpy()
    BLA_IC = df_new.filter(like="BLA_IC_Unit").to_numpy()
    NAc = df_new.filter(like="NAc_Unit").to_numpy() * -1
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
    
    #State
    state = df_new.filter(like='Input').to_numpy()
    
    plt.close('all')
    
    #Plotting set up
    plots = [
        ('BLA_IC', [(BLA_IC[:, i], f'Unit_{i+1}') for i in range(4)], (-0.1, 1.2)),
        ('NAc', [(NAc[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('MC', [(MC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('PFCd_PPC', [(PFCd_PPC[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2)),
        ('PL', [(PL[:, i], f'Unit_{i+1}') for i in range(2)], (-0.1, 1.2))
        ]
    
    n_rows = len(plots) + 5
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
        
    #State plotting
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
        cmap='YlOrRd'
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
        cmap='YlOrRd'
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
        cmap='YlOrRd'
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
        cmap='YlOrRd'
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