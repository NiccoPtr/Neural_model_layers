# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

from params import Parameters
from Model_class import Model
from matplotlib.gridspec import GridSpec
import numpy as np, matplotlib.pyplot as plt, argparse, pandas as pd, os

def plotting(results, idx):
    
    plt.close('all')
    
    res = results[idx]
    
    #Isolating single layers
    MC = np.array(res['MC_output'])
    PFCd_PPC = np.array(res['PFCd_PPC_output'])
    PL = np.array(res['PL_output'])
    W_BLA_IC = np.array(res['W_BLA_IC'])
    W_BLA_IC_NAc = np.array(res['W_BLA_IC_NAc'])
    W_Mani_DLS = np.array(res['W_Mani_DLS'])
    W_Mani_DMS = np.array(res['W_Mani_DMS'])
    state = np.array(res['States_timeline'])
    
    rows, cols = np.ix_([0, 1], [2, 3])
    W_BLA_IC_NAc = W_BLA_IC_NAc[:, rows, cols]
    
    rows, cols = np.ix_([0, 1], [0, 1])
    W_Mani_DLS = W_Mani_DLS[:, rows, cols]
    W_Mani_DMS = W_Mani_DMS[:, rows, cols]
    
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
    
    #Weights plotting -----------------------------
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
            
def parse_args():
    parser = argparse.ArgumentParser(description="BLA_IC simulation")
    parser.add_argument(
        "-p",
        "--inp",
        type=float,
        nargs=6,
        default=[
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                ],
        help="Input values (six*phases floats), inp.shape[1] == len(phases)",
    )
    parser.add_argument(
        "-tr",
        "--trials",
        type=int,
        default=500,
        help="Input amount of trials (int)",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="Input amount of timesteps per trial (int)",
    )
    parser.add_argument(
        "-ph",
        "--phases",
        type=float,
        default=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        help="Input amount of phases thorugh percentage (float) [eg...0.25, 0.5, 0.75, 1.0], max of 8 phases",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default='plot',
        help="Output mode ('plot','save','short_save', 'stream')",
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
    parameters.load("prm_file.json" ,mode = "json")
    parameters.scheduling = {
                            "trials": args.trials,
                            "phases": args.phases,
                            "timesteps": args.timesteps,
                            "states": np.array(args.inp)
                            }
    #-----MC parameters
    parameters.noise['MC']  = 0.4
    parameters.baseline['GPi'] = 0.2
    parameters.Matrices_scalars['MC_MGV'] = 2.0
    parameters.Matrices_scalars['MGV_MC'] = 1.8
    parameters.Str_Learn['eta_DLS'] = 0.001
    parameters.BG_dl_W['STNdl_GPi_W'] = 1.6
    
    #-----PFCd/PPC parameters
    parameters.noise['PFCd_PPC']  = 0.4
    parameters.baseline['GPi_SNpr'] = 0.2
    parameters.Matrices_scalars['PFCd_PPC_P'] = 2.0
    parameters.Matrices_scalars['P_PFCd_PPC'] = 1.8
    parameters.Str_Learn['eta_DMS'] = 0.001
    parameters.BG_dm_W['STNdm_GPiSNpr_W'] = 1.6
    
    #-----PL parameters
    parameters.baseline['SNpr'] = 0.2
    parameters.BG_v_W['STNv_SNpr_W'] = 1.6
    
    #-----BLA_IC_NAc parameters
    parameters.BLA_Learn['eta_b'] = 0.1
    parameters.BLA_Learn['theta_DA'] = 0.5
    parameters.Str_Learn['theta_inp_NAc'] = 0.8
    parameters.Str_Learn['theta_NAc'] = 0.4
    parameters.Str_Learn['eta_NAc'] = 0.1
    
    if len(args.inp) != len(args.phases):
        raise ValueError('Input and Phases must be the same length')
    
    idx = args.index
    model = Model(parameters)
    results = []
    
    for trial in range(parameters.scheduling["trials"]):
        
        print(f'Running trial {trial + 1}')
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
        W_BLA_IC_NAc = []
        W_Mani_DLS = []
        W_Mani_DMS = []
        W_BLA_IC = []
        
        if trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][0]:
            phase = 1
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][1]:
            phase = 2
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][2]:
            phase = 3
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][3]:
            phase = 4
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][4]:
            phase = 5
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][5]:
            phase = 6
            
        elif trial <= parameters.scheduling["trials"] * parameters.scheduling["phases"][6]:
            phase = 7
        else: 
            phase = 8
            
        state = parameters.scheduling["states"][phase - 1]
        
        for t in range(parameters.scheduling["timesteps"]):
            
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
            W_BLA_IC.append(model.BLA_IC.W.copy())
            W_BLA_IC_NAc.append(model.Ws['BLA_IC_NAc'].copy())
            W_Mani_DLS.append(model.Ws['Mani_DLS'].copy())
            W_Mani_DMS.append(model.Ws['Mani_DMS'].copy())
            
            if np.any(action >= model.MC.threshold):
                winner = np.argmax(action)

                if state[0] == 1.0 and winner == 0:
                    state[2:4] = 0.0
                    state[2 + winner] = 1.0
                    
                elif state[1] == 1.0 and winner == 1:
                    state[2:4] = 0.0
                    state[2 + winner] = 1.0
                
                else:
                    state[2:4] = 0.0
            
        result = {
            "Trial": np.ones(parameters.scheduling["timesteps"]) * trial,
            "Phase": np.ones(parameters.scheduling["timesteps"]) * phase,        
            "States_timeline": state_t.copy(),
            'DLS_output': DLS_output.copy(),
            'DMS_output': DMS_output.copy(),
            'BLA_IC_output': BLA_IC_output.copy(),
            'NAc_output': NAc_output.copy(),
            'PL_output': PL_output.copy(),
            'PFCd_PPC_output' : PFCd_PPC_output.copy(),
            "MC_output": MC_output.copy(),
            'W_BLA_IC': W_BLA_IC,
            'W_BLA_IC_NAc': W_BLA_IC_NAc,
            'W_Mani_DLS': W_Mani_DLS,
            'W_Mani_DMS': W_Mani_DMS
            }
        
        print(f'End trial {trial + 1}')
        
        results.append(result)
        
    print(f'Simulation termined: Trials({parameters.scheduling["trials"]}), Timesteps per-trial({parameters.scheduling["timesteps"]})')
        
    if args.mode == "plot":
        plotting(results, idx)
        input("Press Enter to exit")
        
    elif args.mode == 'stream':
        fin_state = state.copy()
        fin_MC_output = model.MC.output.copy()
        mresults = np.hstack((fin_MC_output, fin_state))
        print(("{:10.5f} " * len(mresults)).format(*mresults))
        
    elif args.mode == 'save':
        trial_col = ['Trial']
        phase_col = ['Phase']
        state_cols = [f"Input_{i}" 
                    for i in range(len(state.copy()))]
        MC_out_cols = [f'MC_Unit_{i}'
                       for i in range(model.MC.N)]
        cols = trial_col + phase_col + state_cols + MC_out_cols
        df = pd.DataFrame(columns=cols)
        
        for res in results:
            values = [np.asanyarray(res[k]).reshape(parameters.scheduling["timesteps"], -1)
                     for k in res.keys()]
            values_conc = np.concatenate(values, axis=1)
            df_new = pd.DataFrame(values_conc, columns=df.columns)
            df = pd.concat(
                [df, df_new], 
                ignore_index=True
                )
            
        csv_path = "Model_Simulation.csv"
        
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
            
    elif args.mode == 'short_save':
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
        
        