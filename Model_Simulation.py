# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:46:01 2025

@author: Nicc
"""

from params import Parameters
from Model_class import Model
import numpy as np, matplotlib.pyplot as plt, argparse, pandas as pd, os

def plotting(results, save = False):
    
    rows = 1 + len(results) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, res in enumerate(results):
        
        MC = np.array(res["MC_Output"])
        
        axs[i].plot(MC[:, 0], label = "MC Unit_1")
        axs[i].plot(MC[:, 1], label = "MC Unit_2")
        
        axs[i].set_title(f'Phase: {res["Phase"][0]}, Input: {res["States_timeline"][-1]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    for j in range(len(results), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    if save:
        filename = "Simulation_plot.png"
        plt.savefig(filename, dpi=300)
        
    plt.show()
    
def plotting_perphase(results, save = False):
    
    phases = sorted(set(res["Phase"][0] for res in results))
    
    rows = 1 + len(phases) // 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, phase in enumerate(phases):

        phase_results = [r for r in results if r["Phase"][0] == phase]
        phase_final_res = phase_results[-1]
        
        MC = np.array(phase_final_res["MC_Output"])
        
        axs[i].plot(MC[:, 0], label = "MC Unit_1")
        axs[i].plot(MC[:, 1], label = "MC Unit_2")
        
        axs[i].set_title(f'Phase: {phase_final_res["Phase"][0]}, Input: {phase_final_res["States_timeline"][0]}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)
        
    for j in range(len(phases), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    # if save:
    #     filename = "Simulation_plot.png"
    #     plt.savefig(filename, dpi=300)
        
    plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser(description="BLA_IC simulation")
    parser.add_argument(
        "-p",
        "--inp",
        type=float,
        nargs=6,
        default=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        default=20,
        help="Input amount of trials (int)",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=2000,
        help="Input amount of timesteps per trial (int)",
    )
    parser.add_argument(
        "-ph",
        "--phases",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Input amount of phases thorugh percentage (float) [eg...0.25, 0.5, 0.75, 1.0], max of 8 phases",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default='plot',
        help="Output mode ('plot', 'plot_perphase', 'save','short_save', 'stream')",
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
    
    model = Model(parameters)
    results = []
    
    for trial in range(parameters.scheduling["trials"]):
        
        print(f'Running trial {trial + 1}')
        model.reset_activity()   
        MC_output = []
        state_t = []
        DLS_output = []
        DMS_output = []
        BLA_IC_output = []
        NAc_output = []
        
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
            state_t.append(state.copy())
            DLS_output.append(model.BG_dl.DLS.output.copy())
            DMS_output.append(model.BG_dm.DMS.output.copy())
            BLA_IC_output.append(model.BLA_IC.output.copy())
            NAc_output.append(model.BG_v.NAc.output.copy())
            
            if np.any(action >= model.MC.threshold):
                
                winner = np.argmax(action)

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
            "MC_Output": MC_output.copy()
            }
        
        print(f'End trial {trial + 1}')
        
        results.append(result)
        
    print(f'Simulation termined: Trials({parameters.scheduling["trials"]}), Timesteps({parameters.scheduling["timesteps"]})')
        
    if args.mode == "plot":
        plotting(results)
        input("Press Enter to exit")
        
    elif args.mode == "plot_perphase":
        plotting_perphase(results)
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
        
        