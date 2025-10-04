"""
Created on Mon Aug  4 18:30:29 2025

@author: Utente
"""

from params import parameters
from Layer_types import Leaky_units_exc, Basal_Ganglia_dl
import numpy as np, matplotlib.pyplot as plt

def create_input_levels(input_level_1,
                        input_level_2
                        ) -> list[np.ndarray]:
    """Generates a list of input series (one series per trial setup).

    Each series is a constant input vector for all timesteps.

    Args:
        input_level_1: Tuple of (lower_bound, upper_bound, N) for the first input.
        input_level_2: Tuple of (lower_bound, upper_bound, N) for the second input.

    Returns:
        A list of input levels. Each input level is a pair of floats
            defining the constant inputs for each trial.
    """
    inputs = [np.array([i, j]) for i in np.linspace(*input_level_1) for j in np.linspace(*input_level_2)]
     
    return inputs

def set_layers(parameters):
    
    rng = np.random.RandomState(parameters.seed)
    
    BG_dl = Basal_Ganglia_dl(parameters.N, 
                             parameters.alpha["BG_dl"], 
                             parameters.baseline["DLS"],
                             parameters.baseline["STNdl"],
                             parameters.baseline["GPi"],
                             parameters.BG_dl_W["DLS_GPi_W"], 
                             parameters.BG_dl_W["STNdl_GPi_W"],
                             rng,
                             parameters.noise["BG_dl"])
    
    MGV = Leaky_units_exc(parameters.N, 
                          parameters.alpha["MGV"],
                          parameters.baseline["MGV"],
                          rng,
                          parameters.noise["MGV"])
    
    MGV.update_weights(np.array(parameters.R_matrices["MGV"]))
    
    MC = Leaky_units_exc(parameters.N, 
                         parameters.alpha["MC"], 
                         parameters.baseline["MC"],
                         rng,
                         parameters.noise["MC"])
    
    return BG_dl, MGV, MC
    
def set_env(input_level_1, input_level_2, N):

    inputs = create_input_levels(input_level_1,
                                 input_level_2)    
   
    Ws = {"inp_BGdl" : np.eye(N),
        "BGdl_MGV" : np.eye(N),
        "MGV_MC" : np.eye(N),
        "MC_STNdl" : np.eye(N),
        "MC_DLS" : np.eye(N)}
    
    return inputs, Ws

def set_DA():
    Y = parameters.DA_values["Y"]
    Δ = parameters.DA_values["Δ"]
    DA = parameters.DA_values["DA"]
    return Y, Δ, DA

def run_simulation(input_level_1, input_level_2, max_timesteps):
    
    BG_dl, MGV, MC = set_layers(parameters)
    inputs, Ws = set_env(input_level_1, input_level_2, N=2)
    Y, Δ, DA = set_DA()
    
    results = []
    
    for inp in inputs:
        BG_dl.reset_activity()
        MGV.reset_activity()
        MC.reset_activity()
        
        output_MC_history = []
        activity_MC_history = []
        output_MGV_history = []
        
        for epoch in range(max_timesteps):
            output_BGdl = BG_dl.step(np.dot(Ws["inp_BGdl"], ((Y + Δ * DA) * inp.copy())),
                                    (np.dot(Ws["MC_STNdl"], MC.output.copy())))
            output_MGV = MGV.step(np.dot(Ws["BGdl_MGV"], output_BGdl.copy()))
            output_MC = MC.step(np.dot(Ws["MGV_MC"], output_MGV.copy()))
            
            if np.any(output_MC > 1):
                raise ValueError(f"Clamping failed! Got: {output_MC}")
            
            output_MGV_history.append(np.round(output_MGV.copy(), 4))
            output_MC_history.append(np.round(output_MC.copy(), 4))
            activity_MC_history.append(np.round(MC.activity.copy(), 4))
        
        result_inp = {
            "Inputs": inp.copy(),
            "Final_output": np.round(output_MC.copy(), 4),
            "Output_MGV": output_MGV_history,
            "Output_history": output_MC_history,
            "Activity_history": activity_MC_history
        }
        results.append(result_inp)
    
    return results, inputs, MGV.W.copy()
    

def plotting(results, W, save = False):
    
    unique_inp = sorted(set(
        tuple(res["Inputs"]) for res in results
        ))
    rows = int(np.floor(np.sqrt(len(unique_inp))))
    cols = int(np.ceil(len(unique_inp) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()
    
    for i, inp in enumerate(unique_inp):
        for res in results:
            if tuple(res["Inputs"]) == inp:
                axs[i].plot(res["Output_history"])
        axs[i].set_title(f'Input: {inp}')
        axs[i].legend()
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Activity level')
        axs[i].set_ylim(0, 1)

    plt.tight_layout()
    
    if save:
        W_str = "_".join(map(str, W.flatten()))
        filename = f"Simulation_plot_MGV_{W_str}.png"
        plt.savefig(filename, dpi=300)
        
    plt.show()