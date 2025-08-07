"""
Created on Mon Aug  4 18:30:29 2025

@author: Utente
"""

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

def set_layers():
    
    BG_dl = Basal_Ganglia_dl(N = 2, 
                             alpha = 0.2, 
                             baseline = 0.1, 
                             DLS_GPi_W = 1.0, 
                             STNdl_GPi_W = 1.0)
    MGV = Leaky_units_exc(N = 2, 
                          alpha = 0.2, 
                          baseline = 0.7)
    MGV.update_weights(np.array([[1, -0.8], [-0.8, 1]]))
    MC = Leaky_units_exc(N = 2, 
                          alpha = 0.2, 
                          baseline = 0.1)
    return BG_dl, MGV, MC
    
def set_env(input_level_1, input_level_2, N = 2):

    inputs = create_input_levels(input_level_1,
                                 input_level_2)    
   
    Ws = {"inp_BGdl" : np.eye(N),
        "BGdl_MGV" : np.eye(N),
        "MGV_MC" : np.eye(N),
        "MC_STNdl" : np.eye(N),
        "MC_DLS" : np.eye(N)}
    
    return inputs, Ws

def run_simulation(input_level_1, input_level_2, max_timesteps):
    
    BG_dl, MGV, MC = set_layers()
    inputs, Ws = set_env(input_level_1, input_level_2, N = 2)
    
    results = []
    for inp in inputs:
        BG_dl.reset_activity()
        MGV.reset_activity()
        MC.reset_activity()
        
        output_MC_history = []
        activity_MC_history = []
        output_MC = np.array([0.0, 0.0])
        for epoch in range(max_timesteps):
            output_BGdl = BG_dl.step(np.dot(Ws["inp_BGdl"], inp), np.dot(Ws["MC_DLS"], output_MC))
            output_MGV = MGV.step(np.dot(Ws["BGdl_MGV"], output_BGdl))
            output_MC = MC.step(np.dot(Ws["MGV_MC"], output_MGV))
            
            output_MC_history.append(output_MC)
            activity_MC_history.append(MC.activity.copy())
            
        result_inp = {
                        "Inputs" : inp,
                        "Final_output" : output_MC,
                        "Output_history" : output_MC_history,
                        "Activity_history" : activity_MC_history
                        }
        results.append(result_inp)
            
    return results
    

def plotting(results):
    
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
    plt.show()