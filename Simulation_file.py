import numpy as np

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

def run_simulation(max_timesteps, input_level_1, input_level_2):
    
    timesteps = max_timesteps
    inputs = create_input_levels(input_level_1, input_level_2)
    
    B_G = Basal_Ganglia(N = 2, alpha = 0.1, threshold = 0.8, baseline = 0.0)
    
    results = []
    
    for inp in inputs:
        B_G.reset_act()
        outputs_GPi = []
        for t in range(timesteps):
            output_GPi = B_G.step(inp)
            outputs_GPi.append(output_GPi.copy())
    
        results.append({"Output_GPi" : output_GPi,
                        "Output_history" : outputs_GPi,
                        "Inputs" : inp
                        })
    
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