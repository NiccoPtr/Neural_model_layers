# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 10:42:12 2025

@author: Utente
"""
from params import parameters
from Layer_types import BLA_IC, Leaky_onset_units_exc
import numpy as np, matplotlib.pyplot as plt

def test_Leaky_Onset_unit(parameters, epochs, steps, eta_b, da, theta_da, max_W):
    
    rng = np.random.RandomState(parameters.seed)
    Layer = Leaky_onset_units_exc(N = 2,
                                  tau_uo = 5,
                                  tau_ui = 5,
                                  baseline = 0.0, 
                                  rng = rng, 
                                  noise = 0.0)
    
    inputs = np.zeros([steps, 2])
    inputs[np.arange(steps) > 50, 0] = 1
    inputs[np.arange(steps) <= 60, 1] = 1
    
    results = []
    output_history = []
    activity_dot_history = []
    W_tracking = []
    inp_tracking = []
    
    for epoch in range(epochs):
        Layer.reset_activity()
        for step in range(steps):
            activity_dot = Layer.output.copy() - Layer.step(inputs[step])
            
            pos = np.maximum(0, activity_dot)
            neg = np.maximum(0, - activity_dot)

            delta_W = (eta_b *
                       np.maximum(0, da - theta_da) *
                       np.outer(pos, neg.T) *
                       (max_W - Layer.W))
            
            Layer.W += delta_W
            
            output_history.append(np.round(Layer.output.copy(), 4))
            activity_dot_history.append(np.round(activity_dot.copy(), 4))
            W_tracking.append(np.round(Layer.W.copy(), 4))
            inp_tracking.append(inputs[step])
            
        result = {"Output": output_history.copy(),
                  "Output_derivative": activity_dot_history.copy(),
                  "Input": inp_tracking.copy()}
            
        results.append(result)
        
    return results

def test_BLA_IC(parameters, steps, epochs):
    
    rng = np.random.RandomState(parameters.seed)
    Layer = BLA_IC(
                    N = 4, 
                    tau_uo = 5,
                    tau_ui = 5,
                    baseline = 0.0, 
                    rng = rng, 
                    noise = 0.0,
                    eta_b = 0.01, 
                    tau_t = 10.0, 
                    alpha_t = 0.2, 
                    max_W = 1.0, 
                    theta_da = 0.2
                    )
    
    inputs = np.array([[1.0, 0.0, 0.0, 0.0]])
    
    results = []
    output_history = []
    matrix_history = []
    inp_tracking = []
    Layer.reset_activity()
    
    for step in range(steps):
        for inp in inputs:
            for epoch in range(epochs):
                Layer.step(inp, parameters.DA_values["DA"])
                
                output_history.append((np.round(Layer.BLA_IC_layer.output.copy(), 4)))
                matrix_history.append(np.round(Layer.BLA_IC_layer.W.copy(), 5))
                inp_tracking.append(inp)
                
        result = {"Output": output_history.copy(),
                  "Matrix": matrix_history.copy(),
                  "Input": inp_tracking.copy()}
        
        results.append(result)
        
    return results

def plotting_2(results):
    
    output = np.array(results[-1]["Output"])
    
    plt.plot(output[:, 0], label="Unit 1")
    plt.plot(output[:, 1], label="Unit 2")
    plt.plot(output[:, 2], label="Unit 3")
    plt.plot(output[:, 3], label="Unit 4")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Output")
    plt.title("Unit Outputs")
    plt.show()
    
    inputs = np.array(results[-1]["Input"])
    
    plt.plot(inputs[:, 0], label = "Input_1")
    plt.plot(inputs[:, 1], label = "Input_2")
    plt.plot(inputs[:, 2], label = "Input_3")
    plt.plot(inputs[:, 3], label = "Input_4")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Input")
    plt.title("Inputs over Time")
    plt.show()
    
    matrices = np.array(results[-1]["Matrix"])
    n = matrices.shape[1]
    for i in range(n):       
        for j in range(n):   
            plt.plot(matrices[:, i, j], label=f"{j+1}_{i+1}")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Input")
    plt.title("Evolution of 4x4 Connection Weights")
    plt.show()