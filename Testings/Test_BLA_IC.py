# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 10:42:12 2025

@author: Utente
"""

from params import parameters
from Layer_types import Leaky_onset_units_exc, BLA_IC_Layer
import numpy as np, matplotlib.pyplot as plt

def set_inputs_1(steps, N):
    
    inputs = np.zeros([steps, N])
    b1 = int(0.5 * steps)
    b2 = int(0.55 * steps)
    b3 = int(0.8 * steps)
    inputs[b1:b2, 0] = 3.0
    inputs[b2:b3, :] = 3.0
    
    return inputs

def test_Leaky_Onset_unit(parameters, epochs, steps, eta_b, da, tau_t, alpha_t, max_W, theta_da, inputs):
    
    rng = np.random.RandomState(parameters.seed)
    Layer = Leaky_onset_units_exc(N = 2,
                                  tau_uo = 5,
                                  tau_ui = 5,
                                  baseline = 0.0, 
                                  rng = rng, 
                                  noise = 0.0)
    
    t_dot = np.zeros(Layer.N)
    t = np.zeros(Layer.N)
    results = []
    output_history = []
    output_dot_history = []
    t_dot_history = []
    W_tracking = []
    inp_tracking = []
    
    for epoch in range(epochs):
        Layer.reset_activity()
        for step in range(steps):
            
            t_dot = -t / tau_t + alpha_t * Layer.step(inputs[step])
            t += t_dot
            
            pos = np.maximum(0, t_dot)
            neg = np.maximum(0, - t_dot)

            delta_W = (eta_b *
                       np.maximum(0, da - theta_da) *
                       np.outer(pos, neg.T) *
                       (max_W - Layer.W))
            
            Layer.W += delta_W
            
            output_history.append(np.round(Layer.output.copy(), 4))
            t_dot_history.append(np.round(t_dot.copy(), 4))
            W_tracking.append(np.round(Layer.W.copy(), 4))
            inp_tracking.append(inputs[step])
            
        result = {"Output": output_history.copy(),
                  "Trace_derivative": t_dot_history.copy(),
                  "W_tracking": W_tracking,
                  "Input": inp_tracking.copy()}
            
        results.append(result)
        
    return results

def plotting_1(results):
    
    output = np.array(results[-1]["Output"])
    
    plt.plot(output[:, 0], label="Unit 1")
    plt.plot(output[:, 1], label="Unit 2")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Output")
    plt.title("Unit Outputs")
    plt.show()
    
    output_dot = np.array(results[-1]["Trace_derivative"])
    
    plt.plot(output_dot[:, 0], label="Unit 1")
    plt.plot(output_dot[:, 1], label="Unit 2")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Output_trace")
    plt.title("Trace Derivative Unit Outputs")
    plt.show()

    inputs = np.array(results[-1]["Input"])
    
    plt.plot(inputs[:, 0], label = "Input_1")
    plt.plot(inputs[:, 1], label = "Input_2")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Input")
    plt.title("Inputs over Time")
    plt.show()

    matrices = np.array(results[-1]["W_tracking"])
    n = matrices.shape[1]
    for i in range(n):       
        for j in range(n):   
            plt.plot(matrices[:, i, j], label=f"{j+1}_{i+1}")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Input")
    plt.title("Evolution of 4x4 Connection Weights")
    plt.show()


def set_inputs_2(steps, N):
    
    inputs = np.zeros([steps, N])
    b1 = int(0.2 * steps)
    b2 = int(0.25 * steps)
    b3 = int(0.8 * steps)
    inputs[b1:b3, 0] = 3.0
    inputs[b2:b3, 1] = 3.0
    
    return inputs

def test_BLA_IC(parameters, epochs, steps):
    
    rng = np.random.RandomState(parameters.seed)
    Layer = BLA_IC_Layer(
                    N = 4, 
                    tau_uo = 20,
                    tau_ui = 20,
                    baseline = 0.0, 
                    rng = rng, 
                    noise = 0.0,
                    eta_b = 0.1, 
                    tau_t = 10.0, 
                    alpha_t = 0.2, 
                    max_W = 1.0, 
                    theta_da = 0.2
                    )
    
    inputs = set_inputs_2(steps, Layer.BLA_IC_layer.N)
    
    results = []
    output_history = []
    t_dot_history = []
    matrix_history = []
    inp_tracking = []
    Layer.reset_activity()
    
    for epoch in range(epochs):
        Layer.reset_activity()
        for step in range(steps):
            Layer.step(inputs[step], parameters.DA_values["DA"])
            
            output_history.append((np.round(Layer.BLA_IC_layer.output.copy(), 4)))
            t_dot_history.append(np.round(Layer.t_dot.copy(), 4))
            matrix_history.append(np.round(Layer.BLA_IC_layer.W.copy(), 4))
            inp_tracking.append(inputs[step])
                
        result = {"Output": output_history.copy(),
                  "Trace_derivative": t_dot_history.copy(),
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
    
    t_dot = np.array(results[-1]["Trace_derivative"])
    
    plt.plot(t_dot[:, 0], label="Unit 1")
    plt.plot(t_dot[:, 1], label="Unit 2")
    plt.plot(t_dot[:, 2], label="Unit 3")
    plt.plot(t_dot[:, 3], label="Unit 4")
    plt.legend(loc = "upper right")
    plt.xlabel("Timesteps")
    plt.ylabel("Output_trace")
    plt.title("Trace Derivative Unit Outputs")
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
    
    
def testing_HebbLearning_W(N, epochs, steps):
    
    results = []
    
    for i in range(25, 76, 2):
        if i < 50:
            inputs = np.zeros([200, 2])
            b1 = int(0.5 * 200)
            b2 = int((i/100) * 200)
            b3 = int(0.95 * 200)
            inputs[b2:b1, 1] = 3.0
            inputs[b1:b3, :] = 3.0
        
        else:
            inputs = np.zeros([200, 2])
            b1 = int(0.5 * 200)
            b2 = int((i/100) * 200)
            b3 = int(0.95 * 200)
            inputs[b1:b2, 0] = 3.0
            inputs[b2:b3, :] = 3.0
        
        run = test_Leaky_Onset_unit(parameters,
                                    epochs = 1,
                                    steps = 200,
                                    eta_b = 0.1,
                                    da = 1.0,
                                    tau_t = 10.0,
                                    alpha_t = 0.2,
                                    max_W = 1.0,
                                    theta_da = 0.2, 
                                    inputs = inputs)

        result = {"Matrix_value": run[0]["W_tracking"][-1].copy(),
                  "Inp_time_diff": i - 50
                  }
        
        results.append(result.copy())
    
    return results

def plotting_3(results):
    
    M_Values_1_2 = [W["Matrix_value"][1, 0] for W in results]
    M_Values_2_1 = [W["Matrix_value"][0, 1] for W in results]
    inp_time_diff = [T["Inp_time_diff"] for T in results]
    
    plt.plot(inp_time_diff, M_Values_1_2, label = "1→2")
    plt.legend(loc = "upper right")
    plt.xlabel("Inp_time_diff")
    plt.ylabel("Matrix_value")
    plt.title("Matrix update given input presentation time difference")
    plt.show()
    
    plt.plot(inp_time_diff, M_Values_2_1, label = "2→1")
    plt.legend(loc = "upper right")
    plt.xlabel("Inp_time_diff")
    plt.ylabel("Matrix_value")
    plt.title("Matrix update given input presentation time difference")
    plt.show()
        
        