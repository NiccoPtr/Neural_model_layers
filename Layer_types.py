# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:09:24 2025

@author: Utente
"""

import numpy as np

class Leaky_units_exc:

    def __init__(self, N: int, alpha: float, baseline: float):
        """Initializes the neural unit.

        Args:
            N (int): Number of neurons in the model (must be 2).
            alpha (float): Learning rate for activity update.
            threshold (float): Threshold for action selection.

        """
        self.N = N
        self.W = np.zeros((N, N))
        self.alpha = alpha
        self.baseline = np.ones(N) * baseline
        self.activity = np.array([self.baseline])
        self.output = 0.0

    def update_weights(self, W: np.array):
        """Updates the weight matrix.

        Args:
            W (np.ndarray): Weight matrix of shape (N, N).

        Raises:
            ValueError: If the weight matrix shape is not (N, N).
        """
        if W.shape == (self.N, self.N):
            self.W = W
        else:
            raise ValueError("Weight matrix shape incorrect. Expected shape ({}, {})".format(self.N, self.N))

    def reset_activity(self):
        """Resets neuron activity.

        Args:
            start_activity (np.ndarray): Initial activity vector of length N.

        Raises:
            ValueError: If the start activity vector does not have N elements.
        """
        self.activity *= 0
        self.activity += np.array([self.baseline])
        self.output *= 0
        

    def step(self, inputs):
        """Runs a single timestep, updating activity.

        Args:
            inputs (np.ndarray): Input vector of length N.

        Returns:
            np.ndarray: Updated activity vector.
        """
        net_input = np.dot(self.W, self.output) + inputs
        self.activity += self.alpha * (net_input + self.baseline - self.activity)
        self.output = np.maximum(0, np.tanh(self.activity.copy()))
        return self.output.copy()
    
class Leaky_units_inh(Leaky_units_exc):
    
    def step(self, inputs):
        """Runs a single timestep, updating activity.

        Args:
            inputs (np.ndarray): Input vector of length N.

        Returns:
            np.ndarray: negative Updated activity vector.
        """
        return - super(Leaky_units_inh, self).step(inputs)

class Leaky_onset_units_exc():
    
    def __init__(self, N, W, alpha_uo: float, alpha_ui: float, baseline_uo: float, baseline_ui: float):
        """
        Initialize values for both uo and ui components
        
        Args:
            W_uo - W_ui: weights for intraconnection for each component.
            alpha_uo - alpha_ui (float): Learning rate for activity update for each component.
            baseline_uo - baseline_ui: resting state activity level for each component
        """
        
        self.N = N
        self.W = np.zeros((N, N))
        self.alpha_uo = alpha_uo
        self.alpha_ui = alpha_ui
        self.baseline_uo = np.ones(N) * baseline_uo
        self.baseline_ui = np.ones(N) * baseline_ui
        self.activity_uo = np.array([self.baseline_uo])
        self.activity_ui = np.array([self.baseline_ui])
        self.output = 0.0
        
    def update_weights(self, W: np.array):
        """ 
        Update weight matrix for both uo and ui components
        
        Args:
            - np.array() single value: intraconnection within each component (uo, ui)
        """
        if W.shape == (self.N, self.N):
            self.W = W
        else:
            raise ValueError("Weight matrix shape incorrect. Expected shape ({}, {})".format(self.N, self.N))
        
    def reset_activity(self):
        """Resets neuron activity.

        Args:
            start_activity (np.ndarray): Initial activity vector of length N.

        Raises:
            ValueError: If the start activity vector does not have N elements.
        """
        self.activity_uo *= 0
        self.activitu_ui *= 0
        self.activity_uo += np.array([self.baseline_uo])
        self.activity_ui += np.array([self.baseline_ui])
        self.output *= 0
        
    def step(self, inputs):
        """
        Set inhibitory component ui activity:
            
            - activity_ui will be used as inhibition for input income in uo
        """
        net_input = np.dot(self.W, self.output) + inputs
        self.activity_ui += self.alpha_ui * (net_input + self.baseline_ui - self.activity_ui)
        
        """
        Set  component uo activity:
            
            - activity_uo will be used as output of the onset unit
        """
        self.activity_uo += self.alpha_uo * (np.maximum(0, net_input + self.baseline_uo - self.activity_ui) - self.activity_uo)
        self.output = np.maximum(0, self.activity_uo.copy())
        return self.output.copy()
    
class Leaky_onset_units_inh(Leaky_onset_units_exc):
    
    def step(self, inputs):
        """
        Set inhibitory component ui activity:
            
            - activity_ui will be used as inhibition for input income in uo
            
        Set  component uo activity:
            
            - activity_uo will be used as output of the onset unit
        """
        return - super(Leaky_onset_units_inh, self).step(inputs)
    
    
class Basal_Ganglia_dl:
    
    def __init__(self, N, alpha: float, baseline: float, DLS_GPi_W, STNdl_GPi_W):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.DLS = Leaky_units_inh(N, alpha, baseline)
        self.STNdl = Leaky_units_exc(N, alpha, baseline)
        self.GPi = Leaky_units_inh(N, alpha, baseline = 0.7)
        self.DLS_GPi_W = DLS_GPi_W
        self.STNdl_GPi_W = STNdl_GPi_W
        self.BG_dl_Ws = {
            "DLS_GPi" : np.eye(N).astype(float) * self.DLS_GPi_W,
            "STNdl_GPi" : np.ones((N, N)).astype(float) * self.STNdl_GPi_W
            }

    def reset_activity(self):
        """ 
        Reset activity values for each Layer object through the Layre's function for activity reset
        """
        self.DLS.reset_activity()
        self.STNdl.reset_activity()
        self.GPi.reset_activity()
        
    def step(self, inputs, inp_feedback):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        output_DLS = self.DLS.step((inputs + inp_feedback))
        output_STNdl = self.STNdl.step(inp_feedback)
        
        output_BG_dl = self.GPi.step(np.dot(self.BG_dl_Ws["DLS_GPi"], output_DLS) + np.dot(self.BG_dl_Ws["STNdl_GPi"], output_STNdl))
        
        return output_BG_dl 
    
