# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:09:24 2025

@author: Utente
"""

import numpy as np, matplotlib.pyplot as plt

class Layer_excitatory:

    def __init__(self, N, alpha: float, threshold: float, baseline: float):
        """Initializes the neural model.

        Args:
            N (int): Number of neurons in the model (must be 2).
            alpha (float): Learning rate for activity update.
            threshold (float): Threshold for action selection.

        """
        self.N = N
        self.W = np.zeros((N, N))
        self.alpha = alpha
        self.threshold = threshold
        self.baseline = baseline

    def update_weights(self, W: np.ndarray):
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
        self.activity = np.full(self.N, self.baseline)

    def step(self, inputs):
        """Runs a single timestep, updating activity.

        Args:
            inputs (np.ndarray): Input vector of length N.

        Returns:
            np.ndarray: Updated activity vector.

        Raises:
            ValueError: If weights are not set or input vector length is not N.
        """
        net_input = np.dot(self.W, self.activity) + inputs
        updated_activity = np.tanh(net_input)
        self.activity += self.alpha * (updated_activity + self.baseline - self.activity)
        self.output = np.maximum(0, self.activity)
        return self.output

    
class Layer_inhibit(Layer_excitatory):
    
    def step(self, inputs):
        self.activity = super(Layer_inhibit, self).step(inputs)
        self.output = np.maximum(0, np.tanh(self.activity.copy()))
        return - self.output
    
class Basal_Ganglia:
    
    def __init__(self, N, alpha: float, threshold: float, baseline: float, DLS_GPi_W, STNdl_GPi_W):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.DLS = Layer_inhibit(N, alpha, threshold, baseline)
        self.STNdl = Layer_excitatory(N, alpha, threshold, baseline)
        self.GPi = Layer_excitatory(N, alpha, threshold, baseline = 0.8)
        self.DLS_GPi_W = DLS_GPi_W
        self.STNdl_GPi_W = STNdl_GPi_W
        self.matrices = {
            "DLS_GPi" : np.eye(N).astype(float) * self.DLS_GPi_W,
            "STNdl_GPi" : np.ones((N, N)).astype(float) * self.STNdl_GPi_W
            }

    def reset_act(self):
        """ 
        Reset activity values for each Layer object through the Layre's function for activity reset
        """
        self.DLS.reset_activity()
        self.STNdl.reset_activity()
        self.GPi.reset_activity()
        
    def step(self, inputs):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        output_DLS = self.DLS.step(inputs)
        output_STNdl = self.STNdl.step(inputs)
        
        output_GPi = self.GPi.step(np.dot(self.matrices["DLS_GPi"], output_DLS) + np.dot(self.matrices["STNdl_GPi"], output_STNdl))
        
        return output_GPi
    
