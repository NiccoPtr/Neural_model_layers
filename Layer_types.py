# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:09:24 2025

@author: Utente
"""

import numpy as np

class Leaky_units_exc:

    def __init__(self, N: int, tau: float, baseline: float, rng, noise: float, threshold: float, lesion=False):
        """Initializes the neural unit.

        Args:
            N (int): Number of neurons in the model (must be 2).
            tau (float): Learning rate for activity update.
            threshold (float): Threshold for action selection.

        """
        self.N = N
        self.W = np.zeros((N, N))
        self.tau = tau
        self.rng = rng
        self.noise = noise
        self.baseline = np.ones(N) * baseline
        self.threshold = threshold
        self.activity = self.baseline.copy()
        self.output = np.zeros(N)
        self.lesion = lesion

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
        self.activity += self.baseline.copy()
        self.output *= 0
        
    def step(self, inputs):
        """Runs a single timestep, updating activity.

        Args:
            inputs (np.ndarray): Input vector of length N.

        Returns:
            np.ndarray: Updated activity vector.
        """
        
        net_input = np.dot(self.W, self.output) + (inputs + (self.rng.randn(self.N) * self.noise)) + self.baseline
        self.activity += (1/self.tau) * (net_input - self.activity)
        
        self.output = np.maximum(0, np.tanh(self.activity.copy()))
        
        if self.lesion:
            self.activity *= 0.0
            self.output *= 0.0
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")

        # print(f"  [DEBUG] net_input: {net_input}")
        # print(f"  [DEBUG] updated activity: {self.activity}")
        # print(f"  [DEBUG] output: {self.output}")
        
    
class Leaky_units_inh(Leaky_units_exc):
    
    def step(self, inputs):
        """Runs a single timestep, updating activity.

        Args:
            inputs (np.ndarray): Input vector of length N.

        Returns:
            np.ndarray: negative Updated activity vector.
        """
        super(Leaky_units_inh, self).step(inputs)
        
        self.output = -self.output


class Leaky_onset_units_exc:
    
    def __init__(self, N, tau_uo: float, tau_ui: float, baseline: float, rng, noise: float, lesion=False):
        """
        Initialize values for both uo and ui components
        
        Args:
            W_uo - W_ui: weights for intraconnection for each component.
            tau_uo - tau_ui (float): Learning rate for activity update for each component.
            baseline_uo - baseline_ui: resting state activity level for each component
        """
        
        self.N = N
        self.W = np.zeros((N, N))
        self.tau_uo = tau_uo
        self.tau_ui = tau_ui
        self.rng = rng
        self.noise = noise
        self.baseline = np.ones(N) * baseline
        self.activity_uo = self.baseline.copy()
        self.activity_ui = self.baseline.copy()
        self.output = np.zeros(N)
        self.lesion = lesion
        
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
        self.activity_ui *= 0
        self.activity_uo += self.baseline.copy()
        self.activity_ui += self.baseline.copy()
        self.output *= 0
        
    def step(self, inputs):
        """
        Set inhibitory component ui activity:
            
            - activity_ui will be used as inhibition for input income in uo
        """
        net_input = np.dot(self.W, self.output) + (inputs + (self.rng.randn() * self.noise)) + self.baseline
        
        ui_dot = (1 / self.tau_ui) * (net_input - self.activity_ui)
    
        self.activity_ui += ui_dot
        
        """
        Set  component uo activity:
            
            - activity_uo will be used as output of the onset unit
        """
       
        uo_input = net_input - np.maximum(0, self.activity_ui)
        
        uo_dot = (1 / self.tau_uo) * (uo_input - self.activity_uo)
    
        self.activity_uo += uo_dot
        
        act = np.tanh(self.activity_uo)
    
        self.output = np.maximum(0, act)   
        
        if self.lesion:
            self.activity_ui *= 0.0
            self.activity_uo *= 0.0
            self.output *= 0.0
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity_uo}")

    
class Leaky_onset_units_inh(Leaky_onset_units_exc):
    
    def step(self, inputs):
        """
        Set inhibitory component ui activity:
            
            - activity_ui will be used as inhibition for input income in uo
            
        Set  component uo activity:
            
            - activity_uo will be used as output of the onset unit
        """
        super(Leaky_onset_units_inh, self).step(inputs)
        
        self.output = - self.output
    

class BG_Layer:
    
    def __init__(self, N, tau: float, baseline_Str: float, baseline_STN: float, baseline_GPi_SNpr: float, Str_GPi_SNpr_W, STN_GPi_SNpr_W, rng, noise: float, threshold: float, lesion=False):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.Str = Leaky_units_inh(N, tau, baseline_Str, rng, noise, threshold, lesion)
        self.STN = Leaky_units_exc(N, tau, baseline_STN, rng, noise, threshold)
        self.GPi_SNpr = Leaky_units_inh(N, tau, baseline_GPi_SNpr, rng, noise, threshold)
        self.output_BG = np.zeros(N)
        self.output_Str_pre = np.zeros(N)
        self.output_STN_pre = np.zeros(N)
        self.Str_GPi_SNpr_W = Str_GPi_SNpr_W
        self.STN_GPi_SNpr_W = STN_GPi_SNpr_W
        self.BG_Ws = {
            "Str_GPi_SNpr" : np.eye(N).astype(float) * self.Str_GPi_SNpr_W,
            "STN_GPi_SNpr" : np.ones((N, N)).astype(float) * self.STN_GPi_SNpr_W
            }
        
    def reset_activity(self):
        """ 
        Reset activity values for each Layer object through the Layre's function for activity reset
        """
        self.Str.reset_activity()
        self.STN.reset_activity()
        self.GPi_SNpr.reset_activity()
        self.output_BG *= 0
        self.output_Str_pre *= 0
        self.output_STN_pre *= 0
        
        
    def step(self, inputs, inp_feedback_NAc, inp_feedback_STNv):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        self.Str.step(inputs + inp_feedback_NAc)
        self.STN.step(inp_feedback_STNv)
        
        output_Str = self.Str.output.copy()
        output_STN = self.STN.output.copy()
        
        # print(f"[BGDL] DLS output: {output_DLS}")
        # print(f"[BGDL] STN output: {output_STNdl}")
        
        self.GPi_SNpr.step(np.dot(self.BG_Ws["Str_GPi_SNpr"], self.output_Str_pre) + np.dot(self.BG_Ws["STN_GPi_SNpr"], self.output_STN_pre))
        self.output_BG = self.GPi_SNpr.output.copy()
        
        if np.any(self.output_BG > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
        self.output_Str_pre = output_Str.copy()
        self.output_STN_pre = output_STN.copy()


class BLA_IC_Layer(Leaky_onset_units_exc):
    
    def __init__(self, N, tau_uo, tau_ui, baseline, rng, noise, eta_b, tau_t, alpha_t, theta_da, max_W, lesion=False):
        
        super(BLA_IC_Layer, self).__init__(N, tau_uo, tau_ui, baseline, rng, noise)
        
        self.t = np.zeros(N)
        self.t_dot = np.zeros(N)
        self.tau_t = tau_t
        self.alpha_t = alpha_t
        self.eta_b = eta_b
        self.max_W = max_W
        self.theta_da = theta_da
        self.lesion = lesion

    def learn(self, da):

        self.t_dot = (1 / self.tau_t) * (-self.t + self.alpha_t * self.output)
    
        # self.t_dot = np.clip(self.t_dot, -1e6, 1e6)
    
        self.t += self.t_dot
        # self.t = np.clip(self.t, -1e12, 1e12)
    
        pos = np.maximum(0, self.t_dot)
        neg = np.maximum(0, -self.t_dot)
    
        # pos = np.nan_to_num(pos)
        # neg = np.nan_to_num(neg)
    
        da_term = np.maximum(0, da - self.theta_da)
        # da_term = np.nan_to_num(da_term)
    
        delta_W = (self.eta_b *
                   da_term *
                   np.outer(pos, neg) *
                   (self.max_W - self.W))
    
        # delta_W = np.nan_to_num(delta_W)
    
        self.W += delta_W

    
class SNpc_Layer:
    
    def __init__(self, N: int, tau_i: float, tau_o: float, baseline_i: float, baseline_o: float, SNpci_1_SNpco_1_W, SNpci_2_SNpco_2_W, rng, noise: float, threshold: float):
        
        self.SNpci_1 = Leaky_units_inh(N, tau_i, baseline_i, rng, noise, threshold)
        self.SNpci_2 = Leaky_units_inh(N, tau_i, baseline_i, rng, noise, threshold)
        self.SNpco_1 = Leaky_units_exc(N, tau_o, baseline_o, rng, noise, threshold)
        self.SNpco_2 = Leaky_units_exc(N, tau_o, baseline_o, rng, noise, threshold)
        self.output_1 = np.zeros(N)
        self.output_2 = np.zeros(N)
        self.output_SNpci_1_pre = np.zeros(N)
        self.output_SNpci_2_pre = np.zeros(N)
        
        self.SNpci_1_SNpco_1_W = SNpci_1_SNpco_1_W
        self.SNpci_2_SNpco_2_W = SNpci_2_SNpco_2_W
        self.SNpc_Ws = {
            "SNpci_1_SNpco_1_W" : np.ones(N).astype(float) * self.SNpci_1_SNpco_1_W,
            "SNpci_2_SNpco_2_W" : np.ones(N).astype(float) * self.SNpci_2_SNpco_2_W
            }
        
    def reset_activity(self):
        
        self.SNpci_1.reset_activity()
        self.SNpci_2.reset_activity()
        self.SNpco_1.reset_activity()
        self.SNpco_2.reset_activity()
        self.output_1 *= 0
        self.output_2 *= 0
        self.output_SNpci_1_pre *= 0
        self.output_SNpci_2_pre *= 0
        
    def step(self, inp_NAc, inp_DMS, inp_PPN):
        
        self.SNpci_1.step(inp_NAc)
        output_i_1 = self.SNpci_1.output.copy()
        self.SNpco_1.step(np.dot(self.SNpc_Ws['SNpci_1_SNpco_1_W'], self.output_SNpci_1_pre) + inp_PPN)
        self.output_1 = self.SNpco_1.output.copy()
        
        self.SNpci_2.step(inp_DMS)
        output_i_2 = self.SNpci_2.output.copy()
        self.SNpco_2.step(np.dot(self.SNpc_Ws['SNpci_2_SNpco_2_W'], self.output_SNpci_2_pre) + inp_PPN)
        self.output_2 = self.SNpco_2.output.copy()
        
        if np.any(self.SNpco_1.output.copy() > 1.0) or np.any(self.SNpco_2.output.copy() > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")   
            
        self.output_SNpci_1_pre = output_i_1.copy()
        self.output_SNpci_2_pre = output_i_2.copy()

class BG_v2:

    def __init__(self,
                  N,
                    tau: float,
                      baseline_Str1: float,
                       baseline_Str2: float,
                        baseline_STN: float,
                          baseline_GPi_SNpr: float,
                            baseline_GPe: float,
                              Str1_GPi_SNpr_W,
                                Str2_GPe_W,
                                  STN_GPi_SNpr_W,
                                    STN_GPe_W,
                                      GPe_STN_W,
                                        GPe_GPi_SNpr_W,
                                          rng,
                                            noise: float,
                                              threshold: float):
    
        self.Str1 = Leaky_units_inh(N, tau, baseline_Str1, rng, noise, threshold)
        self.Str2 = Leaky_units_inh(N, tau, baseline_Str2, rng, noise, threshold)
        self.STN = Leaky_units_exc(N, tau, baseline_STN, rng, noise, threshold)
        self.GPi_SNpr = Leaky_units_inh(N, tau, baseline_GPi_SNpr, rng, noise, threshold)
        self.GPe = Leaky_units_inh(N, tau, baseline_GPe, rng, noise, threshold)

        self.output_BG = np.zeros(N)
        self.output_GPe_pre = np.zeros(N)
        self.output_Str1_pre = np.zeros(N)
        self.output_Str2_pre = np.zeros(N)
        self.output_STN_pre = np.zeros(N)
        self.Str1_GPi_SNpr_W = Str1_GPi_SNpr_W
        self.Str2_GPe_W = Str2_GPe_W
        self.STN_GPi_SNpr_W = STN_GPi_SNpr_W
        self.STN_GPe_W = STN_GPe_W
        self.GPe_STN_W = GPe_STN_W
        self.GPe_GPi_SNpr_W = GPe_GPi_SNpr_W
        self.BG_Ws = {
            "Str1_GPi_SNpr" : np.eye(N).astype(float) * self.Str1_GPi_SNpr_W,
            "Str2_GPe" : np.eye(N).astype(float) * self.Str2_GPe_W,
            "STN_GPi_SNpr" : np.ones((N, N)).astype(float) * self.STN_GPi_SNpr_W,
            "STN_GPe" : np.ones((N, N)).astype(float) * self.STN_GPe_W,
            "GPe_STN" : np.eye(N).astype(float) * self.GPe_STN_W,
            "GPe_GPi_SNpr" : np.eye(N).astype(float) * self.GPe_GPi_SNpr_W
            }

    def reset_activity(self):

        self.output_BG *= 0.0
        self.output_GPe_pre *= 0.0
        self.output_Str1_pre *= 0.0
        self.output_Str2_pre *= 0.0
        self.output_STN_pre *= 0.0

        self.Str1.reset_activity()
        self.Str2.reset_activity()
        self.STN.reset_activity()
        self.GPi_SNpr.reset_activity()
        self.GPe.reset_activity()

    def step(self, inp_Str1, inp_Str2, inp_cortex_Str1, inp_cortex_Str2, inp_cortex_STN):

        self.Str1.step(inp_Str1 + inp_cortex_Str1)
        self.Str2.step(inp_Str2 + inp_cortex_Str2)
        self.STN.step(inp_cortex_STN
                        + np.dot(self.BG_Ws['GPe_STN'], self.output_GPe_pre)
                        )
        self.GPe.step(np.dot(self.BG_Ws['Str2_GPe'], self.output_Str2_pre)
                       + np.dot(self.BG_Ws['STN_GPe'], self.output_STN_pre)
                       )
        self.GPi_SNpr.step(np.dot(self.BG_Ws['Str1_GPi_SNpr'], self.output_Str1_pre)
                       + np.dot(self.BG_Ws['STN_GPi_SNpr'], self.output_STN_pre)
                       + np.dot(self.BG_Ws['GPe_GPi_SNpr'], self.output_GPe_pre)
                       )

        self.output_BG = self.GPi_SNpr.output.copy()
        self.output_GPe_pre = self.GPe.output.copy()
        self.output_STN_pre = self.STN.output.copy()
        self.output_Str1_pre = self.Str1.output.copy()
        self.output_Str2_pre = self.Str2.output.copy()
        