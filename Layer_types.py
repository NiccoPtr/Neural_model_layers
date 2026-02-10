# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:09:24 2025

@author: Utente
"""

import numpy as np

class Leaky_units_exc:

    def __init__(self, N: int, tau: float, baseline: float, rng, noise: float, threshold: float):
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
    
    def __init__(self, N, tau_uo: float, tau_ui: float, baseline: float, rng, noise: float):
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
        # net_input = np.clip(net_input, -1e6, 1e6)
        # net_input = np.nan_to_num(net_input)
        
        ui_dot = (1 / self.tau_ui) * (net_input - self.activity_ui)
        # ui_dot = np.nan_to_num(ui_dot)
        # ui_dot = np.clip(ui_dot, -1e6, 1e6)
    
        self.activity_ui += ui_dot
        # self.activity_ui = np.clip(self.activity_ui, -1e6, 1e6)
        # self.activity_ui = np.nan_to_num(self.activity_ui)
        
        """
        Set  component uo activity:
            
            - activity_uo will be used as output of the onset unit
        """
        uo_input = np.maximum(0, net_input - self.activity_ui)
        # uo_input = np.nan_to_num(uo_input)
    
        uo_dot = (1 / self.tau_uo) * (uo_input - self.activity_uo)
        # uo_dot = np.nan_to_num(uo_dot)
        # uo_dot = np.clip(uo_dot, -1e6, 1e6)
    
        self.activity_uo += uo_dot
        # self.activity_uo = np.clip(self.activity_uo, -1e6, 1e6)
        # self.activity_uo = np.nan_to_num(self.activity_uo)
        
        act = np.tanh(self.activity_uo)      
        # act = np.nan_to_num(act)
    
        self.output = np.maximum(0, act)     
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")

    
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
    

class BG_v_Layer:
    
    def __init__(self, N, tau: float, baseline_NAc: float, baseline_STNv: float, baseline_SNpr: float, NAc_SNpr_W, STNv_SNpr_W, rng, noise: float, threshold: float):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.NAc = Leaky_units_inh(N, tau, baseline_NAc, rng, noise, threshold)
        self.STNv = Leaky_units_exc(N, tau, baseline_STNv, rng, noise, threshold)
        self.SNpr = Leaky_units_inh(N, tau, baseline_SNpr, rng, noise, threshold)
        self.output_BG_v = np.zeros(N)
        self.output_NAc_pre = np.zeros(N)
        self.output_STNv_pre = np.zeros(N)
        self.NAc_SNpr_W = NAc_SNpr_W
        self.STNv_SNpr_W = STNv_SNpr_W
        self.BG_v_Ws = {
            "NAc_SNpr" : np.eye(N).astype(float) * self.NAc_SNpr_W,
            "STNv_SNpr" : np.ones((N, N)).astype(float) * self.STNv_SNpr_W
            }
        

    def reset_activity(self):
        """ 
        Reset activity values for each Layer object through the Layre's function for activity reset
        """
        self.NAc.reset_activity()
        self.STNv.reset_activity()
        self.SNpr.reset_activity()
        
    def step(self, inputs, inp_feedback_NAc, inp_feedback_STNv):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        self.NAc.step(inputs + inp_feedback_NAc)
        self.STNv.step(inp_feedback_STNv)
        
        output_NAc = self.NAc.output.copy()
        output_STNv = self.STNv.output.copy()
        
        # print(f"[BGDL] DLS output: {output_DLS}")
        # print(f"[BGDL] STN output: {output_STNdl}")
        
        self.SNpr.step(np.dot(self.BG_v_Ws["NAc_SNpr"], self.output_NAc_pre) + np.dot(self.BG_v_Ws["STNv_SNpr"], self.output_STNv_pre))
        self.output_BG_v = self.SNpr.output.copy()
        
        if np.any(self.output_BG_v > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
        self.output_NAc_pre = output_NAc.copy()
        self.output_STNv_pre = output_STNv.copy()
         

class BG_dm_Layer:
    
    def __init__(self, N, tau: float, baseline_DMS: float, baseline_STNdm: float, baseline_GPi_SNpr: float, DMS_GPiSNpr_W, STNdm_GPiSNpr_W, rng, noise: float, threshold: float):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.DMS = Leaky_units_inh(N, tau, baseline_DMS, rng, noise, threshold)
        self.STNdm = Leaky_units_exc(N, tau, baseline_STNdm, rng, noise, threshold)
        self.GPi_SNpr = Leaky_units_inh(N, tau, baseline_GPi_SNpr, rng, noise, threshold)
        self.output_BG_dm = np.zeros(N)
        self.output_DMS_pre = np.zeros(N)
        self.output_STNdm_pre = np.zeros(N)
        self.DMS_GPiSNpr_W = DMS_GPiSNpr_W
        self.STNdm_GPiSNpr_W = STNdm_GPiSNpr_W
        self.BG_dm_Ws = {
            "DMS_GPiSNpr" : np.eye(N).astype(float) * self.DMS_GPiSNpr_W,
            "STNdm_GPiSNpr" : np.ones((N, N)).astype(float) * self.STNdm_GPiSNpr_W
            }
        

    def reset_activity(self):
        """ 
        Reset activity values for each Layer object through the Layre's function for activity reset
        """
        self.DMS.reset_activity()
        self.STNdm.reset_activity()
        self.GPi_SNpr.reset_activity()
        
    def step(self, inputs, inp_feedback_DMS, inp_feedback_STNdm):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        self.DMS.step(inputs + inp_feedback_DMS)
        self.STNdm.step(inp_feedback_STNdm)
        
        output_DMS = self.DMS.output.copy()
        output_STNdm = self.STNdm.output.copy()
        
        # print(f"[BGDL] DLS output: {output_DLS}")
        # print(f"[BGDL] STN output: {output_STNdl}")
        
        self.GPi_SNpr.step(np.dot(self.BG_dm_Ws["DMS_GPiSNpr"], self.output_DMS_pre) + np.dot(self.BG_dm_Ws["STNdm_GPiSNpr"], self.output_STNdm_pre))
        self.output_BG_dm = self.GPi_SNpr.output.copy()
        
        if np.any(self.output_BG_dm > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
        self.output_DMS_pre = output_DMS.copy()
        self.output_STNdm_pre = output_STNdm.copy()
        
    
class BG_dl_Layer:
    
    def __init__(self, N, tau: float, baseline_DLS: float, baseline_STNdl: float, baseline_GPi: float, DLS_GPi_W, STNdl_GPi_W, rng, noise: float, threshold: float):
        """
        Initialize different layers of neurons of size "N"
           
        Create the different connection matrices for each layer's comunication:
            - required in this case 2 matrices for comunication between 2 neuron layers with the GPi
            - GPi will then take in as inputs this 2 outputs from the 2 neuron layers and return an outcome
        
        Intialize a Baseline value for the each layer:
            - it is a np.array() like vector which keeps the activity state at a certain level even at rest
            - the baseline should be 0.0 for each layer, except for GPi layer
        """
        self.DLS = Leaky_units_inh(N, tau, baseline_DLS, rng, noise, threshold)
        self.STNdl = Leaky_units_exc(N, tau, baseline_STNdl, rng, noise, threshold)
        self.GPi = Leaky_units_inh(N, tau, baseline_GPi, rng, noise, threshold)
        self.output_BG_dl = np.zeros(N)
        self.output_DLS_pre = np.zeros(N)
        self.output_STNdl_pre = np.zeros(N)
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
        
    def step(self, inputs, inp_feedback_DLS, inp_feedback_STNdl):
        """
        Return the output of the 2 layers below GPi layer using the input argument as Input
        
        Use the 2 outputs as input values for GPi activity update and return an outcome
        
        Each layer recalls the origin step function from its origin Class to compute the activity update:
            - modulate the layers's outcomes toward the GPi by using the matrices you initialized within the __init__ function 
            - this is simply duable through multiplication (input * matrix); a Matrix of 1s will let pass the all activity as an input, meanwhile a 0s Matrix will stop the input bringing it down to 0
        """
        self.DLS.step(inputs + inp_feedback_DLS)
        self.STNdl.step(inp_feedback_STNdl)
        
        output_DLS = self.DLS.output.copy()
        output_STNdl = self.STNdl.output.copy()
        
        # print(f"[BGDL] DLS output: {output_DLS}")
        # print(f"[BGDL] STN output: {output_STNdl}")
        
        self.GPi.step(np.dot(self.BG_dl_Ws["DLS_GPi"], self.output_DLS_pre) + np.dot(self.BG_dl_Ws["STNdl_GPi"], self.output_STNdl_pre))
        self.output_BG_dl = self.GPi.output.copy()
        
        if np.any(self.output_BG_dl > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
        self.output_DLS_pre = output_DLS.copy()
        self.output_STNdl_pre = output_STNdl.copy()
        

class BLA_IC_Layer(Leaky_onset_units_exc):
    
    def __init__(self, N, tau_uo, tau_ui, baseline, rng, noise, eta_b, tau_t, alpha_t, theta_da, max_W):
        
        super(BLA_IC_Layer, self).__init__(N, tau_uo, tau_ui, baseline, rng, noise)
        
        self.t = np.zeros(N)
        self.t_dot = np.zeros(N)
        self.tau_t = tau_t
        self.alpha_t = alpha_t
        self.eta_b = eta_b
        self.max_W = max_W
        self.theta_da = theta_da

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
    
    def __init__(self, N: int, tau: float, baseline: float, SNpci_1_SNpco_1_W, SNpci_2_SNpco_2_W, rng, noise: float, threshold: float):
        
        self.SNpci_1 = Leaky_units_inh(N, tau, baseline, rng, noise, threshold)
        self.SNpci_2 = Leaky_units_inh(N, tau, baseline, rng, noise, threshold)
        self.SNpco_1 = Leaky_units_exc(N, tau, baseline, rng, noise, threshold)
        self.SNpco_2 = Leaky_units_exc(N, tau, baseline, rng, noise, threshold)
        self.output_1 = np.zeros(N)
        self.output_2 = np.zeros(N)
        self.output_SNpci_1_pre = np.zeros(N)
        self.output_SNpci_2_pre = np.zeros(N)
        
        self.SNpci_1_SNpco_1_W = SNpci_1_SNpco_1_W
        self.SNpci_2_SNpco_2_W = SNpci_2_SNpco_2_W
        self.SNpc_Ws = {
            "SNpci_1_SNpco_1_W" : np.eye(N).astype(float) * self.SNpci_1_SNpco_1_W,
            "SNpci_2_SNpco_2_W" : np.eye(N).astype(float) * self.SNpci_2_SNpco_2_W
            }
        
    def reset_activity(self):
        
        self.SNpci_1.reset_activity()
        self.SNpci_2.reset_activity()
        self.SNpco_1.reset_activity()
        self.SNpco_2.reset_activity()
        
    def step(self, inp_NAc, inp_DMS, inp_PPN):
        
        self.SNpci_1.step(inp_NAc)
        output_i_1 = self.SNpci_1.output.copy()
        self.SNpco_1.step(self.output_SNpci_1_pre + inp_PPN)
        self.output_1 = self.SNpco_1.output.copy()
        
        self.SNpci_2.step(inp_DMS)
        output_i_2 = self.SNpci_2.output.copy()
        self.SNpco_2.step(self.output_SNpci_2_pre + inp_PPN)
        self.output_2 = self.SNpco_2.output.copy()
        
        if np.any(self.SNpco_1.output.copy() > 1.0) or np.any(self.SNpco_2.output.copy() > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")   
            
        self.output_SNpci_1_pre = output_i_1.copy()
        self.output_SNpci_2_pre = output_i_2.copy()
            