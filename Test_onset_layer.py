# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:27:42 2026

@author: Nicc
"""
import numpy as np, matplotlib.pyplot as plt
from params import Parameters

class Test_onset:
    
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
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
    def step_1(self, inputs):
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
       
        uo_input = net_input - self.activity_ui
        
        uo_dot = (1 / self.tau_uo) * (uo_input - self.activity_uo)
    
        self.activity_uo += uo_dot
        
        self.output = np.tanh(self.activity_uo)     
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
    def step_2(self, inputs):
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
       
        uo_input = net_input - self.activity_ui
        
        uo_dot = (1 / self.tau_uo) * (uo_input - self.activity_uo)
    
        self.activity_uo += uo_dot
        
        act = np.tanh(self.activity_uo)
    
        self.output = np.maximum(0, act)     
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
    def step_3(self, inputs):
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
       
        uo_input =  np.maximum(0, net_input - self.activity_ui)
        
        uo_dot = (1 / self.tau_uo) * (uo_input - self.activity_uo)
    
        self.activity_uo += uo_dot
        
        act = np.tanh(self.activity_uo)
    
        self.output = np.maximum(0, act)     
        
        if np.any(self.output > 1.0):
            raise ValueError(f"[ERROR] Output exceeded 1.0! Output: {self.output}, Activity: {self.activity}")
            
if __name__ == '__main__':
    
    parameters = Parameters()
    parameters.Matrices_scalars['Sat_BLA_IC'] = 20
    rng = np.random.RandomState(parameters.seed)
    
    onset_layer = Test_onset(
        parameters.N["BLA_IC"],
        parameters.tau["BLA_IC"][0],
        parameters.tau["BLA_IC"][1],
        parameters.baseline["BLA_IC"],
        rng,
        parameters.noise["BLA_IC"]
        )
    
    inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    W = np.array(
        [
            [1.0 * parameters.Matrices_scalars['Mani_BLA_IC'], 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 * parameters.Matrices_scalars['Mani_BLA_IC'], 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 * parameters.Matrices_scalars['Food_BLA_IC'], 0.0, -1.0 * parameters.Matrices_scalars['Sat_BLA_IC'], 0.0],
            [0.0, 0.0, 0.0, 1.0 * parameters.Matrices_scalars['Food_BLA_IC'], 0.0, -1.0 * parameters.Matrices_scalars['Sat_BLA_IC']],
        ]
    )
    
    onset_norm = []
    onset_layer.reset_activity()
    for i in range(500):
        if i < 50:
            inp[0] *= 0
        elif i == 50:
            inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
        if i == 500 * 0.18:
            inp[2] = 1.0
            
        onset_layer.step(np.dot(W, inp))
        onset_norm.append(onset_layer.output.copy())
            
    fig = plt.figure()
    plt.plot(onset_norm, label = ['Unit_1', 'Unit_2', 'Unit_3', 'Unit_4'])
    plt.legend(loc = 'upper right')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('Activity')
    plt.title('Normal')
    plt.show()
        
    onset_nomax = []
    onset_layer.reset_activity()
    for i in range(500):
        if i < 50:
            inp[0] *= 0
        elif i == 50:
            inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
        if i == 500 * 0.18:
            inp[2] = 1.0
            
        onset_layer.step_1(np.dot(W, inp))
        onset_nomax.append(onset_layer.output.copy())
            
    fig = plt.figure()
    plt.plot(onset_nomax, label = ['Unit_1', 'Unit_2', 'Unit_3', 'Unit_4'])
    plt.legend(loc = 'upper right')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('Activity')
    plt.title('No Max')
    plt.show()
    
    onset_maxout = []
    onset_layer.reset_activity()
    for i in range(500):
        if i < 50:
            inp[0] *= 0
        elif i == 50:
            inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
        if i == 500 * 0.18:
            inp[2] = 1.0
            
        onset_layer.step_2(np.dot(W, inp))
        onset_maxout.append(onset_layer.output.copy())
            
    fig = plt.figure()
    plt.plot(onset_maxout, label = ['Unit_1', 'Unit_2', 'Unit_3', 'Unit_4'])
    plt.legend(loc = 'upper right')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('Activity')
    plt.title('Max out')
    plt.show()
    
    onset_maxin = []
    onset_layer.reset_activity()
    for i in range(500):
        if i < 50:
            inp[0] *= 0
        elif i == 50:
            inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
        if i == 500 * 0.18:
            inp[2] = 1.0
            
        onset_layer.step(np.maximum(0, np.dot(W, inp)))
        onset_maxin.append(onset_layer.output.copy())
            
    fig = plt.figure()
    plt.plot(onset_maxin, label = ['Unit_1', 'Unit_2', 'Unit_3', 'Unit_4'])
    plt.legend(loc = 'upper right')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('Activity')
    plt.title('Max in')
    plt.show()
    
    onset_old = []
    onset_layer.reset_activity()
    for i in range(500):
        if i < 50:
            inp[0] *= 0
        elif i == 50:
            inp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
        if i == 500 * 0.18:
            inp[2] = 1.0
            
        onset_layer.step_3(np.dot(W, inp))
        onset_old.append(onset_layer.output.copy())
            
    fig = plt.figure()
    plt.plot(onset_old, label = ['Unit_1', 'Unit_2', 'Unit_3', 'Unit_4'])
    plt.legend(loc = 'upper right')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('Activity')
    plt.title('Old onset')
    plt.show()
        
        
        