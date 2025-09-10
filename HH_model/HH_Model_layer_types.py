# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:04:16 2025

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt

class HH_Neural_layer:
    
    def __init__(self, N, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, V_rest):
        
        self.N = N
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.I_Na = 0.0
        self.I_K = 0.0
        self.I_L = 0.0
        self.V_rest = V_rest
        self.V = V_rest
        self.m = self.alpha_m()/(self.alpha_m() + self.beta_m())
        self.h = self.alpha_h()/(self.alpha_h() + self.beta_h())
        self.n = self.alpha_n()/(self.alpha_n() + self.beta_n())
        
    def V_update(self, I, dt):
        dV = (I - self.I_Na - self.I_K - self.I_L) / self.C_m
        self.V += dV * dt
        return self.V
    
    def reset_V(self):
        self.V = self.V_rest.copy()
    
    def I_Na_update(self):
        self.I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        return self.I_Na
    
    def I_K_update(self):
        self.I_K = self.g_K  * self.n**4 * (self.V - self.E_K)
        return self.I_K
    
    def I_L_update(self):
        self.I_L = self.g_L * (self.V - self.E_L)
        return self.I_L
    
    def reset_Na_K_L(self):
        self.I_Na = 0.0
        self.I_K = 0.0
        self.I_L = 0.0
    
    def m_update(self, dt):
        self.m += (self.alpha_m()*(1-self.m) - self.beta_m()*self.m) * dt
        return self.m
        
    def h_update(self, dt):
        self.h += (self.alpha_h()*(1-self.h) - self.beta_h()*self.h) * dt
        return self.h
    
    def n_update(self, dt):
        self.n += (self.alpha_n()*(1-self.n) - self.beta_n()*self.n) * dt
        return self.n
    
    def reset_m_h_n(self):
        self.m *= 0.0
        self.m += self.alpha_m(self.V)/(self.alpha_m(self.V) + self.beta_m(self.V))
        self.h *= 0.0
        self.h += self.alpha_h(self.V)/(self.alpha_h(self.V) + self.beta_h(self.V))
        self.n *= 0.0
        self.n += self.alpha_n(self.V)/(self.alpha_n(self.V) + self.beta_n(self.V))
    
    def alpha_m(self):
        Vhh = self.V - self.V_rest
        return 0.1*(25 - Vhh)/(np.exp((25 - Vhh)/10) - 1)

    def beta_m(self):
        Vhh = self.V - self.V_rest
        return 4.0*np.exp(-Vhh/18)

    def alpha_h(self):
        Vhh = self.V - self.V_rest
        return 0.07*np.exp(-Vhh/20)

    def beta_h(self):
        Vhh = self.V - self.V_rest
        return 1.0/(np.exp((30 - Vhh)/10) + 1)

    def alpha_n(self):
        Vhh = self.V - self.V_rest
        return 0.01*(10 - Vhh)/(np.exp((10 - Vhh)/10) - 1)

    def beta_n(self):
        Vhh = self.V - self.V_rest
        return 0.125*np.exp(-Vhh/80) 