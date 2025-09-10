# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:04:16 2025

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley model parameters
C_m  = 1.0   # membrane capacitance, uF/cm^2
g_Na = 120.0 # maximum sodium conductance, mS/cm^2
g_K  = 36.0  # maximum potassium conductance, mS/cm^2
g_L  = 0.3   # leak conductance, mS/cm^2
E_Na = 50.0  # sodium reversal potential, mV
E_K  = -77.0 # potassium reversal potential, mV
E_L  = -54.387 # leak reversal potential, mV

V_rest = -65.0 # reference resting potential (mV)

# Time parameters
dt = 0.01
T = 50.0
time = np.arange(0, T+dt, dt)

# Functions for gating variables (shifted so V is absolute mV)
def alpha_m(V):
    Vhh = V - V_rest
    return 0.1*(25 - Vhh)/(np.exp((25 - Vhh)/10) - 1)

def beta_m(V):
    Vhh = V - V_rest
    return 4.0*np.exp(-Vhh/18)

def alpha_h(V):
    Vhh = V - V_rest
    return 0.07*np.exp(-Vhh/20)

def beta_h(V):
    Vhh = V - V_rest
    return 1.0/(np.exp((30 - Vhh)/10) + 1)

def alpha_n(V):
    Vhh = V - V_rest
    return 0.01*(10 - Vhh)/(np.exp((10 - Vhh)/10) - 1)

def beta_n(V):
    Vhh = V - V_rest
    return 0.125*np.exp(-Vhh/80)

# Initialize state variables
V = V_rest
m = alpha_m(V)/(alpha_m(V)+beta_m(V))
h = alpha_h(V)/(alpha_h(V)+beta_h(V))
n = alpha_n(V)/(alpha_n(V)+beta_n(V))

I_ext = np.zeros_like(time)
I_ext[1000:2000] = 10  # inject 10 uA/cm^2 between 10 ms and 20 ms

V_trace = []
m_trace, h_trace, n_trace = [], [], []
INa_trace, IK_trace, IL_trace = [], [], []

# Simulation loop
for t, I in zip(time, I_ext):
    # Compute currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K  = g_K  * n**4 * (V - E_K)
    I_L  = g_L * (V - E_L)

    # Update V
    dV = (I - I_Na - I_K - I_L) / C_m
    V += dV * dt

    # Update gating variables
    dm = (alpha_m(V)*(1-m) - beta_m(V)*m) * dt
    dh = (alpha_h(V)*(1-h) - beta_h(V)*h) * dt
    dn = (alpha_n(V)*(1-n) - beta_n(V)*n) * dt
    m += dm
    h += dh
    n += dn

    V_trace.append(V)
    m_trace.append(m); h_trace.append(h); n_trace.append(n)
    INa_trace.append(I_Na); IK_trace.append(I_K); IL_trace.append(I_L)

# Plot
plt.figure(figsize=(10,8))

plt.subplot(4,1,1)
plt.plot(time, V_trace)
plt.title('Membrane potential')
plt.ylabel('V (mV)')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(time, m_trace, label='m')
plt.plot(time, h_trace, label='h')
plt.plot(time, n_trace, label='n')
plt.ylabel('gates')
plt.legend(); plt.grid(True)

plt.subplot(4,1,3)
plt.plot(time, INa_trace, label='I_Na')
plt.plot(time, IK_trace, label='I_K')
plt.plot(time, IL_trace, label='I_L')
plt.ylabel('Currents (outward positive)')
plt.legend(); plt.grid(True)

plt.subplot(4,1,4)
plt.plot(time, I_ext - 80, 'r--', label='I_ext (shifted)')
plt.xlabel('Time (ms)')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
