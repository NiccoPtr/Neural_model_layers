# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:04:16 2025

@author: Utente
"""
from HH_Model_layer_types import HH_Neural_layer as HH
import numpy as np
import matplotlib.pyplot as plt

Neur_layer = HH(N = 0.0, 
                C_m  = 1.0,
                g_Na = 120.0,
                g_K  = 36.0,
                g_L  = 0.3,
                E_Na = 50.0,
                E_K  = -77.0,
                E_L  = -54.387,
                V_rest = -65.0)

dt = 0.01
T = 50.0
time = np.arange(0, T+dt, dt)
I_ext = np.zeros_like(time)
I_ext[1000:2000] = 10

V_trace = []
m_trace, h_trace, n_trace = [], [], []
INa_trace, IK_trace, IL_trace = [], [], []

for t, I in zip(time, I_ext):
    # Compute currents
    Neur_layer.I_Na_update()
    Neur_layer.I_K_update()
    Neur_layer.I_L_update()

    # Update V
    Neur_layer.V_update(I, dt)

    # Update gating variables
    Neur_layer.m_update(dt)
    Neur_layer.h_update(dt)
    Neur_layer.n_update(dt)

    V_trace.append(Neur_layer.V)
    m_trace.append(Neur_layer.m); h_trace.append(Neur_layer.h); n_trace.append(Neur_layer.n)
    INa_trace.append(Neur_layer.I_Na); IK_trace.append(Neur_layer.I_K); IL_trace.append(Neur_layer.I_L)
    
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