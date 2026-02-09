# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:13:28 2026

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from Layer_types import BLA_IC_Layer
from params import Parameters

def plotting(res):
    
    fig, ax = plt.subplots(1, 1)
    
    BLA_IC = np.array(res['BLA_IC_output']) 
    
    ax.plot(BLA_IC[:, 0], label = 'Unit_1')
    ax.plot(BLA_IC[:, 1], label = 'Unit_2')
    ax.plot(BLA_IC[:, 2], label = 'Unit_3')
    ax.plot(BLA_IC[:, 3], label = 'Unit_4')
        
    ax.set_title('BLA_IC simulation')
    ax.legend()
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Activity level')
    ax.set_ylim(0, 1)

    plt.show()    
    
    fig, ax = plt.subplots(1, 1)
    
    BLA_IC = np.array(res['Trace']) 
    
    ax.plot(BLA_IC[:, 0], label = 'Unit_1')
    ax.plot(BLA_IC[:, 1], label = 'Unit_2')
    ax.plot(BLA_IC[:, 2], label = 'Unit_3')
    ax.plot(BLA_IC[:, 3], label = 'Unit_4')
        
    ax.set_title('Trace simulation')
    ax.legend()
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Activity level')
    ax.set_ylim(0, 1)

    plt.show()  
    
def parse_args():
    parser = argparse.ArgumentParser(description="BLA_IC simulation")
    parser.add_argument(
        "-p",
        "--inp",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="Input values (six floats)",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="Number of timesteps",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="plot",
        help="Output mode ('plot', 'save', 'stream')",
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0.0,
        help="Insert BG_dl noise in simulation",
    )
    parser.add_argument(
        '-da',
        '--dopamine',
        type=float,
        default=0.0,
        help='Insert Dopaminergic modulation for BLA_IC learning')
    
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    inp = np.array(args.inp)
    da = args.dopamine
    timesteps = args.timesteps

    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.noise['BLA_IC'] = args.noise

    rng = np.random.RandomState(parameters.seed)
    
    BLA_IC = BLA_IC_Layer(parameters.N["BLA_IC"],
                          parameters.tau["BLA_IC"][0],
                          parameters.tau["BLA_IC"][1],
                          parameters.baseline["BLA_IC"],
                          rng,
                          parameters.noise["BLA_IC"],
                          parameters.BLA_Learn["eta_b"],
                          parameters.BLA_Learn["tau_t"],
                          parameters.BLA_Learn["alpha_t"],
                          parameters.BLA_Learn["theta_DA"],
                          parameters.BLA_Learn["max_W"])
    
    
    W = np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 5.0, 0.0, -10.0, 0.0],
                  [0.0, 0.0, 0.0, 5.0, 0.0, -10.0]])
        
    BLA_IC.reset_activity()
    BLA_IC_output = []
    t_ = []
    _input_ = []
    
    for t in range(timesteps):
        
        BLA_IC.step(np.dot(W, inp))
        BLA_IC.learn(da)
        
        BLA_IC_output.append(BLA_IC.output.copy())
        t_.append(BLA_IC.t.copy())
        _input_.append(inp.copy())

    result = {
              'BLA_IC_output': BLA_IC_output.copy(),
              'Trace': t_.copy(),
              'Inputs_timeline': _input_.copy()
              }
    
    if args.mode == "plot":
        plotting(result)
    elif args.mode == "stream":
        mresults = np.hstack([result[key] for key in result.keys()])
        for row in mresults:
            print(("{:5.3f} " * len(row)).format(*row))
