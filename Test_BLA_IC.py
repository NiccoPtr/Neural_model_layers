# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:13:28 2026

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from BLA_IC_simulation import BLA_IC_sm
from params import Parameters

def plotting(results, deltas):
    
    # fig, ax = plt.subplots(1, 1)
    
    # BLA_IC = [res['BLA_IC_output'] for res in results]
    # for bla in BLA_IC:
    
    #     bla = np.array(bla)
    #     ax.plot(bla[:, 0], label = 'Unit_1')
    #     ax.plot(bla[:, 1], label = 'Unit_2')
    #     ax.plot(bla[:, 2], label = 'Unit_3')
    #     ax.plot(bla[:, 3], label = 'Unit_4')
        
    # ax.set_title('BLA_IC simulation')
    # ax.legend()
    # ax.set_xlabel('Timestep')
    # ax.set_ylabel('Activity level')
    # ax.set_ylim(0, 1)

    # plt.show()    
    
    # fig, ax = plt.subplots(1, 1)
    
    # trace_BLA_IC = [res['Trace'] for res in results]
    
    # ax.plot(trace_BLA_IC[:, 0], label = 'Unit_1')
    # ax.plot(trace_BLA_IC[:, 1], label = 'Unit_2')
    # ax.plot(trace_BLA_IC[:, 2], label = 'Unit_3')
    # ax.plot(trace_BLA_IC[:, 3], label = 'Unit_4')
        
    # ax.set_title('Trace simulation')
    # ax.legend()
    # ax.set_xlabel('Timestep')
    # ax.set_ylabel('Activity level')
    # ax.set_ylim(0, 1)

    # plt.show()  
    
    fig, ax = plt.subplots(1, 1)
    
    W_max = [res['Max_weight'] for res in results]
    
    # ref_idx = results[0]['Index_max_weight']

    # W_max = [
    #     res['Max_weight']
    #     if np.array_equal(res['Index_max_weight'], ref_idx)
    #     else 0
    #     for res in results
    # ]
    
    ax.plot(deltas, W_max, label = 'Max_weight timeline')
    ax.set_title('Max_weight timeline')
    ax.legend()
    ax.set_xlabel('Deltas')
    ax.set_ylabel('Max_W value')
    ax.set_ylim(0, 0.01)
    ax.set_xlim(deltas[0], deltas[-1])
    
    plt.show()
    
def best_delta(results, deltas):
    
    max_weights = np.array([res['Max_weight'] for res in results])
    idxs = np.where(max_weights == max_weights.max())[0]
    
    return deltas[idxs]
    
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
        default=500,
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
        default=2.0,
        help='Insert Dopaminergic modulation for BLA_IC learning')
    
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    da = args.dopamine
    timesteps = args.timesteps

    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.noise['BLA_IC'] = args.noise

    rng = np.random.RandomState(parameters.seed)
    deltas = np.arange(20, 20 + 60 * 8, 4, dtype=float)
    
    results = []
    
    for i in range(len(deltas)):
        
        delta = deltas[i]
        inp = np.array(args.inp)
        bla = BLA_IC_sm(parameters, rng)
            
        bla.reset_activity()
        BLA_IC_output = []
        t_ = []
        Weight_timeline = []
        _input_ = []
        
        for t in range(timesteps):
            
            if t == delta:
                inp[0] = 1.0
                
            if t == timesteps // 2:
                inp[2] = 1.0
                
            bla.step(inp, da)
            
            BLA_IC_output.append(bla.BLA_IC.output.copy())
            t_.append(bla.BLA_IC.t.copy())
            Weight_timeline.append(bla.BLA_IC.W.copy())
            _input_.append(inp.copy())
            W_max = bla.BLA_IC.W.max().copy()
            idx_W_max = np.array(np.where(bla.BLA_IC.W == W_max)).ravel()

    
        result = {
                  'BLA_IC_output': BLA_IC_output.copy(),
                  'Trace': t_.copy(),
                  'Weight_timeline': Weight_timeline,
                  'Max_weight': W_max,
                  'Index_max_weight': idx_W_max,
                  'Inputs_timeline': _input_.copy(),
                  'Delta': delta
                  }
        
        results.append(result)
        
    best_delta = best_delta(results, deltas)
    
    if args.mode == "plot":
        plotting(results, deltas)
    elif args.mode == "stream":
        mresults = np.hstack([result[key] for key in result.keys()])
        for row in mresults:
            print(("{:5.3f} " * len(row)).format(*row))
            
   
        
