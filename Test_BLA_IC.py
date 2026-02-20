# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:13:28 2026

@author: Nicc
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os

from BLA_IC_simulation import BLA_IC_sm
from params import Parameters

plt.ion()

def plotting(result):
    
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    
    BLA_IC = np.array(result['BLA_IC_output'])
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
    
    t = np.array(result['Trace'])
    ax.plot(t[:, 0], label = 'Unit_1')
    ax.plot(t[:, 1], label = 'Unit_2')
    ax.plot(t[:, 2], label = 'Unit_3')
    ax.plot(t[:, 3], label = 'Unit_4')
        
    ax.set_title('Trace simulation')
    ax.legend()
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Activity level')

    plt.show()  
    
    fig, ax = plt.subplots(1, 1)
    
    W_timeline = np.array(result['Weight_timeline'])
    im = ax.imshow(
        W_timeline.reshape(-1, 4 * 4).T, interpolation="none", aspect="auto", vmin=0, vmax=0.02
    )
    ax.set_title("Weights learning")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Connections")
    ax.set_yticks(np.arange(16), [f"w{j}{i}" for j in range(4) for i in range(4)])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

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
        "-mani",
        "--manipulanda",
        type=float,
        default=0,
        help="Insert 0 for manipulanda lever, 1 for manipulanda chain",
    )
    parser.add_argument(
        "-f",
        "--food",
        type=float,
        default=2,
        help="Insert 2 for food_1, 3 for food_2",
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
        default="stream",
        help="Output mode ('plot', 'save','short_save', 'stream')",
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
        help='Insert Dopaminergic modulation for BLA_IC learning'
        )
    parser.add_argument(
        '-de',
        '--delta',
        type=float,
        default=160.0,
        help='Insert delta for manipulanda input onset'
        )
    
    return parser.parse_args()    

if __name__ == "__main__":
    
    args = parse_args()
    da = args.dopamine
    timesteps = args.timesteps
    mani = args.manipulanda
    food = args.food
    delta = args.delta
    inp = np.array(args.inp)
    
    parameters = Parameters()
    parameters.load("prm_file.json" ,mode = "json")
    parameters.noise['BLA_IC'] = args.noise
    rng = np.random.RandomState(parameters.seed)
    
    bla = BLA_IC_sm(parameters, rng)   
    bla.reset_activity()

    BLA_IC_output = []
    t_ = []
    Weight_timeline = []
    _input_ = []
        
    for t in range(timesteps):
        
        if t == delta:
            inp[mani] = 1.0
            
        if t == timesteps // 2:
            inp[food] = 1.0
            
        bla.step(inp, da)
    
        BLA_IC_output.append(bla.BLA_IC.output.copy())
        t_.append(bla.BLA_IC.t.copy())
        Weight_timeline.append(bla.BLA_IC.W.copy().flatten())
        _input_.append(inp.copy())
    
    result = {
              'Delta': np.ones(timesteps) * delta,
              'Inputs_timeline': _input_.copy(),
              'BLA_IC_output': BLA_IC_output.copy(),
              'Trace': t_.copy(),
              'Weight_timeline': Weight_timeline
              }
        
    if args.mode == "plot":
        plotting(result)
        input("Press Enter to exit")
        
    elif args.mode == "stream":
        inp_end = inp.copy()
        W_end = bla.BLA_IC.W.copy().flatten()
        mresults = np.hstack((inp_end, W_end, delta))
        print(("{:10.5f} " * len(mresults)).format(*mresults))
        
    elif args.mode == 'save':
        input_cols = [f"Input_{i}" 
                      for i in range(len(inp.copy()))]
        ouput_cols = [f'Output_Unit_{i}' 
                      for i in range(bla.BLA_IC.N)]
        trace_cols = [f'Trace_Unit_{i}' 
                      for i in range(bla.BLA_IC.N)]
        W_cols = [f'Weight_{x}_{y}' 
                  for x in range(bla.BLA_IC.W.shape[0])
                  for y in range(bla.BLA_IC.W.shape[1])]
        delta_col = ['Delta']
        cols = delta_col + input_cols + ouput_cols + trace_cols + W_cols
        
        values = [np.asanyarray(result[k]).reshape(timesteps, -1)
                 for k in result.keys()]
        values_conc = np.concatenate(values, axis=1)
        df = pd.DataFrame(values_conc, columns=cols)
        
        csv_path = "BLA_IC_Testing.csv"
        
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    elif args.mode == 'short_save':
        fin_inp = inp.copy()
        fin_W = bla.BLA_IC.W.copy().flatten()
            
        input_cols = [f"Input_{i}" for i in range(len(fin_inp))]
        W_cols = [f'Weight_{x}_{y}' 
                  for x in range(bla.BLA_IC.W.shape[0])
                  for y in range(bla.BLA_IC.W.shape[1])]
        delta_col = [str('Delta')]
        values = np.concatenate([fin_inp, fin_W, [delta]])
        columns = input_cols + W_cols + delta_col
        
        df = pd.DataFrame([values], columns=columns)
        
        csv_path = "BLA_IC_short_test.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)