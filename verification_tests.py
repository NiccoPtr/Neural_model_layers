# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:10:51 2026

@author: Nicc
"""

import argparse

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Verification tests")
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        help="Range defining seeds to verify; (i.e 20)",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=int,
        required=True,
        help="Verification ID Number",
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    fin_results = []
    
    for seed in range(1, args.seeds + 1):
        df = pd.read_csv(f"C:/Users/Nicc/Desktop/CNR_Model/testing/test_seed{seed}/Test_Simulation.csv")
        
        for data in range(int(((df.iloc[-1]['Trial'] + 1) // 2) + 1), int(((df.iloc[-1]['Trial'] + 1) * 0.75))):
            verification = {}
            verification["Seed"] = seed
            
            lim_1 = data
            lim_2 = data + 25.0
            
            cond_1 = df[
                (df["Trial"] == lim_1)
                ].sort_values("Timestep").copy()
            cond_1 = cond_1[cond_1['Input_0'] == 1.0]
            MC_cond1 = cond_1.filter(like="MC_Unit").to_numpy()
            inp_cond1 = cond_1.filter(like="Input").to_numpy()
            inp_cond1 = inp_cond1[-1]
            winner_1 = np.argmax(MC_cond1[-1])
            
            cond_2 = df[
                (df["Trial"] == lim_2)
                ].sort_values("Timestep").copy()
            cond_2 = cond_2[cond_2['Input_0'] == 1.0]
            MC_cond2 = cond_2.filter(like="MC_Unit").to_numpy()
            inp_cond2 = cond_2.filter(like="Input").to_numpy()
            inp_cond2 = inp_cond2[-1]
            winner_2 = np.argmax(MC_cond2[-1])
            
            if winner_1 == 0 and inp_cond1[-1] == 1.0:
                verification["Condition_1"] = 1.0
            
            elif winner_1 == 1 and inp_cond1[-2] == 1.0:
                verification["Condition_1"] = 1.0
                
            elif winner_1 == 0 and inp_cond1[-2] == 1.0:
                verification["Condition_1"] = 0.0
                
            elif winner_1 == 1 and inp_cond1[-1] == 1.0:
                verification["Condition_1"] = 0.0
            
            if winner_2 == 0 and inp_cond2[-1] == 1.0:
                verification["Condition_2"] = 1.0
            
            elif winner_2 == 1 and inp_cond2[-2] == 1.0:
                verification["Condition_2"] = 1.0
                
            elif winner_2 == 0 and inp_cond2[-2] == 1.0:
                verification["Condition_2"] = 0.0
                
            elif winner_2 == 1 and inp_cond2[-1] == 1.0:
                verification["Condition_2"] = 0.0
                
            fin_results.append(verification.copy())
        
    sum_cond1 = 0.0 
    sum_cond2 = 0.0
    
    for ver in fin_results:
         sum_cond1 += ver["Condition_1"]
         sum_cond2 += ver["Condition_2"]
        
    res_cond1 = sum_cond1 / len(fin_results)
    res_cond2 = sum_cond2 / len(fin_results)
    
    output_path = f"verification/verification_results_{args.id}.txt"

    with open(output_path, "w") as f:
        f.write("Verification Test results\n")
        f.write(f"Seeds from 1 to {args.seeds}\n\n")
        
        f.write("Controlling for MC final decision given Satiation conditions at 2 given trials (1 per condition):\n")
        f.write("Condition_1 = Satiation_1\n")
        f.write("Condition_2 = Satiation_2\n\n")
        
        f.write(f"Condition 1 success rate: {res_cond1:.4f}\n")
        f.write(f"Condition 2 success rate: {res_cond2:.4f}\n")
        
        
        
        
        