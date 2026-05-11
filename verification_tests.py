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
        default=40,
        help="Range defining seeds to verify; (i.e 20)",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=int,
        default=6,
        help="Verification ID Number",
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    fin_results = []
    single_trials = []
    
    print('Starting verification')
    for seed in range(1, args.seeds + 1):
        
        print(f'Reading file with seed {seed}')
        df = pd.read_csv(f"C:/Users/Nicc/Desktop/CNR_Model/testings/testing_{args.id}/test_seed{seed}/Test_Simulation.csv")
        ver_single_seed = {}
        
        print('Starting data extraction')
        for data in range(1, int(((df.iloc[-1]['Trial'] * 0.5))) + 1):
            verification = {}
            verification["Seed"] = seed
            verification["Trial"] = data
            
            lim_n = data
            lim_1 = data + int(((df.iloc[-1]['Trial'] * 0.5))) + 1
            
            cond_n = df[
                (df["Trial"] == lim_n)
                ].sort_values("Timestep").copy()
            cond_n = cond_n[cond_n['Input_0'] == 1.0]
            MC_condn = cond_n.filter(like="MC_Unit").to_numpy()
            inp_condn = cond_n.filter(like="Input").to_numpy()
            inp_condn = inp_condn[-1]
            winner_n = np.argmax(MC_condn[-1])
            
            cond_1 = df[
                (df["Trial"] == lim_1)
                ].sort_values("Timestep").copy()
            cond_1 = cond_1[cond_1['Input_0'] == 1.0]
            MC_cond1 = cond_1.filter(like="MC_Unit").to_numpy()
            inp_cond1 = cond_1.filter(like="Input").to_numpy()
            inp_cond1 = inp_cond1[-1]
            winner_1 = np.argmax(MC_cond1[-1])
            
            verification["Neutral_condition"] = winner_n
            ver_single_seed[f"Neutral_condition_t{data}"] = winner_n
            
            if winner_1 == 0 and inp_cond1[-1] == 1.0:
                verification["Condition_1"] = 1.0
                ver_single_seed[f"Condition_1_t{int(data) + int(((df.iloc[-1]['Trial'] * 0.5)))}"] = 1.0
            
            elif winner_1 == 1 and inp_cond1[-2] == 1.0:
                verification["Condition_1"] = 1.0
                ver_single_seed[f"Condition_1_t{int(data) + int(((df.iloc[-1]['Trial'] * 0.5)))}"] = 1.0
                
            elif winner_1 == 0 and inp_cond1[-2] == 1.0:
                verification["Condition_1"] = 0.0
                ver_single_seed[f"Condition_1_t{int(data) + int(((df.iloc[-1]['Trial'] * 0.5)))}"] = 0.0
                
            elif winner_1 == 1 and inp_cond1[-1] == 1.0:
                verification["Condition_1"] = 0.0
                ver_single_seed[f"Condition_1_t{int(data) + int(((df.iloc[-1]['Trial'] * 0.5)))}"] = 0.0
                
            fin_results.append(verification.copy())
        
        ver_single_seed['Seed'] = seed
        single_trials.append(ver_single_seed.copy())
        
        print(f'Data extraction file with seed {seed} termined successfully')
        
    print('Strating conditions sums')
    sum_n = 0.0
    sum_cond1 = 0.0 
    
    for ver in fin_results:
        sum_n += ver["Neutral_condition"]
        sum_cond1 += ver["Condition_1"]
        
    res_condn = sum_n / len(fin_results)
    res_cond1 = sum_cond1 / len(fin_results)
    
    print('Sums termined successfully')
    
    print('Isolating single trial success rate')
    
    results_single_ts = []
    for ver in single_trials:
        sum_n_t = 0.0
        sum_cond1_t = 0.0
        for data in range(1, int(((df.iloc[-1]['Trial'] * 0.5))) + 1):
            
            sum_n_t += ver[f"Neutral_condition_t{data}"]
            sum_cond1_t += ver[f"Condition_1_t{int(data) + int(((df.iloc[-1]['Trial'] * 0.5)))}"]
            
        result_t = {}
        result_t['Seed'] = ver['Seed']
        result_t["Results_Neut_Cond"] = sum_n_t / int(((df.iloc[-1]['Trial'] * 0.5)))
        result_t["Results_Cond1"] = sum_cond1_t / int(((df.iloc[-1]['Trial'] * 0.5)))
        
        results_single_ts.append(result_t.copy())
        
    print('Single trial success rates saved successfully')
    
    print('Isolating unsuccessful testings seeds')
    failed_seeds = []
    for ver in fin_results:
        if ver["Condition_1"] == 0.0:
            if ver['Seed'] not in failed_seeds:
                failed_seeds.append(ver['Seed'])
            
    print('Unsuccessful seeds saving termined successfully')
    
    print('Collecting failed trials per seed')
    failed_trials = []
    for ver in fin_results:
        if ver['Seed'] in failed_seeds:
            if ver['Condition_1'] == 0.0:
                failed_trials.append(
                    ('Seed_' + str(ver['Seed']),
                     'Trial_' + str(int(ver['Trial']) + int((df.iloc[-1]['Trial'] * 0.5)) + 1))
                    )
                
    print('Unsuccessful seeds trials saving termined successfully')
    
    print("Calculating Standard Deviation for all seeds")
    
    tot_n = []
    tot_1 = []
    for res in results_single_ts:
        tot_n.append(res["Results_Neut_Cond"])
        tot_1.append(res["Results_Cond1"])
    
    mean_1 = np.mean(tot_1)
    
    std_n = np.std(tot_n)
    std_1 = np.std(tot_1)
    
    print('Creating .txt results file')
    output_path = f"verification/verification_results_{str(args.id)}.txt"

    with open(output_path, "w") as f:
        f.write("Verification Test results\n")
        f.write(f"Seeds from 1 to {args.seeds}\n\n")
        
        f.write("Controlling for MC final decision given Satiation:\n")
        f.write("Neutral_Condition = No Satiation\n")
        f.write("Condition_1 = Satiation_2\n\n")
        
        f.write(f"Neutral Condition average actions selected and Standard Deviation: {res_condn:.4f}, {std_n:.4f}\n")
        f.write(f"Condition 1 total success rate: {res_cond1:.4f}\n")
        f.write(f"Condition 1 Mean & Standard Deviation: {mean_1:.4f}, {std_1:.4f}\n\n")
        
        f.write("Success rate values per single seed\n\n")
        for res in results_single_ts:
            f.write(f"Simulation Seed: {res['Seed']}\n")
            f.write(f"Neutral condition average action selected: {res['Results_Neut_Cond']:.4f}\n")
            f.write(f"Condition 1 success rate: {res['Results_Cond1']:.4f}\n\n")
        
        f.write("Unsuccessful simulations seeds:\n")
        f.write(f"{failed_seeds}\n\n")
        
        f.write("Unsuccessful simulations trials per unsuccessful seed:\n")
        for data in range(len(failed_trials)):
            f.write(f"{failed_trials[data]}\n")    
        
    print('Verification and results saving termined successfully')
        