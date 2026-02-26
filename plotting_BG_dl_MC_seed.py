# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:48:26 2026

@author: Nicc
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="MGV_MC_short_test.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--column",
        type=str,
        required=True,
        help="Column name to plot (y-axis), use: Inp_DLS_W_x_y (2*2)"
    )
    parser.add_argument(
        "--x",
        type=str,
        default='Seed',
        help="Optional x-axis column (e.g. Seed)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.csv)

    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found")

    if args.x is not None and args.x not in df.columns:
        raise ValueError(f"X column '{args.x}' not found")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))

    if args.x:
        sns.lineplot(
            data=df,
            x=args.x,
            y=args.column,
            marker="o"
        )
        plt.xlabel(args.x)
    else:
        sns.lineplot(
            data=df,
            y=args.column,
            marker="o"
        )
        plt.xlabel("Run")

    plt.ylabel(args.column)
    plt.title(f"{args.column} across runs")

    plt.tight_layout()
    plt.show()
