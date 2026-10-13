
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:07:18 2026

@author: Nicc
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from params import Parameters
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Verification tests")

    parser.add_argument(
        "-i",
        "--id",
        type=str,
        default="1",
        help="Verification ID Number",
    )

    parser.add_argument(
        "-p",
        "--save_plot",
        type=str,
        choices=["yes", "show", "no"],
        required=True,
        help="'yes' to save the plot, 'show' to display it, or 'no' for neither",
    )

    parser.add_argument(
        "-d",
        "--decision",
        type=str,
        choices=["Lever", "Chain"],
        required=True,
        help="Decision to plot: 'Lever' or 'Chain'",
    )

    return parser.parse_args()


def mean_act(x, thr):
    res = x.iloc[900:, :].mean(axis=0)
    res = res.argmax() if sum(res >= thr) == 1 else 2

    return res


def p_to_stars(p):
    if not np.isfinite(p):
        return "n.s."
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


if __name__ == "__main__":

    args = parse_args()
    parameters = Parameters()

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "CNR_model_data" / "PIT"

    INST_DIR = DATA_DIR / "instrumental_training"
    PIT_DIR = DATA_DIR / "PIT_testing"
    RESULTS_DIR = DATA_DIR / "results"

    # --------------------------------------------------
    # Load parameters
    # --------------------------------------------------

    param_file = (
        INST_DIR
        / f"instrumental_training_{args.id}"
        / "inst_sim_seed1"
        / "prm_file.json"
    )

    if not param_file.exists():
        param_file = (
            INST_DIR
            / f"instrumental_training_{args.id}"
            / "inst_sim_seed31"
            / "prm_file.json"
        )

    if param_file.exists():
        parameters.load(param_file, mode="json")
        print("Imported parameters successfully")
    else:
        raise FileNotFoundError(
            f"Parameters file not found: {param_file}"
        )

    # --------------------------------------------------
    # Load PIT testing data
    # --------------------------------------------------

    files = sorted(
        glob.glob(
            str(
                PIT_DIR
                / f"PIT_testing_{args.id}"
                / "pit_test_seed*"
                / "PIT_Test_Simulation.csv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No PIT testing files found for ID {args.id}"
        )

    thr = parameters.threshold["MC"]

    dfs = []

    for file in files:

        # Extract the actual seed number from the directory name.
        match = re.search(r"pit_test_seed(\d+)", file)

        if match is None:
            raise ValueError(f"Could not identify seed in path: {file}")

        seed = int(match.group(1))

        temp_df = pd.read_csv(file)
        temp_df["Seed"] = seed

        dfs.append(temp_df)

    raw_df = pd.concat(dfs, ignore_index=True)

    # --------------------------------------------------
    # Identify the decision made on each trial
    # --------------------------------------------------

    df = (
        raw_df.groupby(["Seed", "Phase", "Trial"])[
            ["MC_Unit_0", "MC_Unit_1"]
        ]
        .apply(lambda x: mean_act(x, thr))
        .reset_index(name="Decision")
    )

    df.replace(
        {
            "Decision": {
                0: "Lever",
                1: "Chain",
                2: "None",
            },
            "Phase": {
                1: "BS",
                2: "C",
                3: "IC",
            },
        },
        inplace=True,
    )

    # --------------------------------------------------
    # Count decisions for each seed and phase
    # --------------------------------------------------

    counts_df = (
        df.groupby(["Seed", "Phase", "Decision"])
        .size()
        .unstack("Decision", fill_value=0)
        .reindex(columns=["Lever", "Chain", "None"], fill_value=0)
        .reset_index()
    )

    counts_df.rename(columns={"None": "NONE"}, inplace=True)

    # --------------------------------------------------
    # Prepare the data for plotting
    # --------------------------------------------------

    plot_df = counts_df[["Seed", "Phase", args.decision]].copy()

    plot_df.rename(
        columns={args.decision: "Freq"},
        inplace=True,
    )

    phase_order = ["BS", "C", "IC"]

    plot_df["Phase"] = pd.Categorical(
        plot_df["Phase"],
        categories=phase_order,
        ordered=True,
    )

    # --------------------------------------------------
    # Paired statistical comparisons between phases
    # --------------------------------------------------

    stats_df = plot_df.pivot(
        index="Seed",
        columns="Phase",
        values="Freq",
    ).reindex(columns=phase_order)

    comparisons = [
        ("BS", "C"),
        ("BS", "IC"),
        ("C", "IC"),
    ]

    pvals = {}

    for phase1, phase2 in comparisons:

        paired = stats_df[[phase1, phase2]].dropna()

        if len(paired) < 2:
            p = np.nan
            print(
                f"{phase1} vs {phase2}: insufficient paired seeds "
                f"({len(paired)})"
            )
        else:
            _, p = ttest_rel(
                paired[phase1],
                paired[phase2],
            )

        pvals[(phase1, phase2)] = p

        print(
            f"{args.decision}: {phase1} vs {phase2}, "
            f"n={len(paired)}, p={p:.6g}"
            if np.isfinite(p)
            else f"{args.decision}: {phase1} vs {phase2}, p=NaN"
        )

    # --------------------------------------------------
    # Plot the selected decision
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=plot_df,
        x="Phase",
        y="Freq",
        order=phase_order,
        errorbar="sd",
        ax=ax,
    )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Number of decisions")
    ax.set_title(f"{args.decision} decisions across PIT conditions")

    # --------------------------------------------------
    # Add significance brackets and stars
    # --------------------------------------------------

    summary = plot_df.groupby(
        "Phase", observed=False
    )["Freq"].agg(["mean", "std"])

    # Account for both bar heights and their SD error bars.
    upper_bounds = (
        summary["mean"].fillna(0)
        + summary["std"].fillna(0)
    )

    y_max = max(upper_bounds.max(), 1)
    offset = y_max * 0.12
    bracket_height = y_max * 0.025

    phase_positions = {
        "BS": 0,
        "C": 1,
        "IC": 2,
    }

    for i, (phase1, phase2) in enumerate(comparisons):

        x1 = phase_positions[phase1]
        x2 = phase_positions[phase2]

        y = y_max + offset * (i + 1)
        star = p_to_stars(pvals[(phase1, phase2)])

        # Draw the comparison bracket.
        ax.plot(
            [x1, x1, x2, x2],
            [y - bracket_height, y, y, y - bracket_height],
            lw=1.2,
            color="black",
        )

        ax.text(
            (x1 + x2) / 2,
            y + bracket_height * 0.15,
            star,
            ha="center",
            va="bottom",
        )

    ax.set_ylim(
        0,
        y_max + offset * (len(comparisons) + 0.8),
    )

    fig.tight_layout()

    # --------------------------------------------------
    # Save or show the figure
    # --------------------------------------------------

    if args.save_plot == "yes":

        save_dir = RESULTS_DIR / f"simulation_ID_{args.id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_path = (
            save_dir
            / f"analysis_ID_{args.id}_{args.decision}_barplot.png"
        )

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Plot saved to: {plot_path}")

    elif args.save_plot == "show":
        plt.show()

    plt.close(fig)
