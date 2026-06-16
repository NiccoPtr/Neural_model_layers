#!/usr/bin/env bash

#Use even numbers for ID for Non-lesion simulation
#Use AREA_pre/post_odd numbers for ID for Lesion simulation

conditions=(
    "None None 00"
    "PL None PL_pre_00"
    "None PL PL_post_00"
)

seed_start=1
seed_end=6

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR=$(pwd)

for condition in "${conditions[@]}"; do

    read lesion_pre lesion_post id <<< "$condition"

    echo "======================================"
    echo "Running condition:"
    echo "PRE lesion  = $lesion_pre"
    echo "POST lesion = $lesion_post"
    echo "ID          = $id"
    echo "======================================"

    TRAIN_DIR="/c/Users/Nicc/Desktop/CNR_Model/trainings/training_$id"
    TEST_DIR="/c/Users/Nicc/Desktop/CNR_Model/testings/testing_$id"

    # ==========================================
    # TRAINING SCHEDULING
    # ==========================================

    scheduling=$(cat << EOF
{
    "trials": 80,
    "timesteps": 1000,
    "states": [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "phases": [0.5, 1.0]
}
EOF
)

# # ==========================================
#     # TRAINING LOOP
#     # ==========================================

    for seed in $(seq $seed_start 1 $seed_end); do

        SIM="${TRAIN_DIR}/sim_seed${seed}"

        mkdir -p "$SIM"

        cd "$SIM"

        cp "$CURR_DIR/prm_file.json" "$SIM/"

        echo "$scheduling" > scheduling.json

        echo "Running TRAINING simulation seed=$seed"

        python ${SRC}/Model_Simulation.py \
            -d scheduling.json \
            -s $seed \
            -l $lesion_pre

        cd "$CURR_DIR"

    done

# ==========================================
    # TEST SCHEDULING
    # ==========================================

    scheduling=$(cat << EOF
{
    "trials": 80,
    "timesteps": 1000,
    "states": [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ],
    "phases": [0.5, 1.0]
}
EOF
)

    # ==========================================
    # TEST LOOP
    # ==========================================

    for seed in $(seq $seed_start 1 $seed_end); do

        SIM="${TEST_DIR}/test_seed${seed}"

        mkdir -p "$SIM"

        cd "$SIM"

        echo "$scheduling" > scheduling.json

        echo "Running TEST simulation seed=$seed"

        python ${SRC}/Test_Simulation.py \
            -d scheduling.json \
            -i $id \
            -s $seed \
            -l $lesion_post

        cd "$CURR_DIR"

    done

    # ==========================================
    # ANALYSIS
    # ==========================================

    python ${SRC}/results_analysis_fast.py \
        -p yes \
        -i $id

done