#!/usr/bin/env bash
set -e

#Use even numbers for ID for Non-lesion simulation
#Use AREA_pre/post_odd numbers for ID for Lesion simulation

conditions=(
    "None None 00"
#    "NAc None NAc_pre_1"
#    "None NAc NAc_post_1"
#    "BLA None BLA_pre_1"
#    "None BLA BLA_post_1"
#    "DMS None DMS_pre_1"
#    "None DMS DMS_post_1"
#    "PL None PL_pre_1"
#    "None PL PL_post_1"
    "BLA BLA BLA_pre_post_00"
    "DMS DMS DMS_pre_post_00"
    "NAc NAc NAc_pre_post_00"
    "PL PL PL_pre_post_00"
)

seed_start=1
seed_end=10

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR="$SRC"
DATA_DIR="${SRC}/../CNR_model_data/PIT"

for condition in "${conditions[@]}"; do

    read lesion_pre lesion_post id <<< "$condition"

    echo "======================================"
    echo "Running condition:"
    echo "PRE lesion  = $lesion_pre"
    echo "POST lesion = $lesion_post"
    echo "ID          = $id"
    echo "======================================"

    INST_DIR="${DATA_DIR}/instrumental_training/instrumental_training_${id}"
    PAV_DIR="${DATA_DIR}/pavlovian_training/pavlovian_training_${id}"
    PIT_DIR="${DATA_DIR}/PIT_testing/PIT_testing_${id}"

    # ==========================================
    # INSTRUMENTAL SCHEDULING
    # ==========================================

    scheduling=$(cat << EOF
{
    "trials": 100,
    "timesteps": 1000,
    "states": [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "phases": [0.5, 1.0]
}
EOF
)

# # ==========================================
#     # INSTRUMENTAL TRAINING LOOP
#     # ==========================================

    for seed in $(seq $seed_start 1 $seed_end); do

        SIM="${INST_DIR}/inst_sim_seed${seed}"

        mkdir -p "$SIM"

        cd "$SIM"

        cp "$CURR_DIR/prm_file.json" "$SIM/"

        echo "$scheduling" > scheduling.json

        echo "Running INSTRUMENTAL simulation seed=$seed"

        python ${SRC}/Instrumental_learning.py \
            -d scheduling.json \
            -s $seed \
            -l $lesion_pre

        cd "$CURR_DIR"

    done

    # ==========================================
    # PAVLOVIAN SCHEDULING
    # ==========================================

    scheduling=$(cat << EOF
{
    "trials": 100,
    "timesteps": 1000,
    "states": [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "phases": [0.5, 1.0]
}
EOF
)

# # ==========================================
#     # PAVLOVIAN TRAINING LOOP
#     # ==========================================

    for seed in $(seq $seed_start 1 $seed_end); do

        SIM="${PAV_DIR}/pav_sim_seed${seed}"

        mkdir -p "$SIM"

        cd "$SIM"

        cp "$CURR_DIR/prm_file.json" "$SIM/"

        echo "$scheduling" > scheduling.json

        echo "Running PAVLOVIAN simulation seed=$seed"

        python ${SRC}/Pavlovian_Conditioning.py \
            -d scheduling.json \
            -s $seed \
            -l $lesion_pre

        cd "$CURR_DIR"

    done

# ==========================================
    # PIT SCHEDULING
    # ==========================================

    scheduling=$(cat << EOF
{
    "trials": 100,
    "timesteps": 1000,
    "states": [
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "phases": [0.5, 1.0]
}
EOF
)

    # ==========================================
    # PIT TEST LOOP
    # ==========================================

    for seed in $(seq $seed_start 1 $seed_end); do

        SIM="${PIT_DIR}/pit_test_seed${seed}"

        mkdir -p "$SIM"

        cd "$SIM"

        echo "$scheduling" > scheduling.json

        echo "Running PIT test simulation seed=$seed"

        python ${SRC}/PIT_Test_Simulation.py \
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