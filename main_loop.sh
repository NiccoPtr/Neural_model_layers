#!/usr/bin/env bash

TRAIN_DIR="/c/Users/Nicc/Desktop/CNR_Model/training"
TEST_DIR="/c/Users/Nicc/Desktop/CNR_Model/testing"
seed_start=1
seed_end=20

scheduling=$(cat << EOF 
{
    "trials": 200,
    "timesteps": 1000,
    "states": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
		],
    "phases": [0.25, 0.5, 0.75, 1.0]
}   
EOF
)

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR=$(pwd)

for seed in $(seq $seed_start 1 $seed_end); do
    SIM="${TRAIN_DIR}/sim_seed${seed}"
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Model_Simulation.py -m save -d scheduling.json -s $seed

    cd $CURR_DIR  
done

scheduling=$(cat << EOF 
{
    "trials": 100,
    "timesteps": 1000,
    "states": [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
		[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		],
    "phases": [0.5, 0.75, 1.0]
}   
EOF
)

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR=$(pwd)

for seed in $(seq $seed_start 1 $seed_end); do
    SIM="${TEST_DIR}/test_seed${seed}"
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Test_Simulation.py -m save -d scheduling.json -s $seed

    cd $CURR_DIR  
done
