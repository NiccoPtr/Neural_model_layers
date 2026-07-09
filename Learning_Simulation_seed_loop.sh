#!/usr/bin/env bash

scheduling=$(cat << EOF 
{
    "trials": 50,
    "timesteps": 1000,
    "states": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
		],
    "phases": [0.5, 1.0]
}   
EOF
)

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR=$(pwd)

for seed in $(seq 1 1 1); do
    SIM=sim_seed${seed}
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Model_Simulation.py -d scheduling.json -s $seed -l None

    cd $CURR_DIR  
done