#!/usr/bin/env bash

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

for seed in $(seq 6 1 6); do
    SIM=sim_seed${seed}
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Model_Simulation.py -m save -d scheduling.json -s $seed

    cd $CURR_DIR  
done