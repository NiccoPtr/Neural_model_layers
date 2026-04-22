#!/usr/bin/env bash

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

for seed in $(seq 1 1 9); do
    SIM=test_seed${seed}
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Test_Simulation.py -m save -d scheduling.json -s $seed

    cd $CURR_DIR  
done
