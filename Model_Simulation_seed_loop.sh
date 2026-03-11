#!/usr/bin/env bash

scheduling=$(cat << EOF 
{
    "trials": 1,
    "timesteps": 10,
    "states": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    "phases": [0]
}   
EOF
)

params=$(cat << EOF 
{
        "Str_Learn": {
            "eta_DLS": __DLS__,
            "eta_DMS": __DMS__,
            "eta_NAc": 0.05,
            "theta_DA_DLS": 0.3,
            "theta_DA_DMS": 0.3,
            "theta_DA_NAc": 0.5,
            "theta_DLS": 0.5,
            "theta_DMS": 0.5,
            "theta_NAc": 0.5,
            "theta_inp_DLS": 0.5,
            "theta_inp_DMS": 0.5,
            "theta_inp_NAc": 0.9,
            "max_W_DLS": 1,
            "max_W_DMS": 1,
            "max_W_NAc": 2
        }
    }
EOF
)




SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC


CURR_DIR=$(pwd)

for eta_dls in 0.01 0.02 0.03; do
    for eta_dms in 0.01 0.02 0.03; do
        SIM=sim${eta_dls}_${eta_dms}
        mkdir -p $SIM
        cd $SIM

        echo "$scheduling" > scheduling.json
        echo "$params" | sed -E "s/__DLS__/${eta_dls}/; s/__DMS__/${eta_dms}/" > sim_params.json
        for seed in $(seq 0 1 5); do
            echo 'Running simulation with seed=' $seed
            python ${SRC}/Model_Simulation.py -m save -d scheduling.json -s $seed
        done
        cd $CURR_DIR
    done
done
