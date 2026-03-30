#!/usr/bin/env bash

scheduling=$(cat << EOF 
{
    "trials": 20,
    "timesteps": 1000,
    "states": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
		[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		],
    "phases": [0.25, 0.5, 0.75, 1.0]
}   
EOF
)

params=$(cat << EOF 
{
	"tau":{
            "MC": __tauMC__,
            "PFCd_PPC": __tauPFCd_PPC__,
            "PL": __tauPL__,
            "MGV": 6,
            "P": 6,
            "DM": 6,
            "BG_dl": 6,
            "BG_dm": 6,
            "BG_v": 6,
            "BLA_IC": [__tauBLA_IC_uo__, 10],
            "SNpc": 6,
            "PPN": [2, 10],
            "LH": [2, 10],
            "VTA": 6
        },

        "noise":{
            "BG_dl": 0.0,
            "BG_dm": 0.0,
            "BG_v": 0.0,
            "MGV": 0.0,
            "MC": __nMC__,
            "BLA_IC": 0.0,
            "SNpc": 0.0,
            "PPN": 0.0,
            "LH": 0.0,
            "VTA": 0.0,
            "P": 0.0,
            "DM": 0.0,
            "PL": __nPL__,
            "PFCd_PPC": __nPFCd_PPC__
        },

	"baseline":{
            "PPN": 0.0,
            "LH": 0.0,
            "VTA": 0.0,
            "BLA_IC": 0.0,
            "SNpc": __bSNpc__,
            "DLS": 0.0,
            "STNdl": 0.0,
            "GPi": __bGPi__,
            "DMS": 0.0,
            "STNdm": 0.0,
            "GPi_SNpr": __bGPi_SNpr__,
            "NAc": 0.0,
            "STNv": 0.0,
            "SNpr": __bSNpr__,
            "MGV": 0.0,
            "MC": 0.0,
            "P": 0.0,
            "PFCd_PPC": 0.0,
            "DM": 0.0,
            "PL": 0.0
        },

	"Matrices_scalars":{
            "Mani_DLS": 0.0,
            "Mani_DMS": 0.0,
            "Mani_BLA_IC": 5.0,
            "Food_PPN": 10.0,
            "Food_BLA_IC": 5.0,
            "Food_LH": 10.0,
            "Sat_BLA_IC": 50.0,
            "PPN_SNpco": 20.0,
            "BLA_IC_NAc": 0.0,
            "BLA_IC_LH": 5.0,
            "LH_VTA": 20.0,
            "NAc_SNpci_1": 6.0,
            "DMS_SNpci_2": 10.0,
            "GPi_MGV": 1.5,
            "GPi_SNpr_P": 1.5,
            "SNpr_DM": 1.5,
            "MGV_MC": __wMGV_MC__,
            "P_PFCd_PPC": __wP_PFCd_PPC__,
            "DM_PL": __wDM_PL__,
            "PL_DM": __wPL_DM__,
            "PL_NAc": 1.2,
            "PL_STNv": 1.6,
            "PL_PFCd_PPC": 0.2,
            "PFCd_PPC_P": __wPFCd_PPC_P__,
            "PFCd_PPC_DMS": 1.2,
            "PFCd_PPC_STNdm": 1.6,
            "PFCd_PPC_PL": 1.0,
            "PFCd_PPC_MC": 1.0,
            "MC_MGV": __wMC_MGV__,
            "MC_DLS": 1.2,
            "MC_STNdl": 1.6,
            "MC_PFCd_PPC": 0.2
        },

	"DA_values":{
            "Y_DLS": 0.2,
            "Y_DMS": 0.5,
            "Y_NAc": 0.8,
            "delta_DLS": 4.0,
            "delta_DMS": 6.5,
            "delta_NAc": 1.5
        },

	"Str_Learn":{
            "eta_DLS": __eta_DLS__,
            "eta_DMS": __eta_DMS__,
            "eta_NAc": __eta_NAc__,
            "theta_DA_DLS": 0.3,
            "theta_DA_DMS": 0.3,
            "theta_DA_NAc": 0.5,
            "theta_DLS": __theta_DLS__,
            "theta_DMS": __theta_DMS__,
            "theta_NAc": __theta_NAc__,
            "theta_inp_DLS": 0.5,
            "theta_inp_DMS": 0.5,
            "theta_inp_NAc": __theta_inp_NAc__,
            "max_W_DLS": 1,
            "max_W_DMS": 1,
            "max_W_NAc": 2
        },

	"BG_dl_W":{"DLS_GPi_W": 1.8, "STNdl_GPi_W": __wSTNdl_GPi__},

        "BG_dm_W":{"DMS_GPiSNpr_W": 1.8, "STNdm_GPiSNpr_W": __wSTNdm_GPiSNpr__},

        "BG_v_W":{"NAc_SNpr_W": 1.8, "STNv_SNpr_W": __wSTNv_SNpr__},

	"SNpc_W":{"SNpci_1_SNpco_1_W": 1.0, "SNpci_2_SNpco_2_W": 1.0},

	"BLA_Learn":{"eta_b": __eta_b__, "alpha_t": 50.0, "tau_t": __tau_t__, "theta_DA": __theta_DA__, "max_W": 2}
}
EOF
)

__tauMC__=10
__tauPFCd_PPC__=10
__tauPL__=10
__tauBLA_IC_uo__=2

__nMC__=0.4
__nPL__=0.4
__nPFCd_PPC__=0.4

__bGPi__=0.3
__bGPi_SNpr__=0.3
__bSNpr__=0.3
__bSNpc__=0.3

__wMGV_MC__=1.8
__wMC_MGV__=2.0
__wP_PFCd_PPC__=1.8
__wPFCd_PPC_P__=2.0
__wDM_PL__=1.8
__wPL_DM__=2.0

__eta_DLS__=0.001
__eta_DMS__=0.001
__eta_NAc__=0.1
__theta_DLS__=0.12
__theta_DMS__=0.12
__theta_NAc__=0.12
__theta_inp_NAc__=0.8

__wSTNdl_GPi__=1.6
__wSTNdm_GPiSNpr__=1.6
__wSTNv_SNpr__=1.6

__eta_b__=0.1
__tau_t__=20
__theta_DA__=0.5

SRC=$(dirname "$0"| xargs realpath)
export PYTHONPATH=$SRC
export PATH=$PATH:$SRC

CURR_DIR=$(pwd)

for seed in $(seq 0 1 0); do
    SIM=sim_seed${seed}
    mkdir -p $SIM
    cd $SIM

    echo "$scheduling" > scheduling.json
    params=$(echo "$params" | sed -E "s/__tauMC__/${__tauMC__}/")
    params=$(echo "$params" | sed -E "s/__tauPFCd_PPC__/${__tauPFCd_PPC__}/")
    params=$(echo "$params" | sed -E "s/__tauPL__/${__tauPL__}/")
    params=$(echo "$params" | sed -E "s/__tauBLA_IC_uo__/${__tauBLA_IC_uo__}/")

    params=$(echo "$params" | sed -E "s/__nMC__/${__nMC__}/")
    params=$(echo "$params" | sed -E "s/__nPL__/${__nPL__}/")
    params=$(echo "$params" | sed -E "s/__nPFCd_PPC__/${__nPFCd_PPC__}/")

    params=$(echo "$params" | sed -E "s/__bGPi__/${__bGPi__}/")
    params=$(echo "$params" | sed -E "s/__bGPi_SNpr__/${__bGPi_SNpr__}/")
    params=$(echo "$params" | sed -E "s/__bSNpr__/${__bSNpr__}/")
    params=$(echo "$params" | sed -E "s/__bSNpc__/${__bSNpc__}/")

    params=$(echo "$params" | sed -E "s/__wMGV_MC__/${__wMGV_MC__}/")
    params=$(echo "$params" | sed -E "s/__wMC_MGV__/${__wMC_MGV__}/")
    params=$(echo "$params" | sed -E "s/__wP_PFCd_PPC__/${__wP_PFCd_PPC__}/")
    params=$(echo "$params" | sed -E "s/__wPFCd_PPC_P__/${__wPFCd_PPC_P__}/")
    params=$(echo "$params" | sed -E "s/__wDM_PL__/${__wDM_PL__}/")
    params=$(echo "$params" | sed -E "s/__wPL_DM__/${__wPL_DM__}/")

    params=$(echo "$params" | sed -E "s/__eta_DLS__/${__eta_DLS__}/")
    params=$(echo "$params" | sed -E "s/__eta_DMS__/${__eta_DMS__}/")
    params=$(echo "$params" | sed -E "s/__eta_NAc__/${__eta_NAc__}/")
    params=$(echo "$params" | sed -E "s/__theta_DLS__/${__theta_DLS__}/")
    params=$(echo "$params" | sed -E "s/__theta_DMS__/${__theta_DMS__}/")
    params=$(echo "$params" | sed -E "s/__theta_NAc__/${__theta_NAc__}/")
    params=$(echo "$params" | sed -E "s/__theta_inp_NAc__/${__theta_inp_NAc__}/")

    params=$(echo "$params" | sed -E "s/__wSTNdl_GPi__/${__wSTNdl_GPi__}/")
    params=$(echo "$params" | sed -E "s/__wSTNdm_GPiSNpr__/${__wSTNdm_GPiSNpr__}/")
    params=$(echo "$params" | sed -E "s/__wSTNv_SNpr__/${__wSTNv_SNpr__}/")

    params=$(echo "$params" | sed -E "s/__eta_b__/${__eta_b__}/")
    params=$(echo "$params" | sed -E "s/__tau_t__/${__tau_t__}/")
    params=$(echo "$params" | sed -E "s/__theta_DA__/${__theta_DA__}/")
    
    echo "$params" > sim_params.json
    echo "Running simulation with seed= $seed"
    python ${SRC}/Model_Simulation.py -m save -d scheduling.json -s $seed

    cd $CURR_DIR  
done
      

