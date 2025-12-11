# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:59:11 2025

@author: Nicc
"""

from params import parameters
from Layer_types import *
import numpy as np, matplotlib.pyplot as plt

rng = np.random.RandomState(parameters.seed)

MGV = Leaky_units_exc(parameters.N["MGV"], 
                      parameters.tau["MGV"],
                      parameters.baseline["MGV"],
                      rng,
                      parameters.noise["MGV"],
                      parameters.threshold["MGV"])

MC = Leaky_units_exc(parameters.N["MC"], 
                     parameters.tau["MC"], 
                     parameters.baseline["MC"],
                     rng,
                     parameters.noise["MC"],
                     parameters.threshold["MC"])

BG_dl = BG_dl_Layer(parameters.N["BG_dl"], 
                    parameters.tau["BG_dl"], 
                    parameters.baseline["DLS"],
                    parameters.baseline["STNdl"],
                    parameters.baseline["GPi"],
                    parameters.BG_dl_W["DLS_GPi_W"], 
                    parameters.BG_dl_W["STNdl_GPi_W"],
                    rng,
                    parameters.noise["BG_dl"],
                    parameters.threshold["BG_dl"])

Ws = {"inp_DMS": np.eye(parameters.N["BG_dl"]),
      "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
      "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
      "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
      "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
      "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"]
      }

BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
MGV_output_pre = np.zeros(parameters.N["MGV"])
MC_output_pre = np.zeros(parameters.N["MC"])

inp = np.array([0.0, 1.0])

for i in range(1000):
    
    BG_dl.step(np.dot(Ws["inp_DMS"], inp),
               np.dot(Ws["MC_DLS"], MC_output_pre),
               np.dot(Ws["MC_STNdl"], MC_output_pre))
    
    MGV.step(np.dot(Ws["GPi_MGV"], BG_dl_output_pre +
             np.dot(Ws["MC_MGV"], MC_output_pre)))
    
    MC.step(np.dot(Ws["MGV_MC"], MGV_output_pre))
    
    BG_dl_output_pre = BG_dl.output_BG_dl.copy()
    
    MGV_output_pre = MGV.output.copy()
    
    MC_output_pre = MC.output.copy()


