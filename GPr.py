
from Layer_types import BG_dl_v2, Leaky_units_exc
import numpy as np

class GPr():

    def __init__(self, parameters, rng):

        self.BG_dl = BG_dl_v2(parameters.N["BG_dl"], 
                            parameters.tau["BG_dl"], 
                            parameters.baseline["DLS_1"],
                            parameters.baseline["DLS_2"],
                            parameters.baseline["STNdl"],
                            parameters.baseline["GPi"],
                            parameters.baseline["GPe"],
                            parameters.BG_dl_W["DLS_1_GPi_W"], 
                            parameters.BG_dl_W["DLS_2_GPe_W"],
                            parameters.BG_dl_W["STNdl_GPi_W"],
                            parameters.BG_dl_W["STNdl_GPe_W"],
                            parameters.BG_dl_W["GPe_STNdl_W"],
                            parameters.BG_dl_W["GPe_GPi_W"],
                            rng,
                            parameters.noise["BG_dl"],
                            parameters.threshold["BG_dl"])
        
        self.MGV = Leaky_units_exc(parameters.N["MGV"], 
                              parameters.tau["MGV"],
                              parameters.baseline["MGV"],
                              rng,
                              parameters.noise["MGV"],
                              parameters.threshold["MGV"])
        
        self.MC = Leaky_units_exc(parameters.N["MC"], 
                             parameters.tau["MC"], 
                             parameters.baseline["MC"],
                             rng,
                             parameters.noise["MC"],
                             parameters.threshold["MC"])
        
        self.Ws = {"inp_DLS": np.ones([parameters.N["BG_dl"], parameters.N["BG_dl"]]) * parameters.Matrices_scalars["Mani_DLS"], 
              "MC_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["MC_MGV"],
              "MGV_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars["MGV_MC"],
              "GPi_MGV": np.eye(parameters.N["MGV"]) * parameters.Matrices_scalars["GPi_MGV"],
              "MC_DLS": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_DLS"],
              "MC_STNdl": np.eye(parameters.N["BG_dl"]) * parameters.Matrices_scalars["MC_STNdl"],
              "PFCd_PPC_MC": np.eye(parameters.N["MC"]) * parameters.Matrices_scalars['PFCd_PPC_MC']
              }
        
        self.BG_dl_output_pre = np.zeros(parameters.N["BG_dl"])
        self.MGV_output_pre = np.zeros(parameters.N["MGV"])
        self.MC_output_pre = np.zeros(parameters.N["MC"])

    def reset_activity(self):

        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()

    def update_output_pre(self):
        
        self.BG_dl_output_pre = self.BG_dl.output_BG_dl.copy()
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()

    def step(self, parameters, inp, da, PFCd_PPC_inp = (0.0, 0.0)):
        
        self.BG_dl.step(
            (parameters.DA_values["Y_DLS"] + da) * np.dot(self.Ws["inp_DLS"], inp),
            np.dot(self.Ws["MC_DLS"], self.MC_output_pre),
            np.dot(self.Ws["MC_STNdl"], self.MC_output_pre)
            )
        
        self.MGV.step(np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre) +
                 np.dot(self.Ws["MC_MGV"], self.MC_output_pre))
        
        self.MC.step(np.dot(self.Ws["MGV_MC"], self.MGV_output_pre) +
                    np.dot(self.Ws['PFCd_PPC_MC'], np.array(PFCd_PPC_inp)))
            
        self.update_output_pre()
