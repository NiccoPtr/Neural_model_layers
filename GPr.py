
from Layer_types import BG_dl_v2, Leaky_units_exc
import numpy as np

class GPr:

    def __init__(self, parameters):

        self.parameters = parameters

        rng = np.random.RandomState(self.parameters.seed)

        self.BG_dl = BG_dl_v2(self.parameters.N["BG_dl"], 
                            self.parameters.tau["BG_dl"], 
                            self.parameters.baseline["DLS"],
                            self.parameters.baseline["STNdl"],
                            self.parameters.baseline["GPi"],
                            self.parameters.baseline["GPe"],
                            self.parameters.BG_dl_W["DLS_1_GPi_W"], 
                            self.parameters.BG_dl_W["DLS_2_GPe_W"],
                            self.parameters.BG_dl_W["STNdl_GPi_W"],
                            self.parameters.BG_dl_W["STNdl_GPe_W"],
                            self.parameters.BG_dl_W["GPe_STNdl_W"],
                            self.parameters.BG_dl_W["GPe_GPi_W"],
                            rng,
                            self.parameters.noise["BG_dl"],
                            self.parameters.threshold["BG_dl"])
        
        self.MGV = Leaky_units_exc(self.parameters.N["MGV"], 
                              self.parameters.tau["MGV"],
                              self.parameters.baseline["MGV"],
                              rng,
                              self.parameters.noise["MGV"],
                              self.parameters.threshold["MGV"])
        
        self.MC = Leaky_units_exc(self.parameters.N["MC"], 
                             self.parameters.tau["MC"], 
                             self.parameters.baseline["MC"],
                             rng,
                             self.parameters.noise["MC"],
                             self.parameters.threshold["MC"])
        
        self.Ws = {"inp_DLS": np.eye(self.parameters.N["BG_dl"]), 
              "MC_MGV": np.eye(self.parameters.N["MGV"]) * self.parameters.Matrices_scalars["MC_MGV"],
              "MGV_MC": np.eye(self.parameters.N["MC"]) * self.parameters.Matrices_scalars["MGV_MC"],
              "GPi_MGV": np.eye(self.parameters.N["MGV"]) * self.parameters.Matrices_scalars["GPi_MGV"],
              "MC_DLS": np.eye(self.parameters.N["BG_dl"]) * self.parameters.Matrices_scalars["MC_DLS"],
              "MC_STNdl": np.eye(self.parameters.N["BG_dl"]) * self.parameters.Matrices_scalars["MC_STNdl"],
              "PFCd_PPC_MC": np.eye(self.parameters.N["MC"]) * self.parameters.Matrices_scalars['PFCd_PPC_MC']
              }
        
        self.BG_dl_output_pre = np.zeros(self.parameters.N["BG_dl"])
        self.MGV_output_pre = np.zeros(self.parameters.N["MGV"])
        self.MC_output_pre = np.zeros(self.parameters.N["MC"])

    def reset_activity(self):

        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()

    def update_output_pre(self):
        
        self.BG_dl_output_pre = self.BG_dl.output_BG_dl.copy()
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()

    def step(self, inp, da, PFCd_PPC_inp = (0.0, 0.0)):
        
        self.BG_dl.step(
            (self.parameters.DA_values["Y_DLS"] + da) * np.dot(self.Ws["inp_DLS"], inp),
            ((1/(self.parameters.DA_values["Y_DLS"] + da)) * np.dot(self.Ws["inp_DLS"], inp)),
            np.dot(self.Ws["MC_DLS"], self.MC_output_pre),
            np.dot(self.Ws["MC_STNdl"], self.MC_output_pre)
            )
        
        self.MGV.step(np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre) +
                 np.dot(self.Ws["MC_MGV"], self.MC_output_pre))
        
        self.MC.step(np.dot(self.Ws["MGV_MC"], self.MGV_output_pre) +
                    np.dot(self.Ws['PFCd_PPC_MC'], np.array(PFCd_PPC_inp)))
            
        self.update_output_pre()