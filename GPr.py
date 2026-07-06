
from Layer_types import BG_v2, Leaky_units_exc
import numpy as np

class GPr:

    def __init__(self, parameters):

        self.parameters = parameters

        rng = np.random.RandomState(self.parameters.seed)

        self.BG_dl = BG_v2(
            self.parameters.N["BG_dl"], 
            self.parameters.tau["BG_dl"], 
            self.parameters.baseline["DLS_1"],
            self.parameters.baseline["DLS_2"],
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
            self.parameters.threshold["BG_dl"]
        )
        
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
                                                            
        self.Ws = {
            "inp_DLS_1": np.eye(self.parameters.N["BG_dl"]),
            "inp_DLS_2": np.ones((self.parameters.N["BG_dl"], self.parameters.N["BG_dl"])) * 0.5,
            "GPi_MGV": np.eye(self.parameters.N["MGV"]) * self.parameters.Matrices_scalars["GPi_MGV"],
            "MC_MGV": np.eye(self.parameters.N["MGV"]) * self.parameters.Matrices_scalars["MC_MGV"],
            "MGV_MC": np.eye(self.parameters.N["MC"]) * self.parameters.Matrices_scalars["MGV_MC"],
            "MC_DLS_1": np.eye(self.parameters.N["BG_dl"]) * self.parameters.Matrices_scalars["MC_DLS_1"],
            "MC_DLS_2": np.eye(self.parameters.N["BG_dl"]) * self.parameters.Matrices_scalars["MC_DLS_2"],
            "MC_STNdl": np.eye(self.parameters.N["BG_dl"]) * self.parameters.Matrices_scalars["MC_STNdl"]   
              }
        
        self.BG_dl_output_pre = np.zeros(self.parameters.N["BG_dl"])
        self.MGV_output_pre = np.zeros(self.parameters.N["MGV"])
        self.MC_output_pre = np.zeros(self.parameters.N["MC"])

    def reset_activity(self):

        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()

    def update_output_pre(self):
        
        self.BG_dl_output_pre = self.BG_dl.output_BG.copy()
        self.MGV_output_pre = self.MGV.output.copy()
        self.MC_output_pre = self.MC.output.copy()

    def step(self, inp, da):
        
        self.BG_dl.step(
            (self.parameters.DA_values["Y_DLS_1"] + self.parameters.DA_values["delta_DLS_1"] * da) * np.dot(self.Ws["inp_DLS_1"], inp),
            ((1/(self.parameters.DA_values["Y_DLS_2"] + self.parameters.DA_values["delta_DLS_2"] * da)) * np.dot(self.Ws["inp_DLS_2"], inp)),
            (self.parameters.DA_values["Y_DLS_1"] + self.parameters.DA_values["delta_DLS_1"] * da) * np.dot(self.Ws["MC_DLS_1"], self.MC_output_pre),
            ((1/(self.parameters.DA_values["Y_DLS_2"] + self.parameters.DA_values["delta_DLS_2"] * da)) * np.dot(self.Ws["MC_DLS_2"], self.MC_output_pre)),
            np.dot(self.Ws["MC_STNdl"], self.MC_output_pre)
            )
        
        self.MGV.step(np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre) +
                 np.dot(self.Ws["MC_MGV"], self.MC_output_pre))
        
        self.MC.step(np.dot(self.Ws["MGV_MC"], self.MGV_output_pre))
            
        self.update_output_pre()