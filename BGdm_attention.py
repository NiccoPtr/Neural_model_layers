

from Layer_types import BG_v2, Leaky_units_exc
import numpy as np

class CT_BG():
    
    def __init__(self, parameters, rng):

        self.parameters = parameters
        
        self.BG_dm = BG_v2(
            self.parameters.N["BG_dm"], 
            self.parameters.tau["BG_dm"], 
            self.parameters.baseline["DMS_1"],
            self.parameters.baseline["DMS_2"],
            self.parameters.baseline["STNdm"],
            self.parameters.baseline["GPi_SNpr"],
            self.parameters.baseline["GPe"],
            self.parameters.BG_dm_W["DMS_1_GPi_SNpr_W"], 
            self.parameters.BG_dm_W["DMS_2_GPe_W"],
            self.parameters.BG_dm_W["STNdm_GPi_SNpr_W"],
            self.parameters.BG_dm_W["STNdm_GPe_W"],
            self.parameters.BG_dm_W["GPe_STNdm_W"],
            self.parameters.BG_dm_W["GPe_GPi_SNpr_W"],
            rng,
            self.parameters.noise["BG_dm"],
            self.parameters.threshold["BG_dm"]
        )
        
        self.P = Leaky_units_exc(parameters.N["P"], 
                              parameters.tau["P"],
                              parameters.baseline["P"],
                              rng,
                              parameters.noise["P"],
                              parameters.threshold["P"])
        
        self.PFCd_PPC = Leaky_units_exc(parameters.N["PFCd_PPC"], 
                             parameters.tau["PFCd_PPC"], 
                             parameters.baseline["PFCd_PPC"],
                             rng,
                             parameters.noise["PFCd_PPC"],
                             parameters.threshold["PFCd_PPC"])
        
        self.Ws = {
            "inp_DMS_1": np.ones([parameters.N["BG_dm"], parameters.N["BG_dm"]]) * parameters.Matrices_scalars["Mani_DMS"], 
            "inp_DMS_2": np.ones([parameters.N["BG_dm"], parameters.N["BG_dm"]]) * parameters.Matrices_scalars["Mani_DMS"], 
            "PFCd_PPC_P": np.eye(parameters.N["P"]) * parameters.Matrices_scalars["PFCd_PPC_P"],
            "P_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars["P_PFCd_PPC"],
            "GPi_SNpr_P": np.eye(parameters.N["P"]) * parameters.Matrices_scalars["GPi_SNpr_P"],
            "PFCd_PPC_DMS_1": np.eye(parameters.N["BG_dm"]) * parameters.Matrices_scalars["PFCd_PPC_DMS_1"],
            "PFCd_PPC_DMS_2": np.eye(parameters.N["BG_dm"]) * parameters.Matrices_scalars["PFCd_PPC_DMS_2"],
            "PFCd_PPC_STNdm": np.eye(parameters.N["BG_dm"]) * parameters.Matrices_scalars["PFCd_PPC_STNdm"],
            "MC_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars['MC_PFCd_PPC'],
            "PL_PFCd_PPC": np.eye(parameters.N["PFCd_PPC"]) * parameters.Matrices_scalars['PL_PFCd_PPC']
              }
        
        self.W_learn_mask = np.ones([parameters.N["BG_dm"], parameters.N["BG_dm"]])
        
        self.BG_dm_output_pre = np.zeros(parameters.N["BG_dm"])
        self.P_output_pre = np.zeros(parameters.N["P"])
        self.PFCd_PPC_output_pre = np.zeros(parameters.N["PFCd_PPC"])

    def reset_activity(self):

        self.BG_dm.reset_activity()
        self.P.reset_activity()
        self.PFCd_PPC.reset_activity()
        
    def update_output_pre(self):
        
        self.BG_dm_output_pre = self.BG_dm.output_BG.copy()
        self.P_output_pre = self.P.output.copy()
        self.PFCd_PPC_output_pre = self.PFCd_PPC.output.copy()
        
    def delta_Str_learn_1(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W):
        
        DA_term = np.maximum(0, DA - theta_DA_str)
        delta_W_inp_str = (eta_str *
                           DA_term * 
                           np.outer(
                               np.maximum(0, v_str - theta_str),
                               np.maximum(0, v_inp - theta_inp_str)
                               ) *
                           (max_W_str - W))
        
        delta_W_inp_str *= mask
        
        return delta_W_inp_str
    
    def delta_Str_learn_2(self, eta_str, DA, v_str, v_inp, theta_DA_str, theta_str, theta_inp_str, mask, max_W_str, W, lambda_xor = 0.1):
        
        DA_term = np.maximum(0, DA - theta_DA_str)

        pre = np.maximum(0, v_inp - theta_inp_str)
        post = np.maximum(0, v_str - theta_str)

        hebb = np.outer(post, pre)

        A = pre[None, :]
        B = post[:, None]

        xor_term = (A + B) - (2 * A * B)

        delta_W = (
            eta_str
            * DA_term
            * (hebb - lambda_xor * xor_term)
            * (max_W_str - W)
        )

        delta_W *= mask

        return delta_W
    
    def learning(self, parameters, da, inp):
        
        self.delta_W_inp_DMS_1 = self.delta_Str_learn_1(parameters.Str_Learn["eta_DMS_1"],
                                           da,
                                           self.BG_dm.output_Str1_pre * -1,
                                           inp,
                                           parameters.Str_Learn["theta_DA_DMS_1"],
                                           parameters.Str_Learn["theta_DMS_1"],
                                           parameters.Str_Learn["theta_inp_DMS_1"],
                                           self.W_learn_mask,
                                           parameters.Str_Learn["max_W_DMS"],
                                           self.Ws["inp_DMS_1"]
                                           )
        
        self.Ws['inp_DMS_1'] += self.delta_W_inp_DMS_1

        self.delta_W_inp_DMS_2 = self.delta_Str_learn_2(parameters.Str_Learn["eta_DMS_2"],
                                           da,
                                           self.BG_dm.output_Str2_pre * -1,
                                           inp,
                                           parameters.Str_Learn["theta_DA_DMS_2"],
                                           parameters.Str_Learn["theta_DMS_2"],
                                           parameters.Str_Learn["theta_inp_DMS_2"],
                                           self.W_learn_mask,
                                           parameters.Str_Learn["max_W_DMS"],
                                           self.Ws["inp_DMS_2"]
                                           )
        
        self.Ws['inp_DMS_2'] += self.delta_W_inp_DMS_2
        self.Ws['inp_DMS_2'] = np.maximum(self.Ws['inp_DMS_2'], 0)

    def step(self, parameters, inp, da, MC_inp = (0.8, 0.2), PL_inp = (0.8, 0.2), learn = True):
        
        self.BG_dm.step(
            (self.parameters.DA_values["Y_DMS_1"] + self.parameters.DA_values["delta_DMS_1"] * da) * np.dot(self.Ws["inp_DMS_1"], inp),
            ((1/(self.parameters.DA_values["Y_DMS_2"] + self.parameters.DA_values["delta_DMS_2"] * da)) * np.dot(self.Ws["inp_DMS_2"], inp)),
            (self.parameters.DA_values["Y_DMS_1"] + self.parameters.DA_values["delta_DMS_1"] * da) * np.dot(self.Ws["PFCd_PPC_DMS_1"], self.PFCd_PPC_output_pre),
            ((1/(self.parameters.DA_values["Y_DMS_2"] + self.parameters.DA_values["delta_DMS_2"] * da)) * np.dot(self.Ws["PFCd_PPC_DMS_2"], self.PFCd_PPC_output_pre)),
            np.dot(self.Ws["PFCd_PPC_STNdm"], self.PFCd_PPC_output_pre)
            )
        
        self.P.step(np.dot(self.Ws["GPi_SNpr_P"], self.BG_dm_output_pre) +
                 np.dot(self.Ws["PFCd_PPC_P"], self.PFCd_PPC_output_pre))
        
        self.PFCd_PPC.step(np.dot(self.Ws["P_PFCd_PPC"], self.P_output_pre) +
                    np.dot(self.Ws['MC_PFCd_PPC'], np.array(MC_inp)) +
                    np.dot(self.Ws['PL_PFCd_PPC'], np.array(PL_inp))
                    )
        
        if learn:
            self.learning(parameters, da, inp)
            
        self.update_output_pre()