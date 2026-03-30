# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:38:31 2025

@author: Nicc
"""

import numpy as np

from Layer_types import (BG_dl_Layer, BG_dm_Layer, BG_v_Layer, BLA_IC_Layer,
                         Leaky_onset_units_exc, Leaky_units_exc, SNpc_Layer)


class Model:

    def __init__(self, parameters):

        self.parameters = parameters

        rng = np.random.RandomState(self.parameters.seed)

        """
        Layers set up
        """

        self.PPN = Leaky_onset_units_exc(
            self.parameters.N["PPN"],
            self.parameters.tau["PPN"][0],
            self.parameters.tau["PPN"][1],
            self.parameters.baseline["PPN"],
            rng,
            self.parameters.noise["PPN"],
        )

        self.LH = Leaky_onset_units_exc(
            self.parameters.N["LH"],
            self.parameters.tau["LH"][0],
            self.parameters.tau["LH"][1],
            self.parameters.baseline["LH"],
            rng,
            self.parameters.noise["LH"],
        )

        self.VTA = Leaky_units_exc(
            self.parameters.N["VTA"],
            self.parameters.tau["VTA"],
            self.parameters.baseline["VTA"],
            rng,
            self.parameters.noise["VTA"],
            self.parameters.threshold["VTA"],
        )

        self.BLA_IC = BLA_IC_Layer(
            self.parameters.N["BLA_IC"],
            self.parameters.tau["BLA_IC"][0],
            self.parameters.tau["BLA_IC"][1],
            self.parameters.baseline["BLA_IC"],
            rng,
            self.parameters.noise["BLA_IC"],
            self.parameters.BLA_Learn["eta_b"],
            self.parameters.BLA_Learn["tau_t"],
            self.parameters.BLA_Learn["alpha_t"],
            self.parameters.BLA_Learn["theta_DA"],
            self.parameters.BLA_Learn["max_W"],
        )

        self.SNpc = SNpc_Layer(
            self.parameters.N["SNpc"],
            self.parameters.tau["SNpc"],
            self.parameters.baseline["SNpc"],
            self.parameters.SNpc_W["SNpci_1_SNpco_1_W"],
            self.parameters.SNpc_W["SNpci_2_SNpco_2_W"],
            rng,
            self.parameters.noise["SNpc"],
            self.parameters.threshold["SNpc"],
        )

        self.BG_dl = BG_dl_Layer(
            self.parameters.N["BG_dl"],
            self.parameters.tau["BG_dl"],
            self.parameters.baseline["DLS"],
            self.parameters.baseline["STNdl"],
            self.parameters.baseline["GPi"],
            self.parameters.BG_dl_W["DLS_GPi_W"],
            self.parameters.BG_dl_W["STNdl_GPi_W"],
            rng,
            self.parameters.noise["BG_dl"],
            self.parameters.threshold["BG_dl"],
        )

        self.MGV = Leaky_units_exc(
            self.parameters.N["MGV"],
            self.parameters.tau["MGV"],
            self.parameters.baseline["MGV"],
            rng,
            self.parameters.noise["MGV"],
            self.parameters.threshold["MGV"],
        )

        self.MC = Leaky_units_exc(
            self.parameters.N["MC"],
            self.parameters.tau["MC"],
            self.parameters.baseline["MC"],
            rng,
            self.parameters.noise["MC"],
            self.parameters.threshold["MC"],
        )

        self.BG_dm = BG_dm_Layer(
            self.parameters.N["BG_dm"],
            self.parameters.tau["BG_dm"],
            self.parameters.baseline["DMS"],
            self.parameters.baseline["STNdm"],
            self.parameters.baseline["GPi_SNpr"],
            self.parameters.BG_dm_W["DMS_GPiSNpr_W"],
            self.parameters.BG_dm_W["STNdm_GPiSNpr_W"],
            rng,
            self.parameters.noise["BG_dm"],
            self.parameters.threshold["BG_dm"],
        )

        self.P = Leaky_units_exc(
            self.parameters.N["P"],
            self.parameters.tau["P"],
            self.parameters.baseline["P"],
            rng,
            self.parameters.noise["P"],
            self.parameters.threshold["P"],
        )

        self.PFCd_PPC = Leaky_units_exc(
            self.parameters.N["PFCd_PPC"],
            self.parameters.tau["PFCd_PPC"],
            self.parameters.baseline["PFCd_PPC"],
            rng,
            self.parameters.noise["PFCd_PPC"],
            self.parameters.threshold["PFCd_PPC"],
        )

        self.BG_v = BG_v_Layer(
            self.parameters.N["BG_v"],
            self.parameters.tau["BG_v"],
            self.parameters.baseline["NAc"],
            self.parameters.baseline["STNv"],
            self.parameters.baseline["SNpr"],
            self.parameters.BG_v_W["NAc_SNpr_W"],
            self.parameters.BG_v_W["STNv_SNpr_W"],
            rng,
            self.parameters.noise["BG_v"],
            self.parameters.threshold["BG_v"],
        )

        self.DM = Leaky_units_exc(
            self.parameters.N["DM"],
            self.parameters.tau["DM"],
            self.parameters.baseline["DM"],
            rng,
            self.parameters.noise["DM"],
            self.parameters.threshold["DM"],
        )

        self.PL = Leaky_units_exc(
            self.parameters.N["PL"],
            self.parameters.tau["PL"],
            self.parameters.baseline["PL"],
            rng,
            self.parameters.noise["PL"],
            self.parameters.threshold["PL"],
        )

        """
        Outputs pre at timestep n-1
        """

        self.PPN_output_pre = np.zeros(self.parameters.N["PPN"])

        self.LH_output_pre = np.zeros(self.parameters.N["LH"])

        self.VTA_output_pre = np.zeros(self.parameters.N["VTA"])

        self.BLA_IC_output_pre = np.zeros(self.parameters.N["BLA_IC"])

        self.SNpco_output_pre_1 = np.zeros(self.parameters.N["SNpc"])

        self.SNpco_output_pre_2 = np.zeros(self.parameters.N["SNpc"])

        self.BG_dl_output_pre = np.zeros(self.parameters.N["BG_dl"])

        self.DLS_output_pre = np.zeros(self.parameters.N["BG_dl"])

        self.MGV_output_pre = np.zeros(self.parameters.N["MGV"])

        self.MC_output_pre = np.zeros(self.parameters.N["MC"])

        self.BG_dm_output_pre = np.zeros(self.parameters.N["BG_dm"])

        self.DMS_output_pre = np.zeros(self.parameters.N["BG_dm"])

        self.P_output_pre = np.zeros(self.parameters.N["P"])

        self.PFCd_PPC_output_pre = np.zeros(self.parameters.N["PFCd_PPC"])

        self.BG_v_output_pre = np.zeros(self.parameters.N["BG_v"])

        self.NAc_output_pre = np.zeros(self.parameters.N["BG_v"])

        self.DM_output_pre = np.zeros(self.parameters.N["DM"])

        self.PL_output_pre = np.zeros(self.parameters.N["PL"])

        """
        Set Network's weights'
        """

        self.set_Ws()

    def reset_activity(self):
        """
        Reset layers' activity without setting matrices back to starting values
        """

        self.PPN.reset_activity()
        self.LH.reset_activity()
        self.VTA.reset_activity()
        self.BLA_IC.reset_activity()
        self.SNpc.reset_activity()
        self.BG_dl.reset_activity()
        self.MGV.reset_activity()
        self.MC.reset_activity()
        self.BG_dm.reset_activity()
        self.P.reset_activity()
        self.PFCd_PPC.reset_activity()
        self.BG_v.reset_activity()
        self.DM.reset_activity()
        self.PL.reset_activity()

    def set_Ws(self):
        """
        Matrices set up
        """

        self.Ws = {
            "inp_BLA_IC": np.array(
                [
                    [1.0 * self.parameters.Matrices_scalars['Mani_BLA_IC'], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0 * self.parameters.Matrices_scalars['Mani_BLA_IC'], 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0 * self.parameters.Matrices_scalars['Food_BLA_IC'], 0.0, -1.0 * self.parameters.Matrices_scalars['Sat_BLA_IC'], 0.0],
                    [0.0, 0.0, 0.0, 1.0 * self.parameters.Matrices_scalars['Food_BLA_IC'], 0.0, -1.0 * self.parameters.Matrices_scalars['Sat_BLA_IC']],
                ]
            ),
            "Mani_DLS": np.array(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            )
            * self.parameters.Matrices_scalars["Mani_DLS"],
            "Mani_DMS": np.array(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            )
            * self.parameters.Matrices_scalars["Mani_DMS"],
            "Food_PPN": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            * self.parameters.Matrices_scalars["Food_PPN"],
            "Food_LH": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            * self.parameters.Matrices_scalars["Food_LH"],
            "PPN_SNpco": np.array([1.0]) * self.parameters.Matrices_scalars["PPN_SNpco"],
            "BLA_IC_NAc": np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
            * self.parameters.Matrices_scalars["BLA_IC_NAc"],
            "BLA_IC_LH": np.array([0.0, 0.0, 1.0, 1.0])
            * self.parameters.Matrices_scalars["BLA_IC_LH"],
            "LH_VTA": np.array([1.0]) * self.parameters.Matrices_scalars["LH_VTA"],
            "NAc_SNpci_1": np.eye(self.parameters.N["SNpc"])
            * self.parameters.Matrices_scalars["NAc_SNpci_1"],
            "DMS_SNpci_2": np.eye(self.parameters.N["SNpc"])
            * self.parameters.Matrices_scalars["DMS_SNpci_2"],
            "GPi_MGV": np.eye(self.parameters.N["MGV"])
            * self.parameters.Matrices_scalars["GPi_MGV"],
            "GPi_SNpr_P": np.eye(self.parameters.N["P"])
            * self.parameters.Matrices_scalars["GPi_SNpr_P"],
            "SNpr_DM": np.eye(self.parameters.N["DM"])
            * self.parameters.Matrices_scalars["SNpr_DM"],
            "MGV_MC": np.eye(self.parameters.N["MC"]) * self.parameters.Matrices_scalars["MGV_MC"],
            "P_PFCd_PPC": np.eye(self.parameters.N["PFCd_PPC"])
            * self.parameters.Matrices_scalars["P_PFCd_PPC"],
            "DM_PL": np.eye(self.parameters.N["PL"]) * self.parameters.Matrices_scalars["DM_PL"],
            "PL_DM": np.eye(self.parameters.N["DM"]) * self.parameters.Matrices_scalars["PL_DM"],
            "PL_NAc": np.eye(self.parameters.N["BG_v"])
            * self.parameters.Matrices_scalars["PL_NAc"],
            "PL_STNv": np.eye(self.parameters.N["BG_v"])
            * self.parameters.Matrices_scalars["PL_STNv"],
            "PL_PFCd_PPC": np.eye(self.parameters.N["PFCd_PPC"])
            * self.parameters.Matrices_scalars["PL_PFCd_PPC"],
            "PFCd_PPC_P": np.eye(self.parameters.N["P"])
            * self.parameters.Matrices_scalars["PFCd_PPC_P"],
            "PFCd_PPC_DMS": np.eye(self.parameters.N["BG_dm"])
            * self.parameters.Matrices_scalars["PFCd_PPC_DMS"],
            "PFCd_PPC_STNdm": np.eye(self.parameters.N["BG_dm"])
            * self.parameters.Matrices_scalars["PFCd_PPC_STNdm"],
            "PFCd_PPC_PL": np.eye(self.parameters.N["PL"])
            * self.parameters.Matrices_scalars["PFCd_PPC_PL"],
            "PFCd_PPC_MC": np.eye(self.parameters.N["MC"])
            * self.parameters.Matrices_scalars["PFCd_PPC_MC"],
            "MC_MGV": np.eye(self.parameters.N["MGV"]) * self.parameters.Matrices_scalars["MC_MGV"],
            "MC_DLS": np.eye(self.parameters.N["BG_dl"])
            * self.parameters.Matrices_scalars["MC_DLS"],
            "MC_STNdl": np.eye(self.parameters.N["BG_dl"])
            * self.parameters.Matrices_scalars["MC_STNdl"],
            "MC_PFCd_PPC": np.eye(self.parameters.N["PFCd_PPC"])
            * self.parameters.Matrices_scalars["MC_PFCd_PPC"],
        }

        """
        Masks used for matrices learning in order to avoid unwanted connections' learning
        """

        self.Ws_learn_masks = {
            "Mani_DLS": np.array(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            "Mani_DMS": np.array(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            "BLA_IC_NAc": np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
        }

    def delta_Str_learn_USV(
        self,
        eta_str,
        DA,
        v_str,
        v_inp,
        theta_DA_str,
        theta_str,
        theta_inp_str,
        mask,
        max_W_str,
        W,
    ):

        DA_term = np.maximum(0, DA - theta_DA_str)[:, None]
        delta_W_inp_str = (
            eta_str
            * DA_term
            * np.outer(
                np.maximum(0, v_str - theta_str), np.maximum(0, v_inp - theta_inp_str)
            )
            * (max_W_str - W)
        )

        delta_W_inp_str *= mask

        return delta_W_inp_str

    def delta_Str_learn_SV(
        self,
        eta_str,
        DA,
        v_str,
        v_inp,
        theta_DA_str,
        theta_str,
        theta_inp_str,
        mask,
        max_W_str,
        W,
    ):

        DA_term = np.maximum(0, DA - theta_DA_str)
        DA_term = np.clip(DA_term, 0, 1e6)
        DA_term = np.nan_to_num(DA_term)
        DA_term = DA_term[:, None]

        post = np.maximum(0, v_str - theta_str)
        pre = np.maximum(0, v_inp - theta_inp_str)

        post = np.clip(post, 0, 1e6)
        pre = np.clip(pre, 0, 1e6)

        post = np.nan_to_num(post)
        pre = np.nan_to_num(pre)

        hebb = np.outer(post, pre)
        hebb = np.nan_to_num(hebb)

        w_diff = max_W_str - W
        w_diff = np.nan_to_num(w_diff)
        w_diff = np.clip(w_diff, -1e6, 1e6)

        delta_W = eta_str * DA_term * hebb * w_diff

        delta_W = np.nan_to_num(delta_W)
        delta_W *= mask

        return delta_W

    def learning(self, _input_):
        """
        Set learning for internal layers' and evironmental matrices
        """

        self.BLA_IC.learn(self.VTA_output_pre)

        delta_W_BLA_IC_NAc = self.delta_Str_learn_USV(
            self.parameters.Str_Learn["eta_NAc"],
            self.VTA_output_pre,
            self.NAc_output_pre * -1,
            self.BLA_IC_output_pre,
            self.parameters.Str_Learn["theta_DA_NAc"],
            self.parameters.Str_Learn["theta_NAc"],
            self.parameters.Str_Learn["theta_inp_NAc"],
            self.Ws_learn_masks["BLA_IC_NAc"],
            self.parameters.Str_Learn["max_W_NAc"],
            self.Ws["BLA_IC_NAc"],
        )
        self.Ws["BLA_IC_NAc"] += delta_W_BLA_IC_NAc

        delta_W_Mani_DMS = self.delta_Str_learn_USV(
            self.parameters.Str_Learn["eta_DMS"],
            self.SNpco_output_pre_1,
            self.DMS_output_pre * -1,
            _input_,
            self.parameters.Str_Learn["theta_DA_DMS"],
            self.parameters.Str_Learn["theta_DMS"],
            self.parameters.Str_Learn["theta_inp_DMS"],
            self.Ws_learn_masks["Mani_DMS"],
            self.parameters.Str_Learn["max_W_DMS"],
            self.Ws["Mani_DMS"],
        )
        self.Ws["Mani_DMS"] += delta_W_Mani_DMS

        delta_W_Mani_DLS = self.delta_Str_learn_USV(
            self.parameters.Str_Learn["eta_DLS"],
            self.SNpco_output_pre_2,
            self.DLS_output_pre * -1,
            _input_,
            self.parameters.Str_Learn["theta_DA_DLS"],
            self.parameters.Str_Learn["theta_DLS"],
            self.parameters.Str_Learn["theta_inp_DLS"],
            self.Ws_learn_masks["Mani_DLS"],
            self.parameters.Str_Learn["max_W_DLS"],
            self.Ws["Mani_DLS"],
        )
        self.Ws["Mani_DLS"] += delta_W_Mani_DLS

    def update_output_pre(self):
        """
        Update ouput_pre values based on current values
        """

        self.PPN_output_pre = self.PPN.output.copy()

        self.LH_output_pre = self.LH.output.copy()

        self.VTA_output_pre = self.VTA.output.copy()

        self.BLA_IC_output_pre = self.BLA_IC.output.copy()

        self.SNpco_output_pre_1 = self.SNpc.output_1.copy()

        self.SNpco_output_pre_2 = self.SNpc.output_2.copy()

        self.BG_dl_output_pre = self.BG_dl.output_BG_dl.copy()

        self.DLS_output_pre = self.BG_dl.DLS.output.copy()

        self.MGV_output_pre = self.MGV.output.copy()

        self.MC_output_pre = self.MC.output.copy()

        self.BG_dm_output_pre = self.BG_dm.output_BG_dm.copy()

        self.DMS_output_pre = self.BG_dm.DMS.output.copy()

        self.P_output_pre = self.P.output.copy()

        self.PFCd_PPC_output_pre = self.PFCd_PPC.output.copy()

        self.BG_v_output_pre = self.BG_v.output_BG_v.copy()

        self.NAc_output_pre = self.BG_v.NAc.output.copy()

        self.DM_output_pre = self.DM.output.copy()

        self.PL_output_pre = self.PL.output.copy()

    def step(self, _input_, learning=True):
        """
        Compute step for each layer using the output_pre values
        """
        _input_ = np.asanyarray(_input_)

        self.BLA_IC.step(np.dot(self.Ws["inp_BLA_IC"], _input_))

        self.BG_dm.step(
            (
                self.parameters.DA_values["Y_DMS"]
                + (self.parameters.DA_values["delta_DMS"] * self.SNpco_output_pre_1)
            )
            * np.dot(self.Ws["Mani_DMS"], _input_),
            np.dot(self.Ws["PFCd_PPC_DMS"], self.PFCd_PPC_output_pre),
            np.dot(self.Ws["PFCd_PPC_STNdm"], self.PFCd_PPC_output_pre),
        )

        self.BG_dl.step(
            (
                self.parameters.DA_values["Y_DLS"]
                + (self.parameters.DA_values["delta_DLS"] * self.SNpco_output_pre_2)
            )
            * np.dot(self.Ws["Mani_DLS"], _input_),
            np.dot(self.Ws["MC_DLS"], self.MC_output_pre),
            np.dot(self.Ws["MC_STNdl"], self.MC_output_pre),
        )

        self.PPN.step(np.dot(self.Ws["Food_PPN"], _input_))

        self.LH.step(
            np.dot(self.Ws["Food_LH"], _input_)
            + np.dot(self.Ws["BLA_IC_LH"], self.BLA_IC_output_pre)
        )

        self.VTA.step(np.dot(self.Ws["LH_VTA"], self.LH_output_pre))

        self.BG_v.step(
            (
                self.parameters.DA_values["Y_NAc"]
                + (self.parameters.DA_values["delta_NAc"] * self.VTA_output_pre)
            )
            * np.dot(self.Ws["BLA_IC_NAc"], self.BLA_IC_output_pre),
            np.dot(self.Ws["PL_NAc"], self.PL_output_pre),
            np.dot(self.Ws["PL_STNv"], self.PL_output_pre),
        )

        self.SNpc.step(
            np.dot(self.Ws["NAc_SNpci_1"], self.NAc_output_pre),
            np.dot(self.Ws["DMS_SNpci_2"], self.DMS_output_pre),
            np.dot(self.Ws["PPN_SNpco"], self.PPN_output_pre),
        )

        self.DM.step(
            np.dot(self.Ws["SNpr_DM"], self.BG_v_output_pre)
            + np.dot(self.Ws["PL_DM"], self.PL_output_pre)
        )

        self.P.step(
            np.dot(self.Ws["GPi_SNpr_P"], self.BG_dm_output_pre)
            + np.dot(self.Ws["PFCd_PPC_P"], self.PFCd_PPC_output_pre)
        )

        self.MGV.step(
            np.dot(self.Ws["GPi_MGV"], self.BG_dl_output_pre)
            + np.dot(self.Ws["MC_MGV"], self.MC_output_pre)
        )

        self.PL.step(
            np.dot(self.Ws["DM_PL"], self.DM_output_pre)
            + np.dot(self.Ws["PFCd_PPC_PL"], self.PFCd_PPC_output_pre)
        )

        self.PFCd_PPC.step(
            np.dot(self.Ws["P_PFCd_PPC"], self.P_output_pre)
            + np.dot(self.Ws["PL_PFCd_PPC"], self.PL_output_pre)
            + np.dot(self.Ws["MC_PFCd_PPC"], self.MC_output_pre)
        )

        self.MC.step(
            np.dot(self.Ws["MGV_MC"], self.MGV_output_pre)
            + np.dot(self.Ws["PFCd_PPC_MC"], self.PFCd_PPC_output_pre)
        )

        if learning:
            self.learning( _input_)

        self.update_output_pre()
