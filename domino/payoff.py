import logging

import numpy as np
import pandas as pd

from .calc_mappings import pattern_d2a, pattern_d2b, pattern_d4, neigh_list
from .sliding_window import sliding_window_view


def payoff_table(state_arr, params):
    def log_payoff():
        view: np.ndarray = sliding_window_view(np.pad(state_arr, 1), (3, 3)).reshape(
            [-1, 3, 3])
        df = pd.DataFrame({
            "idx": np.arange(state_arr.size) + 1,
            "i": np.indices(state_arr.shape)[0].reshape(-1) + 1,
            "j": np.indices(state_arr.shape)[1].reshape(-1) + 1,
            "state": state_arr.flat,
            "neigh": list(map(''.join, view[:, neigh_list[0], neigh_list[1]].astype(int).astype(str))),
            "num1": view[:, neigh_list[0], neigh_list[1]].sum(1),
            "corr1pat1": "-",
            "pay1": "-",
            "corr1pat2": "-",
            "pay2": "-",
            "corr1pat3": "-",
            "pay3": "-",
            "cum_pay": payoff_array.flat
        })
        for ind, (pay, ones) in enumerate(zip(payoff_d_list, ones_d_list), start=1):
            col_pay = "pay%d" % ind
            col_ones = "corr1pat%d" % ind
            df[col_pay][d_inds] = pay
            df[col_ones][d_inds] = ones

        logging.custom(" CALCULATE PAYOFF ".center(80, "#"))
        logging.custom(df.to_string(index=False))

    def log_payoff_table():
        logging.custom(" PAYOFF ARRAY ".center(80, "#"))
        logging.custom(pd.DataFrame(payoff_array).to_string(index=False, header=False))

    view: np.ndarray = sliding_window_view(np.pad(state_arr, 1), (3, 3)).reshape(
        [-1, 3, 3])
    d_inds, = np.nonzero(view[:, 1, 1] == 0)
    c_inds, = np.nonzero(view[:, 1, 1])

    payoff_d_list = []
    ones_d_list = []
    for pattern in (pattern_d2a, pattern_d2b, pattern_d4):
        num_correct_d = np.logical_not(
            np.logical_or(view[d_inds, :, :], pattern)
        ).sum(axis=1).sum(axis=1) - 1
        num_incorrect_d = np.logical_and(np.logical_not(view[d_inds, :, :]), pattern).sum(axis=1).sum(axis=1)
        num_correct_c = np.logical_and(view[d_inds, :, :], pattern).sum(axis=1).sum(axis=1)
        num_incorrect_c = 8 - num_correct_d - num_incorrect_d - num_correct_c
        payoff = params.dd_reward * num_correct_d + params.dd_penalty * num_incorrect_d + \
                 params.dc_reward * num_correct_c + params.dc_penalty * num_incorrect_c
        if params.if_special_penalty:
            ind_of_0 = view[d_inds, :, :].sum(axis=1).sum(axis=1) == 0
            payoff[ind_of_0] = params.special_penalty
        payoff_d_list.append(payoff)
        ones_d_list.append(num_correct_c)

    num_correct_d_for_c = np.logical_not(view[c_inds, :, :]).sum(axis=1).sum(axis=1)
    payoff_array = np.zeros(view.shape[0])
    payoff_array[c_inds] = (params.cd_reward - params.cc_penalty) * num_correct_d_for_c + params.cc_penalty * 8
    payoff_array[d_inds] = np.maximum.reduce(payoff_d_list)
    payoff_array = payoff_array.reshape(state_arr.shape)
    if params.sharing:
        view: np.ndarray = sliding_window_view(np.pad(payoff_array, 1), (3, 3)).reshape(
            [-1, 9])
        payoff_array = view.mean(axis=1).reshape(state_arr.shape)

    log_payoff()
    log_payoff_table()
    return payoff_array
