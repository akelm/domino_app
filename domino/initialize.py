import logging
from itertools import accumulate

import numpy as np
import pandas as pd

from .calc_mappings import CurrentState, strategies_str, strat_translate, strat_to_ind
from .parameters import KType
from .payoff import payoff_table
logger = logging.getLogger('custom')


def initialize(params):
    def log_initialize():
        logger.debug("\n\n")
        logger.debug(" INITIALIZE ".center(80, "*"))
        logger.debug("\n")
        logger.debug(" STATE ARRAY ".center(80, "#"))
        logger.debug(pd.DataFrame(state_array).to_string(index=False, header=False))
        logger.debug(" STRATEGY ARRAY ".center(80, "#"))
        logger.debug(pd.DataFrame(strat_translate(strat_array)).to_string(index=False, header=False))

    if params.ca_state is not None:
        state_array = params.ca_state
    else:
        state_array: np.ndarray = np.random.uniform(0, 1, params.mrows * params.ncols).reshape(
            [params.mrows, params.ncols])
        mask = state_array <= params.p_init_c
        state_array[mask] = 1
        state_array[np.logical_not(mask)] = 0

    state_array = state_array.astype(int)
    if params.ca_strat is not None:
        strat_array = strat_to_ind(params.ca_strat)
    else:
        if params.k_change == KType.constant:
            # k const
            steps: list = list(accumulate([0, params.all_c, params.all_d, params.k_d, params.k_c, params.k_dc]))
            labels = ['allC', 'allD', '%dD' % params.k_const, '%dC' % params.k_const, '%dDC' % params.k_const]
        else:
            # k var
            k_spread = params.k_var_1 - params.k_var_0 + 1
            steps: list = list(accumulate([0, params.all_c, params.all_d] + [params.k_d / k_spread] * k_spread + [
                params.k_c / k_spread] * k_spread + [params.k_dc / k_spread] * k_spread))
            spread_range = range(params.k_var_0, params.k_var_1 + 1)
            labels = ['allC', 'allD'] + ['%dD' % k for k in spread_range] + \
                     ['%dC' % k for k in spread_range] + \
                     ['%dDC' % k for k in spread_range]

        steps[0] -= np.finfo(float).eps
        strat_array = np.random.uniform(0, steps[-1], state_array.size).reshape(state_array.shape)
        for lower, higher, label in zip(steps[:-1], steps[1:], labels):
            strat_array[(lower < strat_array) & (strat_array <= higher)] = strategies_str.index(label)
        strat_array = strat_array.astype(int)

    if params.log_to_debug:
        log_initialize()

    payoff_array = payoff_table(state_array, params)

    return CurrentState(states=state_array, strategies=strat_array, payoff=payoff_array)
