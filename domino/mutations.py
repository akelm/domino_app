import logging
from functools import partial

import numpy as np
import pandas as pd

from . import calc_mappings
from .calc_mappings import strategy_mutation_dict
from .sliding_window import sliding_window_view


def mutation(state_arr: np.ndarray, strat_arr, params):
    def log_mutation():
        df = pd.DataFrame({
            "i": np.indices(state_arr.shape).reshape([-1, 2])[:, 0] + 1,
            "j": np.indices(state_arr.shape).reshape([-1, 2])[:, 1] + 1,
            "state": state_arr.flat,
            "state_mut": state_mutated.flat,
            "neigh": list(map(''.join, view[:, calc_mappings.neigh_list_flat].astype(int).astype(str))),
            "num1": view[:, calc_mappings.neigh_list_flat].sum(1),
            "0neigh_mut": state_zeros_mutated.flat,
            "strat": calc_mappings.strat_translate(strat_arr.flat),
            "strat_mut": calc_mappings.strat_translate(new_strat_arr.flat),
        })

        logging.custom(" STATE MUTATION ".center(80, "#"))
        logging.custom(df.to_string(index=False))

    def log_strategy():
        logging.custom(" MUTATED STRATEGY ".center(80, "#"))
        logging.custom(pd.DataFrame(calc_mappings.strat_translate(new_strat_arr)).to_string(index=False, header=False))

    def log_state():
        logging.custom(" MUTATED STATE ".center(80, "#"))
        logging.custom(pd.DataFrame(state_zeros_mutated).to_string(index=False, header=False))

    # normal state mutation
    prob_arr: np.ndarray = calc_mappings.rng.uniform(0, 1, state_arr.size).reshape(state_arr.shape)
    state_mutated = state_arr.copy()
    state_mutated[prob_arr < params.p_state_mut] = np.logical_not(state_mutated[prob_arr < params.p_state_mut])
    # mutation for 0-00000000 neigh
    view: np.ndarray = sliding_window_view(
        np.pad(state_mutated, 1, constant_values=0), (3, 3)).reshape(
        [-1, 9])
    # all 0  mutation
    all_zeros = (view.sum(axis=0) == 0)
    state_zeros_mutated = state_mutated.copy()
    state_zeros_mutated.flat[all_zeros] = calc_mappings.rng.uniform(0, 1, all_zeros.size) < params.p_0_neigh
    # all 1  mutation
    p_1_neigh = 0
    all_ones = (view.sum(axis=0) == 9)
    state_ones_mutated = state_zeros_mutated.copy()
    state_ones_mutated.flat[all_ones] = calc_mappings.rng.uniform(0, 1, all_zeros.size) < p_1_neigh

    prob_arr: np.ndarray = np.random.uniform(0, 1, strat_arr.size).reshape(strat_arr.shape)
    mask = prob_arr < params.p_strat_mut
    rand_choice = partial(np.random.choice, size=1)
    new_strat_arr = strat_arr.copy()
    new_strat_arr[mask].flat = list(map(rand_choice, map(strategy_mutation_dict.get, new_strat_arr[mask].flat)))
    log_mutation()
    log_state()
    log_strategy()
    return state_ones_mutated, new_strat_arr
