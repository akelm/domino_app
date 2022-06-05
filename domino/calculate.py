import logging
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict
from pprint import pformat

from .calc_mappings import strat_filename, state_filename
from .add_log_level import addLoggingLevel
from .sliding_window import sliding_window_view

try:
    addLoggingLevel("custom", 51, methodName="custom")
    logging.basicConfig(filename='debug.txt', level="custom", format="%(message)s", filemode='w')
except AttributeError:
    pass

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

np.set_printoptions(linewidth=sys.maxsize, threshold=sys.maxsize)

from . import calc_mappings

from .competition import change_strategy
from .initialize import initialize
from .mutations import mutation
from .parameters import Parameters, CompetitionType, KType
from .payoff import payoff_table

from .plotting import history_to_img
from .statistics import statistics_single, multirun_statistics


def log_iter(i: int):
    logging.custom("\n\n")
    logging.custom((" ITERATION %d " % (i + 1)).center(80, "*"))
    logging.custom("\n")


def log_exper(i: int):
    logging.custom("\n\n")
    logging.custom((" EXPERIMENT %d " % (i + 1)).center(80, "*"))
    logging.custom("\n")


def calc(params: Parameters):
    def log_params():
        logging.custom(" PARAMETERS ".center(80, "#"))
        logging.custom("\n")
        logging.custom(pformat(asdict(params)))

    log_params()

    calc_mappings.rng = np.random.default_rng(params.seed)
    stats = []
    shutil.rmtree('img', ignore_errors=True)
    os.makedirs("img", exist_ok=True)
    shutil.rmtree(calc_mappings.results_loc, ignore_errors=True)
    shutil.rmtree(calc_mappings.std_results_loc, ignore_errors=True)
    for exp_num in range(params.num_of_exper):
        log_exper(exp_num)
        current = initialize(params)
        history = [deepcopy(current)]
        for iteration in range(params.num_of_iter):
            log_iter(iteration)
            logging.debug("iteration %d in calc.py" % iteration)
            current = iterate(current, params)
            history.append(deepcopy(current))
        logging.debug("finished iterations in calc.py")
        stats.append(statistics_single(history))
        logging.debug("statistics finished in calc.py")
        history_to_img(history, exp_num)
        logging.debug("plotting finished in calc.py")
    multirun_statistics(stats)


def new_state_array(state_arr, strat_arr):
    def log_new_states():
        df = pd.DataFrame({
            "idx": np.arange(state_arr.size) + 1,
            "i": np.indices(state_arr.shape)[0].reshape(-1) + 1,
            "j": np.indices(state_arr.shape)[1].reshape(-1) + 1,
            "state": state_arr.flat,
            "strat": calc_mappings.strat_translate(strat_arr.flat),
            "num1": num_c,
            "num0": 8 - num_c,
            "new_state": new_states.flat
        })
        logging.custom(" CHANGE STATE ".center(80, "#"))
        logging.custom(df.to_string(index=False))

    def log_state_arr():
        logging.custom(" STATE ARRAY ".center(80, "#"))
        logging.custom(pd.DataFrame(new_states).to_string(index=False, header=False))

    view: np.ndarray = sliding_window_view(np.pad(state_arr, 1), (3, 3)).reshape(
        [-1, 3, 3])
    num_c = np.logical_and(np.logical_not(calc_mappings.pattern_c), view).sum(axis=(1, 2))
    function_mapping = map(calc_mappings.strategies_fun.__getitem__, strat_arr.reshape([-1]))
    new_states = np.array([f(x) for f, x in zip(function_mapping, num_c)]).reshape(state_arr.shape)

    log_new_states()
    log_state_arr()
    return new_states


def iterate(current, params):
    # change state
    new_states = new_state_array(current.states, current.strategies)

    # payoff  with opt sharing
    new_payoff = payoff_table(new_states, params)

    # change strategies with competition
    new_strat = change_strategy(new_payoff, current.strategies, params.synchronization, params.competition_type,
                                params.min_payoff)
    # mutate
    new_states, new_strat = mutation(new_states, new_strat, params)
    # new_strat = strategy_mutation(new_strat, params.p_strat_mut)

    new_current = calc_mappings.CurrentState(payoff=new_payoff, strategies=new_strat, states=new_states)

    return new_current


if __name__ == "__main__":
    params = Parameters(mrows=5, ncols=5, p_init_c=0.5, sharing=False, competition_type=CompetitionType.tournament,
                        p_state_mut=0,
                        p_strat_mut=0, p_0_neigh=0, num_of_iter=2, num_of_exper=1, seed=None, dd_penalty=0,
                        dc_penalty=0, dd_reward=1, dc_reward=1, cd_reward=1, cc_penalty=0, if_special_penalty=True,
                        special_penalty=-10, all_c=0.1,
                        all_d=0.1, k_d=0.3, k_c=0.3, k_dc=0.3, k_change=KType.constant, k_const=4, k_var_0=0, k_var_1=7,
                        species=0,
                        synchronization=1.0, debug=False, state_filename=state_filename,
                        strat_filename=strat_filename)
    calc(params)
