from domino.ca_gui import logging
import string
from enum import Enum, auto

from dataclasses import dataclass, field, InitVar, asdict, make_dataclass, fields
from datetime import datetime

import numpy as np

from domino.add_log_level import addLoggingLevel
from domino.calc_mappings import debug_loc


class CompetitionType(Enum):
    tournament = 0
    proportional = 1
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

class KType(Enum):
    constant = 0
    variable = 1
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

@dataclass
class Parameters:
    mrows: int = 20
    ncols: int = 20
    p_init_c: float = 0.5
    sharing: int = False
    competition_type: CompetitionType = CompetitionType.tournament
    p_state_mut: float = 0
    p_strat_mut: float = 0
    p_0_neigh: float = 0
    num_of_iter: int = 500
    num_of_exper: int = 100
    if_seed: int = False
    seed: int = None
    dd_penalty: float = 0
    dc_penalty: float = 0
    dd_reward: float = 0
    dc_reward: float = 0
    cd_reward: float = 0
    cc_penalty: float = 0
    if_special_penalty: int = True
    special_penalty: float = 0

    all_c: float = 0.2
    all_d: float = 0.2
    k_d: float = 0.2
    k_c: float = 0.2
    k_dc: float = 0.2
    k_change: KType = KType.constant
    k_const: int = 4
    k_var_0: int = 0
    k_var_1: int = 8

    species: int = 1
    synchronization: float = 1

    log_to_debug: int = False
    load_init_files: int = False
    ca_state: np.ndarray = field(default=None, init=False)
    ca_strat: np.ndarray = field(default=None, init=False)

    state_filename: InitVar[str] = None
    strat_filename: InitVar[str] = None

    min_payoff: float = field(default=0, init=False)

    def __post_init__(self, state_filename=None, strat_filename=None):
        if all((self.load_init_files, state_filename is not None, strat_filename is not None)):
            self.ca_state = np.loadtxt(state_filename, dtype=int)
            self.ca_strat = np.loadtxt(strat_filename, dtype=str)

            self.ca_strat = np.char.upper(self.ca_strat)
            self.ca_strat = np.char.translate(self.ca_strat, "*", deletechars=string.whitespace + string.punctuation)
            if self.ca_strat.shape != self.ca_state.shape:
                raise Exception("ca_state and ca_strat have diff size.")
            self.mrows, self.ncols = self.ca_state.shape
        # try:
        #     addLoggingLevel("custom", 49, methodName="custom")
        # except:
        #     pass
        # if not log_to_debug:
        #     logging.getLogger('custom').disabled = True
        # else:
        #     logging.basicConfig()
        #     logging.basicConfig(filename=debug_loc, level="custom", format="%(message)s", filemode='w')
        #     # logging.getLogger('custom').disabled = False
        # if self.log_to_debug:
        #     logger = logging.getLogger('custom')
        #     fh = logging.FileHandler(debug_loc)
        #     fh.setLevel(logging.DEBUG)
        #     # create formatter and add it to the handlers
        #     formatter = logging.Formatter("%(message)s")
        #     fh.setFormatter(formatter)
        #     # add the handlers to logger
        #     logger.addHandler(fh)
        self.min_payoff = min(self.dd_penalty,
                              self.dc_penalty,
                              self.dd_reward,
                              self.dc_reward,
                              self.cd_reward,
                              self.cc_penalty) * 8
        if self.if_special_penalty:
            self.min_payoff = min(self.min_payoff, self.special_penalty)

        if not self.if_seed:
            self.seed = int(datetime.now().timestamp())

        self.mrows = int(self.mrows)
        self.ncols = int(self.ncols)
        self.sharing = int(self.sharing)
        self.competition_type = CompetitionType(self.competition_type)
        self.num_of_iter = int(self.num_of_iter)
        self.num_of_exper = int(self.num_of_exper)
        self.if_seed = int(self.if_seed)
        self.seed = abs(int(self.seed)) if self.seed is not None else self.seed
        self.if_special_penalty = int(self.if_special_penalty)
        self.k_change = KType(self.k_change)
        self.k_const = int(self.k_const)
        self.k_var_0 = int(self.k_var_0)
        self.k_var_1 = int(self.k_var_1)
        self.species = int(self.species)
        self.synchronization = int(self.synchronization)
        self.load_init_files = int(self.load_init_files)

    def freeze(self):
        return FrozenParameters(**asdict(self))


FrozenParameters = make_dataclass("FrozenParameters", [(f.name, f.type) for f in fields(Parameters)], frozen=True)
