import string
from dataclasses import dataclass, field, InitVar, asdict, make_dataclass, fields
from datetime import datetime

import numpy as np


class CompetitionType:
    proportional = 0
    tournament = 1


class KType:
    constant = 0
    variable = 1


@dataclass
class Parameters:
    mrows: int = 20
    ncols: int = 20
    p_init_c: float = 0.5
    sharing: int = False
    competition_type: int = CompetitionType.tournament
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
    k_change: int = KType.constant
    k_const: int = 4
    k_var_0: int = 0
    k_var_1: int = 7

    species: int = 1
    synchronization: float = 1

    debug: int = False
    ca_state: np.ndarray = field(default=None, init=False)
    ca_strat: np.ndarray = field(default=None, init=False)

    state_filename: InitVar[str] = None
    strat_filename: InitVar[str] = None

    min_payoff: float = field(default=0, init=False)

    def __post_init__(self, state_filename=None, strat_filename=None):
        if all((self.debug, state_filename is not None, strat_filename is not None)):
            self.ca_state = np.loadtxt(state_filename, dtype=int)
            self.ca_strat = np.loadtxt(strat_filename, dtype=str)

            self.ca_strat = np.char.upper(self.ca_strat)
            self.ca_strat = np.char.translate(self.ca_strat, "*", deletechars=string.whitespace + string.punctuation)
            if self.ca_strat.shape != self.ca_state.shape:
                raise Exception("ca_state and ca_strat have diff size.")
            self.mrows, self.ncols = self.ca_state.shape

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
        self.p_init_c = int(self.p_init_c)
        self.sharing = int(self.sharing)
        self.competition_type = int(self.competition_type)
        self.num_of_iter = int(self.num_of_iter)
        self.num_of_exper = int(self.num_of_exper)
        self.if_seed = int(self.if_seed)
        self.seed = abs(int(self.seed)) if self.seed is not None else self.seed
        self.if_special_penalty = int(self.if_special_penalty)
        self.k_change = int(self.k_change)
        self.k_const = int(self.k_const)
        self.k_var_0 = int(self.k_var_0)
        self.k_var_1 = int(self.k_var_1)
        self.species = int(self.species)
        self.synchronization = int(self.synchronization)
        self.debug = int(self.debug)

    def freeze(self):
        return FrozenParameters(**asdict(self))


FrozenParameters = make_dataclass("FrozenParameters", [(f.name, f.type) for f in fields(Parameters)], frozen=True)
