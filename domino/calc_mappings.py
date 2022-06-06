import os
from collections import namedtuple
from functools import partial
from typing import List

import numpy as np


def k_d(k, num_c):
    return 1 if (8 - num_c) <= k else 0


def k_c(k, num_c):
    return 1 if num_c <= k else 0


def k_dc(k, num_c):
    return 1 if (8 - num_c) >= k else 0


def all_d(_):
    return 0


def all_c(_):
    return 1


neigh_list = [[0, 0, 1, 2, 2, 2, 1, 0], [1, 2, 2, 2, 1, 0, 0, 0]]
neigh_list_flat = [8, 1, 2, 7, 0, 3, 6, 5, 4]
neigh_flat_translate = np.vectorize(neigh_list_flat.__getitem__)
neigh_flat_reverse = np.vectorize(neigh_list_flat.index)

strategies_str: List[str]
strategies_str, strategies_fun = zip(
    *(((str(k) + key), partial(fun, k)) for key, fun in zip(('D', 'C', 'DC'), (k_d, k_c, k_dc)) for k in range(8)))
strategies_str += ('allD', 'allC')
strategies_str_lower = tuple(map(str.lower, strategies_str))
strategies_fun += (all_d, all_c)

strat_translate = np.vectorize(strategies_str.__getitem__)
strat_to_ind = np.vectorize(lambda x: strategies_str_lower.index(x.lower()))

strategy_mutation_str = {
    'allC': ['allD'],
    'allD': ['allC']
}
strategy_mutation_str.update({str(k) + key: ['allC', 'allD'] for k, key in enumerate(('D',) * 8)})
strategy_mutation_str.update(
    {str(k) + key: ['allC', 'allD', str(k) + 'D', str(k) + 'DC'] for k, key in enumerate(('C',) * 8)})
strategy_mutation_str.update(
    {str(k) + key: ['allC', 'allD', str(k) + 'D', str(k) + 'C'] for k, key in enumerate(('DC',) * 8)})

strategy_mutation_dict = {strategies_str.index(key): list(map(strategies_str.index, value)) for key, value in
                          strategy_mutation_str.items()}

CurrentState = namedtuple("CurrentState", {"states", "strategies", "payoff"})

pattern_c = np.array((
    (0, 0, 0),
    (0, 1, 0),
    (0, 0, 0)
))

pattern_d2a = np.array((
    (0, 0, 0),
    (1, 0, 1),
    (0, 0, 0)
))

pattern_d2b = np.array((
    (0, 1, 0),
    (0, 0, 0),
    (0, 1, 0)
))

pattern_d4 = np.array((
    (1, 0, 1),
    (0, 0, 0),
    (1, 0, 1)
))

rng: np.random.Generator

import inspect
stack = inspect.stack()
lowest_py = ''
top_pyw = []
for frame in stack:
    ext = os.path.splitext(frame.filename)[1]
    if ext.lower() == ".py" and "domino_app" in ext.lower():
        lowest_py = frame.filename
    if ext.lower() == ".pyw" and "domino_app" in ext.lower():
        top_pyw = frame.filename

res_dir = os.path.dirname(top_pyw[0] if top_pyw else lowest_py)
img_file_pattern = os.path.join(res_dir, 'img' , "exp_%d_%s_%d.png")
img_file_labels = ('state', 'strategy', "payoff")
results_loc = os.path.join(res_dir, "results.txt")
debug_loc = os.path.join(res_dir, "debug.txt")
std_results_loc = os.path.join(res_dir, "std_results.txt")
state_filename = os.path.join(res_dir, "CA_STATE.txt")
strat_filename = os.path.join(res_dir, "CA_STRATEGIES.txt")

