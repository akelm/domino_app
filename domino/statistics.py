from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from dataclasses import asdict

from . import calc_mappings
from .calc_mappings import CurrentState, strategies_str, pattern_c, results_loc, std_results_loc, results_loc_m
from .sliding_window import sliding_window_view

headers = "f_C f_C_corr av_SUM f_allC f_allD f_kD f_kC f_kDC f_strat_ch".split() + \
          ["f_%dD" % k for k in range(8)] + ["f_%dC" % k for k in range(8)] + \
          ["f_%dDC" % k for k in range(8)]


# neighbourhood_inds=np.array([
#     (-1,0), (-1,1), (0,1), (1,1), (1, 0), (1, -1), (0, -1), (-1,-1)
# ], dtype=int)
#
#
#
# def get_neighbourhood(x,y, arr):
#     return np.pad(arr,1)[  neighbourhood_inds[:,0] +x +1, neighbourhood_inds[:,1] + y +1]

def statistics_single(history: List[CurrentState]):
    if history:
        df = pd.DataFrame(np.nan, index=list(range(len(history))), columns=headers)
        # correct_solutions = calc_mappings.correct_solutions(*history[0].states.shape)
        for ind, previous, current in zip(range(len(history)), [history[0]] + history[:-1], history):
            f_C = current.states.sum() / current.states.size
            view: np.ndarray = sliding_window_view(np.pad(current.states, 1), (3, 3)).reshape(
                [-1, 3, 3])
            f_C_corr = np.all(view == pattern_c, axis=(1, 2)).sum() / current.states.size

            # f_C_corr = np.max([(cs * current.states).sum()/cs.sum() for cs in correct_solutions])

            av_SUM = current.payoff.sum() / current.payoff.size / 8
            f_allC = (current.strategies == strategies_str.index('allC')).sum() / current.strategies.size
            f_allD = (current.strategies == strategies_str.index('allD')).sum() / current.strategies.size

            kD_list = np.array([(current.strategies == strategies_str.index('%dD' % k)).sum() for k in range(8)])
            f_kD = kD_list.sum() / current.strategies.size
            f_kD_arr = kD_list / kD_list.sum() if kD_list.sum() != 0 else kD_list

            kC_list = np.array([(current.strategies == strategies_str.index('%dC' % k)).sum() for k in range(8)])
            f_kC = kC_list.sum() / current.strategies.size
            f_kC_arr = kC_list / kC_list.sum() if kC_list.sum() != 0 else kC_list

            kDC_list = np.array([(current.strategies == strategies_str.index('%dDC' % k)).sum() for k in range(8)])
            f_kDC = kDC_list.sum() / current.strategies.size
            f_kDC_arr = kDC_list / kDC_list.sum() if kDC_list.sum() != 0 else kDC_list

            f_strat_ch = (current.strategies != previous.strategies).sum() / current.strategies.size

            df.loc[ind] = [f_C, f_C_corr, av_SUM, f_allC, f_allD, f_kD, f_kC, f_kDC, f_strat_ch, *f_kD_arr.tolist(),
                           *f_kC_arr.tolist(), *f_kDC_arr.tolist()]

        return df


cols_multirun = "f_C f_C_corr av_SUM f_allC f_allD f_kD f_kC f_kDC f_strat_ch".split()
cols_av = ["av_" + s for s in cols_multirun]
cols_std = ["std_" + s for s in cols_multirun]
cols_merged = list(chain.from_iterable(zip(cols_av, cols_std)))

def multirun_statistics(stats: List[pd.DataFrame], params):
    if stats:
        len_st_gt_1 = len(stats) > 1
        res_file = results_loc_m if len_st_gt_1 else results_loc
        with open(res_file, 'w') as file:
            file.write(
                "# " + repr(asdict(params)).replace("}", "").replace("{", "").replace(",", "\n#")+"\n\n"
            )
            single_df: pd.DataFrame
            for idx, single_df in enumerate(stats):
                file.write("# EXPERIMENT %d\n" % (idx+1))
                single_df.to_csv(file, mode='a',  sep='\t', float_format="%0.3f", index_label="#iter", line_terminator="\n"*3)
                file.write("\n\n")
        if len_st_gt_1:
            with open(std_results_loc, 'w') as file:
                file.write("# " + repr(asdict(params)).replace("}", "").replace("{", "").replace(",", "\n#") + "\n"*3)
                common_array = np.stack([df[cols_multirun].values for df in stats], axis=2)
                df = pd.DataFrame(np.nan, index=list(range(common_array.shape[0])), columns=cols_merged)
                df[cols_av] = common_array.mean(axis=2)
                df[cols_std] = common_array.std(axis=2)
                df.to_csv(file, mode='a', sep='\t', float_format="%0.3f", index_label="#iter")
