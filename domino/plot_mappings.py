import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from .calc_mappings import strategies_str

vmin_dict = {'state': 0, 'strategy': 0, "payoff": -8, "kD": 0, "kC": 0, "kDC": 0}
vmax_dict = {'state': 1, 'strategy': 28, "payoff": 8, "kD": 28, "kC": 28, "kDC": 28}
# new colormap
strategy_all = ListedColormap(("blue", "red"))
strategy_all_blk = ListedColormap(("black", "black"))

strategy_kD = cm.get_cmap('Greens', 32)

dimming_factor = 0.30
dimming_factor_all = 0.20

dimming = np.array([dimming_factor, dimming_factor, dimming_factor, 1])
dimming_all = np.array([dimming_factor_all, dimming_factor_all, dimming_factor_all, 1])

strategy_kC = []
strategy_kDC = []
strategy_other = []
for i in np.linspace(0.4, 1, 9):
    strategy_kC.append([0, i, i, 1])
    strategy_kDC.append([i, 0, i, 1])
    strategy_other.append([0, 0, 0, 1])

strategy_kC.reverse()
strategy_kDC.reverse()

new_colors = np.vstack((
    strategy_kD(np.linspace(0.5, 1, 9)),
    strategy_kC,
    strategy_kDC,
    strategy_all([0, 1])
))

kD_colors = np.vstack((
    strategy_kD(np.linspace(0.5, 1, 9)),
    dimming*strategy_kC,
    dimming*strategy_kDC,
    dimming_all*strategy_all([0, 1])
))

kC_colors = np.vstack((
    dimming*strategy_kD(np.linspace(0.5, 1, 9)),
    strategy_kC,
    dimming*strategy_kDC,
    dimming_all*strategy_all([0, 1])
))

kDC_colors = np.vstack((
    dimming*strategy_kD(np.linspace(0.5, 1, 9)),
    dimming*strategy_kC,
    strategy_kDC,
    dimming_all*strategy_all([0, 1])
))

'''
        allC-czerwony :: Reds,
        allD-jasno niebieski :: Blues,
        kD – spektrum koloru zielonego :: Greens
        kC - spektrum koloru jasno niebieskiego (cyjan), :: BuPu
        kDC- spektrum magenty ; :: RdPu
'''

strat_cmp = ListedColormap(new_colors)

kD_colors_map = ListedColormap(kD_colors)
kC_colors_map = ListedColormap(kC_colors)
kDC_colors_map = ListedColormap(kDC_colors)

assert len(strat_cmp.colors) == len(strategies_str)

state_cmp = ListedColormap(("blue", "red"))
rdbu = cm.get_cmap('BrBG', 17)
newcolors_rdbu = rdbu(np.linspace(0, 1, 17))
payoff_cmp = ListedColormap(np.array([1, 1, 1, 2]) - newcolors_rdbu ** (np.array([3, 2, 3, 1]) * 0.6))

colormaps_dict = {'state': state_cmp, 'strategy': strat_cmp, "payoff": payoff_cmp, "kD": kD_colors_map, "kC": kC_colors_map, "kDC": kDC_colors_map}
colorbar_dict = {'state': True, 'strategy': True, "payoff": True, "kD": True, "kC": True, "kDC": True}
colorbar_labels = {'state': '0 1'.split(), 'strategy': strategies_str, "payoff": list(map(str, np.arange(-8, 9))), "kD":strategies_str,"kC":strategies_str,"kDC":strategies_str}
colorbar_ticks = {'state': [0, 1], 'strategy': np.arange(0, len(strategies_str)), "payoff": np.arange(-8, 9), "kD":np.arange(0, len(strategies_str)),"kC":np.arange(0, len(strategies_str)),"kDC":np.arange(0, len(strategies_str))}
