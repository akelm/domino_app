import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from .calc_mappings import strategies_str

vmin_dict = {'state': 0, 'strategy': 0, "payoff": -8}
vmax_dict = {'state': 1, 'strategy': 25, "payoff": 8}
# new colormap
winter = cm.get_cmap('winter', 8)
cool = cm.get_cmap('cool', 8)
Wistia = cm.get_cmap('Wistia', 8)
binary = cm.get_cmap('binary', 2)

new_colors = np.vstack((np.array([0.9, 0.9, 0.9, 1]) * winter(np.linspace(0, 1, 8)) ** np.array([4, 0.5, 0.9, 1]),
                        cool(np.linspace(0, 1, 8)),
                        Wistia(np.linspace(0, 1, 8)),
                        binary([0, 1])))
strat_cmp = ListedColormap(new_colors)
state_cmp = cm.get_cmap('binary', 2)
rdbu = cm.get_cmap('BrBG', 17)
newcolors_rdbu = rdbu(np.linspace(0, 1, 17))
payoff_cmp = ListedColormap(np.array([1, 1, 1, 2]) - newcolors_rdbu ** (np.array([3, 2, 3, 1]) * 0.6))

colormaps_dict = {'state': state_cmp, 'strategy': strat_cmp, "payoff": payoff_cmp}
colorbar_dict = {'state': True, 'strategy': True, "payoff": True}
colorbar_labels = {'state': '0 1'.split(), 'strategy': strategies_str, "payoff": list(map(str, np.arange(-8, 9)))}
colorbar_ticks = {'state': [0, 1], 'strategy': np.arange(0, len(strategies_str)), "payoff": np.arange(-8, 9)}
