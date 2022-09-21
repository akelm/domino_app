import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import matplotlib
import numpy as np

from .calc_mappings import img_file_pattern, img_file_labels

matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.ioff()
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .plot_mappings import colormaps_dict, vmax_dict, vmin_dict, colorbar_dict, colorbar_ticks, colorbar_labels


def history_to_img(history, exper_num):
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     for ind, current in enumerate(history):
    #         executor.submit(self.state_to_img, current, ind, exper_num)

    for ind, current in enumerate(history):
        state_to_img(current, ind, exper_num)

def state_to_ram(state, label):
    yticks_len, xticks_len = state.states.shape
    xticks = np.arange(xticks_len + 1) - 0.5
    yticks = np.arange(yticks_len + 1) - 0.5
    xrange = xticks[0], xticks[-1]
    yrange = yticks[0], yticks[-1]

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.15)

    ax.cla()
    cax.cla()

    if label == "state":
        arr = state.states
    elif label == "strategy":
        arr = state.strategies
    else: #if label == "payoff":
        arr = state.payoff

    ims = ax.imshow(arr, cmap=colormaps_dict[label], vmax=vmax_dict[label], vmin=vmin_dict[label])
    if colorbar_dict[label]:
        cbar = fig.colorbar(ims, cax=cax, orientation='vertical')
        cbar.set_ticks(colorbar_ticks[label])
        cbar.ax.set_yticklabels(colorbar_labels[label])

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(color='blue', linestyle='--', linewidth=0.5)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.invert_yaxis()

    buffer = BytesIO()
    fig.savefig(buffer, bbox_inches='tight')

    plt.close(fig)

    return buffer

def state_to_img(current, ind, exper_num):
    start = time.perf_counter()

    yticks_len, xticks_len = current.states.shape
    xticks = np.arange(xticks_len + 1) - 0.5
    yticks = np.arange(yticks_len + 1) - 0.5
    xrange = xticks[0], xticks[-1]
    yrange = yticks[0], yticks[-1]

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.15)

    for label, arr in zip(img_file_labels, (current.states, current.strategies, current.payoff)):
        ax.cla()
        cax.cla()

        ims = ax.imshow(arr, cmap=colormaps_dict[label], vmax=vmax_dict[label], vmin=vmin_dict[label])
        if colorbar_dict[label]:
            cbar = fig.colorbar(ims, cax=cax, orientation='vertical')
            cbar.set_ticks(colorbar_ticks[label])
            cbar.ax.set_yticklabels(colorbar_labels[label])

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(color='blue', linestyle='--', linewidth=0.5)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_yticklabels("")
        ax.set_xticklabels("")
        ax.invert_yaxis()
        filename = img_file_pattern % (exper_num + 1, label, ind)
        fig.savefig(filename, bbox_inches='tight')

    plt.close(fig)

    end = time.perf_counter()
    print(f'Finished {ind} in {round(end - start, 2)}')
