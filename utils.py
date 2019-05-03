import numpy as np
from matplotlib import pyplot as plt


def get_dimension_list(matrix):
    dims = []
    while isinstance(matrix, list) and matrix is not None:
        dims.append(len(matrix))
        matrix = matrix[0]
    number_of_dimensions = len(dims)
    return number_of_dimensions

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def plot_detail(fig, ax, x_label, y_label, tick_fontsize, xfontstyle='normal', aspect=1):
    '''For plots with arrows at the border'''
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(aspect*abs(xmax - xmin) / abs(ymax - ymin))

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.yticks([])
    #ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    # xlw = 1
    # ylw = xlw / (ymax - ymin) * (xmax - xmin) * height / width
    xlw = 1./140*(ymax-ymin)# axis line width
    ylw = 1./140*(xmax-xmin)# axis line width
    ohg = 0.3 # arrow overhang

    # # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', width = xlw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)
    ax.arrow(xmin, ymin, 0., ymax - ymin, fc='k', ec='k', width=ylw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False)

    ax.set_xlabel(x_label, fontsize=tick_fontsize, fontstyle=xfontstyle)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=tick_fontsize)
