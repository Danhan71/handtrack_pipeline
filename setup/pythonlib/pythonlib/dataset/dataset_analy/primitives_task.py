""" 
OLD
Analyses that care about primitives used in the task, e.g,, 
prims in grid, extract each prim, and plot information about the beh strokes
assigned to that prim

See notebook: 220710_analy_spatial_timecourse_exploration

"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plotscatter_locations_each_shape(DS, list_task_kind=None):
    """ For each shape, plot distirbutisons of locations (scatterplot)
    """
    from pythonlib.tools.pandastools import filterPandas
    if list_task_kind is None:
        list_task_kind = ["prims_on_grid"]
    F = {
        "task_kind":list_task_kind
    }
    # "task_kind":["prims_single", "prims_on_grid"],
    dfthis = DS.dataset_slice_by_mult(F)

    # fig1 = sns.catplot(data=dfthis, x="gridloc_x", y="gridloc_y", kind="strip", hue="task_kind", 
    #             col="shape_oriented", col_wrap=4)

    fig = sns.catplot(data=dfthis, x="gridloc_x", y="gridloc_y", kind="strip", hue="gridsize", 
                col="shape_oriented", col_wrap=4)

    return fig




