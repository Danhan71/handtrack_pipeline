""" plots, takles in dataframe, Dataset().Dat,
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.listtools import sort_mixed_type

def plot_dat_grid_inputrowscols(df, strokes_ver="strokes_beh", max_n_per_grid=None,
    col_labels = None, row_labels=None, strokes_by_order=False, plotfuncbeh=None, 
    max_cols = 40, max_rows=40, plot_task=True, xlabels = None, titles_each_cell=None):
    """ in each grid position, plots a single trial; (e.. strokes)
    df must have two columns, row and col, which indicate where to plot each
    trial.
    by default, top-left is 0,0
    INPUTS:
    - strokes_by_order, then colors strokes by ordinal (from blue to pink, based on "cool")
    - xlabels, list of strings, same len as df, for x axis labels. Useful since y axis is used for
    labeleing row, and titles are used for labeling columns.
    RETURNS:
    - figbeh, figtask
    NOTES:
    - plots both strokes_ver and "strokes_task"
    """


    from pythonlib.tools.plottools import plotGridWrapper

    # only keep trials that are within the col range
    a = len(df)
    df = df[df["col"]<max_cols]
    df = df[df["row"]<max_rows].reset_index(drop=True)
    if len(df)<a:
        print("Old len(df):", a)
        print("New, removing if too many col or row:", len(df))

    # extract data to plot
    strokeslist = df[strokes_ver].values
    rowlist = df["row"].values
    collist = df["col"].values
    if titles_each_cell is not None:
        titles = df[titles_each_cell].values.tolist()
    else:
        titles = None

    if plotfuncbeh is None:
        if strokes_by_order:
            plotfuncbeh = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, alpha=0.4, add_stroke_number=False, 
                force_onsets_same_col_as_strokes=True, naked=True)
        else:
            plotfuncbeh = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, alpha=0.4, add_stroke_number=False,
                force_onsets_same_col_as_strokes=True, naked=True)

    figbeh = plotGridWrapper(strokeslist, plotfuncbeh, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid,
        col_labels = col_labels, row_labels=row_labels, xlabels=xlabels, titles=titles)


    # Also plot tasks
    if plot_task:
        strokestasklist = df["strokes_task"].values
        # plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, naked=True)
        plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, naked=True)
        figtask = plotGridWrapper(strokestasklist, plotfunc, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid,
            col_labels = col_labels, row_labels=row_labels)
    else:
        figtask = None    

    return figbeh, figtask

############## HELPERS THAT CALL plot_dat_grid_inputrowscols
def _plot_beh_grid_flexible_helper(dfthis, row_group, col_group="trial", row_levels = None, col_levels=None,
    max_n_per_grid=1, plotfuncbeh=None, max_cols = 40, max_rows = 40, plot_task=True, 
    plotkwargs={}, strokes_by_order=False, xlabel_trialcode = True, titles_each_cell=None,
    sort_colvar=False):
    """ [GOOD] flexible helper, can choose what variable to group by.
    INPUTS:
    - row_group and col_group are what variable to group trials by along rows or columns. e..g,
    row_group = "character" means that each row is a different character. In general, can use anything
    that is a column in D.Dat and has discrete levels.
    --- Can also enter:
    ----- "trial" in which case rows/columns will be separte trials, generally in chron order.
    ----- "trial_shuffled" shuffles within level. useful if too many trials.
    - row_levels, col_levels, list of levels, where if None, then will auto get all levels. 
    - xlabel_trialcode, bool (True), on x axis labels each trial's trialcode. If on, then cannot make plot
    tight (since will not see the trialcode).
    """
    from pythonlib.tools.pandastools import filterPandas

    if strokes_by_order:
        plotkwargs["strokes_by_order"] = True

    def _assign_row_col_inds(dfthis, group, levels, new_col_name, other_group=None):
        # If give levels, and if not covers all trials, then will give uncovered trials -1.
        if group in ["trial", "trial_shuffled"]:
            # then should be trialnum, based on the other group
            assert other_group is not None
            from pythonlib.tools.pandastools import append_col_with_index_number_in_group
            dfthis = append_col_with_index_number_in_group(dfthis, other_group, colname = new_col_name,
                randomize = group=="trial_shuffled")
            labels = range(0, max(dfthis[new_col_name])+1)
            if xlabel_trialcode:
                trialcode_list = dfthis["trialcode"].to_list()
            else:
                trialcode_list = None
        else:
            if levels is None:
                levels = sort_mixed_type(dfthis[group].unique().tolist())
            # only keep dataset that is in these levels
            dfthis = filterPandas(dfthis, {group:levels}) # make a copy, otherwise some rows will not have anything
            map_ = {lev:i for i, lev in enumerate(levels)}
            max_i = len(levels)
            def mapper(x):
                """ x is a df row --> index which row to plot
                # If this trial is in map_, then give index. otherwise 
                give it a -1
                """
                if x[group] in map_.keys():
                    return map_[x[group]]
                else:
                    assert False, "since made copy, all rows should have mapping."

            dfthis = applyFunctionToAllRows(dfthis, mapper, new_col_name)
            labels = levels
            if xlabel_trialcode:
                trialcode_list = dfthis["trialcode"].to_list()
            else:
                trialcode_list = None

        return dfthis, labels, trialcode_list

    if sort_colvar:
        if col_levels is None:
            col_levels = sort_mixed_type(dfthis[col_group].unique().tolist())
        else:
            col_levels = sort_mixed_type(col_levels)

    dfthis, row_labels, trialcode_list_1 = _assign_row_col_inds(dfthis, row_group, row_levels, "row", col_group)
    dfthis, col_labels, trialcode_list_2 = _assign_row_col_inds(dfthis, col_group, col_levels, "col", row_group)
    assert trialcode_list_1 == trialcode_list_2

    # if labels too long, prune
    if False:
        col_labels = [t[:10] + ".." + t[-5:] if isinstance(t, str) and len(t)>14 else t for t in col_labels]
        row_labels = [t[:10] + ".." + t[-5:] if isinstance(t, str) and len(t)>14 else t for t in row_labels]

    # tasknames too long, so prune for plotting
    # tasklist_titles = [t[:10] + ".." + t[-5:] for t in tasklist]
    # Plot

    # col_labels = 

    # print(col_labels)
    # print(row_labels)
    # print(dfthis["trial"])
    # assert False

    xlabels = trialcode_list_1 if xlabel_trialcode else None
    
    figbeh, figtask = plot_dat_grid_inputrowscols(dfthis, max_n_per_grid=max_n_per_grid, plotfuncbeh=plotfuncbeh, 
        col_labels=col_labels, row_labels=row_labels, xlabels = trialcode_list_1,
        max_cols=max_cols, max_rows=max_rows, titles_each_cell=titles_each_cell, plot_task=plot_task,
        **plotkwargs)

    return figbeh, figtask

def plot_beh_grid_flexible_helper(D, row_group, col_group="trial", row_levels = None, col_levels=None,
    max_n_per_grid=1, plotfuncbeh=None, max_cols = 40, max_rows = 40, plot_task=True, 
    plotkwargs={}, strokes_by_order=False, xlabel_trialcode = True, titles_each_cell=None):
    """ 
    see _plot_beh_grid_flexible_helper
    """
    from pythonlib.tools.pandastools import filterPandas

    dfthis = D.Dat
    return _plot_beh_grid_flexible_helper(dfthis, row_group, col_group, row_levels, 
        col_levels, max_n_per_grid, plotfuncbeh, max_cols, max_rows, plot_task, 
        plotkwargs, strokes_by_order, xlabel_trialcode, titles_each_cell)


def plot_beh_grid_grouping_vs_task(df, row_variable, tasklist, row_levels=None, plotkwargs = {},
    plotfuncbeh=None, max_n_per_grid=1, max_n_trials=50, max_cols=150):
    """
    Helper (USEFUL) for plotting 2d grid of beh (strokes), with columns as unique tasks, and
    rows as flexible grouping variable.
    INPUTS:
    - df (from D.Dat)
    - row_variable, str that is a col name in df, which will dictate row for each datapont.
    - row_levels, list of levels of row)variable. Optional. if pass in, then will only
    have this many rows (each corresponding to one level). Otherwise will have as many rows
    as there are unique levels
    RETURNS:
    - figbeh, figtask
    """
    print(" ** TODO: this should call plot_beh_grid_flexible_helper. It is redundant")
    from pythonlib.dataset.plots import plot_dat_grid_inputrowscols
    dfthis = df[df["character"].isin(tasklist)]

    # @KGG 3/30/22
    # if the dataframe is sufficiently big, then subset it into a random sample 
    if len(dfthis) > max_n_trials:
        # in plottools.py/plotGridWrapper, *need* row 0 and col 0 for plotting to work
        df_0 = dfthis.iloc[0]
        # arbitrary choice to sample 30% of data
        df_sample = dfthis.sample(int(len(dfthis)*0.3))
        df_this = pd.concat([df_0, df_sample])

    # get levels for rows
    row_levels = sort_mixed_type(dfthis[row_variable].unique().tolist())
    # print("row levels:")
    # print(row_levels)
    assert len(row_levels)<40, "40+ row levesls, are you sure"

    # Row mapper
    print("Row levels, mapped to column indices:")
    row_mapper_ = {row_levels[i]:i for i in range(len(row_levels))}
    print(row_mapper_)
    def row_mapper(x):
        """ x is a df row --> index which row to plot"""
        return row_mapper_[x[row_variable]]
    def col_mapper(x):
        tmp = [i for i, t in enumerate(tasklist) if t==x["character"]]
        assert len(tmp)==1
        return tmp[0]

    # Assign row and col varibles
    dfthis = applyFunctionToAllRows(dfthis, row_mapper, "row")
    dfthis = applyFunctionToAllRows(dfthis, col_mapper, "col")

    # tasknames too long, so prune for plotting
    if False:
        tasklist_titles = [t[:10] + ".." + t[-5:] for t in tasklist]
    else:
        # instead, make text smaller
        tasklist_titles = [t for t in tasklist]

    # Plot
    figbeh, figtask = plot_dat_grid_inputrowscols(dfthis, max_n_per_grid=max_n_per_grid, 
        col_labels = tasklist_titles, row_labels=row_levels, plotfuncbeh=plotfuncbeh, 
        max_cols = max_cols,
        **plotkwargs)

    return figbeh, figtask


def plot_beh_grid_singletask_alltrials(D, task, row_variable, row_levels=None, plotkwargs = {},
    plotfuncbeh=None, max_cols = 40, max_rows = 40):
    """ given a singel task, plots all trials in df in a grid, where rows split into row_levels (grouped by row_variable)
    and columns are trials, in order encountered in df.
    INPUTS:
    - df, usually make this D.Dat, or related
    """

    # df = D.Dat[D.Dat["character"]==task]
    import pandas as pd

    dfplot, row_levels = D.analy_singletask_df(task, row_variable, row_levels=row_levels, return_row_levels=True)

    # df = df[df["character"] == task]

    # # what are rows
    # if row_levels is None:
    #     row_levels = sorted(df[row_variable].unique().tolist())

    # # Skip if dont have data across all rows.
    # if not all([l in df[row_variable].unique().tolist() for l in row_levels]):
    #     print("SKIPPING, since not have data across levels:", task)
    #     return

    # out = []
    # for i, lev in enumerate(row_levels):
    #     dfthis = df[df[row_variable]==lev]
    #     dfthis = dfthis.reset_index(drop=True)
    #     dfthis["col"] = dfthis.index.tolist()
    #     dfthis["row"] = i
    #     out.append(dfthis)

    # dfplot = pd.concat(out)

    if "max_cols" in plotkwargs:
        max_cols = plotkwargs["max_cols"]
        del plotkwargs["max_cols"]
    if "max_rows" in plotkwargs:
        max_rows = plotkwargs["max_rows"]
        del plotkwargs["max_rows"]

    # PLOT
    figb, figt = plot_dat_grid_inputrowscols(dfplot, row_labels=row_levels, plotfuncbeh=None, 
        max_cols = max_cols, max_rows = max_rows, **plotkwargs)

    return figb, figt

def plot_beh_waterfall_singletask_alltrials(D, task, row_variable, row_levels=None, plotkwargs = {},
    plotfuncbeh=None):
    """ given a singel task, plots all trials in df in a waterfall, where organizes them based on row_levels, in order encountered in df.
    this will dictate order, and label in plot.
    INPUTS:
    - df, usually make this D.Dat, or related
    """
    from pythonlib.drawmodel.strokePlots import plotDatWaterfallWrapper
    from pythonlib.tools.stroketools import strokesVelocity

    dfplot, row_levels = D.analy_singletask_df(task, row_variable, row_levels=row_levels, return_row_levels=True)

    # PLOT WATERFALL
    strokes_list = dfplot["strokes_beh"].values
    onsets_list = dfplot["go_cue"].values
    row_list = dfplot["row"].values
    row_list = [row_levels[i] for i in row_list]
    col_list = dfplot["col"].values
    labels = [f"c_{c}|r_{r}" for c,r in zip(col_list, row_list)]

    # Get velocities
    # TODO: assuming that fs for first params will apply across all.
    fs = D.Metadats[0]["filedata_params"]["sample_rate"]
    strokespeed_list = [strokesVelocity(strokes, fs, clean=True)[1] for strokes in strokes_list]
    strokespeed_list = [[ss[:,0] for ss in s] for s in strokespeed_list] # only keep the speed, not the time.

    n = len(strokes_list)
    fig, ax = plt.subplots(1, 1, figsize=(n*1.5, 5))
    # plotDatWaterfallWrapper(strokes_list, onset_time_list=onsets_list, ax=ax, ylabels=labels)
    plotDatWaterfallWrapper(strokes_list, onset_time_list=onsets_list, strokes_ypos_list=strokespeed_list, ax=ax, ylabels=labels)
    return fig


def plot_timecourse_overlaid(D, features_list, xval="tvalfake", grouping=None, doscatter=True, domean=True,
        grouping_each_panel = None, col_wrap = 4):
    """
    Plot timecourse, (feature vs. tval) to summarize an expt. Separate 
    columns (character) and rows (taskgroup) and colors (epochs).
    Overlays indivitual trials and within day mean. 
    - xval, string, for what to use on x axis:
    --- "tvalfake", then is time
    --- "epoch" then is grouping across trials, based on epoch.
    - features_list, list of strings, columns of D.Dat, where values are scalars
    - grouping, string, where defines levels (for colored epochs)
    - grouping_each_panel, list of str, each level of this varibable will have its own panel.
    --- if len==2, then is interpreted as [row, col]
    --- if len==1, then is interpreted as [col], with colwrap used.

    NOTE:
    - this used in model expt timecurse analyses
    """
    from pythonlib.tools.snstools import timecourse_overlaid
    if grouping_each_panel is None:
        grouping_each_panel = ["task_stagecategory", "taskgroup"]
        
    DF = D.Dat
    # YLIM = [0, 6]

    if len(grouping_each_panel)==2:
        row = grouping_each_panel[0]
        col = grouping_each_panel[1]
    elif len(grouping_each_panel)==1:
        col = grouping_each_panel[0]
        row = None
    elif len(grouping_each_panel)==0:
        row=None
        col=None
    else:
        assert False
        
    figlist = []
    for f in features_list:
        # feat = f"FEAT_{f}"
        feat = f
        fig = timecourse_overlaid(DF, feat, xval=xval, row=row, col=col, grouping=grouping,
         doscatter=doscatter, domean=domean, col_wrap=col_wrap)
        figlist.append(fig)
    return figlist

def plot_beh_day_vs_trial(D, prim_list, SAVEDIR, Nmin_toplot=5, max_cols=40):
    """ Plot raw behavior across days (rows) and trials (columns). 
    Useful for seeing progression over learning (rows).
    Each figure a unique scale and orientation
    INPUT:
        SAVEDIR, base dir. each prim is a subdir.
        Nmin_toplot = 5 # only ploit if > this many trials across all days.
        max_cols = 40 # max to plot in single day
    """
    assert False, "DONE! see simple summary"
    assert False, "see analy/primtiives - port plot to here"


def plot_one_trial_per_level(D):
    """ plot a grid, where each location is one (random) trial for a level that is assigned
    to that grid loc. levels are levels under one grouping variable.
    - e.g., one trial per unique task.
    """
    assert False, "see analy --> primtives, port here."


def plotwrapper_draw_grid_rows_cols(df, rowvar, colvar, n_examples_per_sublot=1,
        plot_task=False, xlabel_trialcode=False, sort_colvar=False):
    """ [Good], plot grid of exmaple strokes, where you specify the categorical variable whose
    levels make up the rows and columns
    PARAMS:
    - df, dataframe holding data, must have these columns:
    --- "strokes_beh", each item a strokes (list ofa rray)
    --- <rowvar>
    --- <colvar>
    --- [optiona] "strokes_task", the image.
    """

    assert "strokes_beh" in df.columns
    try:
        assert rowvar in df.columns
        assert colvar in df.columns
    except Exception as err:
        print(rowvar, rowvar in df.columns)
        print(colvar, colvar in df.columns)
        raise err
    
    if plot_task:
        assert "strokes_task" in df.columns
    if xlabel_trialcode:
        assert "trialcode" in df.columns

    figbeh, figtask = _plot_beh_grid_flexible_helper(df, rowvar, colvar, plotfuncbeh=None,
                              plot_task=plot_task, max_n_per_grid=n_examples_per_sublot,
                              xlabel_trialcode = xlabel_trialcode, sort_colvar=sort_colvar)
    return figbeh, figtask
