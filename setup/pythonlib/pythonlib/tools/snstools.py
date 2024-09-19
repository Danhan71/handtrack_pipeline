""" for seaborn plotting"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def rotateLabel(ax, rotation=45, horizontalalignment="right"):
    """ seaborn, maek sure to add labels for catplot
    ax = sns.catplot(...)
    """

    fig = ax

    # PRoblem, this sometimes deltes it, I think
    # for a in fig.axes.flat:
    #     a.set_xticklabels(a.get_xticklabels(), rotation=rotation, 
    #         horizontalalignment=horizontalalignment)

    # This works...
    for ax in fig.axes.flat:
    #     ax.set_xticks(ax.get_xticks(), rotation=45)
        list_text = [this.get_text() for this in ax.get_xticklabels()]
        if len(list_text)>0:
            ax.set_xticklabels(list_text,rotation=rotation, horizontalalignment="right")

def addLabel(ax):
    """ seaborn, maek sure to add labels for catplot"""
    for a in ax.axes.flat:
        a.set_xticklabels(a.get_xticklabels(), rotation=45, 
            horizontalalignment="right")


def addTextLabelToPoints(ax, df, x, y, name):
    """ 
    adds on top of points labels in text.
    e.g., ax = sns.scatterplot(x="x", y="y", hue="category", data=df, hue_order=sorted(list(set(tasks_all_categories))))
    addTextLabelToPoints(ax, df, "x", "y", "name"),
    i.e,., tell me the x, y, and text strings to use 
    """
    
    # add annotations one by one with a loop
    for line in range(df.shape[0]):
        ax.text(df[x].values[line], df[y].values[line], df[name].values[line], 
                 horizontalalignment='left', size='medium', color='black',
                alpha=0.6)

def plotgood_lineplot(data, xval, yval, line_grouping, include_scatter=False,
    color_single=None, 
    lines_add_ci=False,
    rowvar=None, colvar=None, col_wrap=None, 
    height=4, aspect=1,
    include_mean = False, 
    relplot_kw = None):
    """ Flexible plotter for lineplots, where x is categorical (or small num of
    discrete vals) and y is scalar. Overlays lines on the same plot, and does summaries
    across lines. Can also overlay scatter of each datrapt.
    Common use: for each character draw a line representing its value across n epochs (x values).
    THen overlay the mean over characters.
    NOTE: This supercedes relPlotOverlayLineScatter and relplotOverlaid, which vbasically do the
    same thing, but the latter makes all lines the same color.
    PARAMS:
    - data, dataframe
    - xval, string, name of x var.
    - yval, string,
    - line_grouping, string, each unique level of this var gets its own line.
    - include_scatter, bool, if True, then lines are overlaid on scatters of each datapt.
    - color_single, either None (lines are each diff color) or string color (e..g, "k") applied to
    all lines. Scatter pts are always colored.
    - lines_add_ci, bool, if True, then each line includes shaded error bar. (default is 68% ci)
    - rowvar, colvar, str, variables that define subplots
    RETURNS:
    - handle to figure
    NOTE: see https://stackoverflow.com/questions/46598371/overlay-a-line-function-on-a-scatter-plot-seaborn
    for overlaying scatter and line.
    """

    assert line_grouping!=xval, 'a mistake sometimes made...'
    if relplot_kw is None:
        relplot_kw = {}
    if color_single:
        # then turn off legend
        legend=False
    else:
        legend = True

    # METHOD 1 - 
    # g = sns.FacetGrid(data, row=rowvar, hue=line_grouping, col=colvar,
    #                 sharex=True, sharey=True, height=height, aspect=aspect,
    #                 legend_out=True)

    # catlist = set(data[line_grouping])
    # if color_single is not None:
    #     palette = {cat:color_single for cat in catlist}
    # else:
    #     palette = None
    # # g = g.map(sns.scatterplot, xval, yval, palette=palette)
    # g = g.map(sns.lineplot, xval, yval,  ["k"], palette=palette, legend="full")

    # METHOD 2
    # catlist = set(data[line_grouping])
    # palette = {cat:color_single for cat in catlist}
    # g = sns.relplot(kind='scatter', x=xval, y=yval, data=data, 
    #     hue = line_grouping,
    #     height=height, aspect=aspect)
    # # g.map_dataframe(sns.lineplot, xval, yval, color='g')

    # if color_single is not None:
    #     palette = {cat:color_single for cat in catlist}
    # else:
    #     palette = None
    # g.map_dataframe(sns.lineplot, xval, yval, hue=line_grouping, palette=palette)

    # METHOD 3 - use axes and pass ax into axes level plotting functions.

    # METHOD 4
    catlist = set(data[line_grouping])
    if color_single is not None:
        palette = {cat:color_single for cat in catlist}
    else:
        palette = None
    
    if lines_add_ci:
        errorbar=("ci", 68)
    else:
        errorbar = None

    if rowvar:
        relplot_kw["row"] = rowvar
    if colvar:
        relplot_kw["col"] = colvar
    if col_wrap:
        relplot_kw["col_wrap"] = col_wrap

    g = sns.relplot(data=data, kind='line', x=xval, y=yval,
        hue = line_grouping, 
        height=height, aspect=aspect,
        palette=palette, errorbar=errorbar,
        legend=legend,
        **relplot_kw)
    # g = sns.relplot(data=data, kind='line', x=xval, y=yval,
    #     hue = line_grouping, 
    #     height=height, aspect=aspect,
    #     palette=palette, errorbar="se", 
    #     legend=legend,
    #     **relplot_kw)

    if include_scatter:
        # usually the dots you want to allow their orig color
        # g.map_dataframe(sns.scatterplot, xval, yval, hue=line_grouping, palette=palette)
        g.map_dataframe(sns.scatterplot, xval, yval, 
            hue=line_grouping, alpha=0.5, legend=legend)
    
    if include_mean:
        g.map_dataframe(sns.lineplot, xval, yval, alpha=0.5, legend=legend)

    return g


def relplotOverlaid(df, line_category, color, **relplotkwargs):
    """ if want to plot single lines for each cataegory in 
    line_category, and all the same color. sns I think forces you to 
    either average over all categories in line_category (i.e., one
    output per facet, or hue, etc, or they will be different colors if
    use hue=line_category. Here can make all the same color. 
    """
    # print(relplotkwargs)
    # assert False
    assert False, "superceded by plotgood_lineplot"
    catlist = set(df[line_category])
    palette = {cat:color for cat in catlist}
    relplotkwargs["legend"]= False # since all same color, legend is useulse..
    relplotkwargs["palette"]= palette
    relplotkwargs["data"] = df
    return sns.relplot(**relplotkwargs)


def relPlotOverlayLineScatter(data, x, y, hue=None, row=None, col=None, palette=None,
    height=3, aspect=3):
    """
    Overlay line and scatterplot.
    row="block"
    """
    assert False, "superceded by plotgood_lineplot"

    # g = sns.FacetGrid(data, row=row, sharex=True, sharey=True, height=3, aspect=3,
    #                  legend_out=True)
    g = sns.FacetGrid(data, row=row, hue=hue, col=col, sharex=True, sharey=True, height=height, aspect=aspect,
                     legend_out=True)

    catlist = set(data[hue])
    palette = {n:"k" for n in catlist}

    # print(x)
    # print(y)
    # print(hue)
    # print(palette)
    # assert False
    # g = g.map(sns.scatterplot, x, y, hue=hue, palette=palette)
    # g = g.map(sns.lineplot, x, y, hue=hue, palette=palette, legend="full")
    g = g.map(sns.scatterplot, x, y, palette=palette)
    g = g.map(sns.lineplot, x, y,  palette=palette, legend="full")
    return g


def pairplot_corrcoeff(data, x_vars=None, y_vars = None, hue=None, vars=None, aspect=1,
    corrver="spearman"):
    """ like sns.pairplot, but overlaying pearsons r and p.
    """
    from scipy import stats
    def corrfunc(x, y, **kws):
        if corrver=="pearson":
            r, p = stats.pearsonr(x, y)
        elif corrver=="spearman":
            r, p = stats.spearmanr(x, y)
        else:
            assert False

        ax = plt.gca()
        ax.annotate(f"r={r:.2f}|p={p:.4f}", xy=(.1, .9), xycoords=ax.transAxes)

    g = sns.pairplot(data=data, x_vars = x_vars, y_vars = y_vars, aspect=aspect,
                      kind="reg", hue=hue)
    # g.map_upper(plt.scatter, s=10)
    # g.map_diag(sns.distplot, kde=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map(corrfunc)
    return g

def map_function_tofacet(fig, func):
    """
    fig is a facetgrid plot, e.g., catplot
    func(ax), does stuff with ax,applies to each facet
    """

    def F(*pargs, **kws):
        ax = plt.gca()
        func(ax)
        # ax.axhline(0, alpha=0.2, **kws)

    fig.map(F)


def timecourse_overlaid(df, feat, xval="tvalfake", YLIM=None, row=None, col=None, 
    grouping=None,
    ALPHA = 0.5, doscatter=True, domean=True, squeeze_ylim=True, col_wrap=4):
    """ plot timecourse, both scatter of pts and overlay means"""
    
    if grouping:
        tmp = df[grouping].unique().tolist()
        colors = ["r", "b", "g", "k"]
        if len(tmp)<=len(colors):
            PALLETE = {t:c for t, c in zip(tmp, colors)}
        else:
            PALLETE=None
    else:
        PALLETE = None

    if YLIM is None and squeeze_ylim and doscatter:
        from .plottools import get_ylim
        YLIM = get_ylim(df[feat])

    if row is None and col is None:
        g = sns.FacetGrid(df, height=4, aspect=2, 
                          sharex=True, sharey=True, ylim=YLIM)
    else:
        g = sns.FacetGrid(df, row=row, col=col, height=4, aspect=2, 
                          sharex=True, sharey=True, ylim=YLIM, col_wrap=col_wrap)

    # Different plots, depening on if timecourse, or summaries.
    if xval=="tvalfake":
        xvalmean = "tvalday"
        scatplot = sns.scatterplot
        meanplot = sns.lineplot
    elif xval=="epoch":
        xvalmean = "epoch"
        scatplot = sns.swarmplot
        meanplot = sns.pointplot
    else:
        print(xval)
        assert False, "not sure what is mean summary for this xval"

    if domean:
        g.map(meanplot, xvalmean, feat, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":2})
    if doscatter:
        if xval=="tvalfake":
            g.map(sns.scatterplot, xval, feat, **{"hue":grouping,
                                                            "marker":"x", 
                                                          "alpha":ALPHA,
                                                                  "s":40, 
                                                                  "palette":PALLETE})
        elif xval=="epoch":
            g.map(sns.swarmplot, xval, feat, **{"hue":grouping, "alpha":ALPHA,
                                                                  "s":4, 
                                                                  "palette":PALLETE})
    g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})

    return g


def get_xticklabels(fig):
    """ Returns teh labels in order for each axis in fig
    PARAMS:
    - fig, a facetgrid object, e.g,, output from sns.catplot
    RETURNS:
    - list of labels, where list is len num axes, and each inner list is 
    len num x labels, in order.
    """
    out = []
    for ax in fig.axes.flatten():
        out.append([lab.get_text() for lab in ax.get_xticklabels()])
    return out
            

def heatmap_mat(datamat, ax=None, annotate_heatmap=True, zlims=(None, None),
        robust=False, diverge=False, labels_row=None, labels_col=None,
                rotation=90, rotation_y=0):
    """
    Plot heatmap, given datamat shape (nrow, ncols).
    """
    df = pd.DataFrame(datamat)

    return heatmap(df, ax, annotate_heatmap, zlims,
                   robust, diverge, labels_row, labels_col, rotation, rotation_y)

def heatmap(df, ax=None, annotate_heatmap=True, zlims=(None, None),
        robust=False, diverge=False, labels_row=None, labels_col=None,
            rotation=90, rotation_y=0, SHAPE="square", norm_method=None):
    """ 
    Plot a heatmap dictated by cols and rows of df, where the cells correspond to values
    in df
    PARAMS:
    - df, wideform dataframe to plot, should be in 2d shape, with rows and columns, the sahpe of 
    the resulting plot. df.Index are rows (from top to bottom), and df.columns are columns
    (left to right). See pandastools.convert_to_2d_dataframe to convert from long-form
    to this wideform.
    - annotate_heatmap, bool, whether puyt text in cell indicating the values
    - diverge, if True, then centers the heat
    RETURNS:
    - fig, 
    - ax, 
    - rgba_values, (nrows, ncols, 4), where rgba_values[0,1], means rgba value for row 0 col 1.
    """

    # NOTE, from neural plot heatmap..
    # sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
    #    robust=robust, vmin=zlims[0], vmax=zlims[1])

    # make a copy, with these columns
    if labels_row is None:
        list_cat_1 = df.index.tolist()
    else:
        list_cat_1 = labels_row

    if labels_col is None:
        list_cat_2 = df.columns.tolist()
    else:
        list_cat_2 = labels_col

    if SHAPE == "rect":
        if len(list_cat_2)>10:
            w = len(list_cat_2)/10*3.5
        else:
            w = 5
        h = 5
    elif SHAPE == "square":
        if len(list_cat_2)>10:
            w = len(list_cat_2)/10*3.5
        else:
            w = 5
        h = w
    else:
        assert False

    # Clip to maximum size.
    SIZEMAX = 20
    aspect = h/w
    if h>SIZEMAX:
        h = SIZEMAX
        w = h/aspect
    if w>SIZEMAX:
        w = SIZEMAX
        h = aspect*w
    
    # print(SHAPE, w,h, ax is None)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(w, h))
    else:
        fig = None

    # print("SIZE:", w,h)

    dfthis = df
    if norm_method=="all_sub":
        # minus mean over all cells
        dfthis = dfthis - dfthis.mean().mean()
        diverge = True
    elif norm_method=="col_div":
        # normalize so that for each col, the sum across rows is 1
        assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
        dfthis = dfthis.div(dfthis.sum(axis=0), axis=1)
    elif norm_method=="row_div":
        # same, but for rows
        assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
        dfthis = dfthis.div(dfthis.sum(axis=1), axis=0)
    elif norm_method=="all_div":
        # divide by sum of all counts
        assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
        dfthis = dfthis/dfthis.sum().sum()
    elif norm_method=="col_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
        diverge = True
    elif norm_method=="col_sub_notdiverge":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
        diverge = False
    elif norm_method=="row_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=1), axis=0)
        diverge = True
    elif norm_method=="row_sub_firstcol":
        # for each item in a given row, subtract the value of the first colum in that row.
        dfthis = dfthis.subtract(dfthis.iloc[:,0], axis=0)
    elif norm_method is None:
        pass
    else:
        print(dfthis)
        print(norm_method)
        assert False
    df = dfthis

    # compute zlims here, just so you can extract colors accruately below.
    z1, z2 = zlims
    if z1 is None:
        z1 = df.min().min()
    if z2 is None:
        z2 = df.max().max()
    if diverge:
        # then center at 0
        z = np.max(np.abs([z1, z2]))
        z1 = -z
        z2 = z
    # Make sure z1 is less than z2
    if z1>z2:
        print(z1, z2)
        assert False, "how is this possible.."
    else:
        # Make sure z1 is less than z2
        z1 = np.min([z2-0.001, z1])

    if diverge:
        # center at 0, and use diverging palletee
        # 
        # center = 0
        # cmap = sns.color_palette("vlag")
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        lab_add = 0.5
    else:
        # center = None
        cmap = sns.color_palette("rocket", as_cmap=True)
        # cmap = sns.color_palette
        lab_add = 0.5

    sns.heatmap(df, annot=annotate_heatmap, ax=ax, vmin=z1, vmax=z2,
        robust=robust, cmap=cmap)

    # Return the colors
    from matplotlib.colors import Normalize
    # Normalize data
    norm = Normalize(vmin=z1, vmax=z2)
    try:
        rgba_values = cmap(norm(df))
    except Exception as err:
        print(df)
        print(len(df))
        raise err

    if len(list_cat_1)<200:
        # otherwise is too slow, too much text.
        ax.set_yticks([i+lab_add for i in range(len(list_cat_1))], list_cat_1, rotation=rotation_y, fontsize=6)
    if len(list_cat_2)<200:
        ax.set_xticks([i+lab_add for i in range(len(list_cat_2))], list_cat_2, rotation=rotation, fontsize=6)

    return fig, ax, rgba_values
