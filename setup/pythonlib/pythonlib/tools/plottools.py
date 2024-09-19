import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.listtools import sort_mixed_type

def axis_xlim_ylim_intervals_modify(ax, interval_size, axis='x'):
    """ Modify the axis lims for this plot.
    PARAMS:
    - interval_size, gaps between ticks.
    """

    # # Define the interval size
    # interval_size = 1

    # Extract the current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Set x and y ticks with the specified interval size based on the extracted limits
    if axis=="x":
        ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + interval_size, interval_size))
    elif axis=="y":
        ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + interval_size, interval_size))
    else:
        print(axis)
        assert False

def radar_plot(ax, thetas, values, color="k", fill=True):
    """ polar radar plot 
    assumes that theta[-1] immedaitely precedes theta[0]. will connect them
    assumes that thetas are evenly spaced 
    assumes that ax craeted using subplot_kw=dict(projection='polar')"""
    
    values = np.concatenate([values, values[0].reshape(1,)])
    thetas = np.concatenate([thetas, thetas[0].reshape(1,)])
    
    width = np.mean(np.diff(thetas[:-1]))
    thetas = thetas + width/2
    ax.plot(thetas, values, '-k', color=color, zorder=3)
    if fill:
        ax.fill(thetas, values, '-k', color=color, alpha=0.5)

def rotate_x_labels(ax, rotation=45):
    # draw the figure (update)
    ax.figure.canvas.draw_idle()
    # plt.draw() # equivalent to above.
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=rotation)

def rotate_y_labels(ax, rotation=45):
    ax.figure.canvas.draw_idle()
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), rotation=rotation)

def rose_plot(ax, angles, bins=16, density="frequency_radius", offset=0, lab_unit="degrees",
              start_zero=True, fill=False, skip_plot=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    from: https://stackoverflow.com/questions/22562364/circular-histogram-for-python
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density=="proportion_area":
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    elif density=="frequency_area":
#         count = count/np.sum(count)
        radius = (count)**.5
    elif density=="proportion_radius":
        radius = count / angles.size
    elif density=="frequency_radius":
#         count = count/np.sum(count)
        radius = count
    else:
        assert False, "not coded"

    if not skip_plot:
        ####################### PLOTS
        # Plot data on ax
        if not fill:
            ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
                   edgecolor='k', fill=fill, linewidth=1)
        else:
            ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
               edgecolor='k', color=[0.7, 0.7, 0.7], fill=fill, linewidth=1)
        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        # -- plot each line?
        ax.plot(angles, 0.9*np.max(radius)*np.ones(angles.size), 'xk')
        # c = np.vstack([count, count[0]])
        # print(bins)

        ax.set_thetagrids([])

        if False:
            # works, but doesn't look great unless havea lotns of data. 
            if start_zero:
                r = np.concatenate([radius, radius[0].reshape(1,)])
                # b = bins[:-1] + widths/2
                # b = np.concatenate([b, bins[-1].reshape(1,)])
                ax.plot(bins+np.mean(widths)/2, r, '-k', color='k')
                ax.fill(bins+np.mean(widths)/2, r, color='k')

        if lab_unit == "radians":
            label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
            ax.set_xticklabels(label)
        elif lab_unit is None:
            ax.set_xticklabels([])
        
    return radius, bins
#     # Fiddle with labels and limits
#     polar_ax.set_xticks([0, np.pi/4, 2*np.pi - np.pi/4])
#     polar_ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$-\pi/4$'])
#     polar_ax.set_rlabel_position(90)


def annotate(s, ax=None, color="k"):
    """ puts on top left corner"""
    if ax is None:
        plt.annotate(s, (0.05, 0.9), color=color, size=12, xycoords="axes fraction")
    else:
        ax.annotate(s, (0.05, 0.9), color=color, size=12, xycoords="axes fraction")


def color_make_map_discrete_labels(labels, which_dim_of_labels_to_use=None, cmap=None):
    """
    Helper to make colors for plotting, mapping from unque item
    in labels to rgba color. Can be continuous or discrete (and will
    check this automatically).
    PARAMS:
    - labels, values, either cont or discrete.
    - which_dim_of_labels_to_use, either NOne (use entire label) or int, which indexes into
    each lab in labels, and just uses that to determine the color. e.g,, (shape, loc) --> are the labels,
    and you just want to color by shape, then use which_dim_of_labels_to_use = 0.
    RETURNS:
    - dict,  mapping from value to color (if discrete), otherw sie None
    - color_type, str, either "cont" or "discrete".
    - colors, list of colors, matching input labels.
    """

    if which_dim_of_labels_to_use is None:
        labels_for_color = labels
    else:
        labels_for_color = [lab[which_dim_of_labels_to_use] for lab in labels]
    labels_color_uniq = sort_mixed_type(list(set(labels_for_color)))

    if len(set([type(x) for x in labels_color_uniq]))>1:
        # more than one type...
        color_type = "discr"
        pcols = makeColors(len(labels_color_uniq))
        _map_lev_to_color = {}
        for lev, pc in zip(labels_color_uniq, pcols):
            _map_lev_to_color[lev] = pc
    # continuous?
    elif len(labels_color_uniq)>50 and isinstance(labels_color_uniq[0], (int)):
        color_type = "cont"
        # from pythonlib.tools.plottools import map_continuous_var_to_color_range as mcv
        # valmin = min(df[var_color_by])
        # valmax = max(df[var_color_by])
        # def map_continuous_var_to_color_range(vals):
        #     return mcv(vals, valmin, valmax)
        # label_rgbs = map_continuous_var_to_color_range(df[var_color_by])
        _map_lev_to_color = None
    elif len(labels_color_uniq)>8 and isinstance(labels_color_uniq[0], (np.ndarray, float)):
        color_type = "cont"
        _map_lev_to_color = None
    else:
        color_type = "discr"
        # label_rgbs = None
        pcols = makeColors(len(labels_color_uniq), cmap=cmap)
        _map_lev_to_color = {}
        for lev, pc in zip(labels_color_uniq, pcols):
            _map_lev_to_color[lev] = pc

    # Return the color for each item
    if _map_lev_to_color is None:
        colors = labels_color_uniq
    else:
        colors = [_map_lev_to_color[lab] for lab in labels_for_color]

    return _map_lev_to_color, color_type, colors


def color_make_pallete_categories(df, category_name, cmap="turbo"):
    """ 
    Make colors for categorical variables.
    PARAMS;
    - df, will use levels for df[category_name] (assumes is categorical)
    - category_name, string key in to df
    RETURNS:
    - pallete dict, where key:val is
    key is level in this categgory
    val is (4,) np array, rgba
    """
    levels = sort_mixed_type(df[category_name].unique().tolist())
    cols_arrays = makeColors(len(levels), cmap=cmap)
    pallete = {}
    for lev, col in zip(levels, cols_arrays):
        pallete[lev] = col
    return pallete


def legend_add_manual(ax, labels, colors, alpha=0.4, loc="upper right"):
    """ Manually add a legend
    PARAMS:
    - labels, list of str
    - colors, list of color, to match labels
    - loc, "best" is adaptive but slow.
    """
    import matplotlib.patches as mpatches
    handles = []
    # prune length of lables
    def _prune(lab):
        if isinstance(lab, str) and len(lab)>20:
            return lab[:20]
        else:
            return lab
    labels = [_prune(lab) for lab in labels]

    for lab, col in zip(labels, colors):
        this = mpatches.Patch(color=col, label=lab, alpha=alpha)
        handles.append(this)
    ax.legend(handles=handles, framealpha=alpha, fontsize=8, loc=loc) # default loc ("best") is slow for large amonts of data.


def makeColors(numcol, alpha=1, cmap=None, ploton=False):
    """ turbo. see below for reason.
    PREVIOUSLY: gets evensly spaced colors. currntly uses jet map.
    PREVIOUSLY: plasma
    """

    # These, in order from best to worst, in comining (i) separation and (ii) not fading out(eg. yellow).
    # turbo
    # brg
    # nipy_spectral
    # rainbow

    import matplotlib.pylab as pl
    import matplotlib.cm as cm

    if cmap is None:
        cmap = "turbo"
        
    if True:
        pcols = getattr(cm, cmap)(np.linspace(0,1, numcol), alpha=alpha)
    else:
        if cmap=="plasma":
            pcols = pl.cm.plasma(np.linspace(0,1, numcol), alpha=alpha)
        elif cmap=="jet":
            pcols = pl.cm.jet(np.linspace(0,1, numcol), alpha=alpha)
        else:
            print(cmaps)
            assert False, "dont know this CMAP"
    # cool

    if ploton:
        fig, ax = plt.subplots()
        ax.scatter(range(len(pcols)), range(len(pcols)), c=pcols)
        ax.set_title('Color list')
        return pcols, fig, ax
    else:
        return pcols

def colorGradient(pos, col1=None, col2=None, cmap="plasma"):
    """ pos between 0 and 1, gets a value between gradient btw
    col1 and col2. altenratively if leave both col1 and 2 None, 
    then will use camp, if pos <0 or >1 wil clamp to this range"""

    if pos<0:
        pos=0.
    elif pos>1:
        pos =1.

    if col1 is None and col2 is None:
        pcols = makeColors(2, cmap=cmap)
        col1=pcols[0]
        col2 = pcols[1]
    elif col1 is None or col2 is None:
        assert False, "either give me both col1 and 2, or neither"
    else:
        col1=np.array(col1)
        col2 = np.array(col2)
    return (pos*(col2-col1) + col1)[:3]

def map_continuous_var_to_color_range(values, valmin=None, valmax=None,
                                      kind = "diverge", custom_cmap = "rocket"):
    """
    Returns rgba values for each item in values.
    PARAMS:
    - values, array/list of scalar values
    - valmin, the left-most color int he cmap will be mapped to this value.
    if None, then uses the min in values.
    - valmax, see above.

    """
    from matplotlib.colors import Normalize
    if kind == "diverge":
        # middle is black..
        cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    elif kind == "circular":
        # perceptually unifrom
        cmap = sns.color_palette("husl", as_cmap=True)
    elif kind == "sequential":
        # purple --> white
        cmap = sns.color_palette("rocket", as_cmap=True)
    elif kind == "custom":
        cmap = sns.color_palette(custom_cmap, as_cmap=True)
    else:
        print(kind)
        assert False

    # What values to anchor at?
    if valmin is None:
        valmin = min(values)
    if valmax is None:
        valmax = max(values)

    # Normalize data (valmin -->0) (valmax --> 1)
    norm = Normalize(vmin=valmin, vmax=valmax)

    # Convert values to colors
    rgba_values = cmap(norm(values))

    return rgba_values

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties

    FROM: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, scale):
    """ see above, confidence_ellipse (helper function for that)"""
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def savefig(fig, path):
    """ helper to save without clipping axis labels
    """
    fig.savefig(path, bbox_inches="tight")
    #plt.show(fig, block=False) # can uncomment if bugs with memory leak

def saveMultToPDF(path, figs):
    """ saved multiple figs (list of fig obejcts)
    to path as pdf. path should not have pdf in the name"""
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{path}.pdf")
    for fig in figs:
        pdf.savefig(fig, dpi=100, bbox_inches='tight')
    pdf.close()

def shadedErrorBar(x, y, yerr=None, ylowupp = None, ax=None, color="tab:blue",
        alpha_fill = 0.2):
    """ Draw plot with error band and extra formatting to match seaborn style
    - pass eitehr yerr (symmetric) or ylowupp, a list of [yupper, ylower], or neither,
    if no plot error
    from https://stackoverflow.com/questions/56203420/how-to-use-custom-error-bar-in-seaborn-lineplot
    """
    a = yerr is not None
    b = ylowupp is not None
    if a or b:
        assert a!=b, "only guive me one of them"

    if a:
        lower = y - yerr
        upper = y + yerr
    elif b:
        lower, upper = ylowupp
    else:
        lower, upper = None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))

    ax.plot(x, y, label='', color=color)
    if a or b:
        ax.plot(x, lower, color=color, alpha=alpha_fill/2)
        ax.plot(x, upper, color=color, alpha=alpha_fill/2)
        ax.fill_between(x, lower, upper, alpha=alpha_fill, color=color)
    # ax.set_xlabel('timepoint')
    # ax.set_ylabel('signal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.show()
    return ax


def _plotScatterXreduced(X, dims_to_take=None, n_overlay_text = 20, ax=None,
                         color="k", textcolor="r", alpha=0.05,
                         plot_text_over_examples=False, return_inds_text=False,
                         SIZE=7, overlay_mean=False,
                         text_to_plot = None, overlay_ci=True, plot_scatter=True,
                         mean_markersize = 10, mean_alpha=0.9):
    """ 
    GOOD - scatter plot of X, taking oclumns of X as vectors.
    PARAMSL
    - X, (N, m) where m>=2
    - dims_to_take, list of 2 ints, the dimensions to plot,.
    - n_overlay_text, int, overlays text for this many random pts.
    """
    import random

    if text_to_plot is not None:
        plot_text_over_examples = True

    # assert labels is None, "not codede, use plotScatterOverlay"
    if dims_to_take is None:
        dims_to_take = [0,1]
    assert len(dims_to_take)==2 or len(dims_to_take)==3

    Xfit = X[:,dims_to_take]

    # 1) Scatter plot all trials
    if ax is None:
        fig, ax = plt.subplots(figsize=(SIZE,SIZE))
    else:
        fig = None
    ALPHA = Xfit.shape[0]/500

    if plot_scatter:
        if len(dims_to_take)==2:
            ax.plot(Xfit[:,0], Xfit[:,1], "o", color=color, alpha=alpha)
        elif len(dims_to_take)==3:
            ax.plot(Xfit[:,0], Xfit[:,1], Xfit[:,2], "o", color=color, alpha=alpha)
        else:
            print(dims_to_take)
            print(Xfit.shape)
            assert False
    # ax.plot(Xfit[:,0], Xfit[:,1], "x", color=color, alpha=alpha)

    if overlay_mean:
        if len(dims_to_take)==2:
            xmean = np.mean(Xfit[:,0])
            ymean = np.mean(Xfit[:,1])
            ax.plot(xmean, ymean, 's', color=color, markersize=mean_markersize, alpha=mean_alpha)
            if overlay_ci:
                confidence_ellipse(Xfit[:,0], Xfit[:,1], ax, n_std=2, edgecolor = color)
        elif len(dims_to_take)==3:
            xmean = np.mean(Xfit[:,0])
            ymean = np.mean(Xfit[:,1])
            zmean = np.mean(Xfit[:,2])
            ax.plot(xmean, ymean, zmean, 's', color=color, markersize=mean_markersize, alpha=mean_alpha)
            # if overlay_ci:
            #     confidence_ellipse(Xfit[:,0], Xfit[:,1], ax, n_std=2, edgecolor = color)
        else:
            print(dims_to_take)
            assert False

    # === pick out random indices, highlight them in the plot, and plot them
    if plot_text_over_examples and text_to_plot is None:
        # plot rand examples, text of ther indices
        indsrand = random.sample(range(Xfit.shape[0]), nplot)
        indsrand = sorted(indsrand)

        for i in indsrand:
            ax.text(Xfit[i,0], Xfit[i,1], i, color=textcolor, fontsize=10)
    elif plot_text_over_examples and not text_to_plot is None:
        # then text_to_plot is a vector of length Xfit.
        assert isinstance(text_to_plot, list) and len(text_to_plot)==Xfit.shape[0]
        for i in range(Xfit.shape[0]):
            ax.text(Xfit[i,0], Xfit[i,1], text_to_plot[i], color=textcolor, fontsize=7)
    else:
        indsrand = None

    if return_inds_text:
        return fig, ax, indsrand
    else:
        return fig, ax


def plotScatterOverlay(X, labels, dimsplot=(0,1), alpha=0.2, ver="overlay",
    downsample_auto=True, ax=None, SIZE=8, overlay_mean=False,
    ncols_separate = 4, plot_text_over_examples=False, text_to_plot=None,
                       map_lev_to_color=None, color_type="discr",
                        overlay_ci=True, plot_scatter=True,
                       mean_markersize=10, mean_alpha=0.9):
    """ overlay multiple datasets on top of each other
    or separate.
    - X, array shape NxD.
    - labels, vector length N, with label for each sample. 
    Will color differnetly by label.
    - downsample_auto, then subsamples in case there are too many datapts
    """

    if labels is None:
        labels = ["IGNORE_LABEL" for _ in range(X.shape[0])]

    assert X.shape[0] == len(labels)

    if color_type=="discr":
        labellist = set(labels)
        if map_lev_to_color is None:
            # Color the labels
            from pythonlib.tools.plottools import makeColors
            # One color for each level of effect var
            pcols = makeColors(len(labellist))
            map_lev_to_color = {}
            for lev, pc in zip(labellist, pcols):
                map_lev_to_color[lev] = pc

    if downsample_auto:
        # Reduce n poiints to plot, if neded.
        import random
        thresh = 20000
        nthis = X.shape[0]
        if nthis> thresh:
            indsthis = sorted(random.sample(range(X.shape[0]), thresh))
            print(f"Randomly subsampling to {thresh}")
            X = X[indsthis,:]
            labels = [labels[i] for i in indsthis]

    if text_to_plot is not None:
        assert len(text_to_plot)==X.shape[0]

    # Plot
    if ver=="overlay":
        if color_type=="discr":
            # One layer on plot for each level
            for i, l in enumerate(labellist):
                col = map_lev_to_color[l]
                inds = [i for i, lab in enumerate(labels) if lab==l]
                Xthis = X[inds, :]
                if text_to_plot is not None:
                    text_to_plot_this = [text_to_plot[i] for i in inds]
                else:
                    text_to_plot_this = None
                _fig, _ax = _plotScatterXreduced(Xthis, dimsplot, ax=ax,
                                               color=col, textcolor=col, alpha=alpha,
                                                 overlay_mean=overlay_mean, overlay_ci=overlay_ci,
                                                 plot_text_over_examples=plot_text_over_examples,
                                                 text_to_plot = text_to_plot_this,
                                                 SIZE=SIZE, plot_scatter=plot_scatter,
                                                 mean_markersize=mean_markersize, mean_alpha=mean_alpha)
                if i==0:
                    # initiate a plot
                    fig = _fig
                    ax = _ax
                # else:
                #     _plotScatterXreduced(Xthis, dimsplot, ax=ax,
                #                          color=col, textcolor=col, alpha=alpha, overlay_mean=overlay_mean,
                #                          plot_text_over_examples=plot_text_over_examples, text_to_plot = text_to_plot_this)

            # Add legend to the last axis
            # ax.legend(labellist)
            if len(map_lev_to_color.keys())<50:
                legend_add_manual(ax, map_lev_to_color.keys(), map_lev_to_color.values(), 0.2)
        elif color_type=="cont":
            assert len(dimsplot)==2
            xs = X[:, dimsplot[0]]
            ys = X[:, dimsplot[1]]
            sns.scatterplot(x=xs.squeeze(), y=ys.squeeze(), hue=labels, alpha=0.8, ax=ax)

            #
            # fig, ax = plt.subplots(1,1)
            # label_rgbs = map_continuous_var_to_color_range(labels)
            #
            # im = ax.scatter(xs, ys, c=label_rgbs, alpha=0.75, marker="x", cmap=label_rgbs)
            # fig.colorbar(im, ax=ax)
            # assert False

            fig = None
        else:
            print(color_type)
            assert False

        return fig, ax

    elif ver in ["separate", "separate_no_background"]:
        assert False, "no need for separate code here. just run outer mult times."
        n = len(labellist)
        nrows = int(np.ceil(n/ncols_separate))
        fig, axes = plt.subplots(nrows, ncols_separate, figsize=(SIZE*ncols_separate, SIZE*nrows))
        for i, (l, ax) in enumerate(zip(labellist, axes.flatten())):
            inds = [i for i, lab in enumerate(labels) if lab==l]
            Xthis = X[inds]

            if text_to_plot is not None:
                text_to_plot_this = [text_to_plot[i] for i in inds]
            # initiate a plot
            if ver=="separate":
                # Then genreate a background of all pts first.
                _plotScatterXreduced(X, dimsplot, ax=ax,
                                     color="k", textcolor="k", alpha=alpha/5, plot_scatter=plot_scatter)
            _plotScatterXreduced(Xthis, dimsplot, ax=ax,
                                 color=pcols[i], textcolor=pcols[i], alpha=alpha, overlay_mean=overlay_mean,
                                 plot_text_over_examples=plot_text_over_examples, text_to_plot = text_to_plot_this, plot_scatter=plot_scatter)
            ax.set_title(f"label: {l}")

        return fig, axes

    else:
        print(ver)
        assert False
        

def plotScatter45(x, y, ax, plot_string_ind=False, dotted_lines="unity", 
    means=False, labels=None, alpha=0.8, marker="x", x_errors=None, y_errors=None,
                  fontsize=4):
    """ scatter plot, but making sure is square, 
    xlim and ylim are identical, and plotting a unity line
    - plot_string_ind, THen plots 0, 1, 2..., on the pts
    - dotted_lines, {"unity", "plus", "none"}, to verlay
    - means, overlay dot for mean
    - labels, list, same len as x and y, to overlay on dots (text), takes 
    precedence over plot_string_ind
    """


    # Repalce nan errors with 0
    if x_errors is not None:
        x_errors = x_errors.copy()
        x_errors[np.isnan(x_errors)] = 0
    if y_errors is not None:
        y_errors = y_errors.copy()
        y_errors[np.isnan(y_errors)] = 0

    # if np.all(x==0.)
    # print(x, y, y_errors, x_errors)
    # ax.plot(x, y, marker, alpha=alpha)
    ax.errorbar(x, y, y_errors, x_errors, linestyle="", marker=marker, alpha=alpha)

    # ax.set_axis("square")
    minimum = np.min([np.min(x), np.min(y)])
    maximum = np.max([np.max(x), np.max(y)])

    # minimum = np.min((ax.get_xlim(),ax.get_ylim()))
    # maximum = np.max((ax.get_xlim(),ax.get_ylim()))
    ran = maximum - minimum
    MIN = minimum - 0.1*ran
    MAX = maximum + 0.1*ran
    # MIN = minimum
    # MAX = maximum
    if dotted_lines=="unity":
        ax.plot([MIN, MAX], [MIN, MAX], '--k', alpha=0.5)
    elif dotted_lines=="plus":
        ax.plot([0, 0], [MIN, MAX], '--k', alpha=0.5)
        ax.plot([MIN, MAX], [0, 0], '--k', alpha=0.5)
    else:
        assert dotted_lines=="none", "what you want?"

    if means:
        xmean = np.mean(x)
        ymean = np.mean(y)
        ax.plot(xmean, ymean, "ok")

    # overlay strings?
    if labels is not None:
        assert len(labels)==len(x)
        for l, xx, yy in zip(labels, x,y):
            ax.text(xx, yy, l, alpha=0.5, fontsize=6)
    else:

        if plot_string_ind:
            for i, (xx, yy) in enumerate(zip(x,y)):
                ax.text(xx, yy, i, fontsize=fontsize)

    if MAX>MIN:
        ax.set_xlim(MIN, MAX)
        ax.set_ylim(MIN, MAX)
    ax.axis("square")
    # ax.set_aspect('equal', adjustable='datalim')

def hist_with_means(ax, vals, **kwargs):
    """ same, but overlays line for mean
    """
    if len(np.unique(vals))==1:
        print(vals)
        assert False, "need variety"
    from scipy.stats import sem
    tmp = ax.hist(vals, **kwargs)
    bins = tmp[1]
    vmean = np.mean(vals)
    vsem = sem(vals)
    ax.axvline(vmean, color="k")
    ax.axvline(vmean-vsem, color="r")
    ax.axvline(vmean+vsem, color="r")
    return bins


def histogramMult(vals_list, nbins, ax=None, separate_plots=False):
    """ overlays multiple historgrams, making sure to 
    use same bins. first finds bins by taking all values and getting 
    common bins.
    Overlays using steps, using standard mpl colors
    INPUT:
    - vals_list, list of vals, each a (N,) array
    RETURNS:
    - fig
    """

    # Find common bins
    allvals = np.concatenate([x[:] for x in vals_list])
    assert not any(np.isnan(allvals))
    assert len(allvals.shape)==1
    bins = getHistBinEdges(allvals, nbins)
    if separate_plots:
        nc = 3
        nr = int(np.ceil(len(vals_list)/3))
        fig, axes= plt.subplots(nr, nc, figsize=(3*nc, 2*nr))
        for x, ax in zip(vals_list, axes.flatten()):
            ax.hist(x, bins=bins, histtype="step", density=True)
            # print(x)
            # print(bins)
    else:
        if ax is None:
            nc = 1
            nr = 1
            fig, ax= plt.subplots(nr, nc, figsize=(3*nc, 2*nr))
        else:
            fig=None
        for x in vals_list:
            # weights = np.ones_like(x) / (len(x))
            hist_prob(ax, x, bins, "step")
            # ax.hist(x, bins=bins, histtype="step", weights=weights)
        
    return fig

def hist_prob(ax, x, bins, histtype="step"):
    """ liek hist, but plots in prob. not prob density.
    """
    weights = np.ones_like(x) / (len(x))
    ax.hist(x, bins=bins, histtype=histtype, weights=weights)


def getHistBinEdges(vals, nbins):
    """ return array of edges containing all vals,
    and having n bins (so values is nbins+1)
    """
    # get range for binning
    xmin = np.min(vals)
    xmax = np.max(vals)
    
    # append a bit to edges to get all data.
    width = xmax-xmin
    delt = 0.01*width/nbins
    xs = np.linspace(xmin-delt, xmax+delt, nbins+1)
    
    return xs

# def plotHistOfLabels(labels):
#     """ Plot historgram of labels,
#     INPUT:
#     - labels, vector of labels, could be string or num, 
#     """
#     import seaborn as sns
#     plt.figure(figsize=(15,5))
#     sns.histplot(data=SF, x="label", hue="animal_dset", stat="probability", multiple="dodge", element="bars", shrink=1.5)



def plotGridWrapper(data, plotfunc, cols=None, rows=None, SIZE=2.5, 
                   origin="lower_left", max_n_per_grid=None, 
                   col_labels = None, row_labels=None, tight=True,
                   aspect=0.8, ncols=6, titles=None, naked_axes=False, 
                   titles_on_y=False, return_axes =False, fig_axes=None,
                   xlabels = None):
    """ wrapper to plot each datapoint at a given
    col and row.
    INPUT:
    - data = list of datapts. must be compatible with 
    plotfunc
    - plotfunct, func, signature is plotfunc(data[ind], ax)
    - cols and rows, where 0,0 is bottom left of screen (assumes 
    0indexed (same length as data). if either is None, then plots data starting from 0,0 and
    going row by row
    --- ncols, only matters if have to automticalyl get cols and rows.
    - max_n_per_grid, then only plots max n per col/row combo. each time will shuffle
    so that is different. Leave none to plot all
    - col_labels, row_labels, list of things to label the cols and rows. indexed by the row or col
    - titles, same length as data. If present, will overwrite row_labels and col_labels.
    - aspect, w/h
    - clean_axes, then no x or y labels.
    - fig_axes, can pass in axes, but must be in exact shape expected given the other input params.
    (fig, axes)
    - xlabels, list of strings, same len as df, for x axis labels.
    RETURNS:
    - fig, 
    NOTE: will overlay plots if multiple pltos on same on.
    """
    if xlabels is not None:
        tight = False # or else cannot see

    if titles is not None:
        assert isinstance(titles, list)
    else:
        # give it dummy. will not actually use
        titles = ["dummy" for _ in range(len(data))]

    if cols is None or rows is None:
        # get so rows is 0, 1, 2, ... and cols is 0,0,0,...
        cols = []
        rows = []
        n = len(data)
        for i in range(n):
            cols.append(i%ncols)
            rows.append(int(np.floor(i/ncols)))

    if isinstance(cols, list):
        cols = np.array(cols)
    if isinstance(rows, list):
        rows = np.array(rows)


    # if lost row=0, then shift all values down. this could happen if removed datapts becuase of 
    # column pruning. subsequent code needs first row to be 0.
    rows = rows - min(rows)
    
    if min(rows)!=0 or min(cols)!=0:
        print(rows, cols)
        print(min(rows), min(cols))
        print(min(rows)!=0)
        print(min(cols)!=0)
        assert False, "if not, messes up plotting of titles"

    if xlabels is None:
        xlabels = [None for _ in range(len(data))]

    # Package all variables that should not be broken apart
    allvars = [[d, r, c, xl, ti] for d, r, c, xl, ti in zip(data, rows, cols, xlabels, titles)]
    def unpack(allvars):
        data = [t[0] for t in allvars]
        rows = [t[1] for t in allvars]
        cols = [t[2] for t in allvars]
        xlabels = [t[3] for t in allvars]
        titles = [t[4] for t in allvars]
        return data, rows, cols, xlabels, titles

    if max_n_per_grid is not None:
        assert isinstance(max_n_per_grid, int)

        # then shuffle
        import random
        # tmp = [[d, r, c] for d, r, c in zip(data, rows, cols)]
        random.shuffle(allvars)
        data, rows, cols, xlabels, titles = unpack(allvars)
        # data = [t[0] for t in tmp]
        # rows = [t[1] for t in tmp]
        # cols = [t[2] for t in tmp]

    nr = int(max(rows)+1)
    nc = int(max(cols)+1)
    
    if origin=="lower_left":
        rows = max(rows)-rows
    else:
        assert origin=="top_left", "not coded"

    if fig_axes is None:
        fig, axes = plt.subplots(nr, nc, sharex=True, sharey=True, 
                                 figsize=(nc*SIZE*aspect, nr*SIZE), squeeze=False)
    else:
        fig, axes = fig_axes[:]

    done = {}
    minrow = min(rows)
    mincol = min(cols)
    for col, row in zip(cols, rows):
        done[(row, col)] = 0

    if all([t=="dummy" for t in titles]):
        titles = None

    # convert to strings, as this can break code later.
    if titles is not None:
        for i, tit in enumerate(titles):
            if isinstance(tit, tuple):
                try:
                    titles[i] = "-".join([str(x) for x in tit])
                except Exception as err:
                    titles[i] = "TUPLE"

    for i, (dat, col, row, xl) in enumerate(zip(data, cols, rows, xlabels)):
        if max_n_per_grid is not None:
            if done[(row, col)]==max_n_per_grid:
                continue

        ax = axes[row][col]
        plotfunc(dat, ax)

        done[(row, col)] += 1

        if xl is not None:
            ax.set_xlabel(xl)

        if titles is not None:
            if isinstance(titles[i], str):
                if titles_on_y:
                    ax.set_ylabel(f"{titles[i]}")
                else:
                    ax.set_title(f"{titles[i]}")
            else:
                if titles_on_y:
                    ax.set_ylabel(f"{titles[i]:.2f}")
                else:
                    ax.set_title(f"{titles[i]:.2f}")
        # else:
        #     if col_labels:
        #         if row==0:
        #             ax.set_title(col_labels[col])
        #     if row_labels:
        #         if col==0:
        #             ax.set_ylabel(row_labels[row])
        # if naked_axes:
        #     ax.set_yticklabels([])
        #     ax.set_xticklabels([])
        #     ax.tick_params(axis='both', which='both',length=0)

    if tight:
        fig.subplots_adjust(wspace=0, hspace=0)

    # Set the labels
    for col in range(max(cols)+1):
        for row in range(max(rows)+1):

            ax = axes[row][col]
            if titles is None:
                # then labels and columns titles, not panel specific.
                if col_labels:
                    if row==0:
                        ax.set_title(col_labels[col])

            if row_labels:
                if col==0:
                    ax.set_ylabel(row_labels[row])

            if naked_axes:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.tick_params(axis='both', which='both',length=0)

            # smaller font size, since can be long tirles
            ax.title.set_fontsize(10)

    # x and y lim should contain all data
    if return_axes:
        return fig, axes
    else:
        return fig


def get_ylim(vals, pertile=(1.5, 98.5)):
    """ helper, to get ylims for plotting, whch remove outliers,
    """
    # Then remove outliers.
    YLIM = np.percentile(vals[~np.isnan(vals)], pertile)
    ydelt = YLIM[1]-YLIM[0]

    # THis, so doesnt cut off lower data
    YLIM[0]-=0.1*ydelt
    YLIM[1]+=0.1*ydelt
    return YLIM

def subplot_helper(ncols, nrowsmax, ndat, SIZE=2, ASPECTWH = 1, 
        sharex=True, sharey=True, ylim=None, xlim=None):
    """
    Makes multiple figures, each with same num of subplots, for cases with too many supblots for a single figure
    function that returns subplots axes.
    PARAMS:
    - ncols, int
    - nrowsmax, int, resets and starts ne figure if get psat this.
    - SIZE, sides of each subplot
    - ASPECTWH, aspect ratio w/h
    - sharex, share, bool.
    - ylim, (2,) array or list.
    RETURNS:
    - getax, a function, called ax = getax(n) where n is the index to subplot.
    - figholder, list of (fig, axes)), length of num plots.
    - nplots, int.
    EXAMPLE:
    - getax, figholder, nplots = subplot_helper(3, 8, 20)
    """
    
    nrows = int(np.ceil(ndat/ncols))
    nplots = int(np.ceil(nrows/nrowsmax))
    if nplots>1:
        nrows_per_plot = nrowsmax
    else:
        nrows_per_plot = nrows
    
    ndats_per_plot = nrows_per_plot * ncols
    
    figholder = []
    for _ in range(nplots):
        fig, axes = plt.subplots(nrows_per_plot, ncols, sharex=sharex, sharey=sharey, 
                                 figsize=(ncols*SIZE*ASPECTWH, nrows_per_plot*SIZE))
        if ylim is not None:
            for ax in axes.flatten():
                ax.set_ylim(ylim)
        if xlim is not None:
            for ax in axes.flatten():
                ax.set_xlim(xlim)
        figholder.append((fig, axes))
    
    def getax(n, return_fig=False):
        """given the datapoint (n), return the fig
        and axis handle
        RETURNS:
        - ax
        --- or fig, ax if return_fig==True
        """
        
        fignum = int(np.ceil((n+1)/ndats_per_plot))-1
        subplotnum = n%ndats_per_plot
        
        fig, axes =  figholder[fignum]
        ax = axes.flatten()[subplotnum]
        if return_fig:
            return fig, ax
        else:
            return ax
    return getax, figholder, nplots
        
def move_legend_outside_axes(ax):
    """ 
    - ax, already should hav elegend defiened
    """
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def share_axes(axes, which="x"):
    """
    MAke these axes have shared axis, will be scalred based on the lasrgest axis.
    """
    
    if which=="x":
        axes.flatten()[0].get_shared_x_axes().join(axes.flatten()[0], *axes.flatten()[1:])
    elif which=="y":
        axes.flatten()[0].get_shared_y_axes().join(axes.flatten()[0], *axes.flatten()[1:])
    else:
        print(which)
        assert False
def plot_beh_codes(codes, times, ax=None, codenames=False,
                  color="k", yval=1):
    """ Help plot behcodes, which are int or str, along x axis correspnding
    to times.
    PARAMS:
    """

    assert codenames==False, "not coded"       
     
    if ax is None:
        fig, ax = plt.subplots()
        
    for c,t in zip(codes, times):
        ax.plot(t, yval, "o", mfc=color, mec=color)
        ax.text(t, yval, c, color="m", fontsize=8)
        
