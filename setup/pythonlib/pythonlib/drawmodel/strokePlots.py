""" things that take in strokes (list of np arrays) and plots """
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.stroketools import fakeTimesteps, strokesInterpolate

def overlayStrokeTimes(ax, strokes, yfrac=0.9, color="k"):
    """ timecourse plots, overlay times of strokes
    - puts line at ypos defined by yfrac, where 0 is bottom
    and 1 is top
    """
    YLIM = ax.get_ylim()
    y = YLIM[0] + (YLIM[1] - YLIM[0])*yfrac
    for s in strokes:
        on = s[0,2]
        off = s[-1,2]
        ax.hlines(y, on, off, color=color)


def getStrokeColorsGradient(strokes, cmap="winter"):
    """ get evently spaced colors, based on stroke orders
    """
    # cmap = "cool"
    from pythonlib.tools.plottools import makeColors
    color_order = makeColors(len(strokes), alpha=1, cmap=cmap)
    if cmap=="cool":
        color_order = np.flipud(color_order)

    # color_order = color_order[2:]
    if not isinstance(strokes, int):
        color_order_by_pt = []
        for i, s in enumerate(strokes):
            # print(i)
            color_order_by_pt.extend([color_order[i] for _ in range(len(s))])
        # pcol = np.array(pcol)
        # print(
        # assert False
    return color_order, color_order_by_pt

def getStrokeColors(strokes, CMAP="jet"):
    """
    a fixed set of colors, always mapping to stroke nums
    will also apply to stroke onset markers and text.
    Returns:
    - color_order is N x 4, where N is max([5, Nstrokes])
    - color_order_by_pt is expanded to match each timepoint in 
    strokes.
    - easy mode: if want just want stanadard lsit of coloors, 
    pass strokes = 5 (e.g., if 5 is max num colors)
    """
    from pythonlib.tools.plottools import makeColors
    NCOL = 5
    assert NCOL==5, "if you chage this, lose stroke-num standandd"
    if isinstance(strokes, int):
        if strokes>NCOL:
            color_order = makeColors(strokes, cmap=CMAP)
        else:
            color_order = makeColors(NCOL, cmap=CMAP)
        color_order_by_pt = None
    else:
        if len(strokes)>NCOL:
            color_order = makeColors(len(strokes), cmap=CMAP)
        else:
            color_order = makeColors(NCOL, cmap=CMAP)

    # -- make strokes 1 and 2 most perceptulaly different
    # color_order = np.concatenate((color_order[0].reshape(1,-1), color_order[::-1]))
    # color_order = np.concatenate((color_order[0].reshape(1,-1), color_order[-1].reshape(1,-1), color_order[1:-1]))
    color_order = np.concatenate((color_order[0].reshape(1,-1), color_order[::-1]))

    # color_order = color_order[2:]
    if not isinstance(strokes, int):
        color_order_by_pt = []
        for i, s in enumerate(strokes):
            # print(i)
            color_order_by_pt.extend([color_order[i] for _ in range(len(s))])
        # pcol = np.array(pcol)
        # print(
        # assert False
    return color_order, color_order_by_pt

def formatDatStrokesPlot(ax, naked_axes=False, mark_stroke_onset=False, 
    add_stroke_number=False, number_from_zero=False, strokes=None, 
    strokes_cols=None, markersize=None):
    """ Applies formatting to plots of strokes, use this usulaly with either
    plotDatStrokesMapColor or plotDatStrokes  
    PARAMS:
    - mark_stroke_onset, add_stroke_number, number_from_zero, all bools, for marking stroke
    onset with numbers. If do this, then need to pass in strokes and strokes_cols
    - strokes, list of np array, 
    - strokes_cols, list of colors. see other code for color format.

    """
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.set_aspect('equal')
    if naked_axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(None)
        ax.tick_params(axis='both', which='both',length=0)

    if mark_stroke_onset:
        for i, (s, col) in enumerate(zip(strokes, strokes_cols)):
            mfc = col
            tcol=col
            markersize = markersize + 0.5
            ax.plot(s[0,0], s[0,1], mec=col, mfc= mfc, markersize=markersize+1.5, 
                marker="o", alpha= 0.75)
            if add_stroke_number:
                if number_from_zero:
                    snum = i
                else:
                    snum = i+1
                ax.text(s[0,0], s[0,1], f"{snum}", color=tcol, fontsize=markersize+7, alpha=0.7)

    
def plotDatStrokesMapColor(strokes, ax, strokes_values, vmin=None, vmax=None, 
    cmap="winter", markersize=6, alpha=0.55, mark_stroke_onset=True, 
    add_stroke_number=True, naked_axes=False, number_from_zero=False):
    """ plot strokes, similar to plotDatStrokes, but the color is proportional
    to value in strokes_values, where first remapped to range (vmin, vmax), and
    uses color gradient based on that range
    INPUT:
    - strokes_values, list of len same as strokes. Can etiehr be:
    --- list of scalars, all pts in a stroke are same color
    --- list of np ararys, denoting color for each pt. 
    """

    from pythonlib.tools.plottools import colorGradient

    if len(strokes)==0:
        return
        
    # What format was passed in for strokes_values?
    if isinstance(strokes_values[0], np.ndarray) and (strokes_values[0].shape[0]>1 or strokes_values[0].shape[1]>1):
        # Then is list of arrays
        val_ver = "list_of_arrays"
        for vals in strokes_values:
            assert not np.any(np.isnan(vals))
    else:
        val_ver = "list_of_vals"
        assert not np.any(np.isnan(strokes_values))

    if val_ver=="list_of_arrays":
        if vmin is None:
            # Use 2th percentile
            pts = np.concatenate(strokes_values)
            vmin = np.percentile(pts, [2])[0]
        if vmax is None:
            # Use 98th percentile
            pts = np.concatenate(strokes_values)
            vmax = np.percentile(pts, [98])[0]
    else:
        vmin = np.min(strokes_values)
        vmax = np.max(strokes_values)

    # If only one pt or one stroke, then vmin and vmax will be the same. this leads to divide by 0 error below.
    # fix by making them a bit diff, so will plot as if maped to stroke 1.
    if vmin==vmax:
        vmin = vmin
        vmax = vmax+0.01

    # First get colors for each stroke
    color_list = []
    if isinstance(strokes_values[0], np.ndarray) and (strokes_values[0].shape[0]>1 or strokes_values[0].shape[1]>1):
        # Then you want to color each pt differently.
        for i, (s, sv) in enumerate(zip(strokes, strokes_values)):
            # cols = np.array([colorGradient((v-vmin)/(vmax-vmin), cmap=cmap) for v in sv])
            sc = ax.scatter(s[:,0], s[:,1], c=sv, vmin=vmin, vmax=vmax, cmap = cmap)
            # plt.colorbar(sc, cax=ax)
            if mark_stroke_onset:
                mfc = "b"
                tcol= "b"
                mec = "b"
                markersize = markersize + 0.5
                ax.plot(s[0,0], s[0,1], mec=mec, mfc= mfc, markersize=markersize+1.5, 
                    marker="o", alpha= 0.75)
                if add_stroke_number:
                    ax.text(s[0,0], s[0,1], f"{i+1}", color=tcol, fontsize=markersize+7, alpha=0.7)

    else:
        for v in strokes_values:
            col = colorGradient((v-vmin)/(vmax-vmin), cmap=cmap)
            color_list.append(col)

        # print(color_list)
        # print(v, vmin, vmax)
        # assert False
        for i, (s, col) in enumerate(zip(strokes, color_list)):
            ax.plot(s[:,0], s[:,1], color=col, linewidth=(3/5)*markersize,
                alpha=min([1, 1.5*alpha]))

    formatDatStrokesPlot(ax, naked_axes=naked_axes, mark_stroke_onset=mark_stroke_onset, 
        add_stroke_number=add_stroke_number, number_from_zero=number_from_zero, 
        strokes=strokes, strokes_cols=color_list, markersize=markersize)
    

def plotDatStrokesWrapper(strokes, ax, color=None, mark_stroke_onset=True, 
    add_stroke_number=True, mark_stroke_center=False, alpha=0.55):
    """ [GOOD] WRapper to plot strokes a single color
    PARAMS:
    - color, either None (ordinal) or single color string code.
    """

    if color is None:
        # Use default colors.
        plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, add_stroke_number=add_stroke_number, 
            mark_stroke_onset=mark_stroke_onset, number_from_zero=True,
                       mark_stroke_center=mark_stroke_center, alpha=alpha)
    else:
        # use the inputed color
        plotDatStrokes(strokes, ax, plotver="onecolor", pcol=color, 
            force_onsets_same_col_as_strokes=True, each_stroke_separate=True,
            add_stroke_number=add_stroke_number, mark_stroke_onset=mark_stroke_onset,
            number_from_zero=True,
                       mark_stroke_center=mark_stroke_center, alpha=alpha)
        

def plotDatStrokes(strokes, ax, plotver="strokes", fraction_of_stroke=None,
    add_stroke_number=True, markersize=6, pcol=None, alpha=0.55, 
    interpN=None, each_stroke_separate = False, strokenums_to_plot=None, 
    mark_stroke_onset=True, centerize=False, onsets_by_order=True, clean_unordered=False,
    clean_ordered=False, clean_ordered_ordinal=False, clean_task=False, 
    force_onsets_same_col_as_strokes=False, naked=False, mfc_input=None,
    jitter_each_stroke=False, number_from_zero=False,
                   mark_stroke_center=False):
    """given strokes (i.e. [stroke, stroke2, ...], with stroke2 N x 3)
    various ways of plotting
    fraction_of_stroke, from 0 to 1, indicates how much of the trial (i.e., in terms of time) 
    to plot (e.g., can plot timelapse running this multiple times..)
    - pcol, overwrites any gradation in color with one color
    - interpN = 20, then interpolates to fill each stroke.
    - strokenums_to_plot, list of strokenums to plot. if None then plots all.
    e.g., [0,2], then plots 1st and 34d strokes. This only applies if 
    each_stroke_separate is True, so will force the latter.
    - clean_unordered, clean_ordered, shortcuts to plot nice either one color with no 
    markers (unordered) or colored, with markers (stroke onserts)
    """
    import numpy as np

    if fraction_of_stroke is None:
        fraction_of_stroke = []
    if clean_ordered:
        assert clean_unordered==False, "can only choose one of these 2 options"
        each_stroke_separate= True
    elif clean_unordered:
        each_stroke_separate= False
        pcol="k"
        mark_stroke_onset= False
        onsets_by_order=True
    elif clean_ordered_ordinal:
        assert clean_unordered==False, "can only choose one of these 2 options"
        each_stroke_separate= True
        plotver = "order_gradient"
        force_onsets_same_col_as_strokes = True
    elif clean_task:
        add_stroke_number=False 
        each_stroke_separate = True
        mark_stroke_onset=False
        # pcol = [[0.45, 0.3, 0.3]]
        pcol = [[0.3, 0.2, 0.2]]
        alpha = 1
        plotver="onecolor"


    if strokenums_to_plot is not None:
        each_stroke_separate=True
    
    if len(strokes)==0:
        # print("[pythonlib/plotDatStrokes] EMPTY STROKES - not plotting")
        formatDatStrokesPlot(ax, naked_axes=naked)
        return 
        
    from pythonlib.tools.stroketools import fakeTimesteps, strokesInterpolate
    # this code to deal with single dot problems.
    for i, s in enumerate(strokes):
        if len(s.shape)==1:
            # then is just onetimestep, i.e., shape= (3,)
            assert s.shape[0]==3
            s = s.reshape(1,3)
            strokes[i]=s
        # s = s.reshape(-1, 


    # convert all times to the mean time for each stroek, so that is diff color
    import copy
    strokes2 = [s.copy() for s in strokes]

    # append fake times
    if strokes2[0].shape[1]<3:
        strokes2 = fakeTimesteps(strokes2, None, "in_order")
    if not interpN is None:
        strokes2 =strokesInterpolate(strokes2, interpN)

    # ---- centerize
    if centerize:
        c = np.mean(np.concatenate(strokes2), axis=0)[:2]
        strokes2 = [s-np.r_[c,0] for s in strokes2]


    CMAP = "jet"
    # CMAP = "plasma"
    # CMAP = "winter"

    if isinstance(plotver, list) and len(plotver)==3:
        pcol = np.array(plotver).reshape(1,3)
    elif isinstance(plotver, np.ndarray):
        pcol = plotver.reshape(1,3)
    elif isinstance(plotver, str):
        if plotver=="strokes":
            # make color reflect stroke number
            for s in strokes2:
                if len(s)>0:
                    s[:,2] = np.mean(s[:,2])
        elif plotver=="strokes_order":
            # same as storke,s, but ignore time, use order of arrays
            count = 0
            for s in strokes2:
                if len(s)>0:
                    s[:,2] = count
                    count+=1
        elif plotver=="order":
            # a fixed set of colors, always mapping to stroke nums
            # will also apply to stroke onset markers and text.
            color_order, pcol = getStrokeColors(strokes2, CMAP)
        elif plotver=="onecolor":
            for s in strokes2:
                if len(s.shape)>1:
                    s[:,2] = 0

            # CMAP="coolwarm"
            if pcol is None:
                pcol = [[0.3, 0.3, 0.3]]
        elif plotver=="randcolor":
            pcol = np.random.rand(1,3) * 0.7
        elif plotver=="raw":
            # keep as is, so that color reflects time.
            pass
        elif plotver=="order_gradient":
            assert each_stroke_separate==True, "will get color down there"
        else:
            print(plotver)
            print(type(plotver))
            assert False, "dont know this"
    else:
        print(plotver)
        print(type(plotver))
        assert False, "dont know this"

    if fraction_of_stroke:
        each_stroke_separate = False

    if each_stroke_separate:
        # markersize = (3/5)*markersize
        # color scheme must be different for each stroke.
        if isinstance(plotver, str):
            if plotver in ["strokes", "strokes_order", "order"]:
                color_order, pcol = getStrokeColors(strokes2, CMAP)
            elif plotver in ["order_gradient"]:
                color_order, pcol = getStrokeColorsGradient(strokes2)
            elif plotver in ["onecolor", "randcolor"]:
                color_order = [pcol[0] for _ in range(len(strokes2))]
            else:
                color_order, pcol = getStrokeColors(strokes2, CMAP)            
        else:
            color_order = [pcol[0] for _ in range(len(strokes2))]
        # else:
        #     color_order, pcol = getStrokeColors(strokes2, CMAP)

        for i, s in enumerate(strokes2):
            if strokenums_to_plot is not None:
                if i not in strokenums_to_plot:
                    continue
            if jitter_each_stroke:
                from pythonlib.tools.stroketools import getMinMaxVals
                minx, maxx, miny, maxy = getMinMaxVals(strokes)
                delt = np.mean([maxx - minx, maxy-miny])
                JIT = 0.02*delt*i
                sthis = s+JIT   
            else:
                sthis = s
            ax.plot(sthis[:,0], sthis[:,1], color=color_order[i], linewidth=(3/5)*markersize,
                alpha=min([1, 1.5*alpha]))
    else:
        # Then concatenate and then plot. useful if want to define
        # color based on position along entire task

        # print(strokes2)
        strokescat = np.array([ss for s in strokes2 for ss in s])
        # print(strokescat[0])
        # print(len(strokescat))
        # print(strokescat)
        if fraction_of_stroke:
            T = strokescat.shape[0]
            strokescat = strokescat[:int(np.ceil(fraction_of_stroke * T)),:]
            
            # get timepoints
            tvals = np.array([ss for s in strokes2 for ss in s])[:,2]
            timeon = tvals[0]
            timeoff = tvals[int(np.ceil(fraction_of_stroke * T))-1]
        # print(strokescat)
        # print(strokescat[:,0])
        if not pcol is None:
            # print(pcol.shape)
            ax.scatter(strokescat[:,0], strokescat[:,1], c=pcol, 
                marker="o", alpha=alpha, s=markersize)
        else:
            ax.scatter(strokescat[:,0], strokescat[:,1], c=strokescat[:,2], 
                marker="o", cmap=CMAP, alpha=alpha, s=markersize)


    # ==== ADD MARKER AT STROKE ONSET.
    if mark_stroke_onset:
        for i, s in enumerate(strokes2):
            if strokenums_to_plot is not None:
                if i not in strokenums_to_plot:
                    continue
            if force_onsets_same_col_as_strokes:
                col = color_order[i]
                tcol = color_order[i]
            else:
                if onsets_by_order:
                    color_order, pcol = getStrokeColors(strokes2, CMAP)
                    col = color_order[i]
                    tcol = color_order[i]
                if plotver=="order":
                    col = color_order[i]
                    tcol = color_order[i]
                    # tcol = "k"
                elif each_stroke_separate:
                    col = color_order[i]
                    tcol = color_order[i]
                else:
                    col = [0.7, 0.7, 0.7]
                    tcol = "k"
                    tcol = [0.7, 0.7, 0.7]

            if plotver == "onecolor":
                mfc = col
                # mfc = "w"
                markersize = markersize + 0.5
            else:
                mfc = "w"
            if mfc_input:
                mfc = mfc_input
            # ax.plot(s[0,0], s[0,1], mfc=col, mec= [0.7, 0.7, 0.7], markersize=markersize+3, 
                # marker="s")
            ax.plot(s[0,0], s[0,1], mec=col, mfc= mfc, markersize=markersize+1.5, 
                marker="o", alpha= 0.75)
            # ax.text(s[0,0]-10, s[0,1]-11, f"{i+1}", color='k', fontsize=12)

            if add_stroke_number:
                if number_from_zero:
                    snum = i
                else:
                    snum = i+1
                ax.text(s[0,0], s[0,1], f"{snum}", color=tcol, fontsize=markersize+7, alpha=0.7)
                # ax.text(s[0,0], s[0,1], f"{i+1}", color=col, fontsize=12)

                if mark_stroke_center:
                    cen = np.mean(s, axis=0)
                    ax.text(cen[0], cen[1], f"{snum}", color=tcol, fontsize=markersize+7, alpha=0.7)

    if not isinstance(fraction_of_stroke, list):
        return (timeon, timeoff)

    formatDatStrokesPlot(ax, naked_axes=naked)



def plotDatStrokesVelSpeed(strokes, ax, fs, plotver="speed", lowpass_freq=None,
                          overlay_stroke_periods=False, pcol=None, alpha=0.8,
                          nolegend=False, readjust_x_tolims=False):
    """  wrapper, which both processes storkes and plots.
    pass in strokes and will autoamtically differnetiate to 
    get speed/vecopiotu. 
    - lowpass_freq is the final smoothing. around 5 is good for clear bell-shaped.
    around 
    - [NOT FUNCTIONAL] pcol
    """
    from ..tools.stroketools import strokesVelocity

    if plotver == "raw":
        # x y pos
        plotDatStrokesTimecourse(strokes, ax=ax, plotver=plotver, 
            overlay_stroke_periods=overlay_stroke_periods, alpha=alpha, 
            nolegend=nolegend, color=pcol)
    else:
        if plotver=="speed":
            i=1
        elif plotver=="vel":
            i=0
        else:
            assert False

        S = strokesVelocity(strokes, fs, lowpass_freq=lowpass_freq)[i]
        plotDatStrokesTimecourse(S, ax=ax, plotver=plotver, 
            overlay_stroke_periods=overlay_stroke_periods, alpha=alpha, 
            nolegend=nolegend, color=pcol, readjust_x_tolims=readjust_x_tolims)

def plotDatStrokesTimecourse(strokes, ax, plotver="raw", color=None,
    label=None, overlay_stroke_periods=True, alpha=0.8, nolegend=False, 
    readjust_x_tolims = True):
    """given strokes (i.e. [stroke, stroke2, ...], with stroke2 N x 3)
    various ways of plotting, with time on the x axis
    - if color, then will overwrite color which is naturally different fro 
    dim 1 and 2. pass in length 2 list to aply to dim1 and 2.
    """
    YLIM = []
    if label is None:
        label = plotver

    if len(strokes)>0:
        xyt = np.array([ss for s in strokes for ss in s])
        YLIM = (np.nanmin(xyt[:,(0,1)].reshape((-1,1)))-50, 
                np.nanmax(xyt[:,(0,1)].reshape((-1,1)))+50)
        if plotver in ["raw", "vel"]:
            # then is standard. plot timecourse of strokes
            tdim = 2
            if color is None:
                color=["b", "r"]
            else: 
                color=[color, color]
            # elif len(color)!=2:
            #     color=[color, color]
        elif plotver =="speed":
            tdim = 1
            if color is None:
                color=["b"]
            else:
                color=[color]
        else:
            assert False, "not coded"
                
        for i, s in enumerate(strokes):
            for j in range(tdim):
                if i==0:
                    ax.plot(s[:,tdim], s[:,j], ".-", color=color[j], label=f"dim{j}", alpha=alpha)
                    # ax.plot(s[:,tdim], s[:,1], ".r-", color=col2, label='y')
                else:
                    ax.plot(s[:,tdim], s[:,j], ".-", color=color[j], alpha=alpha)
                    # ax.plot(s[:,tdim], s[:,1], ".r-", color=col2)

        # elif plotver=="speed":
        #     # then is standard. plot timecourse of strokes
        #     tdim =1
        #     xyt = np.array([ss for s in strokes for ss in s])
        #     YLIM = (np.nanmin(xyt[:,(0)].reshape((-1,1)))-50, 
        #             np.nanmax(xyt[:,(0)].reshape((-1,1)))+50)
        #     for i, s in enumerate(strokes):
        #         # print(s)
        #         if i==0:
        #             L=label
        #         else:
        #             L = None
        #         ax.plot(s[:,tdim], s[:,0], ".-", color=color, label=L)


        # -- overlay extracted strokes
        if overlay_stroke_periods:
            for s in strokes:
                if len(s)>0:
                    # ax.plot([s[0,tdim], s[-1,tdim]], [YLIM[1], YLIM[1]], '-m')
                    ax.plot([s[0,tdim], s[-1,tdim]], [0,0], '-m', linewidth=4)

        if not nolegend:
            ax.legend() 
            ax.set_title(plotver)

        # ax.set_xlim((-0.1, strokes[-1][-1,tdim]+0.1))
        ax.set_xlabel("time (sec)")
        # ---
        ax.axhline(color="k", linestyle = "--", alpha=0.5)

    return YLIM



def plotDatStrokesMean(strokeslist, ax, strokenum, Ninterp=70, ellipse_std = 1., color=(0,0,0), alpha=0.6, overlay_elipses=False):
    """ given list of strokes (i.e., list of list of nparray)
    plots for  strokenum a single stroke representing mean.
    optioanlykl adds covariance elipses.
    - interpolates based on timepoints to first align strokes.
    - covariance is taken at particular dinices after alginement. by
    default takes onset, midpt, and offset.
    - if a strokes in strokelist is shorter than strokenum, then will
    ignore it. up to you to make sure passing in reasonable strokenums.
    """
    from pythonlib.drawmodel.strokePlots import plotDatStrokes
    from pythonlib.tools.stroketools import strokesInterpolate, strokes_average
    from pythonlib.tools.plottools import confidence_ellipse

    # pick out that desired stroke num
    stroklist = [s[strokenum] for s in strokeslist if len(s)>strokenum]
    if len(stroklist)==0:
        print(f"[plotStrokesMean] no trials had enough strokes for strokenum={strokenum} - not doing antyhing")
        return

    strok_mean, stroklist_interp_stack = strokes_average(stroklist, Ninterp)

    plotDatStrokes([strok_mean], ax=ax, plotver=color, each_stroke_separate=True, alpha=alpha, add_stroke_number=False, markersize=7)

    # == overlay confidence elipses
    if overlay_elipses:
        pts_to_get_elipse = [0, int(np.round(Ninterp/2)), Ninterp-1]
        for pt in pts_to_get_elipse:
            x = stroklist_interp_stack[:,pt,0]
            y = stroklist_interp_stack[:,pt,1]
            confidence_ellipse(x,y, ax=ax, n_std=ellipse_std, edgecolor=color, facecolor=color, alpha=alpha/2)
            ax.plot(np.mean(x), np.mean(y), "o", color=color, alpha=alpha)

        # if False:
        #     # sanity check, plot
        #     plt.figure(figsize=(10,10))
        #     from pythonlib.drawmodel.strokePlots import plotDatStrokes

        #     ax = plt.subplot(211)
        #     plotDatStrokes(slist, ax=ax, plotver="order", each_stroke_separate=False)


        #     ax = plt.subplot(212)
        #     plotDatStrokes(slist_interp, ax=ax, plotver="order", each_stroke_separate=False)
        #     plotDatStrokes([strok_mean], ax=ax, plotver=[0,0,0], each_stroke_separate=True)



def plotDatWaterfallWrapper(strokes_list, strokes_task_list= None, onset_time_list=None,
                            strokes_ypos_list = None,
                            ax=None, colorver="default",
                            cleanver=False, flipxy=False, chunkmodel=None, chunkmodel_idx = 0,
                             waterfallkwargs=None, ylabels=None):
    """ wrapper to plot waterfall, i.e,., raster where y 
    is trial and x is time in trial. Plots multipel trials (strokes_list)
    INPUTS:
    - strokes_list, list of strokes
    - strokes_task_list, optional, list of strokes for tasks corresponding to storkes_list.
    Only need if colorver is by assignment to task stropkes.
    - onset_time_list, times which will subtract from each strokes, e.g., if want to make sure all algined.
    if None, then does not subtract anything.
    - colorver deternbnes colors for each timestep.
    if pass in string, then must be one of these shortcuts:
    "vel", velocity
    "taskstrokenum_fixed", assigned stroke number from ground truth task.
    if pass in function, then should take in strokes and return strokescolor (same
    size as strokes. the stroknum is taken as is from the task struct. useful
    when is identical task across tirals.
    "taskstrokenum_reordered", same but strokenums assigned 0,1,... based on 
    what touched first by behaviro (useful when there is no set order,e.g. when tasks
    differ over trials)
    - cleanver is shortcut to make params: align true, ...
    - chunkmodel, uses this model to parse the task strokes.
    - chunkmodel_idx if this too large, dont exist, then returns None.
    - ylabels, list of names for each item in strokes_list. if empty then just uses 0 , 1, 2...
    """
    from pythonlib.tools.stroketools import assignStrokenumFromTask
    # from pythonlib.drawmodel.strokePlots import getStrokeColors, plotDatWaterfall

    if waterfallkwargs is None:
        waterfallkwargs = {}
    if ylabels is None:
        ylabels = range(len(strokes_list))
    strokescolors_list = []
    

    pcols = getStrokeColors(max([len(strokes) for strokes in strokes_list]))[0] # default.

    # === COLLECT ALL TRIALS.
    for i, strokes in enumerate(strokes_list):

        # 2) -- Collect hwo to color strokes
        def colorbystroknum(sort_stroknum):
            print(chunkmodel_idx)
            strokes_task = strokes_task_list[i]
            # strokes_task = getTrialsTaskAsStrokes(filedata, t,
            #     chunkmodel = chunkmodel, chunkmodel_idx=chunkmodel_idx)
            # if strokes_task is None:
            #     return None
            strokes_colors = assignStrokenumFromTask(strokes, strokes_task, 
                                                         sort_stroknum=sort_stroknum)
            # strokes_colors = [s for s in stroknums_assigned]
            # print(stroknums_assigned)
            # print(strokes_colors)
            # assert False
            if len(strokes_task)>len(pcols):
                pcolsthis = getStrokeColors(strokes_task)[0]
            else:
                pcolsthis = pcols
            for ii in range(len(strokes_colors)):
                strokes_colors[ii] = pcolsthis[strokes_colors[ii]]
            return strokes_colors

        if isinstance(colorver, str):
            if colorver=="vel":
                assert False, "need to pass in fs..."
                fs = filedata["params"]["sample_rate"]
                _, strokes_speed = strokesVelocity(strokes, fs)
                strokes_colors = [s[:,0] for s in strokes_speed]
            elif colorver =="default":
                strokes_colors = [pcols[ii] for ii in range(len(strokes))]
            elif colorver=="taskstrokenum_fixed":
                strokes_colors = colorbystroknum(False)
            elif colorver=="taskstrokenum_reordered":
                strokes_colors = colorbystroknum(True)
            else: 
                print(colorver)
                assert False, "not coded"
        else:
            assert False, "this a function? not yet coded"

        strokescolors_list.append(strokes_colors)

        # === assign colors
    if ax is None:
        if flipxy:
            W = 20
            H = 7
        else:
            W = 7
            H = 20
        fig, ax = plt.subplots(1,1, figsize=(W, H))

    if onset_time_list is not None:
        assert len(onset_time_list)==len(strokes_list)
        strokes_list_new = []
        for i, (onset, strokes) in enumerate(zip(onset_time_list, strokes_list)):
            strokes = [np.copy(s) for s in strokes]
            for s in strokes:
                s[:, 2] = s[:,2] - onset
            # strokes = [s[:, 2] - onset for s in strokes]
            strokes_list_new.append(strokes)
        strokes_list = strokes_list_new

    plotDatWaterfall(strokes_list, strokescolors_list, ax, ylabels=ylabels, flipxy=flipxy, 
        strokes_ypos_list = strokes_ypos_list, **waterfallkwargs)

    waterfallkwargs["xaxis"] = ""
    if flipxy==False:
        ax.set_ylabel("trial")
        ax.set_xlabel(waterfallkwargs["xaxis"])
    else:
        ax.set_xlabel("trial")
        ax.set_ylabel(waterfallkwargs["xaxis"])
    
    if ax is None:
        return fig


def plotDatWaterfall(strokes_list, strokescolors_list, ax, align_by_firsttouch_time=False, 
                    normalize_strok_lengths=False, ylabels=None, xaxis="time", fakegapdist=100., 
                    flipxy=False, align_all_strok=False, trialorder="asinput", 
                    strokes_ypos_list = None,
                    plotkwargs = None):
    """ low-lebvel waterfall plot, i.e., like raster, 
    where y is trials and x is time in trial.
    each strokes_colors in strokes_colors_list must be same dimensions as
    the corresponding strokes in strokes_list, since it
    a color to each timepoint in strokes.
    - if any strokes is empty, leaveas a blank row (skips)
    INPUTS:
    - strokes_list is lsit of strokes, where strokes is
    is llist of N x 3 (xyt) arrays
    - strokescolors_list is list of strokescolors, where
    strokes_colors is list of N x 3 arrays (rgb)
    - strokes_ypos_list, same len as strokes_list, each list within should be
    same length as strokes, but each array (n,) indicating y position for each timepoint.
    Units dont matter, since will normalize across all strokes so that can all fit ont same plot and
    still be proportiona in the y dimension.
    - xaxis,
    -- "time", then is default, time
    -- "dist", then is distance traveled (will add fake gap dist using fakegapdist)
    - flipxy, then trials on x axis, time on y axis.
    - align_all_strok, then strokes place next to each other, all attached to the same axis.
    - trialorder, how to order trials. 
    -- "fromone", means no gaps, and astart numbnering from 1, 2, ...
    -- "asinput", "means incluers gaops, and as inputed."
    -- "nogaps", means as inputed, but skip gaps.
    """
    # yticks = []
    row_nogaps = 0
    row = 1
    if plotkwargs is None:
        plotkwargs = {"marker":"o", "alpha":0.8, "s":10}

    if strokes_ypos_list is not None:
        # do nomalization.
        tmp = np.concatenate([xx for x in strokes_ypos_list for xx in x])
        minval, maxval = np.percentile(tmp, [2, 98])

        def F(strokes):
            return [0.5*(s-minval)/(maxval-minval) for s in strokes]
        strokes_ypos_list = [F(strokes) for strokes in strokes_ypos_list]            

    for k, (strokes, strokescolors) in enumerate(zip(strokes_list, strokescolors_list)):
        

        if len(strokes)==0:
            continue

        if trialorder=="asinput":
            # ylabels_actual.append(ylabels[k])
            row=k
        elif trialorder=="nogaps":
            row = row_nogaps
            row_nogaps+=1

        if xaxis=="time":
            pass    
        elif xaxis=="dist":
            from pythonlib.tools.stroketools import convertTimeCoord
            strokes = convertTimeCoord(strokes, fakegapdist=fakegapdist)

        
        # - plot each stroke separately
        if align_by_firsttouch_time:
            t0 = strokes[0][0,2]
        else:
            t0 = 0.

        # == Plot each single strok
        for i, (strok, strokcols) in enumerate(zip(strokes, strokescolors)):
            
            tvals = strok[:,2] - t0

            if normalize_strok_lengths:
                if xaxis=="time":
                    ontime = (i) + 0.5*i
                    tvals = np.linspace(ontime, ontime+1, len(tvals))
                elif xaxis=="dist":
                    # then need to do interpolation, since x vals not
                    # yet evenly sampled.
                    assert False, " not done!! need to interpolate teh colors so matches strok after strok is interpolated..."
                    strok = strokesInterpolate2([strok], ["npts", 100])[0]
                    strokcols = strokesInterpolate2([strokcols], ["npts", 100])[0]
                    tvals = strok[:,2] - t0
                    ontime = (i) + 0.5*i
                    tvals = np.linspace(ontime, ontime+1, len(tvals))


            if align_all_strok:
                tvals -= tvals[0]
                N = len(strokes)-1
                W = 0.5
                rowthis = row - W/2 + i*W/N
            else:
                rowthis = row



    #         plt.plot(tvals, t*np.ones_like(tvals), color=pcols[i])
    #         try:
            if strokes_ypos_list is not None:
                ythis = strokes_ypos_list[k][i] + rowthis
            else:
                ythis = rowthis*np.ones_like(tvals)
            # print(ythis)
            # print(len(strokes_ypos_list))
            # print(len(strokes_ypos_list[k]))
            # asdfsfsad
            if flipxy:
                x = ythis
                y = tvals
            else:
                x = tvals
                y = ythis
            # ax.scatter(tvals, row*np.ones_like(tvals), c=strokcols,  cmap="plasma", 
            #     **plotkwargs)



            if flipxy:
                marker = "_"
            else:
                marker = "|"
            marker = "o"
            if False:
                ax.plot(x[0], y[0], mec="k", mfc= strokcols[0][:3],
                    markersize=plotkwargs["s"], marker=marker, alpha= 1)
            else:
                ax.plot(x[0], y[0], mec="k", mfc= "none",
                    markersize=plotkwargs["s"], marker=marker, alpha= 1)
            if max(strokcols.shape)<=4:
                # then is color for entire
                ax.scatter(x, y, color=strokcols,  cmap="plasma", 
                    **plotkwargs)
            else:
                # then is color for each pt.
                ax.scatter(x, y, c=strokcols,  cmap="plasma", 
                    **plotkwargs)
            if flipxy:
                ax.axvline(rowthis, color="k", alpha=0.15)
            else:
                ax.axhline(rowthis, color="k", alpha=0.15)
    #         except:
    #             plt.scatter(tvals, t*np.ones_like(tvals), c="k")

        if False:
            # flatten strokes into timepoints of samples
            tvals = np.concatenate([s[:,2] for s in strokes], axis=0)
            plt.plot(tvals, t*np.ones_like(tvals), color=col)
    
    if True:
        if flipxy==False:
            ax.set_yticks(range(row+1))
            # ax.set_yticks(yticks)
            if trialorder=="asinput":
                if ylabels is not None:
                    ax.set_yticklabels(ylabels)
            #         ax.ylabels("trial")
        else:
            ax.set_xticks(range(row+1))
            # ax.set_yticks(yticks)
            if trialorder=="asinput":
                if ylabels is not None:
                    ax.set_xticks(ylabels)
            #         ax.ylabels("trial")



def plotStroksInGrid(stroklist, ncols=5, titlelist = None):
    """ quick plot, in grid, centered."""
    assert False, "see dataset_strokes.plot_examples_grid, and replace here."
    nrows = int(np.ceil(len(stroklist)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2), sharex=True, sharey=True)
    
    for i, (strok, ax) in enumerate(zip(stroklist, axes.flatten())):
        plotDatStrokes([strok], ax)
        if titlelist is not None:
            ax.set_title(titlelist[i])
        else:
            ax.set_title(i)
    
    MIN = np.min([np.min(s[:, [0,1]]) for s in stroklist])
    MAX = np.max([np.max(s[:, [0,1]]) for s in stroklist])
    M = np.max([np.abs(MIN), np.abs(MAX)])
        
    XLIM = [-M, M]
    YLIM = [-M, M]
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    
    return fig


def plotSketchpad(spad, ax):
    """ overlay in gray sketchpad, in format
    2 x 2, [[-x, +x],[-y +y]]
    """
    X = spad
    ax.hlines(X[1,:], X[0,0], X[0,1], color="k", alpha=0.3)
    ax.vlines(X[0,:], X[1,0], X[1,1], color="k", alpha=0.3)

def overlay_stroke_on_plot_mult_rand(strokes, xs, ys, ax, nrand=10, align_to="onset", color=None):
    """
    Helper to overlay multipel small icons of strokes on a single axis, sampling random ones if needed.
    E,g,m to overlay on a scatterplot the strokes associated with each datapt.
    :param strokes:
    :param xs:
    :param ys:
    :param nrand:
    :param align_to:
    :param color:
    :return:
    """
    import random

    assert len(strokes)==len(xs)==len(ys)

    inds = range(len(strokes))

    if len(inds)>nrand:
        inds = random.sample(inds, nrand)

    for i in inds:
        strok = strokes[i][:, :2]
        # times = strokes[i][:, 2]
        x = xs[i]
        y = ys[i]
        overlay_stroke_on_plot(strok, x, y, ax, align_to=align_to, color=color)


def overlay_stroke_on_plot(strok, x, y, ax, SIZE = 0.04, align_to="onset", color=None,
                           alpha=0.5):
    """
    Overlay a strok onto a plot, scaling, shearing, to fit aspect ratio and size of plot,
    and aligning to onset of a pt. Useful for showing example drawings or task for
    indiv datapts.
    PARAMS:
    - strok, np array, (:, 3)
    - x, y, coor to plot
    - SIZE, frac of screen to make the size of the strok.
    - align_to, str, whether to align strok to pt by strok's onset or center
    """

    if color is None:
        color="k"
    from pythonlib.tools.stroketools import strokes_bounding_box_dimensions
    # Get dimensions of figure, so can correctly scale stroke
    XLIM = ax.get_xlim()
    YLIM = ax.get_ylim()
    xdiff = XLIM[1] - XLIM[0]
    ydiff = YLIM[1] - YLIM[0]
    d_ax = (xdiff**2 + ydiff**2)**0.5

    # Decide if align at onset (beh) or center (task)
    from pythonlib.tools.stroketools import strokes_alignonset, strokes_centerize
    import matplotlib.pyplot as plt
    if align_to == "onset":
        strok = strokes_alignonset([strok])[0]
    elif align_to == "center":
        strok = strokes_centerize([strok])[0]
    elif align_to is None:
        pass
    else:
        print(align_to)
        assert False

    # Rescale stroke
    w, h, d = strokes_bounding_box_dimensions([strok])
    rescale = SIZE * d_ax/d
    aspect = [1, ydiff/xdiff]
    strok = strok * rescale # rescale to page
    strok = strok * aspect # to make sure strok aspect is visually accurate
    strok[:,0] += x
    strok[:,1] += y # move to location
    # strok = strok + np.array([x,y])

    # Plot
    ax.plot(strok[:,0], strok[:,1], '-', alpha=alpha, color=color)
    # ax.scatter(strok[:,0], strok[:,1], c=times)
    # if align_to=="onset":
    ax.plot(x, y, 'o', mfc="none", alpha=0.3, color=color)


