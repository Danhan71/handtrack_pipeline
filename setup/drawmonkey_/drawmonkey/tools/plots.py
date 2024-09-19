""" Generally useful plotting code"""
import numpy as np
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from tools.utilsh5 import group2dict
from tools.utils import *
from tools.calc import *

from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse

# def plotTrialSingleOverview(data, TrialRecord, MLConfig, t):
def plotTrialSingleOverview(filedata, t):

    """filedata corresponds to a single .h5 file and t is the trial to plot"""
    params = filedata["params"]
    TrialRecord = filedata["TrialRecord"]
    # data = filedata["data"]
    resolution = params["resolution"]
    beh_codes = params["beh_codes"]


    # 1) plot things related to the trial's data
    # extract x and y. (1) convert from deg to pixlels and (2) rotate so it is in perpsective of animal
    xyt = getTrialsTouchData(filedata, t, post_go_only=False)
    dat_touch_x = xyt[:,0]
    dat_touch_y = xyt[:,1]
    dat_t = xyt[:,2]

    xyt = getTrialsTouchData(filedata, t, post_go_only=True)
    if len(xyt)>0:
        x_go = xyt[:,0]
        y_go = xyt[:,1]
        t_go = xyt[:,2]
    else:
        x_go, y_go, t_go = [], [], []

    # summary of trial outcomes
    to = getTrialsOutcomesAll(filedata, t)
    trialoutcomes = to["trialoutcomes"]
    fracinkgotten = to["fracinkgotten"]
    errorcode = to["errorcode"]

    # behavioral codes
    trial_behcodes = getTrialsBehCodes(filedata, t)
    dat_behcodes_num = trial_behcodes["num"]
    dat_behcodes_time = trial_behcodes["time"]


    ##################### FIGURES
    fig = plt.figure(figsize=(10,20))
    gridspec.GridSpec(4,3)

    ax = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=1)
    plotTrialSimpleTimecourse(filedata, t, ax=ax)

    ax = plt.subplot2grid((4,3), (1,0), colspan=3, rowspan=1)
    ax = plotTrialSimpleTimecourse(filedata, t, ax=ax, plotver="speed")
    
    # overlay more greatly smoothed
    strokes_vels = getTrialsStrokesVelocity(filedata, t, hi_smoothing=True)
    for sv in strokes_vels:
        ax.plot(sv[:,1], sv[:,0], '-r', linewidth=3, alpha=0.6, label="hi_smoothing")


    ax = plt.subplot2grid((4,3), (2,0), colspan=3, rowspan=1)
    ax = plotTrialSimpleTimecourse(filedata, t, ax=ax, plotver="xyvels")

    ax = plt.subplot2grid((4,3), (3,0), colspan=1, rowspan=1)
    plotTrialSimple(filedata, t, ax=ax, post_go_only=False)

    ax = plt.subplot2grid((4,3), (3,1), colspan=2, rowspan=1)
    plotTrialSimple(filedata, t, ax=ax, zoom=True, plotver="order", kwargs={'each_stroke_separate': True}, use_peanut_params={"replaynum":None, "active":True})

    # 4) show trial outcomes

    # ax = plt.subplot2grid((3,3), (2,2))
    # if False:
    #     for i, key in enumerate(trialoutcomes.keys()):
    #         plt.text(0,i, f"{key}:{trialoutcomes[key]}")
    #     plt.ylim((-0.5, 3.5))
    #     plt.xlim((-0.2, 2))
    # else:
    #     plotTrialSimple(filedata, t, ax=ax, post_go_only=True, zoom=True)

    return fig





def plotTrialSimpleTimecourse(filedata, trial, ax=None, post_go_only=False, 
    overlay_strokes=True, plotver="raw", ignore_behcode_if_off_xaxis=True,
    plottitle=True, clean=False):
    """single panel plot, useful for viewing many trials
    try to give it axes. by default only plots 
    touches occuring after go cue."""
    if ax is None:
        fig = plt.figure(figsize=(15,5))
        ax = plt.subplot(1,1,1)

    resolution = filedata["params"]["resolution"]
    
    if clean:
        post_go_only=False
        overlay_strokes=False
        plotver="xyvels"
        ignore_behcode_if_off_xaxis=True
        plottitle=False

    # summary of trial outcomes
    to = getTrialsOutcomesAll(filedata, trial)
    trialoutcomes = to["trialoutcomes"]
    fracinkgotten = to["fracinkgotten"]
    errorcode = to["errorcode"]
    trial_behcodes = getTrialsBehCodes(filedata, trial)
    dat_behcodes_num = trial_behcodes["num"]
    dat_behcodes_time = trial_behcodes["time"]

    # 3) Plot ground truth task
    task = getTrialsTask(filedata, trial)
    task_x = task["x_rescaled"]
    task_y = task["y_rescaled"]

    # plot fixation point

    # a) plot the actual task
#     ax.plot(task_x, task_y, 'xk')
    
    if int(errorcode[0])==0:
        titlecol = "b"
    elif int(errorcode[0])==5:
        titlecol = "c"
    elif int(errorcode[0]) in (3,4):
        titlecol = "r"
    elif int(errorcode[0])==6:
        titlecol = [0.8, 0.4, 0.1]


    if plotver=="empty":
        YLIM = [-1, 1]
        # print(1)
        XLIM = [0, 1]
        return ax
    else:
        # print(2)
        if plotver=="raw":
            strokes = getTrialsStrokes(filedata, trial)
            YLIM = plotDatStrokesTimecourse(strokes, ax, plotver=plotver)
        elif plotver=="speed":
            strokes_vels = getTrialsStrokesVelocity(filedata, trial)
            ax = plotTrialSimpleTimecourse(filedata, trial, ax=ax, plotver="empty")
            YLIM = plotDatStrokesTimecourse(strokes_vels, ax, plotver="speed")
        elif plotver=="xyvels":
            try:
                strokes_vels = getTrialsStrokesVelocityXY(filedata, trial)
            except:
                return ax
            ax = plotTrialSimpleTimecourse(filedata, trial, ax=ax, plotver="empty")
            
            # x and y plot separately
            strokes_vels_x = [s[:,[0,2]] for s in strokes_vels]
            strokes_vels_y = [s[:,[1,2]] for s in strokes_vels]
            YLIM = plotDatStrokesTimecourse(strokes_vels_x, ax, plotver="speed", color="b", label="vel_x", 
                overlay_stroke_periods=False)
            YLIM = plotDatStrokesTimecourse(strokes_vels_y, ax, plotver="speed", color="r", label="vel_y", 
            overlay_stroke_periods=overlay_strokes)
        else:
            print(plotver) 
            assert False, "not coded"
        XLIM = plt.xlim()

    # Overlay times of behavioral codes.
    if YLIM:
        for i, (num,time) in enumerate(zip(dat_behcodes_num, dat_behcodes_time)):
    #             plt.plot([time, time], (-resolution[0]/2, resolution[0]/2), "-k")
            if ignore_behcode_if_off_xaxis:
                if time>XLIM[1]+0.5:
                    continue
            ax.plot([time, time], (YLIM[0], YLIM[1]), ":k")            
            ax.text(time, (i*(YLIM[1]-YLIM[0])/len(dat_behcodes_num))+YLIM[0], filedata["params"]["beh_codes"][num], rotation=45)
        if ignore_behcode_if_off_xaxis:
            ax.set_xlim(right=min(XLIM[1]+1, plt.xlim()[1]))
        else:
            ax.set_xlim(right=max(dat_behcodes_time)+0.2)
        # print(min(XLIM[1]+1, plt.xlim()[1]))
    # print(to['trialoutcomes'])
    # ax.set_title(f"t{trial},{task['str']},error {int(errorcode[0])},\n{to['trialoutcomes']['failure_mode']} ({fracinkgotten})\nnumraise({to['trialoutcomes']['num_finger_raises']})", color=titlecol)
    if plottitle:
        ax.set_title(f"t{trial},{task['str']}, error {int(errorcode[0])},\n{to['trialoutcomes']['failure_mode']} ({fracinkgotten}), numraise({to['trialoutcomes']['num_finger_raises']})", color=titlecol)
    return ax

def plotTaskWrapper(filedata, trial, ax=None, chunkmodel=None):
    """ simple wrapper to make nice llooking plot of task.
    also colors chunks correctly."""

    return plotTrialSimple(filedata, trial, ax=ax,plot_drawing_behavior=False, 
                    taskchunkkwargs={"chunkmodel":chunkmodel}, 
                    taskplot_colorstroke=True, task_alpha=0.8, 
                   taskplot_kwargs = {"LineWidth":5}, zoom=False, 
                    nakedplot=True, empty_title=True)


def plotTrialSimple(filedata, trial, ax=None, post_go_only =True,  plotver="raw", 
    window_rel_go_reward=(-0.1, 0.1), zoom=False,
    fraction_of_stroke=None, nakedplot=False, add_stroke_number=True,
    plot_task_stimulus=True, plot_drawing_behavior=True, plot_fix=True, 
    use_peanut_params=None, only_first_n_strokes=None,
    overlay_guide_dots=True, markersize=6, alpha=0.55, interpN=200, 
    kwargs = None, return_color=False, 
    task_alpha=0.25, empty_title=False, taskchunkkwargs = None, 
    taskplot_colorstroke=False, taskplot_kwargs = None, centerize=False, clean=False, 
    plot_done_button=False, plot_sketchpad=False, overlay_scoring_features=True):
    """single panel plot, useful for viewing many trials
    try to give it axes. by default only plots 
    touches occuring after go cue.
    - use_peanut_params, if want to use strokes defined as overlapping
    peanuts. this is closest to what the moineky was seeinga nd producing. a 
    problem is this only works for things after summer 2020.
    - only_first_n_strokes, if number then oplots at most up to those N strokes.
    - interpN, to interp strokes to smooth.
    - NOTE: centerize currently centers task and behavior seprately, may not be aligned...
    """
    if taskplot_kwargs is None:
        taskplot_kwargs = {}
    if taskchunkkwargs is None:
        taskchunkkwargs= {"chunkmodel":None}
    if use_peanut_params is None:
        use_peanut_params= {"replaynum":None, "active":False}
    if kwargs is None:
        kwargs= {"strokenums_to_plot":None, "each_stroke_separate":False}

    if fraction_of_stroke is None:
        fraction_of_stroke = []
    if clean:
        # plot_fix=False
        kwargs["each_stroke_separate"]=True
        use_peanut_params["active"]=True
    taskplot_kwargs["alpha"]=task_alpha
    if not ax:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)

    # print(filedata.keys())
    resolution = filedata["params"]["resolution"]
    
    # 3) Plot ground truth task
    task = getTrialsTask(filedata, trial)
    task_x = task["x_rescaled"]
    task_y = task["y_rescaled"]
    strokes_task = getTrialsTaskAsStrokes(filedata, trial, **taskchunkkwargs)
    if centerize:
        from pythonlib.tools.stroketools import standardizeStrokes
        strokes_task = standardizeStrokes(strokes_task, ver="centerize")
    if plot_task_stimulus:
        if taskplot_colorstroke:
            from pythonlib.drawmodel.strokePlots import getStrokeColors
            pcols = getStrokeColors(strokes_task)[0]
        else:
            pcols = ["k" for _ in range(len(strokes_task))]

        for pcol, S in zip(pcols, strokes_task):
            ax.plot(S[:,0], S[:,1], "-", color=pcol, **taskplot_kwargs)

    # --- TITLE, BASED ON OUTCOME
    if True:
        # new version, after moved away from error codes, instead using
        # beh eval

        outcome = getTrialsOutcomesWrapper(filedata, trial)
        if getTrialsFixationSuccess(filedata, trial):
            meth = outcome["trial_end_method"]
            behscore = outcome["beh_evaluation"]["beh_multiplier"]
            binaryscore = outcome["beh_evaluation"]["binary_evaluation"]
            reward = outcome["beh_evaluation"]["rew_total"]

            if "modelscore" in outcome["beh_evaluation"]["output"].keys():
                mscore = outcome["beh_evaluation"]["output"]["modelscore"]["value"][0][0]
            else:
                mscore = np.nan

            if meth=="online_abort":
                M = f"{meth}-{outcome['online_abort']['failure_mode']}"
            else:
                M = meth

            replayStats = getTrialsReplayStats(filedata, trial)

            title = f"t{trial}q{getTrialsBloque(filedata,trial)}k{getTrialsBlock(filedata,trial)}|{task['str']}\n{M}|beh{behscore[0][0]:.2f}rew{reward[0][0]:.2f}mod{mscore:.0f}"
            if replayStats is not None:
                nreplays = replayStats["count"][0][0]
                title += f"|rpl:{int(nreplays)}"

            # old coloring scheme - pretty used to this...
            to = getTrialsOutcomesAll(filedata, trial)
            errorcode = to["errorcode"]
            if int(errorcode[0])==0:
                titlecol = "b"
            elif int(errorcode[0])==5:
                titlecol = "c"
            elif int(errorcode[0]) in (3,4):
                titlecol = [0.8, 0.4, 0.1]
            elif int(errorcode[0])==6:
                titlecol = "r"

            # @kgg 221107 - v1 gradient coloring scheme
            # - tried perceptually uniform colormaps (plasma, veridis)
            # - also tried divergent colormap (RdYlGn)
            # - RESULT: hard to read/interpret...
            # 
            # from pythonlib.tools.plottools import colorGradient
            # behscore_final = behscore[0][0]
            # titlecol = colorGradient(behscore_final,cmap='RdYlGn')

            # @kgg 221107 - v2 gradient coloring scheme
            # - tried using discrete colors for 4 buckets
            # - RESULT: not as good as the old scheme
            #
            # behscore_final = behscore[0][0]
            # if 0.0 <= behscore_final < 0.25:
            #     titlecol = "maroon"
            # elif 0.25 <= behscore_final < 0.50:
            #     titlecol = "salmon"
            # elif 0.50 <= behscore_final < 0.75:
            #     titlecol = "skyblue"
            # elif 0.75 <= behscore_final <= 1.0:
            #     titlecol = "mediumaquamarine"
            # else:
            #     assert(False), "possibly a floating point rounding issue"

        else:
            # then did not even fixate
            title = f"t{trial},{task['str']}\nfailed to fixate"
            titlecol = "r"

    else:
        # summary of trial outcomes
        to = getTrialsOutcomesAll(filedata, trial)
        trialoutcomes = to["trialoutcomes"]
        fracinkgotten = to["fracinkgotten"]
        errorcode = to["errorcode"]

        # 4) what color to make the title? to indicate outcome
        if int(errorcode[0])==0:
            titlecol = "b"
        elif int(errorcode[0])==5:
            titlecol = "c"
        elif int(errorcode[0]) in (3,4):
            titlecol = "r"
        elif int(errorcode[0])==6:
            titlecol = [0.8, 0.4, 0.1]

        title = f"t{trial},{task['str']},error {int(errorcode[0])},\n{to['trialoutcomes']['failure_mode']} ({fracinkgotten})"

    # if plotver=="raw":
    #     # just plot the raw data
    #     # extract x and y. (1) convert from deg to pixlels and (2) rotate so it is in perpsective of animal
    #     xyt = getTrialsTouchData(filedata, trial, post_go_only)
    #     if len(xyt)>0:
    #         # b) overlay the drawing
    #         ax.scatter(xyt[:,0], xyt[:,1], c=xyt[:,2], marker="o", cmap="plasma")
    if plot_drawing_behavior:
        if plotver=="empty":
            # just return axes, but with appropriate scaling, labels, etc.
            pass
        else:
            # plot segmented strokes data
            if use_peanut_params["active"] is True:
                # then get strokes from peanut data
                strokes = getTrialsStrokesByPeanuts(filedata, trial, replaynum=use_peanut_params["replaynum"])
            else:
                # default, get from analog touch.
                strokes = getTrialsStrokes(filedata, trial, window_rel_go_reward)
            if only_first_n_strokes is not None:
                strokes = strokes[:only_first_n_strokes]
            if strokes:
                # convert all times to the mean time for each stroek, so that is diff color
                # print(f"alpha (plotdatrstrokes): {alpha}")
                if centerize:
                    strokes = standardizeStrokes(strokes, ver="centerize")
                out = plotDatStrokes(strokes, ax, plotver=plotver, fraction_of_stroke=fraction_of_stroke,
                    add_stroke_number=add_stroke_number, markersize=markersize, alpha=alpha, interpN=interpN, 
                    **kwargs)

        # -- plot fixation point
        if plot_fix:
            try:
                fixpos = getTrialsFix(filedata, trial)["fixpos_pixels"]
                ax.plot(fixpos[0], fixpos[1], 'xk', alpha=0.5)
            except KeyError as err:
                if err.args[1]=="nokey":
                    pass
            except:
                import sys
                print("Unexpected error:", sys.exc_info()[0])
                # import pdb
                # pdb.set_trace()
                raise
        if plot_done_button:
            donepos = getTrialsDoneButtonPos(filedata, trial).squeeze()
            # print(donepos.shape)
            ax.plot(donepos[0], donepos[1], 'xr', alpha=0.5)
        if plot_sketchpad:
            # border of sketchpad
            spad = getTrialsSketchpad(filedata, trial) # [[-x -y], [+x +y]]
            from pythonlib.drawmodel.strokePlots import plotSketchpad
            plotSketchpad(spad.T, ax=ax)
        if overlay_scoring_features:
            # get list of scoring features
            if outcome["beh_evaluation"] is not None:
                feature_dict = outcome['beh_evaluation']['output']
                feature_str = ''
                for f_name in feature_dict:
                    # print("HERE")
                    # for k, v in feature_dict.items():
                    #     print(k, v)
                    # print(f_name)
                    # print(feature_dict[f_name])
                    f_rescale = feature_dict[f_name]['rescale'][0][0]
                    f_raw = feature_dict[f_name]['value'][0][0]
                    feature_str += f_name + ': ' + "{:.2f}".format(f_rescale) + ' [' + "{:.2f}".format(f_raw) + ']' + '\n'
                # print(feature_str)
                xlimits = ax.get_xlim()
                ylimits = ax.get_ylim()
                ax.text(xlimits[0], ylimits[1], feature_str, alpha=0.25, fontsize='x-small',
                        fontweight='normal',ha='left',va='top')


    if overlay_guide_dots:
        gd = getTrialsGuideDots(filedata, trial)
        if gd is not None:
            ax.plot(gd[:,0], gd[:,1], 'x', color=[0.3, 0.6, 0.6], alpha=1, markersize=10)

    if not empty_title:
        ax.set_title(title, color=titlecol)
    
    if zoom==False:
        ax.set_xlim((-resolution[1]/2, resolution[1]/2))
        ax.set_ylim((-resolution[0]/2, resolution[0]/2))
    # ax.axis("equal")
    ax.set_aspect('equal')

    if nakedplot:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='both',length=0)
    if isinstance(fraction_of_stroke, list)==False and plot_drawing_behavior==True:
        return ax, out
    else:
        return ax, None
    
def plotGetBlankCanvas(fd, ax=None, no_xylim = False):
    """ get canvas in proportions of screen """
    if no_xylim:
        nakedplot=True
    else:
        nakedplot=False
    ax, _ = plotTrialSimple(fd, 1, ax=ax, plotver="empty", empty_title=True, 
                         plot_fix = False, nakedplot=nakedplot, plot_task_stimulus = False)
    return ax

def plotTrial2dGrid(dat, fdlist = None, cat1=None, cat2=None, ver="beh", 
    replaynum=1, nstrokes=None, plotargs=None, strokenums_to_plot_alone=None,
    kwargs=None, overlay_stroke_mean=False, 
    plot_task_last_col=False, clean=False, return_placed=False):
    """ plot 2d grid, where each axis defined by cat1/cat2, and
    each cell is overlaid multiple trials. can have options for what 
    to plot in each cell.
    - dat is list of dicts, where keys in dict must correspond to "name_of_category" 
    for both cat1 and 2, and must have a "trial" and "filedata" entries.
    - dat can also be pandas dataframe, same requirements as avbove.
    - if pass fdlist, msut be list of fd, same length as dat, adn will use this instead of
    dat[i]["filedata"]
    - cat1/2 = ["name_of_category", [list of elements]]. cat1 will form 
    y axis.
    - can choose to only entire cat1 or cat2, and wuill then make a single row or col 
    plot, respectively.
    - if leave index 1 empty in cat1 or cat2, then will use entire set of elemnets
    - strokenums_to_plot_alone, a list of ints (0,1,2,..) indexing stroke nums that 
    want to create spearte rows for. then will expand rows. each row will expand into N rows, where N is lengtho f strokenums_to_plot_alone. Examples:
    strokenums_to_plot_alone = [0,1,2]
    strokenums_to_plot_alone = [0, [0,1], [0,1,2]] (here increasinglyu overlays)
    strokenums_to_plot_alone=[0, [0,1], None], (1st, 1st+2nd, all)
    - plot_task_last_col, then plots task ont he last column. decides what the task is based on the
    first trial that satisfies that row. (is sueful if columsna re examples, for instaince)
    """
    import pandas as pd
    if kwargs is None:
        kwargs = {"each_stroke_separate":True}
    if plotargs is None:
        plotargs = {"zoom":True, "plotver":"order", "markersize":8, "alpha":0.6}

    assert cat1 is not None or cat2 is not None, "at least one cat must be etnered"
    if clean:
        kwargs["each_stroke_separate"] = True
        plotargs["empty_title"] = True

    if isinstance(dat, pd.core.frame.DataFrame):
        from pythonlib.tools.pandastools import df2dict
        dat = df2dict(dat)
    # import copy
    # dat = copy.deepcopy(dat)

    # if cat1 is None or cat2 is None:
    #     for d in dat:
    #         d["dummy"]=0
    #     if cat1 is None:
    #         cat1 = ["dummy", [0]]
    #     if cat2 is None:
    #         cat2 = ["dummy", [0]]

    if cat1 is None:
        cat1 = [None, [None]]
    if cat2 is None:
        cat2 = [None, [None]]

    if len(cat1[1])==0:
        cat1[1] = sorted(set([d[cat1[0]] for d in dat]))
    if len(cat2[1])==0:
        cat2[1] = sorted(set([d[cat2[0]] for d in dat]))

    if plot_task_last_col:
        ncols = len(cat2[1])+1
    else:
        ncols = len(cat2[1])

    if strokenums_to_plot_alone is not None:
        nrows = len(strokenums_to_plot_alone)*len(cat1[1])
    else:
        nrows = len(cat1[1])

    fig, axes = plt.subplots(nrows, ncols, sharex=True, 
                             sharey=True, figsize=(len(cat2[1])*4, len(cat1[1])*6), squeeze=False)
    
    # Use fdlist?
    if fdlist is not None:
        assert len(fdlist) == len(dat), "must be matched"
        for d, fd in zip(dat, fdlist):
            d["filedata"] = fd

    if strokenums_to_plot_alone is None:
        # what is current stroke num?
        strokenums_to_plot_alone = [None]

    # to store what was pt in each box.
    placed = {}
    for i, a in enumerate(cat1[1]):
        for j, snum in enumerate(strokenums_to_plot_alone):
            
            rownum = i*len(strokenums_to_plot_alone)+j

            for ii, c in enumerate(cat2[1]):
                
                placed[(i, ii)] = None

                # find all trials with this a and c
                if cat1[0]==None:
                    datthis = [d for d in dat if d[cat2[0]]==c]   
                elif cat2[0]==None:
                    datthis = [d for d in dat if d[cat1[0]]==a]
                else:
                    datthis = [d for d in dat if d[cat1[0]]==a and d[cat2[0]]==c]   

                # plot each trial on same axis
                ax = axes[rownum, ii]
                # ax = axes[i, ii]

                # title for rows and columns.
                if i==0:
                    # first row
                    ax.set_title(f"{cat2[0]}-{c}")
                if ii==0:
                    # then first column
                    ax.set_ylabel(f"{cat1[0]}-{a}")

                if len(datthis)==0:
                    continue

                if isinstance(snum, list):
                    kwargs["strokenums_to_plot"]=snum
                elif snum is None:
                    kwargs["strokenums_to_plot"]=snum
                else:
                    kwargs["strokenums_to_plot"]=[snum]

                for k, d in enumerate(datthis):
                    if k==0:
                        plotargs["add_stroke_number"]=True
                    else:
                        plotargs["add_stroke_number"]=False

                    if ver=="beh":
                        # only task, no beh
                        overlay_guide_dots = True
                        plot_task_stimulus = False
                        plot_drawing_behavior = True
                    elif ver=="behtask":                            
                        overlay_guide_dots = True
                        plot_drawing_behavior = True
                        if k==0:
                            plot_task_stimulus = True
                        else:
                            plot_task_stimulus = False
                    elif ver=="task":
                        overlay_guide_dots = False
                        plot_task_stimulus = True
                        plot_drawing_behavior = False
                        # only need to plot once
                        if k>0:
                            continue
                    else:
                        assert False, "not coded"

                    placed[(i, ii)] = (a, c)
                    plotTrialSimple(d["filedata"], d["trial"], ax, post_go_only=True, 
                                   use_peanut_params={"replaynum":replaynum, "active":True}, 
                                    plot_fix=False, overlay_guide_dots=overlay_guide_dots,
                                   plot_task_stimulus=plot_task_stimulus, plot_drawing_behavior=plot_drawing_behavior, 
                                   only_first_n_strokes = nstrokes, nakedplot=True, kwargs=kwargs,
                                   **plotargs)

                if overlay_stroke_mean:
                    # == overlay mean strokes
                    from pythonlib.drawmodel.strokePlots import plotDatStrokesMean
                    from pythonlib.drawmodel.strokePlots import getStrokeColors
                    strokeslist = [getTrialsStrokesByPeanuts(d["filedata"], d["trial"],replaynum=replaynum) for d in datthis]
                    if kwargs["strokenums_to_plot"]==None:
                        # -- figure out how many strokes
                        strokenum_list = range(max([len(s) for s in strokeslist])+1)
                    else:
                        strokenum_list = kwargs["strokenums_to_plot"]

                    # strokenum_list = [0,1,2]
                    colorlist = getStrokeColors(max(strokenum_list)+1)[0]
                    for strokenum in strokenum_list:
                        plotDatStrokesMean(strokeslist, ax, strokenum=strokenum, color=colorlist[strokenum][:3], 
                            alpha=1., overlay_elipses=False)




            if plot_task_last_col:
                # plot task only on the last column
                # ignore variable that varies by cvolumn
                
                # find all trials with this a and c
                datthis = [d for d in dat if d[cat1[0]]==a]
                if len(datthis)>0:
                    d = datthis[0]

                    overlay_guide_dots = False
                    plot_task_stimulus = True
                    plot_drawing_behavior = False
                    ax = axes[rownum, ii+1]
                    plotTrialSimple(d["filedata"], d["trial"], ax, post_go_only=True, 
                       use_peanut_params={"replaynum":replaynum, "active":True}, 
                        plot_fix=False, overlay_guide_dots=overlay_guide_dots,
                       plot_task_stimulus=plot_task_stimulus, plot_drawing_behavior=plot_drawing_behavior, 
                       only_first_n_strokes = nstrokes, nakedplot=True, kwargs=kwargs, 
                       **plotargs)



    if False:
        # DEBUG CODE
        DAT = []
        for i in range(10):
            DAT.append(
                {"filedata":fd,
                 "trial":i+5,
                 "a":1,
                 "b":1})
        kwargs = {'each_stroke_separate': True, 'strokenums_to_plot': [1]},
        plotTrial2dGrid(DAT, cat1=["a", []], cat2=["b", []], strokenums_to_plot_alone=[1,2,[1, 2]]);
             
    if return_placed:
        return fig, placed
    else:
        return fig
  
def plotMultTrialSimpleClean(filedata, trials_list):
    """ as used in CRCNS grant. no white space, clean, no axes legends, etc
    """
    assert False, "see examples below, but there are many versions, so this code not done."
    fig1 = plotMultTrialsSimple(fdlist, trials, clean=True, plotargs={"centerize":True, "plot_task_stimulus":True, 
                                    "plot_drawing_behavior":False, "nakedplot":True, "add_stroke_number":False}, ncol=int(np.ceil(N**0.5)))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig2 = plotMultTrialsSimple(fdlist, trials, clean=True, plotver="onecolor", plotargs={"centerize":True, "plot_task_stimulus":False, 
                                    "plot_drawing_behavior":True, "nakedplot":True, "add_stroke_number":False}, ncol=int(np.ceil(N**0.5)))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig2b = plotMultTrialsSimple(fdlist, trials, clean=True, plotver="order", plotargs={"centerize":True, "plot_task_stimulus":False, 
                                    "plot_drawing_behavior":True, "nakedplot":True, "add_stroke_number":False}, ncol=int(np.ceil(N**0.5)))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # -- a version with titles for unique trial code
    fdlist = [P.fd(i) for i in indsgood]
    trials = [P.t(i)["trial"] for i in indsgood]
    titles = [P.t(i)["trialcode"] for i in indsgood]
    fig2c = plotMultTrialsSimple(fdlist, trials, titles=titles, clean=False, plotver="onecolor", plotargs={"centerize":True, "plot_task_stimulus":False, 
                                    "plot_drawing_behavior":True, "nakedplot":False, "add_stroke_number":False}, ncol=int(np.ceil(N**0.5)))
    plt.subplots_adjust(wspace=0, hspace=0)

    # --- plot closeup for each trial
    fig3 = plotMultTrialsSimple(fdlist, trials, kind="timecourse", clean=True, ncol=int(np.ceil(N**0.5)))
    plt.subplots_adjust(wspace=0, hspace=0)

def plotMultTrialsSimple2(filedata, trials_list):
    """ good, as in day summary plots
    """
    return plotMultTrialsSimple(filedata, trials_list, zoom=True, 
        strokes_ver="peanuts", plot_fix=False)

def plotMultTrialsSimple(filedata, trials_list, post_go_only=True, 
    kind="static", zoom=False, plotver="order", rand_subset=None, 
    strokes_ver = "default", only_first_n_strokes=None, plot_fix=True, 
    overlay_guide_dots=True, markersize=7, alpha=0.55, titles=None, 
    replaynum=None, empty_title=False, kwargs = None, 
    plotargs=None, clean=False, ncol=4):    
    """plot multiple trials, each showing drawing and task, in a grid
    - plotver only applies if kind is static
    - rand_subset, if None, then ignores. if an integer, then plots this
    many or less if not ebnough trials.
    - if want custom titles, pass in list titles, needs to be same length 
    as trials_list
    - if filedata is list, then assumes that is seprate filedatas one for
    each trial in trials_list
    - clean overides all, and plots clean peanut strokes
    """
    import matplotlib.gridspec as gridspec
    if kwargs is None:
        kwargs= {"each_stroke_separate":True}
    if plotargs is None:
        plotargs = {"centerize":False}

    if clean:
        zoom=True
        # plotver="order"
        # plotver="onecolor"
        strokes_ver = "peanuts"
        plot_fix=False
        overlay_guide_dots=True
        markersize=7
        alpha=0.55
        titles=None
        empty_title=True
        kwargs["each_stroke_separate"] = True

    if rand_subset is not None and rand_subset<len(trials_list):
        import random
        if isinstance(filedata, list) and titles is not None:
            # need to subsample filedata as well
            X = [(fd, t, ti) for fd, t, ti in zip(filedata, trials_list, titles)]
            X = random.sample(X, rand_subset)
            X = sorted(X, key=lambda x:x[1])
            filedata = [x[0] for x in X]
            trials_list = [x[1] for x in X]
            titles = [x[2] for x in X]
        elif isinstance(filedata, list):
            # need to subsample filedata as well
            X = [(fd, t) for fd, t in zip(filedata, trials_list)]
            X = random.sample(X, rand_subset)
            X = sorted(X, key=lambda x:x[1])
            filedata = [x[0] for x in X]
            trials_list = [x[1] for x in X]
        else:
            trials_list = random.sample(trials_list, rand_subset)
            trials_list.sort()

    if isinstance(filedata,list):
        assert len(filedata)==len(trials_list)

    ntrials = len(trials_list)
    if titles is not None:
        assert len(titles) == ntrials, "one title per plot"
    nrow = int(np.ceil(len(trials_list)/ncol))
    if False:
        fig = plt.figure(figsize=(ncol*5,nrow*5))
        gridspec.GridSpec(nrow, ncol)
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*5,nrow*5), sharex=True, sharey=True, squeeze=False)

    for i, t in enumerate(trials_list):
        c = int(i%ncol)
        r = int(np.floor((i)/ncol))

        if isinstance(filedata,list):
            assert len(filedata)==len(trials_list)
            fd = filedata[i]
        else:
            fd = filedata

        if False:
            ax = plt.subplot2grid((nrow, ncol), (r,c))
        else:
            ax = axes[r, c]
        if kind=="static":
            if strokes_ver=="peanuts":
                use_peanut_params={"replaynum":replaynum, "active":True}
            elif strokes_ver=="default":
                use_peanut_params={"replaynum":replaynum, "active":False}

            plotTrialSimple(fd, t, ax, post_go_only, zoom=zoom, plotver=plotver, 
                use_peanut_params=use_peanut_params, only_first_n_strokes=only_first_n_strokes, empty_title=empty_title,
                plot_fix = plot_fix, overlay_guide_dots=overlay_guide_dots, 
                markersize=markersize, alpha=alpha, kwargs=kwargs, **plotargs)
        elif kind=="timecourse":
            if clean:
                plotTrialSimpleTimecourse(fd, t, ax, post_go_only=False,
                    clean=clean)    
            else:
                plotTrialSimpleTimecourse(fd, t, ax, post_go_only=False)
        else:
            print(kind)
            assert False, "not coded"
        
        if titles is not None:
            ax.set_title(titles[i], color="k")
    return fig


def plotTrialTimelapse(filedata, trial, nsteps=10, ncols=8,
    plot_task_stimulus=True):
    """plot timelapses, i..e, evenly spaced in time snapshots of the
    drawing. will by default color based on strokes
    """
    fraclist = [n*1/nsteps for n in range(1,nsteps+1)]
    
    nrows = int(np.ceil((nsteps+2)/ncols))
    fig = plt.figure(figsize=(2*ncols, 2*nrows))
    
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(2*ncols, 2*nrows))
    if nrows>1:
        axes = [a for ax in axes for a in ax]
    # import pdb
    # pdb.set_trace()
    # print(axes)
    ax, out = plotTrialSimple(filedata, trial, ax=axes[0], nakedplot=True, plotver="onecolor", 
        plot_drawing_behavior=False)
    ax.set_title("Stimulus")

    for i, frac in enumerate(fraclist):

        naked=True
        ax, out = plotTrialSimple(filedata, trial, ax=axes[i+1], plotver="onecolor", fraction_of_stroke=frac, 
                       nakedplot=naked, add_stroke_number=False, plot_task_stimulus=plot_task_stimulus)
        t = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")[0]
#         ax.set_title(f"{(out[0]-t):.2f} to {out[1]-t:.2f} sec")
        ax.set_title(f"{out[1]-t:.2f} sec")

        ax.set_xticks([])
        ax.set_yticks([])

    plotTrialSimple(filedata, trial, ax=axes[i+2], nakedplot=True, add_stroke_number=False, plot_task_stimulus=plot_task_stimulus)
    axes[i+2].set_title("All")
    axes[i+2].autoscale()
#     for ax in axes:
#         ax.set_xlim([-400,0])
# #     ax.set_yticks
    return fig

# def plotTrialTimelapse(filedata, trial, nsteps=10, ncols=8,
#     plot_task_stimulus=True):
#     """plot timelapses, i..e, evenly spaced in time snapshots of the
#     drawing. will by default color based on strokes
#     """
#     fraclist = [n*1/nsteps for n in range(1,nsteps+1)]
    
#     nrows = int(np.ceil(nsteps+2/8))
#     fig = plt.figure(figsize=(2*ncols, 2*nrows))
    
#     fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, 
#         figsize=(2*ncols, 2*nrows), sharex=True, sharey=True)

#     ax = plt.subplot(nrows, ncols, 1)
#     ax, out = plotTrialSimple(filedata, trial, ax=ax, plotver="onecolor", plot_drawing_behavior=False)

#     for i, frac in enumerate(fraclist):
#         ax = plt.subplot(nrows, ncols, i+2)
# #         if i>0:
# #             naked=True
# #         else:
# #             naked=False
#         naked=True
#         ax, out = plotTrialSimple(filedata, trial, ax=ax, plotver="onecolor", fraction_of_stroke=frac, 
#                        nakedplot=naked, add_stroke_number=False, plot_task_stimulus=plot_task_stimulus)
#         t = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")[0]
# #         ax.set_title(f"{(out[0]-t):.2f} to {out[1]-t:.2f} sec")
#         ax.set_title(f"{out[1]-t:.2f} sec")

#         ax.set_xticks([])
#         ax.set_yticks([])

#     ax = plt.subplot(nrows, ncols, i+3)
#     plotTrialSimple(filedata, trial, ax=ax, nakedplot=True, add_stroke_number=False, plot_task_stimulus=plot_task_stimulus)
#     ax.set_title("All")

#     for ax in axes:
#         ax.set_xlim([-400,0])
# #     ax.set_yticks
#     return fig


def plotMultTrialsOverview(filedata, saveon=True, overwrite=False, trials = None):
    """iterates over each single trial - for each trial plots using 
    plotTrialSingleOverview"""
    savedir = f"{filedata['params']['figuredir']}/trialsingleoverview"
    import os
    if overwrite==False:
        if os.path.isdir(savedir):
            print(f"[plotmulttrialsoverview] skipping since already exist: {savedir}")
            return  
    os.makedirs(savedir, exist_ok=True)
    if not trials:
        trials = getIndsTrials(filedata)

    for t in trials:
        fig = plotTrialSingleOverview(filedata, t)
        if saveon:
            fig.savefig(f"{savedir}/{t}.pdf")
        plt.close('all')

def plotOverviewSession(filedata):
    """plot and print summaries stats about this session"""
    from tools.utils import getMultTrialsTaskStages
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(11, 4))

    # 2) what are task sets?
    stages_dict = getMultTrialsTaskStages(filedata)

    plt.subplot(1,2,1)
    plt.bar(list(stages_dict.keys()), [stages_dict[key] for key in stages_dict])
    plt.xticks(rotation=45)
    plt.ylabel("number trials (all blocks)")


    # 3) what are blocks?
    BlockParams = filedata["TrialRecord"]["User"]["BlockParams"]
    plotcols = []
    outstring = []
    for block in BlockParams:
        outstring.append('  ')
        outstring.append(f"------- block {block} ----- ")
        # print(' ')
        # print(f"------- block {block} ----- ")
        
        # - what color to plot?
        print(BlockParams[block].keys())
        if "behtype" in BlockParams[block].keys():
            behtype = BlockParams[block]["behtype"]["type"]
            outstring.append(behtype)
        else:
            behtype = " "

        if "TaskSet" in BlockParams[block].keys():
            print(BlockParams[block]["TaskSet"]["numtasks"])
            print(BlockParams[block]["TaskSet"])
            ntasks = np.array([int(BlockParams[block]["TaskSet"]["numtasks"][key][0][0]) for key in BlockParams[block]["TaskSet"]["numtasks"]])
            outstring.append(", ".join([BlockParams[block]["TaskSet"]["tasklist"][key] for key in BlockParams[block]["TaskSet"]["tasklist"]]))
            outstring.append(f"(tot = {sum(ntasks)}) {ntasks}")
        else:
            ntasks = []

        # tasks
        # print([BlockParams[block]["TaskSet"]["tasklist"][key] for key in BlockParams[block]["TaskSet"]["tasklist"]])
        # print(f"(tot = {sum(ntasks)}) {ntasks}")


        thingstoprint = ("MaxFingerRaises")
        # [print(f"{key} =  {BlockParams[block]['params_task'][key]}") for key in BlockParams[block]["params_task"] if key in thingstoprint]
        # [print(f"{key} =  {[x[0] for x in BlockParams[block]['params_task'][key]]}") for key in BlockParams[block]["params_task"] if key in ("inkrate_time")]
        # print(behtype)

        if "params_task" in BlockParams[block].keys():
            keythis = "params_task"
        elif "params" in BlockParams[block].keys():
            keythis = "params"
        inkrate_time = BlockParams[block][keythis]["inkrate_time"]
        [outstring.append(f"{key} =  {BlockParams[block][keythis][key]}") for key in BlockParams[block][keythis] if key in thingstoprint]
        [outstring.append(f"{key} =  {[x[0] for x in BlockParams[block][keythis][key]]}") for key in BlockParams[block][keythis] if key in ("inkrate_time")]
    
        # if "params_task" in BlockParams[block].keys():
        #     inkrate_time = BlockParams[block]["params_task"]["inkrate_time"]
        #     [outstring.append(f"{key} =  {BlockParams[block]['params_task'][key]}") for key in BlockParams[block]["params_task"] if key in thingstoprint]
        #     [outstring.append(f"{key} =  {[x[0] for x in BlockParams[block]['params_task'][key]]}") for key in BlockParams[block]["params_task"] if key in ("inkrate_time")]
        # else:
        #     inkrate_time = " "

        if sum(ntasks) <5:
            color = [0.7, 0.7, 0.7]
        elif behtype=="Trace (instant)":
            color = "b"
        elif behtype=="Trace (pursuit,track)" and min(inkrate_time)>100:
            color = 'c'        
        elif behtype=="Trace (pursuit,track)" and min(inkrate_time)<100:   
            color = 'r'
        else:
            color = 'k'
        plotcols.append(color)
        
        
    import matplotlib.pyplot as plt
    plt.subplot(1,2,2)
    # plt.plot(filedata["TrialRecord"]["BlockCount"], '-k')
    plt.plot(filedata["TrialRecord"]["BlocksPlayed"], '-k')
    plt.title('block progression')
    plt.xlabel("trial num (includes all) (start at 0)")
    plt.ylabel("block num")

    # [print(o) for o in outstring]
    # print(BlockParams)
    # print(filedata["TrialRecord"]["BlocksPlayed"])
    # print(plotcols)
    # for x,y in enumerate(filedata["TrialRecord"]["BlockCount"]):
    for x,y in enumerate(filedata["TrialRecord"]["BlocksPlayed"]):
        plt.plot(x,y, 'o', color=plotcols[int(y-1)])
        
    plt.annotate('g=fewtrials; b=reach(trace); c=trace; r=pursuit', (20,1))

    # --- print things
    [print(o) for o in outstring]

    return fig, outstring


def plotAnalySessionOverview(df, filedata):
    """plot all blocks showing beahvior stats - failure reasons, error rate, etc."""
    import seaborn as sns
    import copy


    plt.figure(figsize=(15,5))
    dfplot = copy.deepcopy(df)
    dfplot["errorcodes"] = [x+(np.random.rand()-0.5)*0.3 for x in dfplot["errorcodes"]]
    axes = sns.catplot(x="block", y="errorcodes", data=dfplot, dodge=False, aspect=1.75, height=10, kind="swarm", hue="errorstring")
    axes.savefig(f"{filedata['params']['figuredir']}/sessionoverview_errorvsblock1.pdf")

    plt.figure(figsize=(15,5))
    dfplot = copy.deepcopy(df)
    dfplot["errorcodes"] = [x+(np.random.rand()-0.5)*0.3 for x in dfplot["errorcodes"]]
    axes = sns.catplot(x="block", y="errorcodes", data=dfplot, dodge=False, kind="swarm", 
                hue="errorstring", col="taskstage", col_wrap=3)
    axes.savefig(f"{filedata['params']['figuredir']}/sessionoverview_errorvsblock2.pdf")

    dfplot = copy.deepcopy(df)
    dfplot["fracinkgotten"] = [x+(np.random.rand()-0.5)*0.04 for x in dfplot["fracinkgotten"]]
    sns.catplot(x="block", y="fracinkgotten", data=dfplot, dodge=False, aspect=2, height=8, kind="swarm", hue="errorcodes")
    plt.ylabel('fracinkgotten (only cases successful fixate)')

    axes = sns.catplot(x="block", y="fracinkgotten", data=dfplot, dodge=False, aspect=2, height=8, kind="boxen")
    axes.savefig(f"{filedata['params']['figuredir']}/sessionoverview_fracinkvsblock1.pdf")

    axes = sns.catplot(x="block", y="fracinkgotten", data=dfplot, dodge=False, kind="swarm",
                hue="errorcodes", col="taskstage", col_wrap=3)
    axes.savefig(f"{filedata['params']['figuredir']}/sessionoverview_fracinkvsblock2.pdf")

    axes = sns.catplot(x="block", y="fracinkgotten", data=dfplot, dodge=False, kind="boxen",
                col="taskstage", col_wrap=3)
    axes.savefig(f"{filedata['params']['figuredir']}/sessionoverview_fracinkvsblock3.pdf")


def loadSingleVideo():
    pass


def plotTask(filedata, task_string, bloque, only_if_go=True, plot_ver="beh_alltrials"):
    """ plot task (without behavior) 
    - can define task in multipel ways. simplest way is
    task string name + bloque (i.e,., compoentns in 'unqiue id')
    """
    
    trials = getIndsTrials(filedata)
    trialsthis = [t for t in trials if getTrialsTask(filedata, t)["str"]==task_string and getTrialsBloque(filedata, t)==bloque]
    
    if plot_ver=="beh_alltrials":
        if only_if_go==True:
            trialsthis = [t for t in trialsthis if getTrialsFixationSuccess(filedata, t)]
        plotMultTrialsSimple(filedata, trialsthis, zoom=True, strokes_ver="peanuts", plot_fix=False,
                    plotver="strokes")
    elif plot_ver=="example_task":
        if len(trialsthis)==0:
            print(f"did not find any task with task_string={task_string} and bloque={bloque}")
        else:
            print(f"foudn these trials with task_string={task_string} and bloque={bloque} [taking first]")
            print(trialsthis)
            t = trialsthis[0] # take first, should all be the asme
            plotTrialSimple(filedata, t, plot_drawing_behavior=False)
    else:
        assert False, "this plot_ver not coded"
    
    for t in trialsthis:
        a = getTrialsTask(filedata, t)["num_presentations"][0][0]
        b = getTrialsTask(filedata, t)["num_successes"][0][0]
        c = getTrialsTask(filedata, t)["ignore"][0][0]
        print([a, b, c])
   

def plotMultTrialsWaterfall(filedata, trials, ax=None, colorver="vel", 
    cleanver=False, flipxy=False, chunkmodel=None, chunkmodel_idx = 0,
     waterfallkwargs=None):
    """ wrapper to plot waterfall, i.e,., raster where y 
    is trial and x is time in trial.
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
    """
    from pythonlib.tools.stroketools import assignStrokenumFromTask
    from pythonlib.drawmodel.strokePlots import getStrokeColors, plotDatWaterfall

    if waterfallkwargs is None:
        waterfallkwargs= {}
    ylabels =[]
    strokes_list = []
    strokescolors_list = []
        
    pcols = getStrokeColors([])[0] # default.

    # === COLLECT ALL TRIALS.
    for t in trials:
        
        ylabels.append(t)
        
        if not getTrialsFixationSuccess(filedata, t):
            # 1) Empty data, this will l;eave blank row.
            strokes_list.append([])
            strokescolors_list.append([])
            continue
        else:

            # 1) -- Collect strokes
            strokes = getTrialsStrokesByPeanuts(filedata, t)


            # 2) -- Collect hwo to color strokes
            def colorbystroknum(sort_stroknum):
                print(chunkmodel_idx)
                strokes_task = getTrialsTaskAsStrokes(filedata, t,
                    chunkmodel = chunkmodel, chunkmodel_idx=chunkmodel_idx)
                if strokes_task is None:
                    return None
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
                for i in range(len(strokes_colors)):
                    strokes_colors[i] = pcolsthis[strokes_colors[i]]
                return strokes_colors

            if isinstance(colorver, str):
                if colorver=="vel":
                    fs = filedata["params"]["sample_rate"]
                    _, strokes_speed = strokesVelocity(strokes, fs, lowpass_freq=None)
                    # _, strokes_speed = strokesVelocity(strokes, fs, lowpass_freq=5) # 1/4/23,
                    strokes_colors = [s[:,0] for s in strokes_speed]
                elif colorver=="taskstrokenum_fixed":
                    strokes_colors = colorbystroknum(False)
                elif colorver=="taskstrokenum_reordered":
                    strokes_colors = colorbystroknum(True)
                else: 
                    print(colorver)
                    assert False, "not coded"
            else:
                assert False, "this a function? not yet coded"

            strokes_list.append(strokes)
            # print(strokes_colors)
            # assert False
            if strokes_colors is None:
                return "failed"
                # this indicates I want to skip this.
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
    plotDatWaterfall(strokes_list, strokescolors_list, ax, ylabels=ylabels, flipxy=flipxy, 
        **waterfallkwargs)
    if flipxy==False:
        ax.set_ylabel("trial")
        ax.set_xlabel(waterfallkwargs["xaxis"])
    else:
        ax.set_xlabel("trial")
        ax.set_ylabel(waterfallkwargs["xaxis"])
    
    if ax is None:
        return fig