""" plots related to the processed data in analy"""
from tools.plots import *
from tools.analy import *

def plotDictCanvasOverlay(stroke_dict, filedata, fieldtoplot, 
                          title="", strokes_to_plot="all", 
                         align_by="", rotate_by="", plotver="raw", 
                         scale_by="", alpha_force = None, ax=None,
                         return_strokes = False):
    """ plots multiple strokes on
    same canvas, showing timesteps
    - aligns multipel strokes by some desired alginemnt point
    - rotates strokes so that they are aligned there too
    - does rotatino first, before doing transformatino
    - rotates so that is facing up
    - return_strokes, useful if want to get what was used to plot (post transforamtions)"""
    from math import pi
    output ={}

    ax = plotTrialSimple(filedata, 1, plotver="empty", nakedplot=True, 
                         add_stroke_number=False, plot_task_stimulus=False, ax=ax, plot_fix=False)[0]
    ax.set_title(title)
    # plt.title(title)
    strokes_all = stroke_dict[fieldtoplot]
    strokes_all_task = stroke_dict["strokes_all_task"]
    if not rotate_by=="":
        angles_all_task = stroke_dict["angles_all_task"]
    else:
        angles_all_task = [[] for _ in range(len(strokes_all))]

    if scale_by=="task_first_stroke":
        # then will scale each task so that the lengh of its
        # first stroke matches the medain length of first stroke
        # across tasks.
        
        # - get median length of first stroke
        first_stroke_lens = [np.linalg.norm(s[0][-1,[0,1]]-s[0][0,[0,1]]) for s in strokes_all_task]
        first_stroke_median = np.median(first_stroke_lens)
        output["first_stroke_median_len"]=first_stroke_median
    else:
        first_stroke_lens = [[] for _ in range(len(strokes_all))]
    
    output["strokes_all"]=[]
    for i, (s, s_task, a_task, fs_len) in enumerate(zip(strokes_all, strokes_all_task, angles_all_task, first_stroke_lens)):
        
        # print(i)
        
        assert len(s)>0, "some strokes are empty. filter trials to exclude this (or deal somehow), before running again."
        # ==== align?
        if align_by=="onset":
            s = [ss-s[0][0,:] for ss in s] # subtract onset
        elif align_by=="onset_task":
            s = [ss-s_task[0][0,:] for ss in s] # subtract onset
        elif align_by=="ownmidpt":
            from pythonlib.tools.stroketools import alignStrokes
            # print(s)
            s = alignStrokes(s, [np.array([0,0]).reshape(-1,1)]) # translates s so that its
            # print(s)
            # assert False
            # center is at 0,0
        else:
            assert align_by=="", "dont know this one..."

        # ==== rotate?
        if rotate_by=="task_first_stroke":
            s = strokesTransform(s, theta=-a_task[0]+pi/2)
        else:
            assert rotate_by=="", "dont know this one"

        # ==== scale?
        if scale_by=="task_first_stroke":
            s = strokesTransform(s, s=first_stroke_median/fs_len)
        else:
            assert scale_by=="", "dont know this"

            
        if "task" in fieldtoplot:
            # the task lines are lighter, fewer points
            alpha=0.65
        else:
            alpha=0.15

        # -- how many strokes to plot?
        pcol = None
        markersize=6
        if strokes_to_plot=="all":
            pass
        elif strokes_to_plot=="first":
            s = [s[0]] # keep just first stroke
        elif strokes_to_plot=="second":
            s = [s[1]] # keep just first stroke
        elif strokes_to_plot=="first_touch":
            s = [s[0][0]]
            pcol = "r"
            markersize=14
            alpha=0.7
        else:
            assert False, "dont know this one"

        if alpha_force:
            alpha=alpha_force

        plotDatStrokes(s, ax=ax, plotver=plotver, add_stroke_number=False, pcol=pcol, 
                      markersize=markersize, alpha=alpha)

        output["strokes_all"].append(s)

    if return_strokes:
        return ax, output
    else:
        return ax


# --- one figure, first bin tasks by spatial location. then plot distribution in rose plot.
def generateSpatialBins(nbins_x, nbins_y, XLIM=(0, 1), YLIM=(0,1)):
    """ e.g., if 2 and 2, then will output x and 
    y bounds that evenly diviude up canvas into 4 
    quadrants.
    xedges = [XLIM[0] ..., XLIM[1]] evenly spaced.
    ***I should have just used linspace...***
    """
    
    def A(nbins, a,b):
        edges = [a]
        for _ in range(nbins):
            edges.append(edges[-1] + (b-a)/nbins)
        return edges
    
    xedges = A(nbins_x, XLIM[0], XLIM[1])
    yedges = A(nbins_y, YLIM[0], YLIM[1])

    return xedges, yedges

def plotDictGridOverlay(stroke_dict, filedata, nbins_x = 2, nbins_y = 3, snum_for_spatial_bin = 0,
    angle_ver_to_plot = "beh", stroke_whose_angle_to_plot = 0, nbins_angle = 6, rose_density = "frequency_area",
    plot_95ci=False):
    """ divides up canvas sptaillay, so one plot
    for trials that are in that spatial bin.
    for each plot also plots stroke angle (rose plot)."""

    # stroke_whose_angle_to_plot: will automatically skip if lacking this stroke num

    # preprocessing of stroke_dict:
        # - update angles and stroke centers in strokedict
    # stroke_dict = _getStrokeDict(removeShort=False)
    # stroke_dict = processReorderStrokes(stroke_dict, filedata)


    # old sctratch notes ====
    # nbins_x = 2
    # nbins_y = 3 # evenly divides up canvas
    # xedges, yedges = generateSpatialBins(nbins_x, nbins_y)    
    # snum = 0
    # plot_task_angle = False
    # stroke_to_use = 0 # will automatically skip if lacking this stroke num
    # nbins = 8
    # DENSITY = "frequency_area"
    # - initialize figure
    # fig = plt.figure(figsize=(5,5))
    # axes = fig.subplots(nrow, ncol, subplot_kw={"projection":"polar"})

    from pythonlib.tools.plottools import rose_plot, radar_plot
    # Get spatial bins
    xedges, yedges = generateSpatialBins(nbins_x, nbins_y)  

    # Preprocess stroke_dict so that have angle and stroke center inforamtion.
    stroke_dict = processAngles(stroke_dict)
    stroke_dict = processCenters(stroke_dict)

    figsall =[]
    for plotver in ["raw", "polar"]:
    #     plotver="raw"
        if plotver=="polar":
            fig, axes = plt.subplots(nbins_y, nbins_x, figsize=(nbins_x*4, nbins_y*4), subplot_kw={"projection":"polar"})
        elif plotver=="raw":
            fig, axes = plt.subplots(nbins_y, nbins_x, figsize=(nbins_x*4, nbins_y*4))
        fig.subplots_adjust(hspace=0, wspace=0)
        figsall.append(fig)

        if nbins_x==1 and nbins_y==1:
            axes = [[axes]]
        elif nbins_x==1:
            # print(axes)
            # print(axes[0])
            axes = [[ax] for ax in axes]
            # print(axes)
        elif nbins_y==1:
            axes = [ax for ax in axes]
        counts = []
        for i, (x1, x2) in enumerate(zip(xedges[:-1], xedges[1:])):
            for ii, (y1, y2) in enumerate(zip(yedges[:-1], yedges[1:])):

                F = {f"center_s{snum_for_spatial_bin}_d0":[x1, x2],  
                      f"center_s{snum_for_spatial_bin}_d1":[y1, y2]}

                stroke_dict_plot = processFilterDat(stroke_dict, filterdict=F, filedata=filedata)
                ax = axes[-(ii+1)][i]
                
                if plotver=="polar":
                    if angle_ver_to_plot=="task":
                        angles_all = stroke_dict_plot["angles_all_task"]
                    elif angle_ver_to_plot=="beh":
                        angles_all = stroke_dict_plot["angles_all"]
                    else:
                        assert False, "not coded"

                    # -- flatten to angles to desired array
                    angles_to_plot = np.array([angles[stroke_whose_angle_to_plot] for angles in angles_all if len(angles)>stroke_whose_angle_to_plot])
                    counts.append(rose_plot(ax, angles_to_plot, bins=nbins_angle, fill=True, density=rose_density, lab_unit=None, start_zero=True)[0])

                    
                    if len(angles_to_plot)>5:
                        # - shuffle control
                        nshuff = 5000
                        # -- also plot "ground truth" to account for the frequencies of those tasks
                        angles_control = stroke_dict_plot["angles_all_task"]
                        # angles_control = angles_all
                        angles_control = [angles[stroke_whose_angle_to_plot] for angles in angles_control if len(angles)>stroke_whose_angle_to_plot]
                        # - 
                        from astropy.stats.circstats import circvar as cv
                        values_shuff = []
                        circ_var_shuff = []
                        # bins_shuff = []
                        for _ in range(nshuff):
                            # for each angle, flip a coin to decide if flip
                            def _flip(angle):
                                from math import pi
                                if np.random.rand()>0.5:
                                    _a = angle+pi
                                    if _a>2*pi:
                                        _a = _a-2*pi
                                    return _a
                                else:
                                    return angle

                            angles_shuff = np.array([_flip(angle) for angle in angles_control])
                            values, bins = rose_plot([], angles_shuff, bins=nbins_angle, fill=True, density=rose_density, lab_unit=None, start_zero=True, skip_plot=True)
                            values_shuff.append(values)
                            circ_var_shuff.append(cv(angles_shuff))



                            # bins_shuff.append(bins)
                        # print(circ_var_shuff)
                        # print(cv(angles_to_plot))
                        p = (np.sum(circ_var_shuff<cv(angles_to_plot))+1)/(nshuff+1)
                        if len(angles_to_plot)>5:
                            from pythonlib.tools.plottools import annotate
                            if p<0.01:
                                # annotate(f"p={p:.2}", color='r')
                                ax.set_title(f"p={p:.2}", color='r')
                            else:
                                # annotate(f"p={p:.2}", color='k')
                                ax.set_title(f"p={p:.2}")
                        # - get 95% CI for the values
                        values_shuff = np.array(values_shuff).T
                        if plot_95ci:
                            values_95ci = np.stack([np.percentile(v, [2.5, 97.5], interpolation="linear") for v in values_shuff])
                            radar_plot(ax, bins[:-1], values_95ci[:,0], color=[0.5, 0.5, 0.7], fill=False)
                            radar_plot(ax, bins[:-1], values_95ci[:,1], color=[0.2, 0.2, 0.6], fill=False)
                        else:
                            values_50ci = np.stack([np.percentile(v, [50], interpolation="linear") for v in values_shuff])
                            # radar_plot(ax, bins[:-1], values_50ci, color=[0.5, 0.5, 0.7], fill=False)
                            if True:
                                radar_plot(ax, bins[:-1], values_50ci[:,0], color=[0.5, 0.5, 0.7], fill=False)




                elif plotver=="raw":
                    plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all_task", strokes_to_plot="first",    
                                          plotver=[0.5, 0.5, 0.5], ax=ax, alpha_force=0.25)
                    plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all", strokes_to_plot="first",    
                                          plotver='raw', ax=ax)
                    
        if plotver=="polar":
            for ax in axes:
                for a in ax:
                    a.set_ylim([0, np.max([np.max(c) for c in counts])])
    return figsall



### ======== WRAPPERS
def plotDictwrapperSplitByAngle(filedata, trials_list, faketimesteps_ver="from_orig",
                          subsample_trials=None):
    """ to plot things over entire canvas at their correct spatial location.
    - have to decide what temporal order to plot the groundtruth strokes
    by default aligns so that task first storke is from (0,0) to (0,1)
    behavior is plotted relative to taht. can choose to have (0,0) align to task edge 
    closest to origin ('from_orig') or choosing edge closest to the first behavioral touch (excluding fixation point) (from_first_touch)
    - subsample_trials = 5, will pick out 5 random trials.
    """
    
    # 1) Preprocess data
    if not subsample_trials is None:
        trials_list = random.sample(trials_list, subsample_trials)
    stroke_dict = getMultTrialsStrokeDict(filedata, trials_list)
    stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all_task", ver=faketimesteps_ver,
                                      replace_key_to_do=True)
#     stroke_dict = processAngles(stroke_dict, stroke_to_use="first_two_points", force_use_two_points=True)
    stroke_dict = processAngles(stroke_dict, stroke_to_use="first_two_points", force_use_two_points=True)

    row, col= 4,4
    fig, axes = plt.subplots(row, col, figsize=(row*4, col*6))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    for i, a in enumerate([[1,5], [2,6], [3,7], [4,8]]):

        # 2) Pull out variables
        filterdict = {
            "angle_bin_task_first_stroke":a}
        stroke_dict_plot = processFilterDat(stroke_dict, filterdict, filedata)

        for j, stroketype in enumerate(["strokes_all", "strokes_all_task"]):
            # 1) not centered
            plotDictCanvasOverlay(stroke_dict_plot, filedata, stroketype, 
                          title="", strokes_to_plot="first", ax=axes[j*2][i])

            # 2) centered
            plotDictCanvasOverlay(stroke_dict_plot, filedata, stroketype, 
                          title="aligned by onset", strokes_to_plot="first",
                         align_by="onset", ax=axes[j*2+1][i])
    return fig, axes


def plotDictwrapperAligned(filedata, trials_list, faketimesteps_ver="from_orig",
                          subsample_trials=None):
    """ to plot things aligned to the first stroke.
    by default aligns so that task first storke is from (0,0) to (0,1)
    behavior is plotted relative to taht.
    can choose to have (0,0) align to task edge closest to origin ('from_orig')
    or choosing edge closest to the first behavioral touch (excluding fixation point) (from_first_touch)
    - subsample_trials = 5, will pick out 5 random trials.
    """
    
    ALIGN_BY = "onset_task"
    ROTATE_BY = "task_first_stroke"
    SCALE_BY = "task_first_stroke"

    # 1) Preprocess data
    if not subsample_trials is None:
        trials_list = random.sample(trials_list, subsample_trials)
    stroke_dict = getMultTrialsStrokeDict(filedata, trials_list)
    stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all_task", ver=faketimesteps_ver,
                                      replace_key_to_do=True)
#     stroke_dict = processAngles(stroke_dict, stroke_to_use="first", force_use_two_points=True)
    stroke_dict = processAngles(stroke_dict)

    fig, axes = plt.subplots(3,2, figsize=(3*5, 2*5))
    fig.subplots_adjust(hspace=0.08, wspace=-0.35)
    # == 1) Plot in original space, trials + first touch point
    ax = axes[0][0]
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                          title="", strokes_to_plot="first", 
                         plotver=[0.8, 0.8, 0.8], ax=ax)

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                          title="", strokes_to_plot="first", 
                         plotver="raw", ax=ax)

    # == 1) Plot in original space, trials + first touch point
    ax = axes[0][1]
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                          title="", strokes_to_plot="first", 
                         plotver=[0.8, 0.8, 0.8], ax=ax)

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                          title="", strokes_to_plot="first_touch", 
                         plotver="onecolor", ax=ax)

    if len(trials_list)<10:
        alpha_task = 0.4
        alpha_beh = 0.3
    else:
        alpha_task = 0.05
        alpha_beh = 0.1

    # == 2) Plot touch, aligned to origin 
    ax = axes[1][0]
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                          title="", strokes_to_plot="first", 
                         plotver=[0.8, 0.8, 0.8], align_by=ALIGN_BY, scale_by=SCALE_BY,
                              rotate_by=ROTATE_BY, alpha_force=alpha_task, ax=ax)
    plt.plot(0,0, 'kx')

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                          title=f"align by {faketimesteps_ver}", strokes_to_plot="first_touch", 
                         plotver="onecolor", align_by=ALIGN_BY, scale_by=SCALE_BY,
                         rotate_by=ROTATE_BY, alpha_force=alpha_beh, ax=ax)
    plt.plot(0,0, 'kx')

    # == 2) Plot strokes, aligned to origin 
    ax = axes[1][1]
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                          title="", strokes_to_plot="first", 
                         plotver=[0.8, 0.8, 0.8], align_by=ALIGN_BY, scale_by=SCALE_BY,
                              rotate_by=ROTATE_BY, alpha_force=alpha_task, ax=ax)
    plt.plot(0,0, 'kx')

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                          title=f"align by {faketimesteps_ver}", strokes_to_plot="first", 
                         plotver="raw", align_by=ALIGN_BY, scale_by=SCALE_BY,
                         rotate_by=ROTATE_BY, alpha_force=alpha_beh, ax=ax)
    plt.plot(0,0, 'kx')

    # == 2) Plot strokes, aligned to own midpoint
    ax = axes[2][0]
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                          title="", strokes_to_plot="first", 
                         plotver=[0.8, 0.8, 0.8], align_by="ownmidpt", scale_by=SCALE_BY,
                              rotate_by=ROTATE_BY, alpha_force=alpha_task, ax=ax)
    plt.plot(0,0, 'kx')

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                          title=f"align by {faketimesteps_ver}", strokes_to_plot="first", 
                         plotver="raw", align_by="ownmidpt", scale_by=SCALE_BY,
                         rotate_by=ROTATE_BY, alpha_force=alpha_beh, ax=ax)
    plt.plot(0,0, 'kx')

    return fig, axes

def plotDictwrapperCondOnSomething(filedata, trials_list, subsample_trials=None, 
                                       REMOVESHORT=False, try_other_dir=False, 
                                   FF = ({"center_s0_d1":[0.4, 0.6],  "center_s1_d1":[0.5, 1]},
                                         {"center_s0_d1":[0.4, 0.6],  "center_s1_d1":[0, 0.5]}), 
                                  faketimesteps = "align_to_behavior", MATCH_SUBJ_TASK_FIRST_STROKE=None,
                                  return_align_output=False):
    """ At first derived from following 9see below) then I made it more general. Now makes one set of plots
    for each filtering operation.
    is best used by uiterating over different ifltering oeprations (FF, each list etnry in FF will be an iterant).
    -- to plot things for 2-line tasks, will plot beahvior on first stroke
    conditioned on what the second stroke is. will assumet aht the first task stroke
    is the one closes to the first touich. will make 2 analyses, one for case where second stroke is 
    above or below the center(first) stroke.
    - REMOVESHORT, should probably remove short strokes?
    - MATCH_SUBJ_TASK_FIRST_STROKE=True # to look at within stroke dynamics [should not change]
    - fangle = [1,2,..., 8] means take all 8 bins for the first stroke angle. (1 is starting from (1,0))

    """
    ## ---- filter so that only get strokes whos centers are within some bounding area
    from tools.analy import processCenters, processFilterDat
    from math import pi
    
    # 1) Preprocess data
    if not subsample_trials is None:
        if subsample_trials<len(trials_list):
            trials_list = random.sample(trials_list, subsample_trials)
    
    # ============= ITERATE over different 2nd stroke conditions
    figsall = []
    ncol = 5
    nrow = len(FF)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3.5, nrow*4))
    fig.subplots_adjust(hspace=0, wspace=0)
    figsall.append(fig)
                
    output_align = []
    for i, f in enumerate(FF):

        # 1) reorder task strokes so that match order of beahvior
        stroke_dict = getMultTrialsStrokeDict(filedata, trials_list)
        if REMOVESHORT:
            # 2) remove short segments
            stroke_dict = processRemoveShortStrokes(stroke_dict)

        # i) reorder all of the task strokes [i.e., based on a model]
        if MATCH_SUBJ_TASK_FIRST_STROKE=="distance":
            stroke_dict = processReorderStrokes(stroke_dict, filedata, method="distance",
                                                     reassign_timestamps=False)
        elif MATCH_SUBJ_TASK_FIRST_STROKE=="prox_to_origing":
            stroke_dict = processReorderStrokes(stroke_dict, filedata, method="prox_to_origin",
                                                     reassign_timestamps=False)
        else:
            assert MATCH_SUBJ_TASK_FIRST_STROKE==None, "not coded"

            
        # === change timing order of ground truth strokes
        if faketimesteps=="align_to_behavior":
            stroke_dict = processFakeTimesteps(stroke_dict, filedata, 
                                               key_to_do="strokes_all_task", ver="from_first_touch",
                                               replace_key_to_do=True)
            # 3) set subsequent strokes to follow from teh first stroke
            stroke_dict = processFakeTimesteps(stroke_dict, filedata, 
                                               key_to_do="strokes_all_task", ver="from_end_of_previous_stroke", 
                                               replace_key_to_do=True)
        else:
            assert faketimesteps==None, "not coded"
                                   
        stroke_dict = processCenters(stroke_dict)
        stroke_dict = processAngles(stroke_dict)
                                   
       #######  DO FILTER

        stroke_dict = processFilterDat(stroke_dict, f, filedata=filedata)

        # 1) plot tasks
        ax = axes[i][0]
        ax = plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              strokes_to_plot="first", plotver="strokes", ax=ax)
        # 1) plot tasks
        ax = axes[i][1]
        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              strokes_to_plot="first", plotver=[0,0,1], ax=ax)
        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              strokes_to_plot="second", plotver=[0.5, 0.5, 0.5], ax=ax)

        # 2) plot behavior over tasks
        ax = axes[i][2]
        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              strokes_to_plot="first", plotver="onecolor", ax=ax)
        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                              strokes_to_plot="first", ax=ax)

        # 2) plot behavior over tasks
        ax = axes[i][3]
        ax = plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              strokes_to_plot="first", plotver="onecolor", ax=ax)
        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                              strokes_to_plot="first_touch", ax=ax)
        
        
        
        
        
            # == 2) Plot touch, aligned to origin 
        ALIGN_BY = "onset_task"
        ROTATE_BY = "task_first_stroke"
        SCALE_BY = "task_first_stroke"

        ax = axes[i][4]
        output_align.append(plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", 
                              title="", strokes_to_plot="first", 
                             plotver="raw", align_by=ALIGN_BY, scale_by=SCALE_BY,
                             rotate_by=ROTATE_BY, alpha_force=0.1, ax=ax, return_strokes=True)[1])
        print(len(stroke_dict["strokes_all"]))        

        plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", 
                              title="", strokes_to_plot="first", 
                             plotver="onecolor", align_by=ALIGN_BY, scale_by=SCALE_BY,
                                  rotate_by=ROTATE_BY, alpha_force=0.02, ax=ax)
    
        
        # 3) plot in rose plot
        figs = plotDictGridOverlay(stroke_dict, filedata, nbins_x=1, nbins_y=1)
        figsall.append(figs[0])
        figsall.append(figs[1])
        
    if return_align_output:
        return figsall, output_align
    return figsall

def plotMakeSaveDir(filedata, name, instead_of_name=None):
    """ currently for plotting from notebook
    saves in /data2/analyses
     - name is usually naem of noteobok. e.g. name ="good_line_analysis_040820"
    """
    if not instead_of_name is None:
        savedir = f"{filedata['params']['figuredir_notebook']}/{name}/{instead_of_name}"
    else:
        savedir = f"{filedata['params']['figuredir_notebook']}/{name}/{filedata['params']['animal']}_{filedata['params']['date']}_{filedata['params']['session']}"
    import os 
    os.makedirs(savedir, exist_ok=True)
    return savedir
