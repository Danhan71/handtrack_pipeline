""" functions used in notebooks for analysis of copy strategy 13.
for sure applies to 13.3 onwards. will work to add functionality for preceding as well.
"""

from tools.utils import *
from tools.plots import *

def gettrials_(fd, df, ALPHA_INVISIBLE = 0.021, clean_strokes=True, unique=False, 
              not_faded=False, trial_range=None):
    """ get list of trials
    - ALPHA_INVISIBLE, for fades, if less than this then is invisible
    - not_faded is positive control, trials without full fade
    - trial_range, inclusive [t1, t2]
    """
    
    # get trials based on fade value
    if not_faded:
        alpha = 0.03
        dfthis = df[(df["fade_samp1"].values>alpha) & (df["fade_guide1task"].values>alpha) & (df["fade_guide1fix"].values>alpha)]
    else:
        dfthis = df[(df["fade_samp1"].values<ALPHA_INVISIBLE) & (df["fade_guide1task"].values<ALPHA_INVISIBLE) & (df["fade_guide1fix"].values<ALPHA_INVISIBLE)]
    trials_list_all = list(dfthis["trial"].values)
    
    # only trials in a certain range
    if trial_range is not None:
        trials_list_all = [t for t in trials_list_all if t>=trial_range[0] and t<=trial_range[1]]
    
    # clean strokes only?
    if clean_strokes:
        print("- only trials with successful fixation")
        trials_list_all = [t for t in trials_list_all if getTrialsFixationSuccess(fd, t)==True]
        
    if unique:
        # then unique trials onlyu
        print("- only unique tasks")
        trials_list_all = removeRedundantTrials(fd, trials_list_all)
        
        
    print("extracted trials")
    print(trials_list_all)
    return trials_list_all



def getGuideDotSets(filedata, trials_list_all, set_variable = "all_same_set", indcenter = 6,
    nstrokes=None):
    """
    ### tell me what defines a set 
    # NOTE: set is identical to gd_id [redundant... actually is gd_id+1]

    set_variable = "all_same_set" # for cs13.3, since all guide dot configs indentical (modulo angle)
    # set_variable = "saved_in_metadat" # for cs13.4,
    indcenter = 6 # for cs13.3 and cs13.4
    - nstrokes, None, then takes all strokes. otherwise takes this many strokes, from onset.
    """

    #####
    def set_(filedata, trial):
        """ get the set which this task came from.
        - only relevant for tasks cs13.4"""
        if set_variable=="all_same_set":
            return 0
        elif set_variable=="saved_in_metadat":
            CAT = getTrialsTask(filedata, trial)["metadat"]["category"]
            idx = CAT.find("set")
            assert CAT[idx+4]=="_", "then set is probably not a single digit"
            return int(CAT[idx+3])

    def center_(filedata, trial):
        if "center" in getTrialsTask(filedata, trial)["metadat"].keys():
            return getTrialsTask(filedata, trial)["metadat"]["center"]
        else:
            print("sure this is the center ind?")
            return getTrialsGuideDots(filedata, t)[indcenter]


    # 1) collect all data into list of dict
    DAT = []
    for t in trials_list_all:
        DAT.append({
            "trial":t,
            "set":set_(filedata, t),
            "angle":getTrialsTask(filedata, t)["metadat"]["angle"][0][0],
            "category":getTrialsTask(filedata, t)["metadat"]["category"],
    #         "strokes":getTrialsStrokesByPeanuts(fd, t),
            "centerpos":center_(filedata, t),
    #         "strokestask":getTrialsTaskAsStrokes(fd, t),
            "guidedotcoords":getTrialsGuideDots(filedata, t),
    #         "score_offline":getTrialsScoreRecomputed(fd, t)
            })

    # 2) reorder the trials
    order_by = "angle"
    if order_by=="angle":
        DAT = sorted(DAT, key=lambda x: x["angle"])

    angle_list = sorted(set([D["angle"] for D in DAT]))
    category_list = set([d["category"] for d in DAT])
    category_list = list(category_list)
    set_list = sorted(set([D["set"] for D in DAT]))


    # == Give each guide-dot configuration a unique id 
    # (different angles will have same id)
    gdConfigs = []

    # 1) Set + Angle
    ct = 0
    for s in set_list:
        
        # get list of possible parameters
        alist = sorted(set([D["angle"] for D in DAT if D["set"]==s]))
        catlist = set([D["category"] for D in DAT if D["set"]==s])
        
        gdConfigs.append({
            "id":ct,
            "angle":alist, # possible angles
            "category":catlist, # possible tasks (name given before applying rotation)
            "set":s
        })        
        
        # assign back to DAT
        for D in DAT:
            if D["angle"] in alist and D["category"] in catlist:
                assert "gd_id" not in D.keys(), "error, this trying to fill twice?"
                D["gd_id"] = ct
        ct+=1
    for D in DAT:
        assert "gd_id" in D.keys()

    for D in DAT:
        D["trial_task"] = D["trial"]

    # == save strokes:
    # save strokes for task and beh, so don't have to extract each time
    REPLAYNUM = 1
    for D in DAT:
        D["strokes_beh"] = getTrialsStrokesByPeanuts(filedata, D["trial"], replaynum=REPLAYNUM)
        D["strokes_task"] = getTrialsTaskAsStrokes(filedata, D["trial"])
        if nstrokes is not None:
            D["strokes_beh"] = D["strokes_beh"][:nstrokes]
        D["nstrokes"] = nstrokes

    # only keep trials with at least the desired num of strokes
    if nstrokes is not None:
        DAT = [D for D in DAT if len(D["strokes_beh"])>=nstrokes]

    # remove empty trials. causes problems later.
    DAT = [D for D in DAT if len(D["strokes_beh"])>0]



    print(" -- gfConfigs:")
    print(gdConfigs)

    return DAT, angle_list, category_list, set_list, gdConfigs

def plotTrialsSortedByCategory(DAT, fd, angle_list, set_list, REPLAYNUM = 1, rand_subset = None):
    # == what variables determine what tasks to lump as a "category" (i.e., plot all together).
    # keep replaynum since if using replays want to plot the first attempt.
    figs = []

    # 2) CATEGORY = (angle, set)
    for a in angle_list:
        for s in set_list:
            trials_this = [d["trial"] for d in DAT if d["angle"]==a and d["set"]==s]

            print(f"plotting these trials for angle: {a}, set {s}")
            print(trials_this)

            fig = plotMultTrialsSimple(fd, trials_this, zoom=True, rand_subset= rand_subset,
                                strokes_ver = "peanuts", plotver="strokes", 
                                only_first_n_strokes=DAT[0]["nstrokes"], plot_fix=False, replaynum=REPLAYNUM)
            figs.append(fig)
                
    # ------- cvommented these out since above willw rok generally. e.g., if want to plot all angles, just
    # make all sets teh same...

    # elif PLOT=="angle":
    #     # 2) CATEGORY = (angle)
    #     for a in angle_list:
    #         trials_this = [d["trial"] for d in DAT if d["angle"]==a]

    #         print(f"plotting these trials for angle: {a}")
    #         print(trials_this)

    #         rand_subset = None
    #         plotMultTrialsSimple(fd, trials_this, zoom=True, rand_subset= rand_subset,
    #                             strokes_ver = "peanuts", plotver="strokes", only_first_n_strokes=3, plot_fix=False)
    #     #     assert FAlse

    # elif PLOT=="taskcat":
    #     # 2) CATEGORY = (task category)
    #     for c in category_list:
    #         trials_this = [d["trial"] for d in DAT if d["category"]==c]

    #         print(f"plotting these trials for cateorty: {c}")
    #         print(trials_this)


    #         rand_subset = None
    #         plotMultTrialsSimple(fd, trials_this, zoom=True, rand_subset= rand_subset,
    #                             strokes_ver = "peanuts", plotver="strokes", only_first_n_strokes=3, plot_fix=False)

    return figs

def plotAngleByCatOverlayTrials_(DAT, fd, anglelist, catlist, ver="beh", replaynum=1):
    # FOR each unique guide position configuration, plot tasks (columns) and angles (rows)
    # for each angle and each category, overlay all trials and plot in a grid.
    # - replaynum to 1, will take peanutpos, except when there are replays in which case take first.
    fig, axes = plt.subplots(len(anglelist), len(catlist), sharex=True, 
                             sharey=True, figsize=(len(catlist)*5, len(anglelist)*5), squeeze=False)

    for i, a in enumerate(anglelist):
        for ii, c in enumerate(catlist):
            # find all trials with this a and c
            DATTHIS = [D for D in DAT if D["angle"]==a and D["category"]==c]

            # plot each trial on same axis
            ax = axes[i, ii]
            for D in DATTHIS:
                if ver=="task":
                    # only task, no beh
                    plotTrialSimple(fd, D["trial"], ax, zoom=True, post_go_only=True, 
                                   use_peanut_params={"replaynum":replaynum, "active":True}, plotver="randcolor", 
                                    plot_fix=False, overlay_guide_dots=True,
                                   plot_task_stimulus=True, plot_drawing_behavior=False, 
                                   only_first_n_strokes = D["nstrokes"])
                elif ver=="beh":
                    plotTrialSimple(fd, D["trial"], ax, zoom=True, post_go_only=True, 
                                   use_peanut_params={"replaynum":replaynum, "active":True}, plotver="randcolor", 
                                    plot_fix=False, overlay_guide_dots=False,
                                   plot_task_stimulus=False, markersize=9, alpha=0.6, 
                                   only_first_n_strokes = D["nstrokes"])
                else:
                    assert False, "not coded"
                    
    return fig



# ==== permutation test
def dist_(strokes_beh, strokes_task, ver1 = "mean", ver2 = "max"):
    from pythonlib.tools.vectools import modHausdorffDistance
    if len(strokes_beh)==0:
        return np.nan

    pos_beh = np.concatenate(strokes_beh)
    pos_task = np.concatenate(strokes_task)

    return modHausdorffDistance(pos_beh, pos_task, dims = [0,1], ver1=ver1, ver2=ver2)

def preparePermutationTest(DAT, fd, gdConfigs, ver1 = "mean", ver2 = "max"):
    # -- a summary statistic for entire dataset (distribution and mean of offline hd)

    def funstat_(DAT, verbose=False):
    #     x = np.array([getTwoTrialsScore(fd, d["trial"], d["trial_task"]) for d in DAT])
        x = np.array([dist_(d["strokes_beh"], d["strokes_task"]) for d in DAT])
        if verbose:
            # return also each trial's value
            return (np.nanmean(x), x)
        else:
            return np.nanmean(x)            
    # print(funstat_(DAT, False))
        
        
    def funshuff_(DAT):
        import copy
        DAT_shuff = copy.deepcopy(DAT)
        
        # all trials that are same guide dot configuration, shuffle those.
        trials_task_gotten = []
        for g in gdConfigs:
            angles = g["angle"]
            gd_id = g["id"]
            cats = g["category"]
            for a in angles:
                # find all trials using this config + angle
                Dthis = [D for D in DAT_shuff if D["gd_id"]==gd_id and D["angle"]==a]

                # shuffle trials for task
                task = [(D["trial_task"], D["strokes_task"]) for D in Dthis]
                random.shuffle(task)
                trials_task_gotten.extend([t[0] for t in task]) # for sanity check after.
    #             if 63 in [t[0] for t in task]:
    #                 print([t[0] for t in task])
    #                 assert False

                # sanity check, make sure all guide positions are the same
                gdpositions = []
                for t in task:
                    gdpositions.append(getTrialsGuideDots(fd, t[0]))
                gdpositions = np.array(gdpositions)
                assert(np.all(np.isclose(np.diff(gdpositions, n=1, axis=0), 0.))), "problem, why guide dots are not the same?"                


                # reassign to D
                for D, t in zip(Dthis, task):
                    D["trial_task"] = t[0]
                    D["strokes_task"] = t[1]
                    
                    # for troubleshooting:
    #                 if D["trial"] == 63:
    #                     print(D["strokes_task"][0][5])

        # check that each trial done once and only once
        assert np.all((np.array(sorted(trials_task_gotten)) == np.array(sorted([D["trial_task"] for D in DAT])))), "then trials not gotten once and only once, in shuffle"
    #     print(np.array(sorted(trials_task_gotten)))
    #     print(np.array(sorted([D["trial_task"] for D in DAT])))
        
        return DAT_shuff

    return funstat_, funshuff_

def checkIfGettingMoreStereotyped(DAT, gdConfigs):
    # Split trials into bins across day (early and late)
    # within each bin, get all pairwise distances between behavior (restricting to 
    # guide doc configs. if he is ignoreing tasks, and just doing same thing, then 
    # should see increse in similarity over the day)

    DISTS = []
    for g in gdConfigs:
        angles = g["angle"]
        for a in angles:
            # get dat for this config
            DATTHIS = [D for D in DAT if D["angle"]==a and D["gd_id"]==g["id"]]

            # split into early and late
            DATTHIS = sorted(DATTHIS, key=lambda x: x["trial"])

            n = round(len(DATTHIS)/2)
            DATTHIS_early = DATTHIS[:n]
            DATTHIS_late = DATTHIS[n:]

            # - get all parwise distances 
            def allpairdists_(DAT):
                n = len(DAT)
                dists = []
                for i in range(n):
                    for ii in range(i+1,n):
                        d = dist_(DAT[i]["strokes_beh"], DAT[ii]["strokes_beh"])
                        dists.append(d)
                return dists
            dists_ee = allpairdists_(DATTHIS_early)
            dists_ll = allpairdists_(DATTHIS_late)
            dists_el = []
            for D1 in DATTHIS_early:
                for D2 in DATTHIS_late:
                    dists_el.append(dist_(D1["strokes_beh"], D2["strokes_beh"]))

            DISTS.append((dists_ee, dists_ll, dists_el))

    fig = plt.figure()
    for D in DISTS:
    #     x = [1,2,3]
        x = ["early-early", "late-late", "early-late"]
        y = [np.mean(DD) for DD in D]
        plt.plot(x,y,"-ok")
        plt.xlabel("comparison (splitting trials in half)")
        plt.ylabel("modHD, pairwise all trials")
        plt.title("each line, unique gd config (angle x gd_id)")

    return fig


def plotPermutationEachTrialResults(DAT, fd, p, stat_actual, stats_shuff):
    # === Get "pval" for each trial.
    from pythonlib.tools.statstools import empiricalPval
    SIDE = "left"
    for i in range(len(stat_actual)):
        if np.isnan(stat_actual[i]):
            p = np.nan
        else:
            p = empiricalPval(stat_actual[i], [s[i] for s in stats_shuff], side=SIDE)
        
        # - put back into DAT
        DAT[i]["pval"]=p
        DAT[i]["stat_actual"] = stat_actual[i]
        DAT[i]["stats_shuff"] = [s[i] for s in stats_shuff]
        DAT[i]["effect_size"] = stat_actual[i] - np.mean([s[i] for s in stats_shuff])

        
    # === PLOT
    fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex=True, squeeze=False, figsize=(20, 5*4))

    # Plot
    ax = axes[0,0]
    x = [D["trial"] for D in DAT]
    y = [D["pval"] for D in DAT]
    ax.plot(x,y,'xk')
    for xx, yy in zip(x,y):
        if yy<1:
            ax.text(xx,yy,xx, color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("p-val, from permutation test")

    # Plot
    ax = axes[1,0]
    x = [D["trial"] for D in DAT]
    y = [D["stat_actual"]-np.mean(D["stats_shuff"]) for D in DAT]
    ax.plot(x,y,'xk')
    for xx, yy in zip(x,y):
        if np.abs(yy)>10:
            ax.text(xx+1,yy,xx, color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("stats_actual - mean(stats_shuff)")

    # Plot
    ax = axes[2,0]
    x = [D["trial"] for D in DAT]
    y = [D["stat_actual"]-np.mean(D["stats_shuff"]) for D in DAT]
    gdset = [D["set"] for D in DAT]
    scatter = ax.scatter(x,y,c=gdset, label=gdset)
    legend1 = ax.legend(*scatter.legend_elements(), title="guidedot sets")
    ax.add_artist(legend1)# for xx, yy in zip(x,y):
    #     if np.abs(yy)>10:
    #         ax.text(xx+1,yy,xx, color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("stats_actual - mean(stats_shuff)")
    ax.axhline(0)

    # Plot
    ax = axes[3,0]
    for i in range(len(DAT)):
        xx = DAT[i]["trial"]
        yy = DAT[i]["stat_actual"]
        ax.axvline(xx, alpha=0.2, linestyle="-")
        ax.plot(np.ones_like(yy)*xx,yy,'or')
        yy = DAT[i]["stats_shuff"]
        ax.plot(np.ones_like(yy)*xx,yy,'xk', alpha=0.2)
        
        yyy = DAT[i]["stat_actual"]-np.mean(DAT[i]["stats_shuff"])
        if np.abs(yyy)>10:
            ax.text(xx,DAT[i]["stat_actual"],xx, color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("dist")


    # Plot
    ax = axes[4,0]
    x = [D["trial"] for D in DAT if ~np.isnan(D["stat_actual"])]
    y = [len(set(D["stats_shuff"])) for D in DAT if ~np.isnan(D["stat_actual"])]
        
    ax.plot(x,y,'xk')
    for xx, yy in zip(x,y):
        if yy>3:
            ax.text(xx,yy,xx, color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("num unique tasks")

    #################### PLOT EXAMPLE TRIALS
        # == pull out the top 20 trials
    # trialrange = [100, 350] # within this range
    # trialrange = [350, 500] # within this range
    # trialrange = [0, 1000] # within this range

    import copy
    DATTHIS = copy.deepcopy(DAT)
    DATTHIS = sorted(DATTHIS, key = lambda x: x["effect_size"])

    trials = [D["trial"] for D in DATTHIS]
    # trials = [t for t in trials if t>=trialrange[0] and t<=trialrange[1]]

    # -- Best 20
    N = 30
    trialsthis = trials[:N]
    print(trialsthis)
    fig_best= plotMultTrialsSimple(fd, trialsthis, zoom=True, plot_fix=False, plotver="strokes",
                        strokes_ver="peanuts", replaynum=1)

    # -- Worst 20
    N = 20
    trialsthis = trials[-N:]
    print(trialsthis)
    fig_worst = plotMultTrialsSimple(fd, trialsthis, zoom=True, plot_fix=False, plotver="strokes",
                        strokes_ver="peanuts", replaynum=1)


    # -- save
    return fig, fig_best, fig_worst
    