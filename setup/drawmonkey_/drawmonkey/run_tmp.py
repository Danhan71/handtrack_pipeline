
# %cd ..
# from tools.utils import * 
# from tools.plots import *
# from tools.analy import *
# from tools.calc import *
# from tools.analyplot import *
# from tools.preprocess import *
# from tools.dayanalysis import *

import os
from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *

from pythonlib.dataset.dataset import Dataset
from pythonlib.dataset.analy import preprocessDat
from pythonlib.tools.pandastools import applyFunctionToAllRows, aggregGeneral, pivot_table
# import pickle
import pandas as pd
from pythonlib.dataset.plots import plot_beh_grid_grouping_vs_task

PLOT_RAW = False
PLOT_RAW_TOPK = False

# option 1: Defualts, to iterate over all:
animal_list = ["Pancho", "Red"]
expt_list = ["plan5"]

# zip these
DO_SEQUENCE_RANK_LIST = [False, True, True, True]
SEQUENCE_RANK_MIN_CONFIDENCE_LIST =  [None, 0.1, 0.1, 0.1]
SEQUENCE_MATCH_KIND_LIST = [None, None, "same", "diff"]

DO_PLOT_RAW = False


for animal in animal_list:
    for expt in expt_list:
#         for DO_SEQUENCE_RANK in DO_SEQUENCE_RANK_LIST:
#             for SEQUENCE_RANK_MIN_CONFIDENCE in SEQUENCE_RANK_MIN_CONFIDENCE_LIST:
#                 for SEQUENCE_MATCH_KIND in SEQUENCE_MATCH_KIND_LIST:
                    
#                     if DO_SEQUENCE_RANK==False and SEQUENCE_RANK_MIN_CONFIDENCE==0.1:
#                         continue
                        
#                     if SEQUENCE_MATCH_KIND is not None:
#                         if SEQUENCE_RANK_MIN_CONFIDENCE is None:
#                             # only work doing seuqnece match if the matches are confindent.
#                             continue
        for DO_SEQUENCE_RANK, SEQUENCE_RANK_MIN_CONFIDENCE, SEQUENCE_MATCH_KIND in zip(DO_SEQUENCE_RANK_LIST,
                                                                                       SEQUENCE_RANK_MIN_CONFIDENCE_LIST,
                                                                                       SEQUENCE_MATCH_KIND_LIST):
                                                                                       

            if DO_SEQUENCE_RANK==False:
                PLOT_RAW = DO_PLOT_RAW
            else:
                PLOT_RAW = False

            #### LOAD
            D = Dataset([])

            D.load_dataset_helper(animal, expt)


            #### PREPROCESS
    #         dfAgg, feature_names, condition, score_col_names= preprocessDat(D)

            D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt, 
                                                                                         get_sequence_rank=DO_SEQUENCE_RANK,
                                                                                        sequence_rank_confidence_min =SEQUENCE_RANK_MIN_CONFIDENCE, 
                                                                                        sequence_match_kind=SEQUENCE_MATCH_KIND)


            if SEQUENCE_RANK_MIN_CONFIDENCE is not None:
                SDIR_MAIN = f"/data2/analyses/main/planning_analyses/{animal}-{expt}-seqrank_{DO_SEQUENCE_RANK}_minseqrankconfid_{SEQUENCE_RANK_MIN_CONFIDENCE}"
            else:
                SDIR_MAIN = f"/data2/analyses/main/planning_analyses/{animal}-{expt}-seqrank_{DO_SEQUENCE_RANK}"
            if SEQUENCE_MATCH_KIND is not None:
                SDIR_MAIN += f"_sequencmatchkind_{SEQUENCE_MATCH_KIND}"
            os.makedirs(SDIR_MAIN, exist_ok=True)
            savedir_figures = f"{SDIR_MAIN}/figures"
            os.makedirs(savedir_figures, exist_ok=True)

            feature_names = FEATURE_NAMES
            vals = feature_names
            condition = GROUPING
            score_col_names = SCORE_COL_NAMES

            # EXTRACT SUMMARIZED DATA
            from pythonlib.tools.pandastools import summarize_featurediff, summarize_feature
            dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(
                D.Dat, GROUPING,GROUPING_LEVELS,FEATURE_NAMES, return_dfpivot=True, do_normalize=True,
                normalize_grouping = ["animal", "expt"])

            dfAgg, dfaggflat = summarize_feature(D.Dat, GROUPING, FEATURE_NAMES, 
                                                 ["character", "animal", "expt"])


            # NOTE DOWN TRIAL NUMBERS.
            out = {}
            tasklist = D.Dat["character"].unique().tolist()
            for task in tasklist:
                out[task] = []
                for lev in GROUPING_LEVELS:
                    dfthis = D.Dat[(D.Dat["character"]==task) & (D.Dat[GROUPING]==lev)]
                    out[task].append(f"{lev} - {dfthis['trialcode'].tolist()}")

            #         for row in dfthis.iterrows():
            #             tmp = f"{row[1][GROUPING]} - {row[1]['trial']}"
            #             out[task].append(tmp)

            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(out, f"{SDIR_MAIN}/trials.yaml")

            out = {}
            for task in tasklist:
                out[task] = []
                for lev in GROUPING_LEVELS:
                    dfthis = D.Dat[(D.Dat["character"]==task) & (D.Dat[GROUPING]==lev)]
                    out[task].append(f"{lev} - {len(dfthis['trialcode'])}")
            out["*TOTAL_N_TASKS"] = len(tasklist)
            out["*TOTAL_N_TRIALS"] = len(D.Dat)
            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(out, f"{SDIR_MAIN}/ntrials.yaml")


#                 dfAgg = aggregGeneral(D.Dat, ["character", GROUPING], FEATURE_NAMES)

            if len(dfAgg)==0:
                continue

            ##### Analyses
            sdir = f"{savedir_figures}/overview"
            os.makedirs(sdir, exist_ok=True)

            ### Summary plots
            import seaborn as sns
            fig = sns.catplot(data=dfAgg, x=condition, y="time_go2raise")
            fig.savefig(f"{sdir}/pairplot_1.pdf")
            fig = sns.catplot(data=dfAgg, x=condition, y="time_raise2firsttouch")
            fig.savefig(f"{sdir}/pairplot_2.pdf")
            fig = sns.catplot(data=dfAgg, x=condition, y="hausdorff")
            fig.savefig(f"{sdir}/pairplot_3.pdf")
            fig = sns.catplot(data=dfAgg, x=condition, y="hdoffline")
            fig.savefig(f"{sdir}/pairplot_4.pdf")

            # All pairwise scatterplots
            if False:
                # too much infomration
                fig = sns.pairplot(data=dfAgg, vars=vals, hue=condition,
                            kind="scatter", diag_kind="hist", height=3)
                fig.savefig(f"/tmp/{animal}.pdf")


            ## TOTAL DISTANCE, TIME
            fig = sns.pairplot(data=dfAgg, vars=["nstrokes","total_distance",  "total_time", "total_speed"], 
                         hue=condition, diag_kind="hist")
            fig.savefig(f"{sdir}/scatterplot_1.pdf")

            ### NEW, USING PLAN TIME CATEGORIES
            # Directly compare variables (take difference)
            def _plot_all_pairwise(dfthis, SET_NUM = None):
                """
                assumes has a column "plan_time_cat" with {long, short}
                """
                from pythonlib.tools.statstools import ttest_paired

                # Note: is lev1 minus lev2
                lev2 = GROUPING_LEVELS[0]
                lev1 = GROUPING_LEVELS[1]

                vars_to_ignore = ["plan_time_cat", "hold_time_string", "saved_setnum", "character", "animal",
                                 "expt"]
                varlist = list(dfthis.columns)
                varlist = [v for v in varlist if v not in vars_to_ignore]
                ncols = 5
                nrows = int(np.ceil(len(varlist)/ncols))

                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))

                if SET_NUM is not None:
                    dfthis = dfthis[dfthis["saved_setnum"]==SET_NUM]

                for var, ax in zip(varlist, axes.flatten()):
                    print(var)
                    # -------------------
                    dftmp = pivot_table(dfthis, index=["character"], columns=[GROUPING], values=[var], flatten_col_names=False)
                    names = dftmp["character"]

                    vals = dftmp[var][lev1] - dftmp[var][lev2]
                    p = ttest_paired(vals, ignore_nan=True)[1]

                    ax.hist(vals, 20)
                    ax.axvline(0, color="r")
                    if p<0.05:
                        ax.set_title(f"{var}|p={p:.3f}", color="r")
                    else:
                        ax.set_title(f"{var}|p={p:.3f}", color="k")
                    ax.set_xlabel(f"{lev2} ==== {lev1}")

                return fig

            fig = _plot_all_pairwise(dfAgg, SET_NUM=None)
            fig.savefig(f"{sdir}/all_comparisons.pdf")
    #         fig.savefig(f"/tmp/all_comparisons.pdf")

            if False:
                ### SPECIFIC PAIRWISE SCATTERPLOTS
                ## DISTNACES
                sns.pairplot(data=dfthis, vars=["nstrokes", "dist_strokes", "dist_gaps"], hue=condition, diag_kind="hist")
                sns.pairplot(data=dfthis, vars=["nstrokes", "dist_per_stroke", "dist_per_gap"], hue=condition, diag_kind="hist")
                print("for same numebr of strokes, using longer gaps")
                print("total distance is prop to num strokes")

                ## STROKE SPEED AND NUMBER
                # sns.displot(data=dfthis, x="nstrokes", y="stroke_speed", hue="block")

                # sns.scatterplot(data=dfthis, x="nstrokes", y="stroke_speed", hue="block")
                # sns.regplot(data=dfthis[dfthis["block"]==17], x="nstrokes", y="stroke_speed")
                # sns.regplot(data=dfthis[dfthis["block"]==18], x="nstrokes", y="stroke_speed")

                sns.pairplot(data=dfthis, vars=["nstrokes", "stroke_speed", "gap_speed"], hue=condition, diag_kind="hist")

                # == is the speed effec tthere, even if control for distance?
                sns.pairplot(data=dfthis, vars=["nstrokes", "stroke_speed", "dist_strokes"], hue=condition, diag_kind="hist")
                sns.pairplot(data=dfthis, vars=["nstrokes", "gap_speed", "dist_gaps"], hue=condition, diag_kind="hist")

                print("Seems like the faster stroke and gap speed is partly explained by the longer strokes and gaps")

                # onset and offset 
                sns.pairplot(data=dfthis, vars=["nstrokes", "onset_speed", "offset_speed"], hue=condition, diag_kind="hist")
                sns.pairplot(data=dfthis, vars=["nstrokes", "onset_speed", "dist_raise2firsttouch"], hue=condition, diag_kind="hist")
                sns.pairplot(data=dfthis, vars=["nstrokes", "offset_speed", "dist_touchdone"], hue=condition, diag_kind="hist")



                # Is reaction time longer if doing more strokes?
                sns.pairplot(data=dfthis, vars=["nstrokes", "time_go2raise", "time_raise2firsttouch"], hue=condition, diag_kind="hist")


            if False:
                    # Get pivoted
                var = "dist_raise2firsttouch"
                # bk2 = 18
                # bk1 = 17

                # -------------------
                dftmp = pivot_table(dfthis, index=["unique_task_name"], columns=[condition], values=[var], flatten_col_names=False)
                names = dftmp["unique_task_name"]
                # sort
                tmp = dftmp[var][bk2] - dftmp[var][bk1]

                tmp = [(i, d, n) for i, (d, n) in enumerate(zip(tmp, names))]
                tmp = sorted(tmp, key=lambda x: x[1])
                inds_sorted = [t[0] for t in tmp]
                d_sorted = [t[1] for t in tmp]
                names_sorted = [t[2] for t in tmp]

                plt.figure()
                plt.plot(range(len(d_sorted)), d_sorted, '-ok')
                plt.axhline(0)
                plt.ylabel(f"bk {bk2} has more positive val")
                # top and bottom n
                print("-----")
                print(var)
                print(f"condition {bk2} - condition {bk1}")

                rank = 1
                task = names_sorted[rank]
                val = d_sorted[rank]
                idx = inds_sorted[rank]
                print("rank, index, taskname, value")
                print(rank, idx, task, val)
                print("-----")
                _plot_unique_task(task)



            if False:
                # === plot a trials movement timings as sanity check
                # compare across blocks...
                # Select a specific task to condition on.

                # bk1 = "1000.0"
                # bk2 = "200.0"

                def _plot_unique_task(task):
                    # idx1 = random.sample(list(np.where(P.pandas()["block"]==bk1)[0]),1)[0]
                    # idx2 = random.sample(list(np.where(P.pandas()["block"]==bk2)[0]),1)[0]
                    idx1 = random.sample(list(np.where((P.pandas()[condition]==bk1) & (P.pandas()["unique_task_name"]==task))[0]),1)[0]
                    idx2 = random.sample(list(np.where((P.pandas()[condition]==bk2) & (P.pandas()["unique_task_name"]==task))[0]),1)[0]

                    print("----- FROM PANDAS")
                    hd1 = P.pandas().iloc[idx1]["hausdorff_positive"]
                    hd2 = P.pandas().iloc[idx2]["hausdorff_positive"]
                    print("hausdorff positives")
                    print(hd1, hd2)

                #     print("time_go2raise")
                #     tmp1 = P.pandas().iloc[idx1]["time_go2raise"]
                #     tmp2 = P.pandas().iloc[idx2]["time_go2raise"]
                #     print(tmp1, tmp2)
                    print("---- Aggregated dataframe")
                    print(f"= {bk1}")
                    print(dfthis[(dfthis["unique_task_name"]==task) & (dfthis[condition]==bk1)])
                    print(f"= {bk2}")
                    print(dfthis[(dfthis["unique_task_name"]==task) & (dfthis[condition]==bk2)])

                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True, sharey=True)
                    fd_t = P.fd_trial(idx1)
                    hd1 = getTrialsScoreRecomputed(*fd_t)
                #     mot = getTrialsMotorTimingStats(*fd_t)
                #     print("----motor stats 1 (recomputed using fd")
                #     print(mot)

                    plotTrialSimpleTimecourse(*fd_t, ax=axes.flatten()[0])
                    fd_t = P.fd_trial(idx2)
                    plotTrialSimpleTimecourse(*fd_t, ax=axes.flatten()[1])
                    hd2 = getTrialsScoreRecomputed(*fd_t)
                #     mot = getTrialsMotorTimingStats(*fd_t)
                #     print("----motor stats 2 (recomputed using fd")
                #     print(mot)

                    print("hausdorff positives (recomputed using extracted fd, def matches figures")
                    print(hd1, hd2)


                    fd_t = P.fd_trial(idx1)
                    plotTrialSimpleTimecourse(*fd_t)
                    fd_t = P.fd_trial(idx2)
                    plotTrialSimpleTimecourse(*fd_t)

                    fd_t = P.fd_trial(idx1)
                    plotTrialSimple(*fd_t, clean=True, plot_done_button=True, plot_sketchpad=True)
                    print(getTrialsMotorTimingStats(*fd_t))
                    fd_t = P.fd_trial(idx2)
                    plotTrialSimple(*fd_t, clean=True, plot_done_button=True, plot_sketchpad=True)
                    print(getTrialsMotorTimingStats(*fd_t))

                task = "mixture2_110-savedset-34-04078"
                print("bk1 - bk2")
                print(bk1, bk2)
                _plot_unique_task(task);

            if PLOT_RAW_TOPK:
                ##### SYSTEMATIC, PLOT multiple tasks, split by plan duration
                sdir = f"{savedir_figures}/raw_tasks_greatest_effect"
                os.makedirs(sdir, exist_ok=True)

                # Extract tasks which have greatest difference along some feature
                import random
                Nplot = 15
                niter = 3
                for v in vals:
                    tasklist = D.analy_get_tasks_strongesteffect(GROUPING, GROUPING_LEVELS, v)
                    for ver in ["bottom", "top"]:
                        if ver=="bottom":
                            tasklistthis = tasklist[-Nplot:]
                        else:
                            tasklistthis = tasklist[:Nplot]

                        row_var = GROUPING
                        row_levs = GROUPING_LEVELS
                        for ii in range(niter):
                            figbeh, figtask = plot_beh_grid_grouping_vs_task(D.Dat, row_var, tasklistthis, row_levs)
                            figbeh.savefig(f"{sdir}/raw-{ver}{Nplot}-{v}_{ii}_beh.pdf")
                            figtask.savefig(f"{sdir}/raw-{ver}{Nplot}-{v}_{ii}_task.pdf")
                    plt.close("all")


            ##### PLOTS - for each task, plot all trials for long and short planning
            if PLOT_RAW:
                sdirthis = f"{savedir_figures}/raw_each_task"
                os.makedirs(sdirthis, exist_ok=True)

                from pythonlib.tools.plottools import saveMultToPDF
                from pythonlib.dataset.plots import plot_beh_grid_singletask_alltrials
                from pythonlib.dataset.plots import plot_beh_waterfall_singletask_alltrials

                tasklist = D.Dat["character"].unique().tolist()
                # ONLY PLOT THOSE WITH DATA ACROSS BOTH LEVELS.
    #             group = "plan_time_cat"
                # levels = ["short", "long"]

                for task in tasklist:
                #     task = "mixture2_10-savedset-34-09627"
                #     df = D.Dat[D.Dat["character"]==task]
                    figb, figt = plot_beh_grid_singletask_alltrials(D, task, GROUPING)    
                #     figb.savefig(f"{sdirthis}/alltrials_splitbyplantime_{task}_beh.pdf")
                #     figt.savefig(f"{sdirthis}/alltrials_splitbyplantime_{task}_task.pdf")

                    # Waterfall
                    figw = plot_beh_waterfall_singletask_alltrials(D, task, row_variable=GROUPING)

                    saveMultToPDF(f"{sdirthis}/alltrials_splitbyplantime_{task}", [figb, figw, figt])
                    plt.close("all")

            ##### Variation in first stroke
            # First, extract only the first trial for each task.
            group = "plan_time_cat"

            # for each task assign a "trial num"
            D.analy_assign_trialnums_within_task([group])

            # only keep first trials
            Dfirsttrial = D.filterPandas({"trialnum_this_task":[0]}, "dataset")

            sdir = f"{savedir_figures}/stroke_by_stroke"
            os.makedirs(sdir, exist_ok=True)

            #### Make sure equal numbers of trials for each task.
            Dtrialsmatch = D.analy_match_numtrials_per_task([group])

            #### Plots, strokes in space, showing strokes, onsets, and offsets.
            from pythonlib.drawmodel.strokePlots import plotDatStrokes
            ntot=4

            for DVER in ["full", "firsttrial", "matchedtrials"]:
                if DVER=="full":
                    Dthis = D
                elif DVER=="firsttrial":
                    Dthis = Dfirsttrial
                elif DVER=="matchedtrials":
                    Dthis = Dtrialsmatch
                else:
                    assert False
                for align_to_onsets in [True, False]:
                    for PLOTVER in ["strokes", "onsets", "offsets"]:
                        for ALPHA in [0.02, 0.05, 0.1, 0.15, 0.2]:
                            fig, axes = plt.subplots(2, ntot, sharex=True, sharey=True, figsize=(ntot*4, 13))
                            for snum in range(ntot):
                            #     snum = 0
                                for i, g in enumerate(Dthis.Dat.groupby(group)):
                                    print(g[0])

                                    strokes_list = g[1]["strokes_beh"].values
                                    strokes_plot = [strokes[snum] for strokes in strokes_list if len(strokes)>snum]

                                    if align_to_onsets:
                                        strokes_plot = [strokes - strokes[0] for strokes in strokes_plot]

                                    ax = axes[i, snum]

                                    if PLOTVER in ["strokes"]:
                                        plotDatStrokes(strokes_plot, ax=ax, pcol="b", plotver=[0,0,0], alpha=ALPHA, add_stroke_number=False,
                                                      mark_stroke_onset=False)
                                    if PLOTVER in ["strokes", "onsets"]:
                                        # plot positions of first points
                                        for s in strokes_plot:
                                            ax.plot(s[0,0], s[0,1], "ro", alpha=ALPHA)
                                    if PLOTVER in ["offsets"]:
                                        for s in strokes_plot:
                                            ax.plot(s[-1,0], s[-1,1], "ro", alpha=ALPHA)

                                    ax.set_xlabel(f"snum: {snum}")
                                    ax.set_ylabel(f"{group} : {g[0]}")

                            fig.savefig(f"{sdir}/dver_{DVER}-algnons_{align_to_onsets}-alpha_{ALPHA}-plotver_{PLOTVER}.pdf")




            ###### Sequences more similar within group than between?
            sdir = f"{savedir_figures}/score_sequences"
            os.makedirs(sdir, exist_ok=True)

            if False:
                # v1, first agg so each task on dpt
                dfagg = aggregGeneral(D.Dat, ["character", GROUPING], values=score_col_names)
                import pandas as pd
                dfaggflat = pd.melt(dfagg, id_vars = ["character", GROUPING])
                #     DatThisAggPairedAllFlat = DatThisAggPairedAllFlat.rename(columns={None:"model"})
                dfaggflat
            else:
                # v2 - use trials as datapoint
                Dtrialsmatch = D.analy_match_numtrials_per_task([GROUPING])
                dfagg = Dtrialsmatch.Dat[score_col_names + ["character", GROUPING]]
                dfaggflat = pd.melt(dfagg, id_vars = ["character", GROUPING])

            import seaborn as sns
            # fig = sns.catplot(data=dfaggflat, x="variable", y="value", kind="point", hue="plan_time_cat", ci=68)
            fig = sns.catplot(data=dfaggflat, x="plan_time_cat", y="value", kind="point", hue="variable", ci=68)
            fig.savefig(f"{sdir}/summary_lineplot_1.pdf")

            # TODO: Shuffle analysis, to get p value for this alignment.

            if "alignment" in FEATURE_NAMES:
                ############################################### [SEQUENCING]
                ##### Are the tasks with largest ssequence difference also those with greatest efficiency advantage for LONG plna?
                sdir = f"{savedir_figures}/score_sequences"
                niter = 3
                Nplot = 15
                dfunc = lambda x, y: x+y
                tasklist = D.analy_get_tasks_strongesteffect(GROUPING, GROUPING_LEVELS, "alignment", dfunc=dfunc)
                for ver in ["bottom", "top"]:
                    if ver=="bottom":
                        tasklistthis = tasklist[-Nplot:]
                    else:
                        tasklistthis = tasklist[:Nplot]

                    for ii in range(niter):
                        figbeh, figtask = plot_beh_grid_grouping_vs_task(D.Dat, GROUPING, tasklistthis, GROUPING_LEVELS)
                        figbeh.savefig(f"{sdir}/raw-{ver}{Nplot}-alignment_{ii}_beh.pdf")
                        figtask.savefig(f"{sdir}/raw-{ver}{Nplot}-alignment_{ii}_task.pdf")

                func = lambda x: np.nanmean(x)
                dfpivot = pivot_table(D.Dat, index=["character"], columns=[GROUPING], values=feature_names, aggfunc=func)

                # collect, one vector for each value, comparing two monkey priors.
                out = {}
                from pythonlib.tools.snstools import pairplot_corrcoeff
                for TAKE_ABS in [True, False]:
                    # 1) alignemnt, take mean
                    out["alignment"] = np.nanmean(np.c_[dfpivot["alignment"][GROUPING_LEVELS[0]].values, 
                                                        dfpivot["alignment"][GROUPING_LEVELS[1]].values], axis=1)
                    # 2) all other features, take difference
                    for val2 in FEATURE_NAMES:
                        if TAKE_ABS:
                            out[f"{val2}-{GROUPING_LEVELS[1]}min{GROUPING_LEVELS[0]}"] = np.abs(dfpivot[val2][GROUPING_LEVELS[1]] - dfpivot[val2][GROUPING_LEVELS[0]])
                        else:
                            out[f"{val2}-{GROUPING_LEVELS[1]}min{GROUPING_LEVELS[0]}"] = dfpivot[val2][GROUPING_LEVELS[1]] - dfpivot[val2][GROUPING_LEVELS[0]]

                    dfsummary = pd.DataFrame(out)
                    dfsummary= dfsummary.dropna().reset_index()

                    # fig = sns.pairplot(data=dfsummary, y_vars = ["alignment"], x_vars = dfsummary.columns)
                #     fig = sns.pairplot(data=dfsummary, x_vars = ["alignment"], y_vars = dfsummary.columns, aspect=2,
                #                       kind="reg")
                    fig = pairplot_corrcoeff(data=dfsummary, x_vars = ["alignment"], y_vars = dfsummary.columns, aspect=2)
                    fig.savefig(f"{sdir}/scatter_alignment_vs_otherfeatures_absval{TAKE_ABS}.pdf")
        #             fig.savefig(f"/tmp/scatter_alignment_vs_otherfeatures_absval{TAKE_ABS}.pdf")
                    print(sdir)


                    # =========== DOES LONGER FIRST REACH PREDICT BETTER SCORE?
                    fig = pairplot_corrcoeff(data=dfsummary, x_vars = ["dist_raise2firsttouch-longminshort"], y_vars = dfsummary.columns, aspect=2)
                    fig.savefig(f"{sdir}/scatter_firstdist_vs_otherfeatures_absval{TAKE_ABS}.pdf")

                    dfsummary["dist_raise2firsttouch-longminshort_abs"] = np.abs(dfsummary["dist_raise2firsttouch-longminshort"])
                    fig = pairplot_corrcoeff(data=dfsummary, x_vars = ["dist_raise2firsttouch-longminshort_abs"], y_vars = dfsummary.columns, aspect=2)
                    fig.savefig(f"{sdir}/scatter_firstdistABS_vs_otherfeatures_absval{TAKE_ABS}.pdf")

                    # ======= diff in n strokes predict diff in score/effiicency?
                    fig = pairplot_corrcoeff(data=dfsummary, x_vars = ["nstrokes-longminshort"], y_vars = dfsummary.columns, aspect=2)
                    fig.savefig(f"{sdir}/scatter_nstrokes_vs_otherfeatures_absval{TAKE_ABS}.pdf")

                plt.close("all")



            #### VARIATION IS MORE IF LONGER PLANNIGN?
            # for each task assign a "trial num"
            D.analy_assign_trialnums_within_task([GROUPING])

            sdir = f"{savedir_figures}/diversity_across_tasks"
            os.makedirs(sdir, exist_ok=True)

            #### Make sure equal numbers of trials for each task.
            Dtrialsmatch = D.analy_match_numtrials_per_task([GROUPING])

            #### Plots, strokes in space, showing strokes, onsets, and offsets.
            from pythonlib.drawmodel.strokePlots import plotDatStrokes
            ntot=4
            from pythonlib.drawmodel.features import stroke2angle, strokeDistances, strokeCircularity
            from scipy.stats import circstd
            from pythonlib.tools.plottools import rose_plot
            out = []
            for snum in range(ntot):
                for g in Dtrialsmatch.Dat.groupby(GROUPING):
            #         print(g[0])
                    strokes_list = [strokes for strokes in g[1]["strokes_beh"].values]


                    strokes_snum = [strokes[snum] for strokes in strokes_list if len(strokes)>snum]

                    if len(strokes_snum)==0:
                        continue

                    ### circ
                    scirc = strokeCircularity(strokes_snum)

                    ### Lengths
                    slens = strokeDistances(strokes_snum)

                    ### Angles
                    angles = stroke2angle(strokes_snum)
                #     print(angles)

                    if False:
                        fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='polar'))
                        rose_plot(ax, np.array(angles), bins=24, fill=True)
                        print(circstd(np.array(angles)))

                    ### Locations
                    onsetslocs = [s[0,:2] for s in strokes_snum]
                    onsetslocs = np.r_[onsetslocs]

                    # std in x and y
            #         print(np.std(onsetslocs, 0))
                    xstd, ystd = np.std(onsetslocs, 0)

                    # SAVE
                    out.append({
                        "group":g[0],
                        "snum":snum,
                        "circstd":circstd(np.array(angles)),
                        "xstd": xstd,
                        "ystd": ystd,
                        "slenstd":np.std(slens),
                        "slensmean":np.mean(slens),
                        "scircstd":np.std(scirc),
                        "scircmean":np.mean(scirc)
                    })


            dftmp = pd.DataFrame(out)
            features = ["circstd", "xstd", "ystd", "slenstd", "slensmean", "scircstd", "scircmean"]
            import seaborn as sns
            fig, axes= plt.subplots(1,len(features), figsize=(len(features)*4,3))
            for y, ax in zip(features, axes.flatten()):
                sns.barplot(data=dftmp, x="snum", y=y, hue="group", ax=ax)
            fig.savefig(f"{sdir}/stroke_by_num_stats_1.pdf")
            sns.pairplot(data=dftmp, x_vars=["snum"], y_vars=features, hue="group", aspect=1.5)
            fig.savefig(f"{sdir}/stroke_by_num_stats_2.pdf")
            # ----
            plt.close("all")