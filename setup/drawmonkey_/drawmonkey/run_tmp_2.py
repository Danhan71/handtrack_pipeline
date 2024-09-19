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
expt_list = ["plandir1", "plandir2", "neuralprep5", "neuralprep6", "neuralprep7", "neuralprep8", "plan3", "plan4", "plan5"]

# zip these
DO_SEQUENCE_RANK_LIST = [False, True, True, True]
SEQUENCE_RANK_MIN_CONFIDENCE_LIST =  [None, 0.1, 0.1, 0.1]
SEQUENCE_MATCH_KIND_LIST = [None, None, "same", "diff"]


for DO_SEQUENCE_RANK, SEQUENCE_RANK_MIN_CONFIDENCE, SEQUENCE_MATCH_KIND in zip(DO_SEQUENCE_RANK_LIST,
                                                                               SEQUENCE_RANK_MIN_CONFIDENCE_LIST,
                                                                               SEQUENCE_MATCH_KIND_LIST):

    from pythonlib.tools.pandastools import summarize_featurediff
    Dlist = []

    for animal in animal_list:
        for expt in expt_list:

            #### LOAD
            D = Dataset([])

            D.load_dataset_helper(animal, expt)

            #### PREPROCESS
    #         dfAgg, feature_names, condition, score_col_names= preprocessDat(D)
#             D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt, 
#                                                                                          get_sequence_rank=DO_SEQUENCE_RANK,
#                                                                                         sequence_rank_confidence_min =SEQUENCE_RANK_MIN_CONFIDENCE 
#                                                                                         )
            D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt, 
                                                                                         get_sequence_rank=DO_SEQUENCE_RANK,
                                                                                        sequence_rank_confidence_min =SEQUENCE_RANK_MIN_CONFIDENCE, 
                                                                                        sequence_match_kind=SEQUENCE_MATCH_KIND)

            Dlist.append(D)


    from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
    ts = makeTimeStamp()
    SDIR_MAIN = f"/data2/analyses/main/planning_analyses/COMBINED/doseqrank_{DO_SEQUENCE_RANK}-seqrankminconfid_{SEQUENCE_RANK_MIN_CONFIDENCE}-seqmatchkind_{SEQUENCE_MATCH_KIND}"
    SDIR_FIGS = f"{SDIR_MAIN}/figs"
    os.makedirs(SDIR_FIGS, exist_ok=True)

    # save params
    prms = {"animals":animal_list, "expts": expt_list, "timestamp":ts}
    writeDictToYaml(prms, f"{SDIR_MAIN}/params.yaml")

    # Combine into single Dataset
    from pythonlib.dataset.dataset import concatDatasets
    Dall = concatDatasets(Dlist)

    ##### All plots with Dlist
    #### Plot overview (one bar plot), diff (long - short) vs. variable

    from pythonlib.tools.pandastools import summarize_featurediff

    # dfthis = D.Dat

    # dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(
    #     dfthis, GROUPING,GROUPING_LEVELS,FEATURE_NAMES, return_dfpivot=True)
    dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(
        Dall.Dat, GROUPING,GROUPING_LEVELS,FEATURE_NAMES, return_dfpivot=True, do_normalize=True,
        normalize_grouping = ["animal", "expt"])

    # === PLOT
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel

    dfthis = dfsummaryflat[dfsummaryflat["variable"].isin(COLNAMES_DIFF)]

    fig = sns.catplot(data=dfthis, x="variable", y="value", row="animal", hue="expt",
               kind="bar", ci=68, aspect=3);
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_diffs.pdf")

    fig = sns.catplot(data=dfthis, x="variable", y="value_norm", row="animal", hue="expt",
               kind="bar", ci=68, aspect=3);
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_diffs_normalized.pdf")

    fig = sns.catplot(data=dfthis, x="variable", y="value", row="animal", kind="bar", ci=68, aspect=3);
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_diffs.pdf")

    fig = sns.catplot(data=dfthis, x="variable", y="value_norm", row="animal", kind="bar", ci=68, aspect=3);
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_diffs_normalized.pdf")



    # ========== PLOT (ACTUAL VALUE, NOT DIFFS)
    from pythonlib.tools.pandastools import summarize_feature
    dfagg, dfaggflat = summarize_feature(Dall.Dat, GROUPING, FEATURE_NAMES, ["character", "animal", "expt"])


    fig = sns.catplot(data=dfaggflat, y="value", x="expt", hue=GROUPING, row="variable", col="animal",
                      kind="boxen", sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_actualvals_1_boxen.pdf")


    fig = sns.catplot(data=dfaggflat, y="value", x="expt", hue=GROUPING, row="variable", col="animal",
                      kind="point", ci=68, sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_actualvals_1_point.pdf")

    fig = sns.catplot(data=dfaggflat, y="value", hue="expt", x=GROUPING, row="variable", col="animal",
                      kind="boxen", sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_actualvals_2_boxen.pdf")


    fig = sns.catplot(data=dfaggflat, y="value", hue="expt", x=GROUPING, row="variable", col="animal",
                      kind="point", ci=68, sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_allexpts_actualvals_2_point.pdf")

    # SAME, but average over all expts
    fig = sns.catplot(data=dfaggflat, y="value", x="animal", hue=GROUPING, row="variable",kind="boxen", sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_actualvals_1_boxen.pdf")


    fig = sns.catplot(data=dfaggflat, y="value", x="animal", hue=GROUPING, row="variable", kind="point", 
                      ci=68, sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_actualvals_1_point.pdf")

    fig = sns.catplot(data=dfaggflat, y="value", hue="animal", x=GROUPING, row="variable",
                      kind="boxen", sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_actualvals_2_boxen.pdf")


    fig = sns.catplot(data=dfaggflat, y="value", hue="animal", x=GROUPING, row="variable", 
                      kind="point", ci=68, sharey=False)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/overviewbars_actualvals_2_point.pdf")

    ### save Dlist temporarily
    sdir = "/data2/analyses/main/planning_analyses/COMBINED/Dlist_testing_210516_fixedrankingeff"
    os.makedirs(sdir, exist_ok=True)
    sdir = f"{sdir}/Dlist_RedPancho_neuralprep5678.pkl"
    import pickle
    with open(sdir, "wb") as f:
        pickle.dump(Dlist, f)

    dfthis = dfsummaryflat[dfsummaryflat["variable"].isin(["alignment-MEAN"])]

    ##### Sequence stuff

    dfthis = dfsummaryflat[dfsummaryflat["variable"].isin(["alignment-MEAN"])]
    # fig = sns.catplot(data=dfthis, x="variable", y="value", row="animal", hue="expt",
    #            kind="bar", ci=68, aspect=3);
    fig = sns.catplot(data=dfthis, x="variable", y="value", row="animal", hue="expt",
               kind="bar", ci=68, aspect=1);
    fig.savefig(f"{SDIR_FIGS}/alignment_meanoverplandurs_1.pdf")

    dfthis = dfsummaryflat[dfsummaryflat["variable"].isin(["alignment-MEAN"])]
    # fig = sns.catplot(data=dfthis, x="variable", y="value", row="animal", hue="expt",
    #            kind="bar", ci=68, aspect=3);
    fig = sns.catplot(data=dfthis, x="variable", y="value", hue="animal", kind="bar", ci=68, aspect=1);
    fig.savefig(f"{SDIR_FIGS}/alignment_meanoverplandurs_2.pdf")


    ##### Correlation between alignment and efficiency scores


    # get correlation coefficients between variables
    GB = ["animal", "expt"]
    dflist = []
    for g in dfsummary.groupby(GB):
        print(g[0])
        dftmp = g[1].corr("spearman")
        dflist.append({
            "group":g[0],
            "df":dftmp})

    xvar_constant = "alignment-MEAN"
    x_vars = ["alignment-MEAN"]
    y_vars = ["nstrokes", "dist_raise2firsttouch", "hdoffline", "total_distance", 
              "dist_per_gap", "dist_per_stroke", "dist_strokes", "dist_gaps", "total_time"]
    y_vars = [f"{y}-longminshort" for y in y_vars]
    y_var_list = y_vars
    out = []
    for X in dflist:
        for y_var in y_var_list:
            x = X["df"][xvar_constant][y_var]
            animal, expt = X["group"]
            out.append({
                "animal":animal,
                "expt":expt,
                "yvar":y_var,
                "xvar":xvar_constant,
                "corrcoeff":x})


    # -- plot all
    dfcorrcoeff = pd.DataFrame(out)
    fig = sns.catplot(data=dfcorrcoeff, x="yvar", y="corrcoeff", row="animal", hue="expt",
                      kind="bar", ci=68, aspect=2)
    plt.axhline(0, color="k", alpha=0.2)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/corrBtwAlignAndOthers_1.pdf")

    # --- plot over expts
    fig = sns.catplot(data=dfcorrcoeff, x="yvar", y="corrcoeff", hue="animal", 
                      kind="bar", ci=68, aspect=2)
    plt.axhline(0, color="k", alpha=0.2)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/corrBtwAlignAndOthers_2.pdf")

    # --- plot over expts
    fig = sns.catplot(data=dfcorrcoeff, x="yvar", y="corrcoeff", kind="bar", ci=68, aspect=2)
    plt.axhline(0, color="k", alpha=0.2)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/corrBtwAlignAndOthers_3.pdf")


    # --- plot over expts
    fig = sns.catplot(data=dfcorrcoeff, x="yvar", y="corrcoeff", kind="bar", ci=68, aspect=2)
    plt.axhline(0, color="k", alpha=0.2)
    rotateLabel(fig)
    fig.savefig(f"{SDIR_FIGS}/corrBtwAlignAndOthers_4.pdf")



    ##### NOte; this section not saving

    print("TODO: error bars for each correlation coefficient")
    if False:
        from pythonlib.tools.snstools import pairplot_corrcoeff
        for animal in ["Red", "Pancho"]:
            for expt in sorted(dfsummary["expt"].unique().tolist()):
                dfthis = dfsummary[(dfsummary["animal"]==animal) & (dfsummary["expt"]==expt)]
                fig = pairplot_corrcoeff(data=dfthis, x_vars = x_vars,
                                         y_vars = y_vars, aspect=2, hue="expt")

    ##### Variabilty of strokes, positions, etc

    res = Dlist[0].Metadats[0]["filedata_params"]["resolution"]
    XPIXNORM = res[1]
    YPIXNORM = res[0]



    def preprocess_dsetlist(Dlist, ver, GROUPING=None):

        if ver=="match_ntrials_per_task":
            # make sure, given grouping, each level has sname number trials
            # for a given task
            Dlist_out =[]
            assert GROUPING is not None
            for D in Dlist:
                D.analy_assign_trialnums_within_task([GROUPING])

                #### Make sure equal numbers of trials for each task.
                Dtrialsmatch = D.analy_match_numtrials_per_task([GROUPING])
                Dlist_out.append(Dtrialsmatch)
            return Dlist_out
        else:
            print(ver)
            assert False, "not coded"


    def dset_varaibility_stats(D, GROUPING, snum_list = range(4)):
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
    #     ntot=4
        from pythonlib.drawmodel.features import stroke2angle, strokeDistances, strokeCircularity
        from scipy.stats import circstd
        from pythonlib.tools.plottools import rose_plot

        # First preprocess so that dataset has matched nuym trials
        out = []
        for snum in snum_list:
            for g in D.Dat.groupby(GROUPING):

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
                xstd, ystd = np.nanstd(onsetslocs, 0)
                # convert to cv
    #             xstd = xstd/np.mean(onsetslocs[:,0])
    #             ystd = ystd/np.mean(onsetslocs[:,1])
                # norm to screen size
                xstd = xstd/XPIXNORM
                ystd = ystd/YPIXNORM

                # SAVE
                out.append({
                    "group":g[0],
                    "snum":snum,
                    "circstd":circstd(np.array(angles)),
                    "xstd": xstd,
                    "ystd": ystd,
                    "slenstd":np.nanstd(slens)/np.mean(slens),
    #                 "slens-MEAN":np.mean(slens),
                    "scircstd":np.nanstd(scirc)/np.mean(scirc),
    #                 "scirc-MEAN":np.mean(scirc)
                })


        dfthis = pd.DataFrame(out)                           
        featurenames = ["circstd", "xstd", "ystd", "slenstd", "scircstd"]

        return dfthis, featurenames

    Dlist_matched = preprocess_dsetlist(Dlist, ver="match_ntrials_per_task", GROUPING=GROUPING)

    out = []
    for D in Dlist_matched:
        dftmp, FEATURE_NAMES_VARIAB = dset_varaibility_stats(D, GROUPING)
        dftmp["animal"] = D.animals()[0]
        dftmp["expt"] = D.expts()[0]
        out.append(dftmp)

    dfvarstats = pd.concat(out)

    dfvarstatsflat = pd.melt(dfvarstats, id_vars = ["animal", "expt", "snum", "group"]).dropna().reset_index(drop=True)

    for s in sorted(dfvarstatsflat["snum"].unique().tolist()):
        dfthis = dfvarstatsflat[dfvarstatsflat["snum"]==s]
        fig = sns.catplot(data=dfthis, hue="group", row="variable", y="value", 
                          col="animal", x="expt", kind="bar", sharey=False)
        rotateLabel(fig)
        fig.savefig(f"{SDIR_FIGS}/variability_snum{s}_1.pdf")

        fig = sns.catplot(data=dfthis, hue="group", row="variable", y="value", x="animal", kind="bar", sharey=False)
        rotateLabel(fig)
        fig.savefig(f"{SDIR_FIGS}/variability_snum{s}_2.pdf")

        fig = sns.catplot(data=dfthis, hue="animal", row="variable", y="value", x="group", kind="point", sharey=False)
        rotateLabel(fig)
        fig.savefig(f"{SDIR_FIGS}/variability_snum{s}_3.pdf")

        fig = sns.catplot(data=dfthis, x="group", row="variable", y="value", kind="point", sharey=False)
        rotateLabel(fig)
        fig.savefig(f"{SDIR_FIGS}/variability_snum{s}_4.pdf")