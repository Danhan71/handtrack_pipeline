""" Related to iming of movements, gaps, etc.
usually working with DatStrokesstrokes, and eg.. PIG, grammar, etc.
"""

import seaborn as sns
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt
import numpy as np
import os
from pythonlib.tools.plottools import savefig
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel


def gapstrokes_preprocess_extract_strokes_gaps(D, params_preprocess=None,
        microstim_version=False, prune_strokes=False,
        VARS_CONTEXT=None, lenient_preprocess=False):
    """
    [GOOD}]
    GAPSTROKES -- analysis of gaps and strokes...
    Extract stroke and gaps and their timings, etc.
    Not related to grammar
    RETURNS:
    - DS, SAVEDIR
    """
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("motortiming_gapstrokes")

    if False:
        if False: # I dont think the below holds. its just for grammarchunks_preprocess_and_plot
            # To do this analysis, you must onlyu include cases with one to one match of beh to task stroke.
            # Otherwise there is mismatch: chekcing if beh matches parses uses "first touch" whereas assigning chunk
            # to beh stroke must to for each beh stroke, regardless of whether is first touch.
            # D.preprocessGood(params=["one_to_one_beh_task_strokes", "correct_sequencing_binary_score"])
            D.preprocessGood(params=["one_to_one_beh_task_strokes"])
        else:
            D.preprocessGood(params=["one_to_one_beh_task_strokes_allow_unfinished"])

        if params_preprocess is not None:
            D.preprocessGood(params=params_preprocess)

        # Generate DatStrokes
        from pythonlib.dataset.dataset_strokes import DatStrokes
        DS = DatStrokes(D)
    else:
        from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
        if lenient_preprocess:
            DS = preprocess_dataset_to_datstrokes(D, "all_no_clean")
            prune_strokes = False
        else:
            DS = preprocess_dataset_to_datstrokes(D, "clean_one_to_one")

    # Condition for gapstrokes analyses.
    DS = _gapstrokes_preprocess_extract_strokes_gaps(DS, microstim_version=microstim_version,
                                                              prune_strokes=prune_strokes,
                                                              VARS_CONTEXT=VARS_CONTEXT)

    return DS, SAVEDIR

    #
    # # Compute motor timing
    # DS.timing_extract_basic()
    # DS.dataset_append_column("block")
    # DS.dataset_append_column("epoch_orig")
    # DS.dataset_append_column("probe")
    # DS.dataset_append_column("epoch")
    #
    # # From pig
    # D.seqcontext_preprocess()
    # D.taskclass_shapes_loc_configuration_assign_column()
    #
    # # Compute image similarity.
    # DS.distgood_compute_beh_task_strok_distances()
    #
    # if prune_strokes:
    #     # Then, ad-hoc params, for removing the really bad strokes.
    #     # These determed from Luca primsingridrand data.
    #
    #     if False:
    #         DS.plot_multiple_after_slicing_within_range_values("distcum", 0, 61, True)
    #         DS.plot_multiple_after_slicing_within_range_values("dist_beh_task_strok", 20, 35, True)
    #         DS.plot_multiple_after_slicing_within_range_values("dist_beh_task_strok", 35, 400, True)
    #
    #     methods = ["stroke_too_short", "beh_task_dist_too_large"]
    #     params = {
    #         "min_stroke_length":60,
    #         "min_beh_task_dist":35
    #     }
    #     DS.clean_preprocess_data(methods=methods, params=params)
    #
    #
    # ############### PREPROCESS
    # vel = DS.Dat["gap_from_prev_dist"]/DS.Dat["gap_from_prev_dur"]
    # DS.Dat["gap_from_prev_vel"] = vel
    #
    # ############# CONTEXT
    # from pythonlib.tools.pandastools import append_col_with_grp_index
    # DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape"], "shape_pre_this", use_strings=False)
    # DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_loc_prev", "gridloc"], "loc_pre_this", use_strings=False)
    # DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"], "locshape_pre_this", use_strings=False)
    # DS.Dat = append_col_with_grp_index(DS.Dat, ["shape", "gridloc", "stroke_index"], "sh_loc_idx", use_strings=True)
    # ## Extract sequential context, user-defiined
    # if VARS_CONTEXT is None:
    #     DS.Dat["context"] = DS.Dat["locshape_pre_this"] # Default
    # else:
    #     DS.Dat = append_col_with_grp_index(DS.Dat, VARS_CONTEXT, new_col_name="context")
    #
    # ##
    # if microstim_version:
    #     print("KEEPING data so that each locshape_pre_this x epoch_orig has at least one datapt per stim condition")
    #     from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars, grouping_append_and_return_inner_items
    #     DS.dataset_append_column("microstim_epoch_code")
    #     # Then enforce that each locshape_pre_this x epoch_orig has at least one datapt per stim condition
    #     # group to make a new var
    #     DS.Dat, _ = extract_with_levels_of_conjunction_vars(DS.Dat, "microstim_epoch_code", ["epoch_orig", "locshape_pre_this", "block"],
    #                                            n_min=1)

    ########### additional motor stuff
    # return DS, SAVEDIR


def _gapstrokes_preprocess_extract_strokes_gaps(DS, microstim_version=False, prune_strokes=False,
        VARS_CONTEXT=None):

    D = DS.Dataset

    # Compute motor timing
    DS.motor_velocity_timing_extract()
    DS.dataset_append_column("block")
    DS.dataset_append_column("epoch_orig")
    DS.dataset_append_column("probe")
    DS.dataset_append_column("epoch")

    # From pig
    D.seqcontext_preprocess()
    D.taskclass_shapes_loc_configuration_assign_column()

    # Compute image similarity.
    DS.distgood_compute_beh_task_strok_distances()

    if prune_strokes:
        # Then, ad-hoc params, for removing the really bad strokes.
        # These determed from Luca primsingridrand data.

        if False:
            DS.plot_multiple_after_slicing_within_range_values("distcum", 0, 61, True)
            DS.plot_multiple_after_slicing_within_range_values("dist_beh_task_strok", 20, 35, True)
            DS.plot_multiple_after_slicing_within_range_values("dist_beh_task_strok", 35, 400, True)

        methods = ["stroke_too_short", "beh_task_dist_too_large"]
        params = {
            "min_stroke_length":60,
            "min_beh_task_dist":35
        }
        DS.clean_preprocess_data(methods=methods, params=params)


    ############### PREPROCESS
    vel = DS.Dat["gap_from_prev_dist"]/DS.Dat["gap_from_prev_dur"]
    DS.Dat["gap_from_prev_vel"] = vel

    ############# CONTEXT
    from pythonlib.tools.pandastools import append_col_with_grp_index
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape"], "shape_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_loc_prev", "gridloc"], "loc_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"], "locshape_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["shape", "gridloc", "stroke_index"], "sh_loc_idx", use_strings=True)
    ## Extract sequential context, user-defiined
    if VARS_CONTEXT is None:
        DS.Dat["context"] = DS.Dat["locshape_pre_this"] # Default
    else:
        DS.Dat = append_col_with_grp_index(DS.Dat, VARS_CONTEXT, new_col_name="context")

    ##
    if microstim_version:
        print("KEEPING data so that each locshape_pre_this x epoch_orig has at least one datapt per stim condition")
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars, grouping_append_and_return_inner_items
        DS.dataset_append_column("microstim_epoch_code")
        # Then enforce that each locshape_pre_this x epoch_orig has at least one datapt per stim condition
        # group to make a new var
        DS.Dat, _ = extract_with_levels_of_conjunction_vars(DS.Dat, "microstim_epoch_code",
                                                            ["epoch_orig", "locshape_pre_this", "block"],
                                                            n_min_across_all_levs_var=1)

    ########### additional motor stuff
    return DS

def gapstrokes_timing_plot_all(DS, savedir, LIST_Y_PLOT=None):
    DS, SAVEDIR = gapstrokes_preprocess_extract_strokes_gaps(D)
    return _gapstrokes_timing_plot_all(DS, SAVEDIR, LIST_Y_PLOT)

def _gapstrokes_timing_plot_all(DS, savedir, LIST_Y_PLOT=None):
    """
    Plot motor timing for gaps and strokes in multiple ways
    PARAMS:
    - DS, extract this using gapstrokes_preprocess_extract_strokes_gaps.
    - savedir, also can extract
    """

    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

    if LIST_Y_PLOT is None:
        # LIST_Y_PLOT = ["time_duration", "gap_from_prev_dur", "gap_from_prev_dur_log10", "gap_from_prev_vel", "velocity"] # all the intersting ones?
        LIST_Y_PLOT = ["time_duration", "gap_from_prev_dur", "gap_from_prev_vel", "velocity"] # ones that usually want to plot

    # Jitter plot, vel vs. stroke index, for each context
    VAR_CONTEXT = "locshape_pre_this"
    VAR = "stroke_index_fromlast"
    VAR_HUE = "stroke_index"


    ####################################
    # Simple, first plot stroke and gap dustributions, comparing them.
    fig, axes = plt.subplots(1,2, figsize=(10,4))

    ax = axes.flatten()[0]
    tmax = np.max(np.r_[DS.Dat["gap_from_prev_dur"], DS.Dat["time_duration"]])
    bins = np.linspace(0, tmax, 50)
    ax.hist(DS.Dat["gap_from_prev_dur"], bins, histtype="step", label="gaps", density=True);
    ax.hist(DS.Dat["time_duration"], bins, histtype="step", label="strokes", density=True);
    ax.legend()
    ax.set_title("all data")
    ax.set_xlabel("duration (sec)")

    ax = axes.flatten()[1]
    tmax = np.percentile(np.r_[DS.Dat["gap_from_prev_dur"], DS.Dat["time_duration"]], [99])[0]
    bins = np.linspace(0, tmax, 50)
    ax.hist(DS.Dat["gap_from_prev_dur"], bins, histtype="step", label="gaps", density=True);
    ax.hist(DS.Dat["time_duration"], bins, histtype="step", label="strokes", density=True);
    ax.legend()
    ax.set_title("zoomed (xmax = 99th percentile)")
    ax.set_xlabel("duration (sec)")

    savefig(fig, f"{savedir}/duration_strokes_vs_gaps_all.pdf")


    ######################################
    for Y in LIST_Y_PLOT:
        # Y = "val"

        ### first, prune to keep only context with at least 2 stroke indices
        n_min = 5
        DFTHIS, _ = extract_with_levels_of_conjunction_vars(DS.Dat, var=VAR, vars_others=[VAR_CONTEXT],
                                                            n_min_across_all_levs_var=n_min,
                                                            lenient_allow_data_if_has_n_levels=2)
        print(len(DS.Dat))
        print(len(DFTHIS))

        # Only keep the first stroke index
        if False:
            DFTHIS = DFTHIS[DFTHIS["stroke_index"] == 0]
        #     DFTHIS = DFTHIS[DFTHIS["stroke_index"].isin([0,1,2])]

        # NORM 1 (subtract last indec)
        def F(x):
            sindlast = max(x["stroke_index_fromlast"]) # the least negative. usually -1.
            tmp = x[x["stroke_index_fromlast"]==sindlast][Y]
            valmean = np.mean(tmp)
            x["valmean"] = valmean
            x[f"{Y}_NORMED"] = x[Y] - valmean

            # which set of stroke indices (from last) exist
            x["set_stroke_index_fromlast"] = [tuple(sorted(set(x["stroke_index_fromlast"]))) for _ in range(len(x))]

            return x
        DFTHIS = DFTHIS.groupby(VAR_CONTEXT).apply(F)

        # NORM 2 (divide last indec)
        def F(x):
            sindlast = max(x["stroke_index_fromlast"]) # the least negative. usually -1.
            tmp = x[x["stroke_index_fromlast"]==sindlast][Y]
            valmean = np.mean(tmp)
            x[f"{Y}_DIVNORMED"] = x[Y]/valmean
            return x
        DFTHIS = DFTHIS.groupby(VAR_CONTEXT).apply(F)

        ###### plot
        for valthis in [Y, f"{Y}_NORMED", f"{Y}_DIVNORMED"]:
            print("Plotting for ", valthis)
            # sns.catplot(data=DFTHIS, x=VAR, y=f"{Y}_NORMED", kind="point", alpha=0.2, row = "set_stroke_index_fromlast", hue="stroke_index")
            # sns.catplot(data=DFTHIS, x=VAR, y=f"{Y}_NORMED", jitter=True, alpha=0.2, row = "set_stroke_index_fromlast", hue="stroke_index")
            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, col=VAR_CONTEXT, col_wrap=4, jitter=True, hue=VAR_HUE)
            # fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, col=VAR_CONTEXT,
            #         row = "set_stroke_index_fromlast", jitter=True, hue=VAR_HUE)
            savefig(fig, f"{savedir}/{valthis}-1.pdf")
            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, kind="point", alpha=0.2, hue= "set_stroke_index_fromlast", ci=68)
            savefig(fig, f"{savedir}/{valthis}-2.pdf")
            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, kind="point", alpha=0.2, hue= VAR_CONTEXT, ci=68)
            savefig(fig, f"{savedir}/{valthis}-3.pdf")
            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, kind="point", alpha=0.2, hue= VAR_CONTEXT, col="stroke_index", ci=68)
            savefig(fig, f"{savedir}/{valthis}-4.pdf")
            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, kind="point", alpha=0.2, row = "set_stroke_index_fromlast", ci=68)
            savefig(fig, f"{savedir}/{valthis}-5.pdf")
            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y=valthis, jitter=True, alpha=0.2, row = "set_stroke_index_fromlast", ci=68)
            savefig(fig, f"{savedir}/{valthis}-6.pdf")
            plt.close("all")


def gapstroke_timing_compare_by_variable(D, VAR, VARS_CONTEXT, params_preprocess,
    n_min = 3, PLOT=True, microstim_version=False):
    """
    [NOTE: is identical to gapstrokes_preprocess_extract_strokes_gaps, except prunes byu conjucntion'
    at end]
    Compare timing of gaps and strokes across variables, controlling for sequential context
    PARAMS;
    - VAR, string, the variable whos levels are compared
    - VARS_CONTEXT, list of str variables, conjucntions determine context, e.g.,
    VARS_CONTEXT = ["CTXT_loc_prev", "gridloc", "epoch"]
    VARS_CONTEXT = ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"]
    e.g., compare timing for probes vs. non-probes.
    - params_preprocess,
    e.g., params_preprocess = ["no_supervision", "one_to_one_beh_task_strokes", "correct_sequencing_binary_score", "only_blocks_with_probes"]
    RETURNS:
    - Saves figures...
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from pythonlib.tools.pandastools import append_col_with_grp_index

    ###### EXTRACT DATASET STROKES, PREPROCESSED
    # restrict to cases without online supervision
    D.grammarmatlab_successbinary_score()
    # D.preprocessGood(params=params_preprocess)

    # Get data strokes
    DS, SAVEDIR = gapstrokes_preprocess_extract_strokes_gaps(D,
        params_preprocess=params_preprocess,
        microstim_version=microstim_version,
        VARS_CONTEXT=VARS_CONTEXT)
    D.preprocessGoodCheckLog(params_preprocess+["one_to_one_beh_task_strokes_allow_unfinished"])

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER(f"motortiming_gapstrokes_byvariable/{VAR}")
    print("Saving figures at: ", SAVEDIR)

    # If the same first shape/loc occurs for both probe=0 and 1
    # If the same first loc occurs for both probe=0 and 1 (e./.g, novel prims)
    # DS.Dat = append_col_with_grp_index(DS.Dat, VARS_CONTEXT, new_col_name="context")

    # Extract only ocntexts that have each level of the VAR (or at lesat lenient_allow_data_if_has_n_levels
    # many levels)
    DS.dataset_append_column(VAR)
    #print("**** len DS (1)", len(DS.Dat))
    # NOTE: This prunes too much, like 50% of trials in one test case.
    DFTHIS, _ = extract_with_levels_of_conjunction_vars(DS.Dat, var=VAR, vars_others=["context"],
                                                        n_min_across_all_levs_var=n_min,
                                                        lenient_allow_data_if_has_n_levels=2)
    #print("**** len DS (2)", len(DFTHIS))

    if len(DFTHIS)>0:
        print(len(DS.Dat))
        print(len(DFTHIS))
        print(len(DFTHIS["context"].unique()))

        if PLOT:
            ############# PLOTS
            # %matplotlib inline
            # yvar = "time_duration"

            # For specific stroke index, and context
            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="gap_from_prev_vel", hue=VAR,
                       col="context", col_wrap=3, alpha=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_vel-1.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="gap_from_prev_vel", hue=VAR,
                       col="context", col_wrap=3, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_vel-2.pdf")

            fig = sns.catplot(data=DFTHIS, x=VAR, y="gap_from_prev_vel", hue="context", kind="point", aspect=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_vel-3.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="gap_from_prev_dur", hue=VAR,
                       col="context", col_wrap=3, alpha=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_dur-1.pdf")

            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="gap_from_prev_dur", hue=VAR,
                       col="context", col_wrap=3, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_dur-2.pdf")


            fig = sns.catplot(data=DFTHIS, x=VAR, y="gap_from_prev_dur", hue="context", kind="point", aspect=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/gap_dur-3.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="time_duration", hue=VAR,
                       col="context", col_wrap=3, alpha=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_dur-1.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="time_duration", hue=VAR,
                       col="context", col_wrap=3, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_dur-2.pdf")

            plt.close("all")

            fig = sns.catplot(data=DFTHIS, x=VAR, y="time_duration", hue="context", kind="point", aspect=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_dur-3.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="velocity", hue=VAR,
                       col="context", col_wrap=3, alpha=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_vel-1.pdf")

            fig = sns.catplot(data=DFTHIS, x="stroke_index", y="velocity", hue=VAR,
                       col="context", col_wrap=3, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_vel-2.pdf")

            fig = sns.catplot(data=DFTHIS, x=VAR, y="velocity", hue="context", kind="point", aspect=0.5)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/stroke_vel-3.pdf")


            fig = sns.pairplot(data=DFTHIS, x_vars=["gap_from_prev_dist"], y_vars=["gap_from_prev_dur"], hue=VAR,
                              plot_kws={"alpha":0.25}, height=6)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/pair-gap_dist-vs-gap_dur-1.pdf")

            plt.close("all")

    return DS, DFTHIS


def grammarchunks_preprocess_and_plot(D, PLOT=True, SAVEDIR=None):
    """
    Analysis of timing at transitions differentiated by whether are within or
    across chunks, for grammar expereints. Does extyraction of chunks, and plots
    """
    if SAVEDIR is None:
        SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("motortiming_grammarchunks")

    # Remove all baseline (i.e., for grammar)
    assert False, "use preprocessGood here"
    D.Dat = D.Dat[~D.Dat["epoch"].isin(["base", "baseline"])].reset_index(drop=True)

    D.grammarparses_successbinary_score_wrapper()

    # To do this analysis, you must onlyu include cases with one to one match of beh to task stroke.
    # Otherwise there is mismatch: chekcing if beh matches parses uses "first touch" whereas assigning chunk
    # to beh stroke must to for each beh stroke, regardless of whether is first touch.
    # D.preprocessGood(params=["one_to_one_beh_task_strokes", "correct_sequencing_binary_score"])
    D.preprocessGood(params=["one_to_one_beh_task_strokes", "correct_sequencing_binary_score", "no_supervision"])
    for ind in range(len(D.Dat)):
        D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)

    # Generate DatStrokes
    from pythonlib.dataset.dataset_strokes import DatStrokes
    assert False, "use preprocess_dataset_to_datstrokes to clean properly"
    DS = DatStrokes(D)

    ############### PREPROCESS
    DS.motor_velocity_timing_extract()
    vel = DS.Dat["gap_from_prev_dist"]/DS.Dat["gap_from_prev_dur"]
    DS.Dat["gap_from_prev_vel"] = vel

    # chunk diff from previous stroke?
    DS.context_chunks_assign_columns()
    # list_chunk_diff = []
    # for ind in range(len(DS.Dat)):
    #     chunk_diff, rank_within_diff = DS.context_chunks_diff(ind, first_stroke_diff_to_zero=True)
    #     list_chunk_diff.append(chunk_diff)
    # DS.Dat["chunk_diff_from_prev"] = list_chunk_diff

    # Epoch
    DS.dataset_append_column("epoch")
    list_epoch = DS.Dat["epoch"].unique().tolist()

    # Extract motor stats
    from pythonlib.tools.pandastools import append_col_with_grp_index
    DS.Dat = append_col_with_grp_index(DS.Dat, ["stroke_index", "chunk_diff_from_prev"], "strokeindex_chunkdiff", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["epoch", "chunk_diff_from_prev"], "epoch_chunkdiff", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape"], "shape_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_loc_prev", "gridloc"], "loc_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"], "locshape_pre_this", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["CTXT_shape_prev", "shape", "chunk_diff_from_prev"], "shape_pre_chunkdiff_this", use_strings=False)

    #################### PLOTS
    if PLOT:
        savedir = f"{SAVEDIR}/vels_agg"
        os.makedirs(savedir, exist_ok=True)
        _plotagg_vel_all(DS, savedir)

        # Plot
        savedir = f"{SAVEDIR}/grammar_chunks"
        os.makedirs(savedir, exist_ok=True)
        _plotscatter_durvsdist_all(DS, savedir)

        plt.close("all")
    return DS

def _plotagg_vel_all(DS, savedir):
    """ summary using catplot, velocity for each shape combo
    """

    # for each (shape/loc/index) get mean speed
    # for each (shape/loc) get mean speed
    if False:
        # NOTE needed, just do catplot below.
        from pythonlib.tools.pandastools import aggregGeneral
        # aggregGeneral(DS.Dat, group=["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"])
        DFAGG = aggregGeneral(DS.Dat, group=["CTXT_shape_prev", "shape",
                                             "stroke_index", "epoch",
                                             "epoch_chunkdiff",
                                             "chunk_diff_from_prev",
                                            "shape_pre_this"],
                      values=["gap_from_prev_vel"])

    fig = sns.catplot(data=DS.Dat, x="stroke_index", y="gap_from_prev_vel", hue="epoch_chunkdiff",
               col="shape_pre_this", jitter=True, alpha=0.4)
    rotateLabel(fig)
    path = f"{savedir}/by_strokeindex-1.pdf"
    savefig(fig, path)

    fig = sns.catplot(data=DS.Dat, x="stroke_index", y="gap_from_prev_vel", hue="epoch_chunkdiff",
               col="shape_pre_this", kind="point")
    rotateLabel(fig)
    path = f"{savedir}/by_strokeindex-2.pdf"
    savefig(fig, path)


    fig = sns.catplot(data=DS.Dat, x="shape_pre_this", y="gap_from_prev_vel", hue="epoch_chunkdiff",
               col="stroke_index", jitter=True, alpha=0.4)
    rotateLabel(fig)
    path = f"{savedir}/by_shapes-1.pdf"
    savefig(fig, path)

    fig = sns.catplot(data=DS.Dat, x="shape_pre_this", y="gap_from_prev_vel", hue="epoch_chunkdiff",
               col="stroke_index", kind="point")
    rotateLabel(fig)
    path = f"{savedir}/by_shapes-2.pdf"
    savefig(fig, path)


def _plotscatter_durvsdist_all(DS, savedir):
    from pythonlib.tools.listtools import stringify_list

    ################ Prune DS so that try to match, as much as possible, the shapes, locations, etc, across epochs.
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    NMIN = 5
    DFTHISGOOD, dict_df = extract_with_levels_of_conjunction_vars(DS.Dat, var="epoch",
                                                                  vars_others=["CTXT_shape_prev", "shape",
                                                                               "CTXT_loc_prev", "gridloc"],
                                                                  n_min_across_all_levs_var=NMIN,
                                                                  lenient_allow_data_if_has_n_levels=2)
    if len(DFTHISGOOD)==0:
        DFTHISGOOD = DS.Dat.copy()

    print("Original len", len(DS.Dat))
    print("New len (after matching context)", len(DFTHISGOOD))
    if False:
        for k,v in dict_df.items():
            print(k, len(v))

    assert len(DS.Dat)>0
    # Get list of stuff
    list_locshape = DFTHISGOOD["locshape_pre_this"].unique().tolist()
    list_loc = DFTHISGOOD["loc_pre_this"].unique().tolist()
    list_shape = DFTHISGOOD["shape_pre_this"].unique().tolist()
    list_epoch = DFTHISGOOD["epoch"].unique().tolist()
    list_strokeindex_chunkdiff = DFTHISGOOD["strokeindex_chunkdiff"].unique().tolist()

    stroke_indices_keep = [2,3,4,5] # to match the epochs
    nmin = 7
    # GOOD (specific shape/loc sequence)
    for x in list_locshape:
        DFTHIS = DFTHISGOOD[(DFTHISGOOD["locshape_pre_this"]==x) & (DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
        if sum([sum(DFTHIS["epoch"]==epoch)>=nmin for epoch in list_epoch])>1:
            # At least 2 epochs have data.
            fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch_chunkdiff",
                         plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
            fig.axes.flatten()[0].set_title(f"{x}")
            fig.axes.flatten()[0].set_ylim(0)
            fig.axes.flatten()[0].set_xlim(0)

            path = f"{savedir}/locshape--{stringify_list(x, True)}.pdf"
            print("Saving fig to: ", path)
            savefig(fig, path)

            plt.close("all")

    # LOCATION
    for x in list_loc:
        DFTHIS = DFTHISGOOD[(DFTHISGOOD["loc_pre_this"]==x) & (DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
        if sum([sum(DFTHIS["epoch"]==epoch)>=nmin for epoch in list_epoch])>1:
            # At least 2 epochs have data.
            fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch_chunkdiff",
                         plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
            fig.axes.flatten()[0].set_title(f"{x}")
            fig.axes.flatten()[0].set_ylim(0)
            fig.axes.flatten()[0].set_xlim(0)

            path = f"{savedir}/loc--{stringify_list(x, True)}.pdf"
            print("Saving fig to: ", path)
            savefig(fig, path)

            plt.close("all")

    # SHAPE
    for x in list_shape:
        DFTHIS = DFTHISGOOD[(DFTHISGOOD["shape_pre_this"]==x) & (DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
        if sum([sum(DFTHIS["epoch"]==epoch)>=nmin for epoch in list_epoch])>1:
            # At least 2 epochs have data.
            fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch_chunkdiff",
                         plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
            fig.axes.flatten()[0].set_title(f"{x}")
            fig.axes.flatten()[0].set_ylim(0)
            fig.axes.flatten()[0].set_xlim(0)

            path = f"{savedir}/shape--{stringify_list(x, True)}.pdf"
            print("Saving fig to: ", path)
            savefig(fig, path)

            plt.close("all")

    # [GOOD] specific shape/loc/index
    for x in list_locshape:
        for strokeindex in stroke_indices_keep:
            DFTHIS = DFTHISGOOD[(DFTHISGOOD["locshape_pre_this"]==x) & (DFTHISGOOD["stroke_index"]==strokeindex)]
            if sum([sum(DFTHIS["epoch"]==epoch)>=nmin for epoch in list_epoch])>1:
                # At least 2 epochs have data.
                fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch_chunkdiff",
                             plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
                fig.axes.flatten()[0].set_title(f"{strokeindex}-{x}")
                fig.axes.flatten()[0].set_ylim(0)
                fig.axes.flatten()[0].set_xlim(0)

                path = f"{savedir}/locshape_strokeindex--{stringify_list(x, True)}.pdf"
                print("Saving fig to: ", path)
                savefig(fig, path)

                plt.close("all")

    for epoch in list_epoch:
        for x in list_locshape:
            DFTHIS = DFTHISGOOD[(DFTHISGOOD["epoch"]==epoch) & (DFTHISGOOD["locshape_pre_this"]==x)]
            if sum([sum(DFTHIS["strokeindex_chunkdiff"]==strokeindex_chunkdiff)>=nmin for strokeindex_chunkdiff in list_strokeindex_chunkdiff])>1:
                fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="strokeindex_chunkdiff",
                             plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
                fig.axes.flatten()[0].set_title(f"{epoch}-{x}")
                fig.axes.flatten()[0].set_ylim(0)
                fig.axes.flatten()[0].set_xlim(0)

                path = f"{savedir}/epoch_{epoch}--locshape_{stringify_list(x, True)}--eachstrokeindex.pdf"
                print("Saving fig to: ", path)
                savefig(fig, path)

                plt.close("all")

    for epoch in list_epoch:
        for x in list_shape:
            DFTHIS = DFTHISGOOD[(DFTHISGOOD["epoch"]==epoch) & (DFTHISGOOD["shape_pre_this"]==x)]
            if sum([sum(DFTHIS["strokeindex_chunkdiff"]==strokeindex_chunkdiff)>=nmin for strokeindex_chunkdiff in list_strokeindex_chunkdiff])>1:
                fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="strokeindex_chunkdiff",
                             plot_kws={"alpha":0.6}, height = 6, aspect=0.75)
                fig.axes.flatten()[0].set_title(f"{epoch}-{x}")
                fig.axes.flatten()[0].set_ylim(0)
                fig.axes.flatten()[0].set_xlim(0)

                path = f"{savedir}/epoch_{epoch}--shape_{stringify_list(x, True)}--eachstrokeindex.pdf"
                print("Saving fig to: ", path)
                savefig(fig, path)

                plt.close("all")


    DFTHIS = DFTHISGOOD[(DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
    fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch_chunkdiff",
                 plot_kws={"alpha":0.3}, height = 8, aspect=0.75)
    fig.axes.flatten()[0].set_ylim(0)
    fig.axes.flatten()[0].set_xlim(0)
    path = f"{savedir}/epoch_chunkdiff.pdf"
    print("Saving fig to: ", path)
    savefig(fig, path)
    plt.close("all")

    DFTHIS = DFTHISGOOD[(DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
    fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="epoch",
                 plot_kws={"alpha":0.3}, height = 8, aspect=0.75)
    fig.axes.flatten()[0].set_ylim(0)
    fig.axes.flatten()[0].set_xlim(0)
    path = f"{savedir}/epoch.pdf"
    print("Saving fig to: ", path)
    savefig(fig, path)
    plt.close("all")

    for epoch in list_epoch:
        DFTHIS = DFTHISGOOD[(DFTHISGOOD["epoch"]==epoch) & (DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
        fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="strokeindex_chunkdiff",
                     plot_kws={"alpha":0.5}, height = 8, aspect=0.75)
        path = f"{savedir}/epoch-{epoch}--strokeindex_chunkdiff.pdf"
        print("Saving fig to: ", path)
        savefig(fig, path)
        plt.close("all")


        DFTHIS = DFTHISGOOD[(DFTHISGOOD["epoch"]==epoch) & (DFTHISGOOD["stroke_index"].isin(stroke_indices_keep))]
        fig = sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="shape_pre_chunkdiff_this",
                     plot_kws={"alpha":0.6}, height = 8, aspect=0.75)
        path = f"{savedir}/epoch-{epoch}--shape_pre_chunkdiff_this.pdf"
        print("Saving fig to: ", path)
        savefig(fig, path)
        plt.close("all")

        # sns.pairplot(data=DFTHIS, x_vars="gap_from_prev_dist", y_vars="gap_from_prev_dur", hue="shape_pre_chunkdiff_this",
        #              kind="kde", plot_kws={"alpha":0.6}, height = 8, aspect=0.75)


def _plot_velocity_all(DS):
    assert False, "IN PROGRESS! not that useful, since dur is not that linear wrt time..."


    for epoch in list_epoch:
        print(epoch)
        dfthis = DFTHISGOOD[DFTHISGOOD["epoch"]==epoch]

        y = "gap_from_prev_vel"
        HUE = "chunk_diff_from_prev"
        x = "stroke_index"
        fig = sns.catplot(data=dfthis, x=x, y=y, hue=HUE, jitter=True, alpha=0.3, aspect=2)
        rotateLabel(fig)

        fig = sns.catplot(data=dfthis, x=x, y=y, hue=HUE, kind="bar", alpha=0.3, aspect=2)
        rotateLabel(fig)

    #     fig = sns.catplot(data=dfthis, x=x, y=y, hue=HUE, alpha=0.3, aspect=2, kind="bar", col="chunk_rank")
    #     rotateLabel(fig)

    #     fig = sns.catplot(data=dfthis, x="chunk_within_rank", y=y, hue=HUE, alpha=0.3, aspect=2, kind="bar", col="chunk_rank")
    #     rotateLabel(fig)

    #     fig = sns.catplot(data=dfthis, x=x, y=y, hue="chunk_rank", alpha=0.3, aspect=2, kind="bar", col="chunk_diff_from_prev")
    #     rotateLabel(fig)


    fig = sns.catplot(data=DS.Dat, x="epoch", y="gap_from_prev_vel", hue="chunk_diff_from_prev",
                      col="shape_pre_this", col_wrap=3, kind="bar", alpha=0.3, aspect=2)
    rotateLabel(fig)


    fig = sns.catplot(data=DFTHISGOOD, x="shape_pre_this", y="gap_from_prev_vel", hue="chunk_diff_from_prev",
                      row="epoch", col="stroke_index", kind="bar", alpha=0.3, aspect=1)
    rotateLabel(fig)

    # fig = sns.catplot(data=DS.Dat, x="shape_pre_this", y="gap_from_prev_vel", hue="epoch", jitter=True,
    #                   alpha=0.2, col="stroke_index", col_wrap=3)
    # rotateLabel(fig)

    fig = sns.catplot(data=DFTHISGOOD, x="shape_pre_this", y="gap_from_prev_vel", hue="epoch", jitter=True, alpha=0.2)
    rotateLabel(fig)

    fig = sns.catplot(data=DFTHISGOOD, x="shape_pre_this", y="gap_from_prev_vel", hue="epoch", kind="point", ci=68)
    rotateLabel(fig)


    sns.pairplot(data=DFTHISGOOD, x_vars="shape_pre_this", y_vars="gap_from_prev_dur", hue="chunk_diff_from_prev",
                 plot_kws={"alpha":0.4}, height = 4, aspect=0.75)
