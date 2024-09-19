""" analysis of characters, including strokiness
Extract prims, gets their dsitance to basis sets of strokes, 
plots.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type

def debug_eyeball_distance_metric_goodness(D):
    """ Script to comapre distnaces by eye to what loloks good. Iterates over distance metrics,
    and for eahc plots example strokes along with printing their scores. See that the scores
    match expected by eye (strokiness).
    NMOTE: in progress, but skeleton is there.
    """
        
    # collect data, once for each distance metric
    # list_distance_ver=["euclidian_diffs", "euclidian", "hausdorff_alignedonset"]
    list_distance_ver = ["dtw_vels_2d"]
    LIST_RES = []
    for dist in list_distance_ver:
        RES = generate_data(D, list_distance_ver=[dist])
        LIST_RES.append(RES)

    # plot for each metric
    list_inds = get_random_inds()
    for RES in LIST_RES:
        plot_example_trials(RES, list_inds=list_inds)


def pipeline_generate_and_plot_all(D, do_plots=True, filter_taskkind = "character",
    list_distance_ver=None, suffix=None):
    """
    Full pipeline to generate and plot and save all, for character analysis.
    """
    
    if list_distance_ver is None:
        # list_distance_ver=("euclidian_diffs", "euclidian", "hausdorff_alignedonset")
        list_distance_ver = ["dtw_vels_2d"]

    savedir = D.make_savedir_for_analysis_figures("character_strokiness")
    if suffix is not None:
        savedir = f"{savedir}/{suffix}"

    assert len(D.Dat)>0

    RES = generate_data(D, filter_taskkind=filter_taskkind,
        list_distance_ver=list_distance_ver,
        which_features="beh_motor_sim")

    if do_plots and RES is not None:

        DS = RES["DS"]
        Dthis = DS.Dataset # Because this is a copy of D that has been modded
        list_strok_basis = RES["list_strok_basis"]
        list_shape_basis = RES["list_shape_basis"]

        if "microstim_epoch_code" in DS.Dataset.Dat.columns:
            plot_epoch_effects_paired_chars_microstim_wrapper(DS, savedir)

        plot_clustering(DS, list_strok_basis, list_shape_basis, savedir)
        plot_learning_and_characters(Dthis, savedir)

        for MIN_SCORE in [0., 0.7]:
            for sorted_unique_shapes in [True, False]:
                plot_prim_sequences(RES, Dthis, savedir, MIN_SCORE,
                    sorted_unique_shapes)

    return RES, savedir

def generate_data(D, version_trial_or_shapemean="trial", 
        trial_summary_score_ver="clust_sim_max",
        list_distance_ver=None, 
        plot_score_hist=False, filter_taskkind = None,
        ds_clean_methods = None, ds_clean_params = None,
        ds_filterdict = None,
        which_features = "beh_motor_sim"):
    """
    Initial data generation: extracts strokes, extracts basis
    set of strokes to compare them to, does similarity matrix
    PARAMS;
    - which_basis_set, which_shapes, params for getting bassis set of 
    strokes (see within)
    - trial_summary_score_ver, str, which statistic to use as the
    summary score (i.e, a strokiness score)
    - which_features, str, HACKY -- extract just a single feature space, 
    event hough multiple are computed, e.g, "beh_motor_sim"
    RETURNS:
    - RES, dict of results. -OR-
    - (or) None, if no tasks left after filter (e.g.,, filter to just chars)
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes

    assert list_distance_ver is None, "does nothing.."

    ### Generate Strokes data
    Dthis = D.copy()
    Dthis.preprocessGood(params=["no_supervision", "remove_online_abort"])

    DS = DatStrokes(Dthis)
    if ds_clean_methods is not None:
        DS.clean_preprocess_data(methods=ds_clean_methods, params=ds_clean_params)
    # Filter to just "character" tasks
    if filter_taskkind:
        DS.filter_dataframe({"task_kind":[filter_taskkind]}, True)
    if ds_filterdict is not None:
        DS.filter_dataframe(ds_filterdict, True)

    if len(DS.Dat)==0:
        # usually beuase not "character" taskkinds
        return None

    ## Score each trial across all features.
    ClustDict, ParamsDict, ParamsGeneral, dfdat = DS.features_wrapper_generate_all_features(version_trial_or_shapemean)

    ## Assign trials to clusters, based on a single feature space.
    DS.clustergood_assign_data_to_cluster(ClustDict, ParamsDict, 
            ParamsGeneral, dfdat,
            which_features = which_features,
            trial_summary_score_ver=trial_summary_score_ver)

    ###  order the prims by complexity
    def map_shape_to_complexity(sh):
        """ Had coded, the complexity of shapes based on how many segments..."""
        lev0 = ["line"]
        lev1 = ["Lcentered", "V", "arcdeep"]
        lev2 = ["squiggle3", "usquare", "zigzagSq", "circle"]

        if any([x in sh for x in lev0]):
            return 0
        elif any([x in sh for x in lev1]):
            return 1
        if any([x in sh for x in lev2]):
            return 2
        else:
            print(sh)
            assert False

    complexities = []
    for i in range(len(DS.Dat)):
        complexities.append(map_shape_to_complexity(DS.Dat.iloc[i]["clust_sim_max_colname"]))
    DS.Dat["clust_sim_max_colname_complexity"]=complexities

    # Prune dataset
    DS.dataset_prune_to_match_self()

    ## OUT
    RES = {
        "version_trial_or_shapemean":version_trial_or_shapemean,
        "ClustDict":ClustDict,
        "ParamsDict":ParamsDict,
        "ParamsGeneral":ParamsGeneral,
        "dfdat":dfdat,
        "trial_summary_score_ver":trial_summary_score_ver,
        "DS":DS,
        "which_features":which_features,
        "list_strok_basis":ParamsDict[which_features]["list_strok_basis"],
        "list_shape_basis":ParamsDict[which_features]["list_shape_basis"],
        "list_distance_ver":list_distance_ver
        # "which_shapes":which_shapes,
        # "Cl":Cl,
    }
    
    if plot_score_hist:        
        plt.figure()
        plt.hist(Cl.Xinput)
    
    return RES


def plot_example_trials(RES, nrand=5, list_inds=None):
    """ Useful plots and prints for evaluating quality of strok-strok distance/simialrity
    matrix calcualtion. 
    Plot n example trials (i.e, extract strokes) overlaid on their dataset trials 
    (e.g., entire character). Also print similarituy and clustering results.
    """

    DS = RES["DS"]
    Cl = RES["Cl"]
    
    # pick random strokes and plot their concentration
    if list_inds is None:
        import random
        list_inds = random.sample(range(len(DS.Dat)), nrand)

    # DS.plot_multiple([1,2,100], titles_by_dfcolumn="clust_sim_max")
    # fig, axes, list_inds = DS.plot_multiple(None, titles_by_dfcolumn="clust_sim_max_colname", nrand=5)
    fig, axes, inds_trials_dataset = DS.plot_multiple_overlay_entire_trial(list_inds)
    fig, axes, inds_trials_dataset = DS.plot_multiple_overlay_entire_trial(list_inds, overlay_beh_or_task="task")

    # # plot task images
    # D.plotMultTrials(inds_trials_dataset, which_strokes="strokes_task")
    # DS.plot_beh_and_aligned_task_strokes(list_inds)

    # 
    # Print info about each stroke
    print("vec labels:", Cl.LabelsCols)
    print("")
    for ind in list_inds:
        simvec = DS.Dat.iloc[ind]["clust_sim_vec"]
        simmax = DS.Dat.iloc[ind]["clust_sim_max"]
        simmaxind = DS.Dat.iloc[ind]["clust_sim_max_ind"]
        simmaxname = DS.Dat.iloc[ind]["clust_sim_max_colname"] 
        simconc = DS.Dat.iloc[ind]["clust_sim_concentration"]
        alignshape = DS.Dat.iloc[ind]["shape"]

        print(ind, int(100*simmax), int(100*simconc), simmaxname)
        print(simmaxind, simvec)
        print(" ")

def _plot_microstim_effect(DS, savedir):
    """ Plot effect of microstim on characters. Each plot is a scatter, with each char being a single datapt
    This only works if "microstim_epoch_code" is a column.
    Saves in (e.g) /gorilla1/analyses/main/character_strokiness/Diego_charstimdiego1_231209/epoch_effects
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    import os

    Dthis = DS.Dataset

    if "microstim_epoch_code" not in Dthis.Dat.columns:
        return

    # Include Other metrics of strokes
    Dthis.extract_beh_features()

    # First, keep only chars that have both epochs
    dfthis, dict_dfthis = extract_with_levels_of_conjunction_vars(Dthis.Dat, "epoch", ["character"],
                                                                  n_min_across_all_levs_var=1)

    # Get names of stim epochs.
    stim_epochs = sort_mixed_type(dfthis["microstim_epoch_code"].unique().tolist())
    stim_epochs_on = [e for e in stim_epochs if not e=="off"]

    sdir = f"{savedir}/epoch_effects_charlevel"
    os.makedirs(sdir, exist_ok=True)
    print("Saving plots at: ", sdir)
    for var_outer in ["task_stagecategory", "epoch_orig"]:
        for feat in ["FEAT_angle_overall", "FEAT_num_strokes_beh", "FEAT_circ", "FEAT_dist", "strokes_clust_score"]:
            for stim_epoch in stim_epochs_on:
                for plot_label in [False, True]:
                    dfres, fig = plot_45scatter_means_flexible_grouping(dfthis, "microstim_epoch_code",
                                                                        "off", y_lev_manip=stim_epoch,
                                                                        var_value=feat, var_subplot=var_outer,
                                                                        var_datapt="character",
                                                           plot_text=plot_label, alpha=0.2, SIZE=4)
                    savefig(fig, f"{sdir}/by_{var_outer}-score_{feat}-scatter-vs-{stim_epoch}-label_{plot_label}.pdf")


            fig = sns.catplot(data=dfthis, x=var_outer, y=feat, hue="microstim_epoch_code", kind="point", errorbar=("ci", 68))
            path =f"{sdir}/by_{var_outer}-score_{feat}-pointplot.pdf"
            print(path)
            savefig(fig, path)
            plt.close("all")

def plot_epoch_effects_paired_chars_microstim_wrapper(DS, savedir, paired_by_char=True,
                                    SAMPLE_RANDOM_SINGLE_TRIAL=False):
    """
    Plot effects of microstim, where chars have both microstim and no-stim trials.
    :param DS:
    :return:
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import convert_to_1d_dataframe_hist
    from scipy.stats import entropy
    from pythonlib.tools.nptools import bin_values_categorical_factorize


    # Extract and preprocess
    DS = DS.copy()
    DS.dataset_append_column("microstim_epoch_code")
    Dthis = DS.Dataset

    # Keep only if char has both epochs
    if paired_by_char:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        Dthis.Dat, _ = extract_with_levels_of_conjunction_vars(Dthis.Dat, "epoch", ["character"],
                                                               n_min_across_all_levs_var=1)
        # - Prune DS so that chars are matched across epochs.
        DS.dataset_replace_dataset(Dthis)
        DS.dataset_prune_self_to_match_dataset()

    ## Pair up chars even more by getting a single trial per char
    if SAMPLE_RANDOM_SINGLE_TRIAL:
        from pythonlib.tools.pandastools import extract_trials_spanning_variable, append_col_with_grp_index
        DS.Dataset.Dat = append_col_with_grp_index(DS.Dataset.Dat, ["character", "epoch"],
                                                   "char_epoch", use_strings=True,
                                                   strings_compact=True)
        inds, chars = DS.Dataset.taskcharacter_extract_examples(var="char_epoch")
        trialcodes = Dthis.Dat.iloc[inds]["trialcode"].tolist()
        assert len(set(trialcodes))==len(trialcodes)

        # slice DS by trialcodes
        from pythonlib.tools.pandastools import slice_by_row_label
        dfthis = slice_by_row_label(DS.Dat, "trialcode", trialcodes)
    else:
        # interim, just use all data
        dfthis = DS.Dat
    DS.Dat = dfthis

    #### Datapt = char (Paired chars (scatter))
    _plot_microstim_effect(DS, savedir)

    #### LOCATION/SHAPE/INDEX distribitions
    _plot_location_shape_index_distributions(DS, savedir)

    #### MOTOR TIMING
    _plot_timing(DS, savedir)

    #### OTHER PLOTS
    sdir = f"{savedir}/epoch_effects_strokelevel"
    os.makedirs(sdir, exist_ok=True)

    ### PLOTS
    fig = sns.catplot(data=dfthis, x="clust_sim_max_colname", y="clust_sim_max", hue="epoch", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{sdir}/clust_sim_max-vs-clust_sim_max_colname-1.pdf")

    fig = sns.catplot(data=dfthis, x="clust_sim_max_colname", y="clust_sim_max", hue="epoch", aspect=1.5,
                      kind="point", errorbar=("ci", 68))
    rotateLabel(fig)
    savefig(fig, f"{sdir}/clust_sim_max-vs-clust_sim_max_colname-2.pdf")

    fig, ax = plt.subplots(1,1)
    sns.histplot(data=dfthis, x="clust_sim_max", hue="microstim_epoch_code", ax=ax)
    savefig(fig, f"{sdir}/hist_1d-clust_sim_max.pdf")

    fig = sns.catplot(data=dfthis, x="microstim_epoch_code", y="clust_sim_max", kind="point")
    rotateLabel(fig)
    savefig(fig, f"{sdir}/pointplot-clust_sim_max.pdf")

    # Histrograms
    fig, ax = plt.subplots()
    epochs = dfthis["epoch"].unique().tolist()
    for ep in epochs:
        dftmp = dfthis[dfthis["epoch"]==ep]
        # shapes, vals, fig, ax = convert_to_1d_dataframe_hist(dftmp, "clust_sim_max_colname", True, ax=ax)
        convert_to_1d_dataframe_hist(dftmp, "clust_sim_max_colname", True, ax=ax)
        # # fig.savefig(f"{sdir}/hist_n_matches.pdf")
    savefig(fig, f"{sdir}/hist_n_matches.pdf")

    for var in ["clust_sim_max_colname", "clust_sim_max_colname_complexity"]:
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, var, "epoch", True, annotate_heatmap=True)
        savefig(fig, f"{sdir}/hist_2d-{var}.pdf")

        _, fig, _, _ = convert_to_2d_dataframe(dfthis, var, "epoch", True, norm_method="col_div",
                                               annotate_heatmap=True)
        savefig(fig, f"{sdir}/hist_2d-{var}-col_div.pdf")

        fig, ax = plt.subplots(1,1)
        sns.histplot(data=dfthis, x=var, hue="epoch", stat="probability",
                     multiple="dodge", ax=ax)
        from pythonlib.tools.plottools import rotate_x_labels
        rotate_x_labels(ax, 45)
        savefig(fig, f"{sdir}/hist_1d-{var}.pdf")


    ##### get entropy of shape names as a function of thresholding
    fig, ax = plt.subplots()
    threshes = np.linspace(0, 1, 20)
    for epoch in epochs:
        entropies = []
        list_n = []
        for thresh in threshes:

            # shapes = dfthis[dfthis["clust_sim_max"]>thresh]["clust_sim_max_colname"].factorize()
            shapes = dfthis[(dfthis["clust_sim_max"]>thresh) & (dfthis["epoch"]==epoch)]["clust_sim_max_colname"].tolist()
            shapes_int = bin_values_categorical_factorize(shapes)
            entropies.append(entropy(shapes_int))
            list_n.append(len(shapes_int))

        ax.plot(threshes, entropies, "-o", label=epoch)
        print("list n: ", epoch, list_n)
    ax.legend()
    ax.set_title("entropy of shape lables")
    ax.set_ylabel("entropy")
    ax.set_xlabel("thresh (keep data above this)")
    savefig(fig, f"{sdir}/entropy-vs-threshold_clust_stim.pdf")

    plt.close("all")


def _plot_timing(DS, savedir, microstim_version=True):
    """
    Plot effect of micrositm on timing...
    :param DS:
    :param savedir:
    :param microstim_version:
    :return:
    """
    # go cue to each stroke
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline, plot_triallevel_results
    from pythonlib.dataset.dataset_analy.motortiming import _gapstrokes_preprocess_extract_strokes_gaps

    sdir = f"{savedir}/timing"
    os.makedirs(sdir, exist_ok=True)

    D = DS.Dataset

    # Dcopy = D.copy()
    # DS, SAVEDIR, dfres, grouping = preprocess_plot_pipeline(Dcopy, microstim_version=True, PLOT=False)

    # microstim_version = True
    # Merge the DS from strokes clust with DS from primtiveness.
    prune_strokes = True
    VARS_CONTEXT = None
    DS = _gapstrokes_preprocess_extract_strokes_gaps(DS, microstim_version, prune_strokes, VARS_CONTEXT)

    # Metric of stroke complexity

    # plot timing where context is simply stroke index (here shape is less relevant)
    contrast = "epoch"
    # savedir = "/tmp"
    if False:
        context = "context"
        # context var: stroke index x assigned shape
        DS.grouping_append_and_return_inner_items(["stroke_index", "clust_sim_max_colname"], new_col_name="context")
    else:
        context = "stroke_index"
    yvars = ["circularity", "time_duration", "velocity", "distcum", "gap_from_prev_dur",
            "gap_from_prev_dist", "gap_from_prev_vel"]
    plot_triallevel_results(DS, contrast, sdir, context, yvars)

def _plot_location_shape_index_distributions(DS, savedir):
    """
    Plot effect of microstim on distribiotn of shapes,. locations.
    :param RES:adfasdfasdf
    :param RES:adfasdfasdf
    :return:
    """
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.tools.plottools import color_make_pallete_categories

    sdir = f"{savedir}/loc_shape_index_distros"
    os.makedirs(sdir, exist_ok=True)

    # Thresholds, for keeping only decent shapes for this plot.
    thresh = 0.62
    max_nstrokes = 4

    # First, threshold it to just those that are good matches.
    # DS = RES["DS"]

    # get the center of the actual beh stroke
    def F(x):
        return x["Stroke"].extract_center()
    DS.Dat = applyFunctionToAllRows(DS.Dat, F, "beh_center")

    # Colors for each stroke index
    cmap = color_make_pallete_categories(DS.Dat, "stroke_index", "turbo")

    ### Plot distrubtions for each epoch
    list_epoch = DS.Dat["epoch"].unique()
    for epoch in list_epoch:
        dfthis = DS.Dat[(DS.Dat["clust_sim_max"]>thresh) & (DS.Dat["stroke_index"]<=max_nstrokes) & (DS.Dat["epoch"]==epoch)].reset_index(drop=True)
        # print(len(DS.Dat))
        # print(len(dfthis))

        # Combine all shapes, plot location
        SIZE = 4
        var = "beh_center"
        fig, ax = plt.subplots(1,1, figsize=(SIZE, SIZE))
        # locations
        locations = np.asarray(dfthis[var].tolist())
        indexes = np.asarray(dfthis["stroke_index"].tolist())
        colors = [cmap[i] for i in indexes]
        ax.scatter(locations[:,0], locations[:,1], c=colors, marker=".", alpha=0.3)
        savefig(fig, f"{sdir}/locations-ALLDATA-{epoch}.pdf")

        # Separate supblots for each level of (shape) or (stroke_index)
        for var_subplot in ["clust_sim_max_colname", "stroke_index"]:
            list_lev = sort_mixed_type(dfthis[var_subplot].unique())
            SIZE = 3
            var = "beh_center"
            ncols = 4
            nrows = int(np.ceil(len(list_lev)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))

            for sh, ax in zip(list_lev, axes.flatten()):
                dfthisthis = dfthis[dfthis[var_subplot]==sh]
                ax.set_title(sh)

                # locations
                locations =  np.asarray(dfthisthis[var].tolist())
                indexes = np.asarray(dfthisthis["stroke_index"].tolist())
                colors = [cmap[i] for i in indexes]
                ax.scatter(locations[:,0], locations[:,1], c=colors, marker=".", alpha=0.25)
                # ax.hist2d(locations[:,0], locations[:,1])
                # ax.plot(locations[:,0], locations[:,1], '.k', alpha=0.2)

            savefig(fig, f"{sdir}/locations-{var_subplot}-{epoch}.pdf")

        # Bias for shapes
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "clust_sim_max_colname",
                                               "stroke_index", True, annotate_heatmap=False)
        savefig(fig, f"{sdir}/2dheat-shape-stroke_index-NONORM-{epoch}.pdf")

        for norm_method in ["row_div", "col_div"]:
            _, fig, _, _ = convert_to_2d_dataframe(dfthis, "clust_sim_max_colname",
                                                   "stroke_index", True, annotate_heatmap=False,
                                                   norm_method=norm_method)
            savefig(fig, f"{sdir}/2dheat-shape-stroke_index-{norm_method}-{epoch}.pdf")


def plot_learning_and_characters(D, savedir, scorename = "strokes_clust_score"):
    """ Plot and save all things related to:
    - strokiness of each trial, grouped by block, char, and day
    - changes in strokiness (i.e., learning)
    (Meant for across day analyssi)
    """ 
    from scipy.stats import linregress as lr
    from pythonlib.tools.nptools import rankItems

    sdir = f"{savedir}/figs_character_strokiness"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving figures at:", sdir)
    
    ## COrrelation between online scores (e.g, strokiness) and offline computed strokiness
    # sdir = f"{savedir}/corr_scores"
    # import os
    # os.makedirs(sdir, exist_ok=True)
    list_blocks = sorted(D.Dat["block"].unique().tolist())
    vars_to_compare = ["rew_total", "strokinessv2", "pacman", "hausdorff", "ft_decim", "numstrokesorig", "ft_minobj", "frac_strokes", "shortness", "score_final"]
    vars_to_compare = [var for var in vars_to_compare if var in D.Dat.columns]
    for bk in list_blocks:
        df = D.Dat[D.Dat["block"]==bk]
        if len(df)>0:
            try:
                if sum(~df[scorename].isna())>0:
                    fig = sns.pairplot(data=df, x_vars=vars_to_compare, y_vars=[scorename])
                    savefig(fig, f"{sdir}/corr-rew_vs_strokiness-bk{bk}.pdf")
                    # fig.savefig(f"{sdir}/corr-rew_vs_strokiness-bk{bk}.pdf")
                plt.close("all")
            except Exception as err:
                print("----", bk)
                print(len(df))
                print(df.keys())
                raise err

    ## Score for each trial by block and date
    fig = sns.catplot(data=D.Dat, x="block", y=scorename, aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-1.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="violin", aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-2.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="point", aspect=2, hue="date")
    fig.savefig(f"{sdir}/score_by_block-3.pdf")

    fig = sns.catplot(data=D.Dat, x="date", y=scorename, kind="point", aspect=2, hue="block")
    fig.savefig(f"{sdir}/score_by_date-1.pdf")

    fig = sns.catplot(data=D.Dat, col="block", col_wrap=4, x="date", y=scorename, aspect=1, kind="point")
    fig.savefig(f"{sdir}/score_by_date-3.pdf")

    fig = sns.relplot(data=D.Dat, col="block", col_wrap=3, x="tvalfake", y=scorename, aspect=2, kind="scatter")
    fig.savefig(f"{sdir}/score_by_tval_withinblock.pdf")

    if "strokinessv2" in D.Dat.columns:
        fig = sns.relplot(data=D.Dat, col="block", col_wrap=3, x="tvalfake", y="strokinessv2", aspect=2, kind="scatter")
        fig.savefig(f"{sdir}/strokinessv2_by_tval_withinblock.pdf")

    plt.close("all")

    ## For each character
    # Plot chars, sorted by score, for ALL and for each category.
    D.taskcharacter_print_score_per_char(scorename, sdir)

    # First, sort characters
    # Sort all characters by score, and plot them in a grid
    list_char, list_score, list_n = D.taskcharacter_find_plot_sorted_by_score(scorename,
                                                                      plot=True,
                                                                      sdir=sdir)

    # # Plot
    # # -- get one trial for each char
    #
    # from pythonlib.tools.pandastools import extract_trials_spanning_variable
    # n_iter = 3
    # for i in range(n_iter):
    #     inds, chars = extract_trials_spanning_variable(D.Dat, "character", list_char)
    #     assert chars == list_char
    #
    #     if len(inds)<60:
    #         indsthis = inds
    #         charsthis = chars
    #         list_score_this = list_score
    #     else:
    #         indsthis = inds[:30] + inds[-30:]
    #         charsthis = chars[:30] + chars[-30:]
    #         list_score_this = list_score[:30] + list_score[-30:]
    #
    #     # -- plot
    #     fig, axes, idxs = D.plotMultTrials2(indsthis, titles=charsthis, SIZE=3);
    #     fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
    #     fig, axes, idxs = D.plotMultTrials2(indsthis, "strokes_task", titles=list_score_this);
    #     fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")
    #
    #     plt.close("all")

    # which carhacters best
    fig = sns.catplot(data=D.Dat, y="character", x=scorename, height=10, hue="block",
               order = list_char)
    fig.savefig(f"{sdir}/char_sorted_score-1.pdf")

    fig = sns.catplot(data=D.Dat, y="character", x=scorename, height=8, col="block", col_wrap=2,
               order = list_char)
    fig.savefig(f"{sdir}/char_sorted_score-2.pdf")

    fig = sns.relplot(data=D.Dat, x="block", y=scorename, col="character", col_wrap=4, 
               col_order=list_char)
    fig.savefig(f"{sdir}/char_learning_block.pdf")
    # fig = sns.catplot(data=D.Dat, x="block", y=scorename, col="character", col_wrap=4, 
    #            col_order=list_char)
    # fig.savefig(f"{sdir}/char_score_by_block-1.pdf")

    fig = sns.relplot(data=D.Dat, x="tvalfake", y=scorename, col="character", col_wrap=4, 
               col_order=list_char, hue="block")
    fig.savefig(f"{sdir}/char_learning_tval.pdf")

    plt.close("all")
    
    ### Change in score over trials (each char one slope)
    list_char_alpha = sorted(D.Dat["character"].unique().tolist())
        
    # First, prune dataset to avoid (i) trials with supervision and (ii) trials
    # without enough strokes, both of which have artifically high strokiness.
    # Each trial, got correct n strokes?
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    def F(x):
        # Returns True if n beh strokes = or > than n task strokes.
        ntask = len(x["strokes_task"])
        nbeh = len(x["strokes_beh"])
        return nbeh>=ntask
    D.Dat = applyFunctionToAllRows(D.Dat, F, "nbeh_match_ntask")

    # For each character, get its change in score
    ONLY_TRIALS_WITH_ENOUGH_STROKES = True
    ONLY_TRIALS_IN_NOSUP_BLOCKS = True

    Dcopy = D.copy()
    if ONLY_TRIALS_IN_NOSUP_BLOCKS:
        Dcopy.preprocessGood(params=["no_supervision"])
    DF = Dcopy.Dat
    print("orignial", len(DF))
    # if ONLY_TRIALS_IN_NOSUP_BLOCKS:
    #     DF = DF[DF["supervision_stage_concise"]=="off|0||0"]
    # print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))
    if ONLY_TRIALS_WITH_ENOUGH_STROKES:
        DF = DF[DF["nbeh_match_ntask"]==True]
    print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))

    if "strokinessv2" in df.columns:
        list_slope = []
        list_slope_sk2 = []
        list_slope_rew = []
        for char in list_char_alpha:
            df = DF[DF["character"]==char]
            t = df["tvalfake"]
            v = df[scorename]
            strokinessv2 = df["strokinessv2"]
            rew_total = df["rew_total"]


            try:
                if len(t)>=5:
                    # convert to to rank
                    t = rankItems(t)

                    if len(np.unique(v))>1:
                        slope = lr(t, v)[0]
                    else:
                        slope = np.nan
                    list_slope.append(slope)

                    # make sure score and strokiness correlate with offline score
                    if len(np.unique(strokinessv2))>1:
                        slope = lr(strokinessv2, v)[0]
                    else:
                        slope = np.nan
                    list_slope_sk2.append(slope)

                    if len(np.unique(rew_total))>1:
                        slope = lr(rew_total, v)[0]
                    else:
                        slope = np.nan
                    list_slope_rew.append(slope)      

                else:
                    list_slope.append(np.nan)
                    list_slope_rew.append(np.nan)
                    list_slope_sk2.append(np.nan)
            except Exception as err:
                print(t)
                print(strokinessv2)
                print(v)
                print(rew_total)
                raise err


        # only keep chars with enough data
        inds = np.where(~np.isnan(list_slope))[0].tolist()
        list_char_alpha = [list_char_alpha[i] for i in inds]
        list_slope = [list_slope[i] for i in inds]
        list_slope_rew = [list_slope_rew[i] for i in inds]
        list_slope_sk2 = [list_slope_sk2[i] for i in inds]

        # Plot (each char)
        fig, axes = plt.subplots(1,3, figsize=(15, len(list_char_alpha)*0.16))

        if len(list_slope)>0:
            ax=axes.flatten()[0]
            ax.plot(list_slope, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/trials)")
            ax.grid(True)

            ax=axes.flatten()[1]
            ax.plot(list_slope_sk2, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")
            ax.grid(True)

            ax=axes.flatten()[2]
            ax.plot(list_slope_rew, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/rew_total)")
            ax.grid(True)

        fig.savefig(f"{sdir}/slope_score_vs_trial-each_char.pdf")

        # Plot, historgram across cahar
        fig, axes = plt.subplots(1,3, figsize=(9,2))

        if len(list_slope)>0:
            ax=axes.flatten()[0]
            ax.hist(list_slope, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/trials)")

            ax=axes.flatten()[1]
            ax.hist(list_slope_sk2, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")

            ax=axes.flatten()[2]
            ax.hist(list_slope_rew, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/rew_total)")

        fig.savefig(f"{sdir}/slope_score_vs_trial-hist.pdf")

        # Does having higher slope for (rew vs. score) predict learning?
        fig, axes = plt.subplots(2,2, figsize=(8,8))

        if len(list_slope)>0:

            ax = axes.flatten()[0]
            ax.plot(list_slope_rew, list_slope, 'ok')
            ax.set_xlabel(f"slope ({scorename}/rew_total)")
            ax.set_ylabel(f"slope ({scorename}/trials)")

            ax = axes.flatten()[1]
            ax.plot(list_slope_sk2, list_slope, 'ok')
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")
            ax.set_ylabel(f"slope ({scorename}/trials)")

        fig.savefig(f"{sdir}/scatter-slopes_vs_slopes.pdf")

    # Close
    plt.close("all")


def plot_clustering(DS, list_strok_basis, list_shape_basis, savedir):
    """ Clustering beh stroke based on max similiaryt to prims. Plots example
    strokes (beh) to get a sense of how clustered are the strokes.
    """

    sdir = f"{savedir}/clustering_by_basis_prims"
    import os
    os.makedirs(sdir, exist_ok=True)

    from pythonlib.tools.pandastools import convert_to_1d_dataframe_hist
    shapes, vals, fig, ax = convert_to_1d_dataframe_hist(DS.Dat, "clust_sim_max_colname", True)
    # fig.savefig(f"{sdir}/hist_n_matches.pdf")
    savefig(fig, f"{sdir}/hist_n_matches.pdf")

    # fig = DS.plot_examples_grid("clust_sim_max_colname", col_levels=shapes, nrows=5)
    fig = DS.plotshape_multshapes_trials_grid("clust_sim_max_colname", 
        col_levels=shapes, nrows=5)
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches.pdf")

    # Plot the basis set stroke
    # sort by order
    list_strok_basis_sorted = [list_strok_basis[list_shape_basis.index(sh)] for sh in shapes]
    fig, axes = DS.plot_multiple_strok(list_strok_basis_sorted, overlay=False, ncols=len(list_strok_basis_sorted), titles=shapes);
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches-basis.pdf")

    figholder = DS.plotshape_multshapes_egstrokes(key_subplots = "clust_sim_max_colname",
                n_examples_total_per_shape = 5, color_by=None, list_shape=None);

    for i, x in enumerate(figholder):
        fig = x[0]
        fig.savefig(f"{sdir}/drawings_examples_overlaid-iter{i}.pdf")

    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    dfthis, fig, ax, rgba_values = convert_to_2d_dataframe(DS.Dat, "character", "clust_sim_max_colname", True, annotate_heatmap=False);
    # fig.savefig(f"{sdir}/hist_n_matches_2d_heat_characters.pdf")
    savefig(fig, f"{sdir}/hist_n_matches_2d_heat_characters.pdf")


    # for each basis strok, get the distribution of scores for beh strokes that match it
    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes)
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-1.pdf")

    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes, kind="bar")
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-2.pdf")

    plt.close("all")

    # For each basis stroke, plot examples, sorted by score
    from pythonlib.tools.pandastools import extract_trials_spanning_variable
    nrand = 40
    trials_dict = extract_trials_spanning_variable(DS.Dat, "clust_sim_max_colname", shapes, 
                                                   n_examples=nrand, return_as_dict=True, 
                                                   method_if_not_enough_examples="prune_subset")[0]

    for sh in shapes:
        inds = trials_dict[sh]
        scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
        
        # sort
        x = [(i, s) for i,s in zip(inds, scores)]
        x = sorted(x, key=lambda x:-x[1])
        inds = [xx[0] for xx in x]
        scores = [xx[1] for xx in x]
        fig, axes, list_inds = DS.plot_multiple(inds, titles=scores, ncols=8)
        fig.savefig(f"{sdir}/behstrokes_matching_basis-{sh}-sorted_by_score.pdf")    

    # Also look in entire pool, regardless of shape
    from pythonlib.tools.listtools import random_inds_uniformly_distributed
    inds = random_inds_uniformly_distributed(DS.Dat["clust_sim_max"].tolist(), nrand)
    scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
        
    # sort
    nrand = 64
    from pythonlib.tools.listtools import random_inds_uniformly_distributed
    inds = list(random_inds_uniformly_distributed(DS.Dat["clust_sim_max"].tolist(), nrand))
    scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
    shapes = DS.Dat.iloc[inds]["clust_sim_max_colname"]

    titles = [f"{sh}|{sc:.2f}" for sh, sc in zip(shapes, scores)]

    fig, axes, list_inds = DS.plot_multiple(inds, titles=titles, ncols=8)
    fig.savefig(f"{sdir}/behstrokes_ALL-sorted_by_score.pdf")    

def plot_prim_sequences(RES, D, savedir, MIN_SCORE = 0., sorted_unique_shapes=False):
    """ Extract for each trial a sequence of strokes (shape names), and plot
    things about these (e.g. distribution across trialsS).
    PARAMS:
    - MIN_SCORE, scalar, only keep trials with mean strokes_clust_score >= this
    - sorted_unique_shapes, bool, then first get sorted(unqiue()) shapes for each trial.
    """
    from pythonlib.tools.plottools import makeColors, rotate_x_labels

    # MIN_SCORE = 0.75
    sdir = f"{savedir}/prim_sequences-MIN_SCORE_{MIN_SCORE}-sorteduniq_{sorted_unique_shapes}"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving at:", sdir)
    DS = RES["DS"]

    def _extract_shapes_beh_order(tc):
        """ get list of shapes for this trial in order of beh strokes
        """        
        # 1) shapes, based on simmat alignment
        shapes_ordered_by_beh_strokes_datseg = DS.dataset_extract_strokeslength_list(tc, "shape")

        # 2) shapes, based on clustering each stroke
        shapes_ordered_by_beh_strokes_cluster = DS.dataset_extract_strokeslength_list(tc, "clust_sim_max_colname")

        return shapes_ordered_by_beh_strokes_datseg, shapes_ordered_by_beh_strokes_cluster

    # For each char, get its "best parse"
    # list_scores = []
    list_shapes_datseg = []
    list_shapes_cluster = []
    for i in range(len(D.Dat)):
        tc = D.Dat.iloc[i]["trialcode"]
        if D.Dat.iloc[i]["strokes_clust_score"]>=MIN_SCORE:
            shapes_datseg, shapes_cluster = _extract_shapes_beh_order(tc)
            list_shapes_cluster.append(shapes_cluster)
            list_shapes_datseg.append(shapes_datseg)
        else:
            list_shapes_cluster.append(None)
            list_shapes_datseg.append(None)
    
    # Remove Nones
    A = []
    B = []
    for sh_c, sh_d in zip(list_shapes_cluster, list_shapes_datseg):
        if sh_c is None:
            continue
        A.append(sh_c)
        # B.append(sh_d)
    list_shapes_cluster = A
    # list_shapes_datseg = A

    # Colors, for plotting ranks.
    n = max([len(x) for x in list_shapes_cluster])
    pcols, fig, ax = makeColors(n, ploton=True)
    fig.savefig(f"{sdir}/legend_rank_colors.pdf")

    # "Waterfall"-like plot, showing sequences ordered by their freqeuncy.
    if sorted_unique_shapes:
        def F(x):
            x = tuple(sorted(set(x)))
            return x
    else:
        def F(x):
            x = tuple(x)
            return x
    for Nprims in [None, 1, 2, 3,4,5]:

        list_shapes_cluster_unique = [F(x) for x in list_shapes_cluster if x is not None]

        # only keep if has Nprims
        if Nprims is not None:
            list_shapes_cluster_unique = [x for x in list_shapes_cluster_unique if len(x)==Nprims]

        from pythonlib.tools.listtools import tabulate_list
        list_shapes_sorted = sorted(tabulate_list(list_shapes_cluster_unique, True), key=lambda x:-x[1])

        # Plot it
        fig, axes = plt.subplots(1,2, figsize=(10, 6))

        # 1) Waterfall
        ax = axes.flatten()[0]
        for i, shapes in enumerate(list_shapes_sorted):
            for j, (sh, col) in enumerate(zip(shapes[0], pcols)):
                ax.plot(sh, i, 'x', color=col, alpha=0.7)
        ax.set_ylabel('rank in ordered by freq (0, most frequent sequence)')
        ax.set_xlabel('shapes in sequence')
        ax.set_title(f"[Nprims={Nprims}], Most common sequences")
        # plt.xticks(rotation=45);
        # ax.draw(ax.figure.canvas.renderer)
        rotate_x_labels(ax, 45)

        #2) Hist, n strokes in sew
        ax = axes.flatten()[1]
        ax.set_xlabel("n shapes in sequence")
        list_n = [len(x[0]) for x in list_shapes_sorted]
        ax.hist(list_n)
        plt.xticks(rotation=45);

        fig.savefig(f"{sdir}/waterfall_hist-Nprims_{Nprims}.pdf")

        plt.close("all")
