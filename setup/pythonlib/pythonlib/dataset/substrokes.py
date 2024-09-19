"""
Works with substrokes, which are segmentations of strokes into smaller (meaningful)
segments

Goal is to create a "fake" dataset, where the strokes are replaced by sbubstroeks, thus
allowing easy sliding into all analyses
"""

import matplotlib.pyplot as plt
import numpy as np
from ..tools.nptools import sort_by_labels as sbl
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
import seaborn as sns
from pythonlib.tools.stroketools import strokesVelocity
import os
import pandas as pd
import seaborn as sns
import random
from pythonlib.tools.plottools import savefig


def _convert_inds_to_frac(inds, Npts):
    """ Given inds (list of ints) converts them
    to fractions (i/Npts), think of this as window
    into stroke durations
    """
    out = tuple([i/Npts for i in inds])
    print(out)
    return out


def database_shape_segmentation_help_decide_values(DS, shape, Npts):
    """ Script to help guide user to hand-pick window values for a given shape
    """
    # Second, visualize this shape
    from pythonlib.tools.nptools import find_peaks_troughs
    # refrac_npts = 9

    refrac_npts = int(Npts/10)
    assert shape in DS.Dat["shape"].unique().tolist()

    inds_ds = DS.find_indices_this_shape(shape)
    print(shape)
    # Get all strokes for this shape, interpolate to have same n pts, and then stack (ndat, NPTS, 3)
    strok_template, strokstacked = DS.cluster_compute_mean_stroke(inds_ds, Npts=Npts, check_same_direction=False, ver="median")

    if len(inds_ds)==0:
        assert False, "Shape has no data"

    print(inds_ds)
    DS.plot_multiple_speed_and_drawing(inds_ds)

    # First, find the time indices of the troughs and peaks in the mean stroke (template)
    fs = 1/np.mean(np.diff(strok_template[:,2])) # is exact.
    _, strokes_speeds = strokesVelocity([strok_template], fs)
    inds_peaks, inds_troughs_template, _ = find_peaks_troughs(strokes_speeds[0][:,0], PLOT=True, refrac_npts=refrac_npts)

    print("Based on peaks and troughs, and variability across trials, decide on windows where expect troughs")
    print(f"Convert those windows to fractions (divide by {Npts}) and then enter by hand into database_shape_segmentation_get()")
    print("Use this as helper: _convert_inds_to_frac()")

    # # Plot
    # fig, axes = plt.subplots(1,3, figsize=(13,5))
    #
    # # Plot velocitues foir all trials
    # ax = axes.flatten()[0]
    # _strokes = [s for s in strokstacked]
    # _strokes_speed = strokesVelocity(_strokes, fs)[1]
    # ts = np.arange(Npts)
    # for sp in _strokes_speed:
    #     ax.plot(ts, sp[:,0], "-", alpha=0.1)
    #
    # ax = axes.flatten()[1]
    # _strokes = [s for s in strokstacked]
    # _strokes_speed = strokesVelocity(_strokes, fs)[1]
    # for sp in _strokes_speed:
    #     ax.plot(sp[:,1], sp[:,0], "-", alpha=0.1)
    #
    # # plot the strokes
    # ax = axes.flatten()[2]
    # DS.plot_single_strok(strok_template, ax=ax, color="k")
    # ax.set_title(f"{shape}-{database_shape_segmentation[shape]}")
    #
    # # print("Indices of troughts (in template):", inds_troughs_template)
    # # _inds_tmp = []
    # # for window in list_frac_windows: # (frac1, frac2)
    # #     # convert frac to inds'
    # #     window_inds = [frac*Npts for frac in window]
    # #
    # #     # find troughs within this iwnodw
    # #     _inds_tmp.extend([i for i in inds_troughs_template if (i>=window_inds[0]) and (i<=window_inds[1])])
    # # inds_troughs_template = sorted(_inds_tmp)
    # # print("-- after restricting to window ", inds_troughs_template)


def database_shape_segmentation_get(animal):
    """ Database hollding window where expect segmentation edges,
    in frac along stroke, by time.
    """
    if animal=="Diego":
        database_shape_segmentation = {

            "zigzagSq-1-1-1": [(0.24, 0.44), (0.54, 0.85)],

            "V-2-2-0": [(0.3, 0.6)],

            "arcdeep-4-3-0": [(0.20, 0.48), (0.52, 0.9)],
            "line-8-1-0": [],
            "line-8-2-0": [(0.42, 0.74)],
            "line-8-3-0": [],
            "line-8-4-0": [(0.64, 0.93)],
            "Lcentered-4-1-0":[(0.33, 0.69)],
            # "arcdeep-4-2-0":[(0.25, 0.55), (0.6, 0.94)],
            # "arcdeep-4-2-0":[(0.25, 0.55), (0.57, 0.94)],
            "arcdeep-4-2-0":[(0.25, 0.94)],

            "V-2-3-0":[(0.39, 0.65)],

            "squiggle3-3-2-0":[(0.1, 0.38), (0.45, 0.75)],

            "circle-6-1-0":[(0.18, 0.42), (0.58, 0.83)],

            "usquare-1-4-0":[(0.5, 0.8)],

            "Lcentered-4-4-0":[(0.4, 0.72)],
            "Lcentered-4-3-0":[(0.42, 0.72)],
            "squiggle3-3-1-1":[(0.18, 0.44)],

            "V-2-4-0":[(0.42, 0.65)],

            "arcdeep-4-4-0":[(0.5, 0.84)],
            "Lcentered-4-2-0":[(0.3, 0.69)],

            "usquare-1-3-0":[(0.27, 0.53)],


            # "usquare-1-2-0":[(0.25, 0.48), (0.52, 0.86)],
            "usquare-1-2-0":[(0.52, 0.86)], # first trough is not common

            "zigzagSq-1-2-0":[(0.16, 0.36), (0.43, 0.7)],

            "zigzagSq-1-2-1":[(0.2, 0.5), (0.6, 0.84)],

            "squiggle3-3-1-0":[(0.1, 0.32), (0.38, 0.62)],
            # "squiggle3-3-1-0":[(0.15, 0.4)],

            "zigzagSq-1-1-0":[(0.18, 0.4), (0.52, 0.77)],

            "squiggle3-3-2-1":[(0.18, 0.44), (0.5, 0.88)]
        }
    elif animal=="Pancho":
        database_shape_segmentation = {
            "line-8-1-0": [],
            "line-8-2-0": [],
            "line-8-3-0": [],
            "line-8-4-0": [],

            "line-9-1-0": [(0.3, 0.84)],
            "line-9-2-0": [],
            "line-9-3-0": [],
            "line-9-4-0": [],

            "line-10-1-0": [],
            "line-10-2-0": [],
            "line-10-3-0": [],
            "line-10-4-0": [],
            "V-2-2-0":[([0.4, 0.6])],
            "Lcentered-6-2-0":[(0.3, 0.54), (0.7, 0.9)],
            "Lcentered-6-3-0":[(0.3, 0.55)],
            "Lcentered-6-4-0":[(0.2, 0.42), (0.6, 0.88)],
            "Lcentered-6-8-0":[(0.2, 0.4)],
            "Lcentered-6-5-0":[(0.34, 0.54)],
            "Lcentered-6-6-0":[(0.4, 0.6)],
            "Lcentered-5-6-0":[(0.4, 0.6)],
            "Lcentered-5-8-0":[(0.25, 0.45)],
            "Lcentered-5-5-0":[(0.1, 0.3), (0.39, 0.63)],

            # "Lcentered-4-2-0":[(0.3, 0.54), (0.7, 0.9)],
            "Lcentered-4-2-0":[(0.3, 0.56)], # Secon peak is rare

            "Lcentered-4-3-0":[(0.28, 0.62)],

            "Lcentered-4-4-0":[(0.2, 0.54), (0.55, 0.92)],

            # "Lcentered-4-4-0":[(0.6, 0.88)],
            # "squiggle3-3-1-0":[(0.16, 0.38), (0.44, 0.66), (0.71, 0.91)],
            "squiggle3-3-1-0":[(0.16, 0.38), (0.44, 0.68)],

            "squiggle3-3-2-1":[(0.15, 0.39), (0.42, 0.80)],

            "circle-6-1-0":[(0.08, 0.28), (0.32, 0.56), (0.58, 0.88)],

            "V-2-4-0":[(0.3, 0.5)],

            "arcdeep-4-1-0":[(0.06, 0.29), (0.36, 0.86)],

            "Lzigzag1-3-5-1":[(0.18, 0.38), (0.4, 0.7)],
            "Lzigzag1-4-5-1":[(0.2, 0.36), (0.4, 0.6)],
            "Lzigzag1-4-6-0":[(0.16, 0.36)],
            # "Lzigzag1-4-6-0":[(0.16, 0.36), (0.6, 0.86)],
            "Lzigzag1-3-6-0":[(0.16, 0.36), (0.50, 0.75)],
            "squiggle3-3-2-0":[(0.16, 0.36), (0.52, 0.74)],
            "Lzigzag1-4-5-0":[(0.16, 0.38), (0.44, 0.64)],
            "Lzigzag1-3-5-0":[(0.16, 0.38), (0.40, 0.64)],
            "V-2-1-0":[(0.1, 0.3), (0.36, 0.58)],
            "arcdeep-4-2-0":[(0.42, 0.82)],
            "arcdeep-4-4-0":[(0.2, 0.43)],
        }
    else:
        assert False

    return database_shape_segmentation


def pipeline_wrapper(D, dosave=True):
    """ entire pipeline, going from D to Dsubs and DSsubs, which are "fake"
    datasets holding substrokes where you would expecte strokes.
    PARAMS:
    - dosave, bool, if True, then saved DSsubs and Dsubs.
    """
    from pythonlib.tools.pandastools import slice_by_row_label
    from pythonlib.dataset.dataset_strokes import DatStrokes
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.vectools import bin_angle_by_direction
    from pythonlib.tools.nptools import bin_values
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.pandastools import append_col_with_grp_index

    PLOT = True
    Npts = 70
    refrac_npts = int(0.1*Npts) # within this refrac, takes ony a single trough maximum.
    # assert Npts==50, "not sure if hard coded.."
    THRESH = 0.65 # throws out any strokes with dtw distnace higher than this.
    animal = D.animals(True)[0]
    date = D.dates(True)[0]
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("substrokes_preprocess")

    # Get database
    database_shape_segmentation = database_shape_segmentation_get(animal)

    # First, smooth all data in dataset
    ## EMpticalyl determined to work for these guys, in imprioving hte getting
    # of troughs.
    print("SMOOTHING STROKES, to preprocess")
    D = D.copy()
    if D.animals(True)[0]=="Diego":
        window_time = 0.35
    elif D.animals(True)[0]=="Pancho":
        window_time = 0.25
    else:
        assert False
    list_strokes_filt = D.strokes_smooth_preprocess(window_time=window_time)
    D.Dat["strokes_beh"] = list_strokes_filt

    # Get DS, cleaned
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    DS = preprocess_dataset_to_datstrokes(D, "clean_one_to_one")

    if len(DS.Dat)<0.8 * len(D.Dat):
        print(len(DS.Dat))
        print(len(D.Dat))
        assert False, "weird..., pruned this much?"

    ##### MAIN LOOP
    list_shapes = DS.Dat["shape"].unique().tolist()
    list_gridsize = DS.Dat["gridsize"].unique().tolist()

    MAP_IND_DS_TO_SUBSTROKES = {}
    MAP_IND_DS_TO_INDS_WITHIN_STROKE = {}
    MAP_IND_DS_TO_FRACS_WITHIN_STROKE = {}
    res_by_shape = []
    inds_ds_used = []
    for shape in list_shapes:
        for gridsize in list_gridsize:
            plot_savedir = f"{SAVEDIR}/preprocess_each_shape/{shape}-{gridsize}"
            os.makedirs(plot_savedir, exist_ok=True)
            print("Doign this shape/gridsize: ", plot_savedir, gridsize)

            a = DS.Dat["shape_oriented"]==shape
            b = DS.Dat["gridsize"]==gridsize
            inds_ds_all = DS.Dat[a & b].index.tolist()
            if len(inds_ds_all)>0:
                list_frac_windows = database_shape_segmentation[shape]

                #### FIRST RUN, to prune trials to get better template.
                # inds_ds = DS.find_indices_this_shape(shape)
                n = len(inds_ds_all)
                _, _, _, MASKS = preprocess_segment_from_DS(DS, inds_ds_all, Npts,
                                                                 list_frac_windows, refrac_npts,
                                                                 PLOT=False,
                                                                 MEAN_check_same_direction=False,
                                                                 # MEAN_ver="median",
                                                                 MEAN_ver="mean",
                                                                 THRESH_dtw_dist=THRESH
                                                                 )
                inds_ds_pruned = list(np.array(inds_ds_all)[MASKS["idxs_keep_final_bool"]])
                assert len(inds_ds_all)==n

                #### SECOND RUN, using subset of datapts, to make a good template
                list_inds_trough_in_dat, strokes_warped, list_dists, MASKS = preprocess_segment_from_DS(DS,
                                                                inds_ds_all, Npts,
                                                                 list_frac_windows, refrac_npts,
                                                                 inds_ds_template=inds_ds_pruned,
                                                                 PLOT=PLOT,
                                                                 plot_savedir=plot_savedir,
                                                                 THRESH_dtw_dist=THRESH
                                                                 )
                assert len(list_inds_trough_in_dat)==len(inds_ds_all)

                # Prune data given these
                inds_keep_both = MASKS["idxs_keep_final_bool"]
                inds_ds_keep = np.array(inds_ds_all)[inds_keep_both]
                list_inds_trough_in_dat = np.array(list_inds_trough_in_dat)[inds_keep_both, :]
                # list_dists = np.array(list_dists)[inds_keep_both]

                # Save results by shape
                res_by_shape.append(
                    {
                        "shape":shape,
                        "gridsize":gridsize,
                        "ntot":len(inds_keep_both),
                        "nkeep":sum(inds_keep_both),
                        "nremove_dtw":sum(~MASKS["idxs_keep_dtw_bool"]),
                        "nremove_windows":sum(~MASKS["idxs_keep_wind_bool"]),
                    }
                )

                ### Extract substrokes
                for ind_ds, inds_trough in zip(inds_ds_keep, list_inds_trough_in_dat):

                    # convert these to time in trial
                    frac_within_strok = [_ind/Npts for _ind in inds_trough]
                    strok = DS.Dat.iloc[ind_ds]["Stroke"]()
                    npts_strok = len(strok)
                    inds_within_strok_segment_edges = [int(frac * npts_strok)+1 for frac in frac_within_strok] # add 1, since want onset.
                    inds_within_strok_segment_edges = [0] + inds_within_strok_segment_edges + [npts_strok] # endpoints

                    # Extract the segments
                    # # get the pts that segment into substrokes
                    # pts_segment_edges = [strok[i, :] for i in inds_within_strok_segment_edges]
                    substrokes = []
                    for i1, i2 in zip(inds_within_strok_segment_edges[:-1], inds_within_strok_segment_edges[1:]):
                        substrokes.append(strok[i1:i2, :])

                    # # collect
                    assert ind_ds not in MAP_IND_DS_TO_SUBSTROKES
                    MAP_IND_DS_TO_SUBSTROKES[ind_ds] = substrokes
                    MAP_IND_DS_TO_INDS_WITHIN_STROKE[ind_ds] = inds_within_strok_segment_edges
                    MAP_IND_DS_TO_FRACS_WITHIN_STROKE[ind_ds] = frac_within_strok

                for _ind in inds_ds_all:
                    assert _ind not in inds_ds_used
                inds_ds_used.extend(inds_ds_all)



    ########## PLOTS, EVALUATIONS
    # Plot summary of pruning across shapes
    dfres = pd.DataFrame(res_by_shape)
    path = f"{SAVEDIR}/alignment_results_keeps.csv"
    dfres.to_csv(path)

    inds_ds_used = sorted(inds_ds_used)
    nused = len(inds_ds_used)

    tmp = sorted(list(MAP_IND_DS_TO_SUBSTROKES.keys()))
    print("Got these inds_ds finally:")
    print("n = ", len(tmp))
    print("the indices: ", tmp)
    if len(tmp) < 0.75*len(DS.Dat):
        print(len(DS.Dat))
        assert False, "why threw out so many?"

    if not len(inds_ds_used)==len(DS.Dat):
        print(inds_ds_used)
        print(len(inds_ds_used))
        print(len(DS.Dat))
        print(dfres)
        assert False, "why didn't use almost al of the data?"

    # Plot example strokes, overlaying segmentation (plot random cases for each shape)
    #  This ensures that the extraction maps correctly to
    for shape in list_shapes:
        plot_savedir = f"{SAVEDIR}/sanity_mapping_back_to_DS"
        os.makedirs(plot_savedir, exist_ok=True)

        inds_ds_all = DS.find_indices_this_shape(shape)
        inds_ds_all = [i for i in inds_ds_all if i in list(MAP_IND_DS_TO_INDS_WITHIN_STROKE.keys())]

        for it in range(1):
            fig, axes, _ = DS.plot_multiple(inds_ds_all, SIZE=4, nrand=20)
            # for each find the pt of segmentation
            for ind_ds, ax in zip(inds_ds_all, axes.flatten()):

                inds_within_stroke = MAP_IND_DS_TO_INDS_WITHIN_STROKE[ind_ds]
                substrokes = MAP_IND_DS_TO_SUBSTROKES[ind_ds]

                # convert these to time in trial
                strok = DS.Dat.iloc[ind_ds]["Stroke"]()

                # get the pts
                pts_segment_edges = [strok[i, :] for i in inds_within_stroke[:-1]] # includes endpoints

                for pt in pts_segment_edges:
                    ax.plot(pt[0], pt[1], "or")

                # overlay substrokes
                for sub in substrokes:
                    ax.plot(sub[:,0], sub[:,1], "x", alpha=0.2)
                    ax.plot(sub[0,0], sub[0,1], "ok", alpha=1)

            savefig(fig, f"{plot_savedir}/{shape}-drawings_example_segmentations-iter_{it}.pdf")
        plt.close("all")

    ##### FINALLY: Map back to D (trials).
    # Collect mapping from (tc, strokeidx) --> strokes (i.e., segmentations)
    for k, v in MAP_IND_DS_TO_SUBSTROKES.items():
        assert len(v)>0

    rows_new_dataset = []
    for ind_dat, row in D.Dat.iterrows():
        nstrokes = len(row["strokes_beh"])
        tc = row["trialcode"]

        # Collect data for each orig stroke
        eachstroke_substrokes = []
        eachstroke_ind_ds = []
        eachstroke_indstroke = []
        eachstroke_shape = []
        for ind_strok in range(nstrokes):
            # Find the ind_ds that holds this stroke
            # if (tc, ind_strok) in DS.Dat["trialcode_strokeidx"].tolist():
            ind_ds = DS.dataset_index_by_trialcode_strokeindex(tc, ind_strok, return_none_if_doesnt_exist=True)
            if ind_ds is None:
                # Then doesnt exist in DS - was pruned.
                pass
            elif ind_ds not in MAP_IND_DS_TO_SUBSTROKES:
                # Then was pruend becuase segmentation was bad
                pass
            else:
                # get its substrokes
                shape = DS.Dat.iloc[ind_ds]["shape"]
                eachstroke_substrokes.append(MAP_IND_DS_TO_SUBSTROKES[ind_ds]) # list of np arary
                eachstroke_ind_ds.append(ind_ds)
                eachstroke_shape.append(shape)
                eachstroke_indstroke.append(ind_strok)

        # Flatten to get results for this trialcode, in a flattened list of substrokes
        if len(eachstroke_substrokes)>0:
            substrokes = []
            inds_ds = []
            shapes = []
            index_within_stroke = []
            index_stroke_orig = []
            for _i in range(len(eachstroke_substrokes)):
                subs = eachstroke_substrokes[_i]
                n = len(subs)

                substrokes.extend(subs)
                inds_ds.extend([eachstroke_ind_ds[_i] for _ in range(n)])
                shapes.extend([eachstroke_shape[_i] for _ in range(n)])
                index_within_stroke.extend([idx_within for idx_within in range(n)])
                index_stroke_orig.extend([eachstroke_indstroke[_i] for _ in range(n)])

            rows_new_dataset.append({
                "ind_dat":ind_dat,
                "trialcode":tc,
                "substrokes":substrokes,
                "inds_ds":inds_ds,
                "shape":shapes,
                "stroke_index_orig":index_stroke_orig,
                "index_within_stroke":index_within_stroke
            })

    ##### Construct new dataset
    # Dataframe, one row for each trialcode.
    df_rows_substrokes = pd.DataFrame(rows_new_dataset)
    Dsubs = D.copy()
    Dsubs.Dat = slice_by_row_label(Dsubs.Dat, "trialcode", df_rows_substrokes["trialcode"].tolist(),
                                   assert_exactly_one_each=True)
    assert np.all(Dsubs.Dat["trialcode"]==df_rows_substrokes["trialcode"])
    # Replace the "strokes" column
    Dsubs.Dat["strokes_beh"] = df_rows_substrokes["substrokes"]

    # Clean up the new dataset
    # Dsubs.cleanup_wrapper("no_pruning_strokes")
    Dsubs.cleanup_wrapper("substrokes")
    Dsubs.extract_beh_features()
    # Check that did not throw anything out
    # NOTE, later code will confirm that n strokes for each trial are aligned too.
    assert np.all(Dsubs.Dat["trialcode"] == df_rows_substrokes["trialcode"])

    ########## GENERATE A NEW DS that uses substrokes.
    columns_to_append = {} # one value for each (substroke)
    for column in ["shape", "index_within_stroke", "stroke_index_orig"]:
        columns_to_append[column] = df_rows_substrokes[column].tolist()

    ##### Construct a new DS, making sure to not prune any strokes, and passing in the correct shape labels
    DSsubs = DatStrokes(Dsubs, columns_to_append=columns_to_append)

    #### DEtermine substrokes features.
    # Get new column: conj of shape and index_within_shape
    DSsubs.Dat = append_col_with_grp_index(DSsubs.Dat, ["shape", "index_within_stroke"], "shape_idxwithin", use_strings=False)

    ##### Plot, sanity check, compare to dataset
    features_motor_extract_and_bin(DSsubs, plot_save_dir=SAVEDIR)

    # # Comapres to both Dsubs and D
    # if False:
    #     ind = 301
    #     DSsubs.plot_single_overlay_entire_trial(ind)
    #
    #     print(DSsubs.Dat.loc[ind, ["shape_idxwithin", "stroke_index_orig", "angle"]])
    #
    #     tc = DSsubs.Dat.iloc[ind]["trialcode"]
    #     print(tc)
    #     ind = D.Dat[D.Dat["trialcode"]==tc].index.tolist()[0]
    #     D.plotSingleTrial(ind)
    #
    #
    # nbins = 4
    # for var in ["circularity", "distcum"]:
    #     DSsubs.Dat[f"{var}_binned"] = bin_values(DSsubs.Dat[var].values, nbins=nbins)
    # for var in ["angle"]:
    #     DSsubs.Dat[f"{var}_binned"] = bin_angle_by_direction(DSsubs.Dat[var].values,
    #                                                        num_angle_bins=nbins)
    # # FInal - label each substroke
    # DSsubs.Dat = append_col_with_grp_index(DSsubs.Dat, ["distcum_binned", "angle_binned"], "dist_angle", use_strings=False)
    #
    # # Save n conjunctions
    # grouping_print_n_samples(DSsubs.Dat, ["index_within_stroke", "circularity_binned", "distcum_binned", "angle_binned", "shape"],
    #                          savepath=f"{SAVEDIR}/substroke_features_groupings.txt", save_convert_keys_to_str = False,
    #                          save_as="text", sorted_by_keys=True)
    #
    # # Plot joint distributions of features
    # fig = sns.pairplot(data=DSsubs.Dat, vars=["circularity", "distcum", "displacement", "angle"], hue="shape_idxwithin", height=3.5)
    # savefig(fig, f"{SAVEDIR}/substroke_features-color_shape_idx.pdf")
    # fig = sns.pairplot(data=DSsubs.Dat, vars=["circularity", "distcum", "displacement", "angle"], hue="shape", height=3.5)
    # savefig(fig, f"{SAVEDIR}/substroke_features-color_shape.pdf")
    # fig = sns.pairplot(data=DSsubs.Dat, vars=["circularity", "distcum", "displacement", "angle"], hue="dist_angle", height=3.5)
    # savefig(fig, f"{SAVEDIR}/substroke_features-color_bin.pdf")
    #
    # plt.close("all")


    ### FINALY SANITY CHECK
    for shape in list_shapes:
        plot_savedir = f"{SAVEDIR}/sanity_final"
        os.makedirs(plot_savedir, exist_ok=True)

        inds_ds_all = DSsubs.find_indices_this_shape(shape)

        for it in range(1):
            if len(inds_ds_all)>20:
                inds_ds_this = random.sample(inds_ds_all, 20)
            else:
                inds_ds_this = inds_ds_all
            titles = DSsubs.Dat.loc[inds_ds_this, ["shape_idxwithin", "stroke_index_orig", "dist_angle"]].values.tolist()
            fig, axes, inds_trials_dataset = DSsubs.plot_multiple_overlay_entire_trial(inds_ds_this, titles=titles, title_font_size=6)
            savefig(fig, f"{plot_savedir}/{shape}-title_shidx_stridx_distang-iter_{it}.pdf")
        plt.close("all")

    # SAVE
    from pythonlib.tools.expttools import writeDictToTxt
    params_save = {
        "Npts":Npts,
        "refrac_npts":refrac_npts,
        "THRESH":THRESH,
        "SAVEDIR":SAVEDIR,
        "MAP_IND_DS_TO_SUBSTROKES":MAP_IND_DS_TO_SUBSTROKES,
        "MAP_IND_DS_TO_INDS_WITHIN_STROKE":MAP_IND_DS_TO_INDS_WITHIN_STROKE,
        "MAP_IND_DS_TO_FRACS_WITHIN_STROKE":MAP_IND_DS_TO_FRACS_WITHIN_STROKE,
        "res_by_shape":res_by_shape,
        "inds_ds_used":inds_ds_used
    }
    writeDictToTxt(params_save, f"{SAVEDIR}/params_save.txt")
    # DSsubs.save(SAVEDIR, columns_keep_in_dataset=["trialcode", "strokes_beh"])
    DSsubs.save(SAVEDIR)

    return Dsubs, DSsubs, SAVEDIR


def features_motor_extract_and_bin(DS, plot_save_dir=None):
    """
    GOOD, run this at very end of substrokes getting.
    Clean extraction of all relevant motor features that characterize substrokes,
    and then binning them, and then plotting (many).
    Verified that these lead to very similar substrokes (by eye) across diff shapes, and
    are accurately quantification of them
    :param DSsubs:
    - plot_save_dir, str, path to save figures. This is slow!
    :return:
    - MOdifies DS.Dat
    """
    from pythonlib.drawmodel.features import strokeCircularity, strokeCircularitySigned
    from pythonlib.tools.nptools import bin_values
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.vectools import get_angle, bin_angle_by_direction, cart_to_polar
    from pythonlib.tools.pandastools import grouping_print_n_samples


    # GOOD BINNING
    DS.Dat["circ_signed"] = strokeCircularitySigned(DS.Dat["strok"].tolist(), prune_outer_flanks_frac=0.1)
    DS.Dat["circularity"] = strokeCircularity(DS.Dat["strok"].tolist(), prune_outer_flanks_frac=0.1)

    # BIN ALL VALUES
    nbins = 2
    for var in ["velocity", "distcum"]:
        DS.Dat[f"{var}_binned"] = bin_values(DS.Dat[var].values, nbins=nbins)
    nbins = 3
    for var in ["circ_signed", "circularity"]:
        DS.Dat[f"{var}_binned"] = bin_values(DS.Dat[var].values, nbins=nbins)
    nbins = 8
    for var in ["angle"]:
        DS.Dat[f"{var}_binned"] = bin_angle_by_direction(DS.Dat[var].values, num_angle_bins=nbins)
    # DS.Dat["angle_binned"] = bin_angle_by_direction(DS.Dat["angle"], num_angle_bins=12)

    # Also a coarse angl ebin
    DS.Dat[f"angle_binned_coarse"] = bin_angle_by_direction(DS.Dat[var].values, num_angle_bins=3)

    # CONTEXT DEFINITION
    # Remove all data for which dont have the entire stroke, since then cant look at sequence context
    tolerance = 0.85
    n1 = len(DS.Dat)
    DS.clean_preprocess_data(["dataset_missing_stroke_in_context"])
    n2 = len(DS.Dat)
    if n2/n1 < tolerance:
        print(n1, n2)
        assert False, "why removed so much?"
    DS.context_define_local_context_motif(version="substrokes_angle")

    # GROUP by conjunction bins
    DS.Dat = append_col_with_grp_index(DS.Dat, ["distcum_binned", "angle_binned",
                                                "circ_signed_binned", "velocity_binned"],
                                       "di_an_ci_ve_bin", False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["distcum_binned", "angle_binned"],
                                       "dist_angle", use_strings=False)
    DS.Dat = append_col_with_grp_index(DS.Dat, ["di_an_ci_ve_bin", "CTXT_prev_next"],
                                       "ss_this_ctxt", use_strings=False)


    ##### Remove substrokes that are too short in space or time.
    # DO This at very end, since it would efefct context extraction.
    methods = ["stroke_too_short", "stroke_too_quick"]
    params = {
        "min_stroke_length":35,
        "min_stroke_dur":0.2
    }
    n1 = len(DS.Dat)
    DS.clean_preprocess_data(methods=methods, params=params)
    n2 = len(DS.Dat)
    assert n2/n1>0.75, "why removed so much data?"

    ##################### PLOTS
    if plot_save_dir is not None:
        # plot and save all conjunctions, can take time.
        # FInal - label each substroke

        # Save n conjunctions
        grouping_print_n_samples(DS.Dat, ["index_within_stroke", "distcum_binned", "angle_binned", "circ_signed_binned", "velocity_binned", "shape"],
                                 savepath=f"{plot_save_dir}/substroke_features_groupings.txt", save_convert_keys_to_str = False,
                                 save_as="text", sorted_by_keys=True)

        # Plot joint distributions of features
        # fig = sns.pairplot(data=DS.Dat, vars=["circ_signed_binned", "distcum", "velocity", "angle"], hue="shape_idxwithin", height=3.5)
        # savefig(fig, f"{plot_save_dir}/substroke_features-color_shape_idx.pdf")
        # fig = sns.pairplot(data=DS.Dat, vars=["circ_signed_binned", "distcum", "velocity", "angle"], hue="shape", height=3.5)
        # savefig(fig, f"{plot_save_dir}/substroke_features-color_shape.pdf")
        # fig = sns.pairplot(data=DS.Dat, vars=["circ_signed_binned", "distcum", "velocity", "angle"], hue="dist_angle", height=3.5)
        # savefig(fig, f"{plot_save_dir}/substroke_features-color_bin.pdf")

        # REmove "circ_signed_binned" as it is now a string.
        fig = sns.pairplot(data=DS.Dat, vars=["distcum", "velocity", "angle"], hue="shape_idxwithin", height=3.5)
        savefig(fig, f"{plot_save_dir}/substroke_features-color_shape_idx.pdf")
        fig = sns.pairplot(data=DS.Dat, vars=["distcum", "velocity", "angle"], hue="shape", height=3.5)
        savefig(fig, f"{plot_save_dir}/substroke_features-color_shape.pdf")
        fig = sns.pairplot(data=DS.Dat, vars=["distcum", "velocity", "angle"], hue="dist_angle", height=3.5)
        savefig(fig, f"{plot_save_dir}/substroke_features-color_bin.pdf")

        plt.close("all")

        ## PLOT example drawings, grouped by shape
        DS.plot_multiple_sorted_by_feature_split_by_othervar("di_an_ci_ve_bin", ["shape_idxwithin"],
                                                             plot_save_dir=plot_save_dir, nmin_plot=4,
                                                             only_plot_if_mult_lev_of_feature=True)
        # for sh_idx_within in DS.Dat["shape_idxwithin"].unique().tolist():
        #     inds = DS.Dat[DS.Dat["shape_idxwithin"]==sh_idx_within].index.tolist()
        #     # for feature in ["circ_signed", "velocity", "distcum", "angle"]:
        #     for feature in ["di_an_ci_ve_bin"]:
        #         fig, axes = DS.plot_multiple_sorted_by_feature(inds, feature, overlay_beh_or_task="beh", nplot=15, SIZE=2)
        #         savefig(fig, f"{plot_save_dir}/grp_shape_{sh_idx_within}-sortby_{feature}.pdf")
        #         plt.close("all")

        ## PLOT example drawings, grouped by shape
        DS.plot_multiple_sorted_by_feature_split_by_othervar("shape_idxwithin", ["di_an_ci_ve_bin"],
                                                             plot_save_dir=plot_save_dir, nmin_plot=4,
                                                             only_plot_if_mult_lev_of_feature=True)
        DS.plot_multiple_sorted_by_feature_split_by_othervar("shape_idxwithin", ["di_an_ci_ve_bin", "CTXT_prev_this_next"],
                                                             plot_save_dir=plot_save_dir, nmin_plot=4,
                                                             only_plot_if_mult_lev_of_feature=True)

    # Conjucntions
    # How mnay cases with variation in shapes, conditioned on substroke motor, including context.
    if plot_save_dir is not None:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        for vars_others in [
            ["di_an_ci_ve_bin"],
            ["di_an_ci_ve_bin", "CTXT_prev_this_next"],
            ]:
            var = "shape_idxwithin"
            # vars_others = ["di_an_ci_ve_bin", "CTXT_prev_this_next"]
            extract_with_levels_of_conjunction_vars(DS.Dat, var, vars_others, n_min_across_all_levs_var=5,
                                                    lenient_allow_data_if_has_n_levels=2, prune_levels_with_low_n=True,
                                                    plot_counts_heatmap_savepath=f"{plot_save_dir}/conj-{var}-vs-{'|'.join(vars_others)}-GOOD.pdf",
                                                    PRINT_AND_SAVE_TO=f"{plot_save_dir}/conj-{var}-vs-{'|'.join(vars_others)}-GOOD.txt")


### CLEAN - so only strokes that (i) have segments with endpoints within winod and (ii) DTW distance not too large
def _indices_within_frac_window(list_frac_windows, vals, inds_troughs_template, Npts):
    """ Return inds only those that are within list_frac_windows
    PARAMS:
    - inds_troughs_template, list of ints, each index into stroke taht is Npts long
    - list_frac_windows, list of 2-tuples, each a window into stroke duration,
    like (frac_on, frac_off).
    RETURNS:
    - list of ints, unqiue and sorted, only those inds within inds_troughs_template which fall into at
    least one of the iwindows.
    """

    _inds_tmp = []
    for window in list_frac_windows: # (frac1, frac2)
        # convert frac to inds'
        window_inds = [frac*Npts for frac in window]

        # find troughs within this iwnodw
        inds_this = [i for i in inds_troughs_template if (i>=window_inds[0]) and (i<=window_inds[1])]

        # if multiple, take the one with lowest val
        if len(inds_this)>1:
            # print(inds_this)
            # print(vals[inds_this])
            inds_this = [inds_this[np.argmin(vals[inds_this])]] # list [ind]

        _inds_tmp.extend(inds_this)
    _inds_tmp = sorted(set(_inds_tmp))
    # print("-- after restricting to window ", _inds_tmp)
    return _inds_tmp

def preprocess_segment_from_DS(DS, inds_ds, Npts, list_frac_windows,
                               refrac_npts,
                               inds_ds_template=None,
                               PLOT=False, PLOT_EACH_TRIAL=False,
                               plot_savedir=None,
                               MEAN_check_same_direction=True,
                               MEAN_ver="mean",
                               DEBUG = False,
                               THRESH_dtw_dist=None,

                               ):
    """

    PARAMS:
    - shape, str into DS.Dat, e.g, = "zigzagSq-1-1-1"
    :param D:
    :param DS:
    :param shape:
    :return:

    # for each strok, collect the time
    # TODO:
    # 0. Remove cases with scores that are too high.
    # 1. Inspect extract timings in heatmap below.
    # 2. Impose regularization  not too large jump? --> or remove these trials.
    # 3. Take mean again after warping and then redo.
    # 4. Extract segments.

    """
    from pythonlib.tools.nptools import find_peaks_troughs
    from pythonlib.tools.stroketools import strokesInterpolate2
    from pythonlib.drawmodel.strokedists import distStrokWrapper

    if inds_ds_template is None:
        inds_ds_template = inds_ds

    ############ GET TEMPLATE AND STROKES
    # Get template stroke
    strok_template, strokstacked_template = DS.cluster_compute_mean_stroke(inds_ds_template, Npts=Npts,
                                                                  check_same_direction=MEAN_check_same_direction,
                                                                  ver=MEAN_ver)

    # Get all strokes for this shape, interpolate to have same n pts, and then stack (ndat, NPTS, 3)
    _, strokstacked = DS.cluster_compute_mean_stroke(inds_ds, Npts=Npts,
                                                                  check_same_direction=False,
                                                                  ver=MEAN_ver)

    # First, find the time indices of the troughs and peaks in the mean stroke (template)
    fs = 1/np.mean(np.diff(strok_template[:,2])) # is exact.
    _, strokes_speeds = strokesVelocity([strok_template], fs)
    inds_peaks, inds_troughs_template, fig = find_peaks_troughs(strokes_speeds[0][:,0], PLOT,
                                                           refrac_npts=refrac_npts)
    if PLOT:
        savefig(fig, f"{plot_savedir}/template_peaks_troughs.pdf")

    # Restrict troughs to user-defined windows
    inds_troughs_template_input = inds_troughs_template
    inds_troughs_template = _indices_within_frac_window(list_frac_windows,
                                                        strokes_speeds[0][:,0],
                                                        inds_troughs_template,
                                                        Npts)
    if not len(inds_troughs_template)==len(list_frac_windows):
        print("------------")
        from pythonlib.tools.expttools import makeTimeStamp
        ts = makeTimeStamp()
        print("Troughs before pruning. One of these should fall within each of the windows", [i/Npts for i in inds_troughs_template_input])
        print("Window:", list_frac_windows)
        print("... Other info:", inds_troughs_template_input, inds_troughs_template, Npts)
        print(f"Fig saved to /tmp/{ts}.png")
        print("This many datapt: ", len(inds_ds_template))

        inds_peaks, inds_troughs_template, fig = find_peaks_troughs(strokes_speeds[0][:,0], PLOT=True,
                                                                refrac_npts=refrac_npts)

        fig.savefig(f"/tmp/{ts}.png")

        fig = DS.plot_multiple_speed_and_drawing(inds_ds_template)
        fig.savefig(f"/tmp/{ts}_drawings.png")

        assert False
    print("-- after restricting to window ", inds_troughs_template)

    # Get stacked velocities across all data
    strokes = [s for s in strokstacked]
    strokes_speed = strokesVelocity(strokes, fs)[1]
    strokstacked_speed =  np.stack(strokes_speed) # stakc them

    # Plot summary so far
    if PLOT:
        fig1 = DS.plot_multiple_speed_and_drawing(inds_ds_template)
        fig2 = DS.plot_multiple_speed_and_drawing(inds_ds)

        # Plot troughs
        # Plot
        fig3, ax = plt.subplots(1,1, figsize=(5,5))
        ax.set_title("DATA (and final TROUGHS)")
        ax.set_xlabel("time index")
        _strokes = [s for s in strokstacked]
        _strokes_speed = strokesVelocity(_strokes, fs)[1]
        ts = np.arange(Npts)
        for sp in _strokes_speed:
            ax.plot(ts, sp[:,0], "-", alpha=0.1)
        # Overlay troughts
        for ind in inds_troughs_template:
            ax.axvline(ind)

        if plot_savedir:
            savefig(fig1, f"{plot_savedir}/strokes_overview_template.pdf")
            savefig(fig2, f"{plot_savedir}/strokes_overview_data.pdf")
            savefig(fig3, f"{plot_savedir}/final_troughs.pdf")
        plt.close("all")

    if False:
        # Without warping, try to find troughs from raw data.
        list_inds_troughs = []
        for i in range(len(strokstacked_speed)):
            speeds = strokstacked_speed[i, :, 0]

            from pythonlib.tools.timeseriestools import smoothDat
            speeds = smoothDat(speeds, 4)

            _, inds = find_peaks_troughs(speeds, True, refrac_npts=refrac_npts)
            list_inds_troughs.append(inds)
            print(inds)
            assert False

    ########## ALIGNMENT OF DATA TO TEMPLATE
    # Replace all times with linearly increasing 0, 1, .... This allows referecing by index
    # not by time.
    NPTS = len(strok_template)
    assert NPTS==Npts
    assert strokstacked.shape[1]==NPTS
    strokstacked[:,:,2] = np.arange(NPTS)
    strok_template[:,2] = np.arange(NPTS)
    strokstacked_speed[:, :, 1] = np.arange(NPTS)

    #### For each datapt, align it to the template and return the pts that map to the
    # troughs of speed for the template.
    strokes_warped = []
    # collect the time of troughs
    list_inds_trough_in_dat = []
    list_dists = []
    for ict, strok in enumerate(strokstacked):
        strok = strok.copy()
        # print(ict)
        # Plot, before alignmnet
        # t_template = np.arange(NPTS)
        # t_dat = np.arange(NPTS)
        # fig, ax = plt.subplots()
        # for i in range(2):
        #     ax.plot(t_template, strok_template[:, i], "-o", label=f"template {i}")
        #     ax.plot(t_dat, strok[:, i], "-o", label=f"dat {i}")
        # ax.legend()

        dist, alignment = distStrokWrapper(strok_template, strok, "dtw_vels_2d",
                                           align_to_onset=True, rescale_ver="stretch_to_1_diag",
                                           dtw_return_alignment=True, DEBUG=DEBUG)

        # Plot, after alignment

        # t_dat[align[:,0]] = t_template[align[:,1]]

        # plot to compare
        # fig, ax = plt.subplots()
        # for i in range(2):
        #     ax.plot(t_template, strok_template[:, i], "-o", label=f"template {i}")
        #     ax.plot(t_dat, strok[:, i], "-o", label=f"dat {i}")
        #     # ax.plot(align[:,1], strok_template[:, i], "-o", label=f"template {i}")
        #     # ax.plot(align[:,0], strok[:, i], "-o", label=f"dat {i}")
        # ax.legend()

        # Find the idnex in data at which the troughs for template occur
        inds_trough_in_dat = []
        for idx_in_template in inds_troughs_template:
            # function mapping from index in template to index in stroke
            list_idx_in_data = [a[1] for a in alignment if a[0]==idx_in_template] # since you can have multiple matches
            idx_in_data = int(np.mean(list_idx_in_data))
            # idx = np.argwhere(align[:,1] == ind)[0][0] # IGNORE
            # print(align[idx,0], idx_in_data)
            inds_trough_in_dat.append(idx_in_data)
        list_inds_trough_in_dat.append(inds_trough_in_dat)

        if True:
            # Warp the data, for plotting.
            strok_new = strok.copy()
            t_template = np.arange(NPTS)
            t_dat = np.arange(NPTS)
            align = np.array(alignment)
            t_dat[align[:,1]] = t_template[align[:,0]]

            # Force ends to be attached to endpoints
            t_dat[0] = 0
            t_dat[1] = NPTS-1

            if t_dat[0]==1.:
                print(align)
                dist, alignment = distStrokWrapper(strok_template, strok, "dtw_vels_2d",
                                   align_to_onset=True, rescale_ver="stretch_to_1_diag",
                                   dtw_return_alignment=True, DEBUG=True)
                assert False

            # Warp all the data, then restack and plot --> visualize that they are better aligned
            strok_new[:,2] = t_dat
            strok_new_interp = strokesInterpolate2([strok_new], ["input_times", np.arange(NPTS)])[0]

            strokes_warped.append(strok_new_interp)

        if PLOT_EACH_TRIAL and dist>0.4 and dist<0.48:
            ind = 0
            # show pre and postp
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            for ind, ax in zip([0,1], axes.flatten()):
                ax.set_title(f"dim {ind}")
                ax.plot(strok_template[:,2], strok_template[:, ind], "-o", label="template", alpha=0.3)
                ax.plot(strok[:,2], strok[:, ind], "-o", label="orig", alpha=0.3)
                ax.plot(strok_new[:,2], strok_new[:, ind], "-o", label="warped", alpha=0.3)
                ax.plot(strok_new_interp[:,2], strok_new_interp[:, ind], "-o", label="warped(interp)", alpha=0.3)

                for idx in inds_troughs_template:
                    ax.axvline(strok_template[idx, 2], label="template", color="b", alpha=0.4)
                for idx in inds_trough_in_dat:
                    ax.axvline(strok_template[idx, 2], label="dat", color="r", alpha=0.4)
                ax.legend()

            # plot the strokes
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            DS.plot_single_strok(strok_template, ax=ax, color="k")
            DS.plot_single_strok(strok, ax=ax)
            print("distanc: ", dist)
            assert False

        list_dists.append(dist)

    ###### PRUNE DATA INDICES, to only those that (i) have lowish distance threshold and
    # (ii) have trough times that fall within window

    # 1) Exclude if segments are too off from template
    inds_keep_wind = []
    for i, inds_trough in enumerate(list_inds_trough_in_dat):
        vals = strokstacked[i, :, 0] # speed
        inds_trough_good = _indices_within_frac_window(list_frac_windows, vals, inds_trough, Npts)
        # print(len(inds_trough), len(inds_trough_good))

        if len(inds_trough_good) < len(inds_trough):
            inds_keep_wind.append(False)
        else:
            inds_keep_wind.append(True)
    inds_keep_wind = np.array(inds_keep_wind)

    # Throw out cases
    distances = np.array(list_dists)
    if THRESH_dtw_dist is None:
        # then dont throw out
        THRESH_dtw_dist = np.max(distances)+1
    inds_keep_dtw = distances<THRESH_dtw_dist

    print("Keep (DTW score):", sum(inds_keep_dtw), " / ", len(inds_keep_dtw))
    print("Keep (out of winodw):", sum(inds_keep_wind), " / ", len(inds_keep_wind))

    # Inds keep both
    inds_keep_both = inds_keep_wind & inds_keep_dtw
    print("Keep (overall):", sum(inds_keep_both), " / ", len(inds_keep_both))

    # Prune data given these
    idxs_keep_wind_bool = inds_keep_wind
    idxs_keep_dtw_bool = inds_keep_dtw
    idxs_keep_final_bool = inds_keep_both

    MASKS = {
        "idxs_keep_wind_bool":idxs_keep_wind_bool,
        "idxs_keep_dtw_bool":idxs_keep_dtw_bool,
        "idxs_keep_final_bool":idxs_keep_final_bool}

    if PLOT:
        # Plot cases kept and thrown out
        inds_ds_array = np.array(inds_ds)

        inds_ds_this = list(inds_ds_array[idxs_keep_final_bool])
        fig1 = DS.plot_multiple_speed_and_drawing(inds_ds_this)

        inds_ds_this = list(inds_ds_array[~idxs_keep_wind_bool])
        fig2 = DS.plot_multiple_speed_and_drawing(inds_ds_this)

        inds_ds_this = list(inds_ds_array[~idxs_keep_dtw_bool])
        fig3 = DS.plot_multiple_speed_and_drawing(inds_ds_this)

        if plot_savedir:
            savefig(fig1, f"{plot_savedir}/strokes_overview_afteralign_keeps.pdf")
            savefig(fig2, f"{plot_savedir}/strokes_overview_afteralign_remove_outside_wind.pdf")
            savefig(fig3, f"{plot_savedir}/strokes_overview_afteralign_remove_dtw_dist.pdf")
            plt.close("all")


    if False: # Instead, clean up strokes before passing in..
        ##### Use reuslts to extract substrokes
        fig, axes = plt.subplots(2,1)

        ax = axes.flatten()[0]
        ax.hist(list_dists, bins=40)


        # Throw out cases
        THRESH = 0.5
        distances = np.array(list_dists)
        inds_keep = distances<THRESH

        nbad = sum([d > THRESH for d in list_dists])
        ntot = len(list_dists)
        print("Exclude:", nbad, " / ", ntot)
        strokstacked = strokstacked[inds_keep, ...]
        array_troughs_in_dat = array_troughs_in_dat[inds_keep, ...]

        strokstacked_warped = strokstacked_warped[inds_keep, ...]
        strokstacked_warped_speed = strokstacked_warped_speed[inds_keep, ...]

        strokstacked_speed =  strokstacked_speed[inds_keep, ...]

    if PLOT:
        array_troughs_in_dat = np.array(list_inds_trough_in_dat)
        strokstacked_warped = np.stack(strokes_warped) # (ntrials, ntime, 3)
        _strokes_speed = strokesVelocity([s for s in strokstacked_warped], fs)[1]
        strokstacked_warped_speed =  np.stack(_strokes_speed)

        ###### PLOT
        def plot_heatmaps(STROKSTACKED, STROKSTACKED_SPEED, list_inds_trough_in_dat):
            """ Plot heatmaps showing how activity aligns aross trials."""
            # Plot again
            fig, axes = plt.subplots(1,5, figsize=(20,30))

            for i, ax in enumerate(axes.flatten()):
                if i<3:
                    ax.set_title(f"dim {i}")
                    ax.imshow(STROKSTACKED[:,:,i])
                elif i==3:
                    # speed
                    ax.set_title(f"speed")
                    ax.imshow(STROKSTACKED_SPEED[:,:, 0])
                elif i==4:
                    # speed
                    ax.set_title(f"speed (time)")
                    ax.imshow(STROKSTACKED_SPEED[:,:, 1])
                else:
                    assert False

                # nrows = STROKSTACKED.shape[0]
                for row_from_top, inds_troughs in zip(range(len(STROKSTACKED)), list_inds_trough_in_dat):
                    # row = nrows - row_from_top
                    for j in range(len(inds_troughs)):
                        ax.plot(inds_troughs[j], row_from_top-0.5, "w.", alpha=0.3)
                        # print(row_from_top)
                    # assert False
                        # print(inds_troughs[j])
                # assert False
            return fig

        def plot_timecourse(STROKSTACKED, STROKSTACKED_SPEED):

            times = np.arange(NPTS)

            # Plot again
            fig, axes = plt.subplots(2,2, figsize=(12,6))

            for i, ax in enumerate(axes.flatten()):
                if i<2:
                    ax.set_title(f"dim {i}")
                    ax.plot(times, STROKSTACKED[:,:,i].T, alpha=0.1)
                    ax.plot(times, np.mean(STROKSTACKED[:,:,i].T, axis=1), "-k", alpha=1)
                elif i==2:
                    # speed
                    ax.set_title(f"speed")
                    ax.plot(times, STROKSTACKED_SPEED[:,:,0].T, alpha=0.1)
                    ax.plot(times, np.mean(STROKSTACKED_SPEED[:,:,0].T, axis=1), "-k", alpha=1)
            return fig

        fig1 = plot_heatmaps(strokstacked, strokstacked_speed, list_inds_trough_in_dat)

        fig2 = plot_heatmaps(strokstacked_warped, strokstacked_warped_speed, list_inds_trough_in_dat)

        fig3 = plot_timecourse(strokstacked, strokstacked_speed)

        fig4 = plot_timecourse(strokstacked_warped, strokstacked_warped_speed)

        # plot distrubtion of distances
        fig5, ax = plt.subplots()
        ax.hist(list_dists, bins=40)
        ax.set_title("DTW distances")

        if plot_savedir:
            savefig(fig1, f"{plot_savedir}/final_heatmaps_data.pdf")
            savefig(fig2, f"{plot_savedir}/final_heatmaps_data_warped.pdf")
            savefig(fig3, f"{plot_savedir}/final_timecourses_data.pdf")
            savefig(fig4, f"{plot_savedir}/final_timecourses_data_warped.pdf")
            savefig(fig5, f"{plot_savedir}/final_dtw_distances_hist.pdf")
        plt.close("all")

    return list_inds_trough_in_dat, strokes_warped, list_dists, MASKS

def load_presaved_using_pipeline(D):
    """ Load subtsrokes data that was previously computeda nd saved using
    pipeline_wrapper().
    REturns:
        - DSsubs, only with trialcodes that are in D.Dat
        - Dsubs, only with trialcodes that are in D.Dat
    """
    import pickle as pkl

    # Given a dataset, load its pruned substrokes versions
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("substrokes_preprocess")

    path = f"{SAVEDIR}/DS.pkl"
    with open(path, "rb") as f:
        DSsubs = pkl.load(f)

    # path = f"{SAVEDIR}/dataset_beh.pkl"
    # with open(path, "rb") as f:
    #     Dsubs = pkl.load(f)

    # prune all to match the inputed dataset
    trialcodes_keep = D.Dat["trialcode"].tolist()
    DSsubs.Dat = DSsubs.Dat[DSsubs.Dat["trialcode"].isin(trialcodes_keep)].reset_index(drop=True)

    # Also extract dataset
    Dsubs = DSsubs.Dataset
    Dsubs.Dat = Dsubs.Dat[Dsubs.Dat["trialcode"].isin(trialcodes_keep)].reset_index(drop=True)

    return DSsubs, Dsubs


