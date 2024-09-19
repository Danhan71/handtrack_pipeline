"""
[GOOD]
Label all strokes, usualyl for character tasks, using match between beh and templates (maximum
similarity score), and save as DS.

SAVES ONLY strokes from "character" taskkinds!

"""

from pythonlib.tools.plottools import savefig
from pythonlib.dataset.dataset_analy.primitives import *
from pythonlib.dataset.dataset_preprocess.primitives import *
from pythonlib.dataset.dataset import Dataset, load_dataset_daily_helper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


if __name__=="__main__":
    # Chars
    # animal = "Pancho"
    # date = "230126"

    animal = sys.argv[1]
    date = sys.argv[2]
    rename_shapes_if_cluster_labels_exist = False # IMportant --> if already have clulsters, make sure dont try to load
    # those...

    D = load_dataset_daily_helper(animal, date, rename_shapes_if_cluster_labels_exist=rename_shapes_if_cluster_labels_exist)

    # WHICH_LEVEL = "shapemean"
    WHICH_LEVEL = "trial"
    WHICH_BASIS_SET = animal
    # WHICH_BASIS_SET = "Diego"
    WHICH_TASK_KIND = None # To do for all strokes
    # WHICH_TASK_KIND = "character"
    # WHICH_TASK_KIND = "prims_on_grid"
    SUBSAMPLE = False
    WHICH_FEATURE = "beh_motor_sim" # For clustering/scoring.

    # Keep only characters
    if WHICH_TASK_KIND is not None:
        D.Dat = D.Dat[D.Dat["task_kind"] == WHICH_TASK_KIND].reset_index(drop=True)

    if SUBSAMPLE:
        # OPTIONAL
        D.subsampleTrials(1, 1)

    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from pythonlib.dataset.dataset_strokes import DatStrokes
    # DS = preprocess_dataset_to_datstrokes(D, "clean_chars")
    DS = preprocess_dataset_to_datstrokes(D, "all_no_clean") # get all strokes.
    SDIR = D.make_savedir_for_analysis_figures_BETTER(f"strokes_clustering_similarity/{WHICH_LEVEL}-basis_{WHICH_BASIS_SET}")

    # SDIR = f"{SDIR}/{WHICH_LEVEL}"
    # print(SDIR)
    #%%

    ##### Perform clustering
    ClustDict, ParamsDict, ParamsGeneral, dfdat = DS.features_wrapper_generate_all_features(WHICH_LEVEL,
                                                                                            which_basis_set=WHICH_BASIS_SET)
    plt.close("all")

    # For each trial, extracting clustering score, etc.
    DS.clustergood_assign_data_to_cluster(ClustDict, ParamsDict,
                ParamsGeneral, dfdat,
                which_features = WHICH_FEATURE,
                trial_summary_score_ver="clust_sim_max")



    ##################### PLOTS
    Cl = ClustDict[WHICH_FEATURE]
    list_shape_basis = ParamsDict[WHICH_FEATURE]["list_shape_basis"]
    list_strok_basis = ParamsDict[WHICH_FEATURE]["list_strok_basis"]

    #### QUICK SMALL PLOTS
    # yvar = "clust_sim_max"
    savedir = f"{SDIR}/sim_score_histograms"
    os.makedirs(savedir, exist_ok=True)
    list_yvar = ["sims_max", "sims_concentration", "sims_concentration_v2", "sims_entropy"]
    ncols = 2
    nrows = int(np.ceil(len(list_yvar)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
    for yvar, ax in zip(list_yvar, axes.flatten()):
        vals = Cl.cluster_extract_data("max_sim")[yvar]
        if yvar=="sims_entropy":
            ax.hist(vals, bins=50);
        else:
            ax.hist(vals, bins=np.linspace(0., max(vals), num=50));
    #     ax.hist(DS.Dat[yvar], bins=np.linspace(0., 1., num=50));
        ax.set_xlabel(yvar)
    path = f"{savedir}/hist-{yvar}.pdf"
    savefig(fig, path)
    plt.close("all")

    ### [Good] main plots
    ##### Plots of heatmaps, raw results, for feature spaces and clustering
    DS.clustergood_plot_raw_results(ClustDict, ParamsDict, ParamsGeneral, dfdat, SDIR)
    plt.close("all")

    ##### Figures previously in characters
    # Plot results for characters
    from pythonlib.dataset.dataset_analy.characters import plot_clustering, plot_learning_and_characters, plot_prim_sequences
    plot_clustering(DS, list_strok_basis, list_shape_basis, SDIR)
    plt.close("all")

    # Get indices spanning range of vals
    savedir = f"{SDIR}/example_trials/{WHICH_FEATURE}"
    os.makedirs(savedir, exist_ok=True)

    valname = "clust_sim_max"
    vals = DS.Dat[valname].tolist()
    nplot = 40
    from pythonlib.tools.listtools import random_inds_uniformly_distributed
    indsplot = random_inds_uniformly_distributed(vals, nplot, return_original_values=False)
    for ind in indsplot:
        val = vals[ind]
        prefix = f"{valname}-{val:.2f}"
        DS.clustergood_plot_single_dat(ind, savedir=savedir, prefix=prefix)
        plt.close("all")

    #%%

    #### DIM REDUCTIONS
    savedir = f"{SDIR}/dim_reduction"
    os.makedirs(savedir, exist_ok=True)

    gmm_n_mixtures = range(4, 20)
    things_to_do = ("tsne", "gmm", "gmm_using_tsne")
    perplist = [15]
    gmm_tsne_perp_to_use = 15
    Cl.cluster_compute_all(gmm_n_mixtures=gmm_n_mixtures, perplist=perplist,
                           things_to_do=things_to_do,
                           gmm_tsne_perp_to_use=gmm_tsne_perp_to_use)

    # Note: these corrs means that should (i) get better features and (ii) do PCA first before clsutering.
    simmat = Cl.Xinput
    simmat_rownorm = simmat/np.sum(simmat, axis=1, keepdims=True)
    cc = np.corrcoef(simmat_rownorm.T)
    fig, _, _, _, _ = Cl._plot_heatmap_data(cc, labels_row=Cl.LabelsCols, labels_col=Cl.LabelsCols);
    savefig(fig, f"{savedir}/xcorr_of_simmat.pdf")

    ##### PCA of sim mat

    # Plot basis set
    if False:
        # THis is plotted elsewhere
        shapes = Params["list_shape_basis"]
        # shapes = RES["list_shape_basis"]
        labels_col = Cl.LabelsCols
        assert shapes == labels_col
        strokes = Params["list_strok_basis"]
        fig, axes = DS.plot_multiple_strok(strokes, overlay=False, titles = shapes, ncols = len(shapes))
    #     fig.savefig(f")

    Cl.cluster_pca_plot_all(savedir=savedir)


    for ver in ["gmm", "gmm_using_tsne"]:
        gmm_n_best, list_n, list_crossval, list_bic = Cl.cluster_gmm_extract_best_n(ver=ver)

        fig, axes = plt.subplots(2,2)

        ax = axes.flatten()[0]
        ax.plot(list_n, list_crossval, "-ok")
        ax.set_ylabel("crossval")
        ax.set_xlabel("gmm_n")

        ax = axes.flatten()[1]
        ax.plot(list_n, list_bic, "-ok")
        ax.set_ylabel("bic")
        ax.set_xlabel("gmm_n")

        savefig(fig, f"{savedir}/gmm_scores_using-ver_{ver}.pdf")

    if False:
        # Plot, label by "alignsim" shapes (not useful for characters)
        list_perp = Cl.cluster_tsne_extract_list_perp()
        for perp in list_perp:
            Cl.cluster_plot_scatter("tsne", perp=perp, dims=[0, 1])

    # Plot in tsne space, using gmm labels
    gmm_n_best, list_n, list_crossval, list_bic = Cl.cluster_gmm_extract_best_n()
    for space in ["tsne", "pca"]:
        for label in ["shape", None, "col_max_sim", "gmm", "gmm_using_tsne"]:
            fig, axes = Cl.cluster_plot_scatter(space, label=label, gmm_n=gmm_n_best, perp=15, dims=[0, 1])
            savefig(fig, f"{savedir}/scatter-space_{space}-label_{label}.pdf")



    ### SAVE
    import pickle
    path = f"{SDIR}/DS.pkl"
    with open(path, "wb") as f:
        pickle.dump(DS, f)
    print("Saved to: ", path)

    plt.close("all")


    # Add shape labels
    # list_shapes_final = []
    # for ind in range(len(DS.Dat)):
    #     shape = DS.Dat.iloc[ind]["clust_sim_max_colname"]
    #     list_shapes_final.append(shape)
    DS.Dat["shape_label"] = DS.Dat["clust_sim_max_colname"]

    # initial angle
    DS.features_compute_velocity_binned()

    #### Save this DS, so can later load into constructing Snippets [e.g., RSA plots]
    params = {
        "NOTEBOOK":"230623_STROKES_CLUSTERING_SIMILARITY",
        "WHICH_LEVEL":WHICH_LEVEL,
        "WHICH_BASIS_SET":WHICH_BASIS_SET,
        "ParamsGeneral":ParamsGeneral,
        "ParamsDict":ParamsDict,
        "WHICH_FEATURE":WHICH_FEATURE}


    DIR = f"/gorilla1/analyses/recordings/main/EXPORTED_BEH_DATA/DS/{animal}/{date}"
    os.makedirs(DIR, exist_ok=True)
    DS.export_dat(DIR, params)


