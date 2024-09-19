""" 7/23/23 - Many plots of stroke clusetring and similarity-to-basis-prims, for 
characters. First "complete" set of anslyes.
QUseitons is wherther Panhco is better fit by Pancho prims, and vice versa for Diego
"""

from pythonlib.tools.plottools import savefig
from pythonlib.dataset.dataset_analy.primitives import *
from pythonlib.dataset.dataset_preprocess.primitives import *
from pythonlib.dataset.dataset import Dataset, load_dataset, load_dataset_daily_helper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


assert False, "moved everything to 230623_STROKES_CLUSTERING_SIMILARITY for code devo there... This is still good, but there more up to tdate"

LIST_BASIS_SET = ["Pancho", "Diego"]
# LIST_BASIS_SET = ["Diego"]

# animal = "Pancho"
# date = "230125"
# WHICH_BASIS_SET = "Diego"
# WHICH_LEVEL = "shapemean"
WHICH_LEVEL = "trial"
SUBSAMPLE = False

if False:
    ############## PRIMS_IN_GRID [MOST, DURING REC IN RIG]
    LIST_ANIMAL_DATE = [
        ("Pancho", "230608"), 
        ("Pancho", "230111"), 
        ("Pancho", "220805"), 
        ("Pancho", "220714"), 
        ("Diego", "230625"), 
        ("Diego", "230624"), 
        ("Diego", "230623"), 
        ("Diego", "230622"), 
        ]
    WHICH_TASK_KIND = "prims_on_grid"
else:
    ############## CHARACTERS
    ### ALL
    # LIST_ANIMAL_DATE = [
    #     ("Pancho", "230125"), 
    #     ("Diego", "230424"),
    #     ("Pancho", "230126"), 
    #     ("Pancho", "230127"), 
    #     ("Diego", "230422"),
    #     ("Diego", "230423"),
    #     ("Diego", "230424"),
    #     ]

    ### TWO SUBSETS
    # LIST_ANIMAL_DATE = [
    #     ("Pancho", "230125"), 
    #     ("Diego", "230424")
    #     ]
    # LIST_ANIMAL_DATE = [
    #     ("Pancho", "230126"), 
    #     ("Pancho", "230127"), 
    #     ("Diego", "230422"),
    #     ("Diego", "230423"),
    #     ("Diego", "230424"),
    #     ]
    LIST_ANIMAL_DATE = [
        ("Pancho", "230127"), 
        ("Diego", "230424"),
        ]
    # WHICH_BASIS_SET = "Pancho"
    WHICH_TASK_KIND = "character"

for (animal, date) in LIST_ANIMAL_DATE:
    for WHICH_BASIS_SET in LIST_BASIS_SET:

        D = load_dataset_daily_helper(animal, date)

        SDIR = D.make_savedir_for_analysis_figures_BETTER(f"strokes_clustering_similarity/{WHICH_LEVEL}-basis_{WHICH_BASIS_SET}")

        # Keep only characters
        D.Dat = D.Dat[D.Dat["task_kind"] == WHICH_TASK_KIND].reset_index(drop=True)

        ##### Exclude the bottom nth percentile of trials based on ft_decim
        # ACTUALLY: just ignore, since doesnt seem like strokiness is worse for worse trials
        if False:
            D.score_visual_distance()
            D.Dat["hdoffline"]

            sns.pairplot(data=D.Dat, vars=["strokinessv2", "beh_multiplier", "hausdorff", "ft_decim", "hdoffline"], plot_kws={"alpha":0.3},
                        kind="kde")

            sns.pairplot(data=D.Dat, vars=["strokinessv2", "beh_multiplier", "hausdorff", "ft_decim", "hdoffline"], plot_kws={"alpha":0.3})

            D.plot_trials_after_slicing_within_range_values("hdoffline", 15, 30)

            D.plot_trials_after_slicing_within_range_values("hdoffline", 0, 10)

            D.plot_trials_after_slicing_within_range_values("beh_multiplier", 0.75, 1)


            D.plot_trials_after_slicing_within_range_values("ft_decim", 0.8, 1)
            # D.plot_trials_after_slicing_within_range_values("hausdorff", -1, -0.4)

            D.plot_trials_after_slicing_within_range_values("ft_decim", 0, 0.5)
            # D.plot_trials_after_slicing_within_range_values("hausdorff", -1, -0.4)

        # Dataset, if frac touch too low

        if WHICH_TASK_KIND=="character":
            # characters, be more leneint
            params = ["remove_online_abort"]
            frac_touched_min = None 
        elif WHICH_TASK_KIND=="single_prim":
            params = ["remove_online_abort", "frac_touched_ok"]
            frac_touched_min = 0.6    
        elif WHICH_TASK_KIND=="prims_on_grid":
            params = ["one_to_one_beh_task_strokes"]
            frac_touched_min = None 
        else:
            print(WHICH_TASK_KIND)
            assert False

        D.preprocessGood(params=params, frac_touched_min=frac_touched_min)

        # MEthods for cleaning..
        if WHICH_TASK_KIND=="single_prim" and WHICH_LEVEL == "shapemean":
            methods = ["remove_if_multiple_behstrokes_per_taskstroke", "prune_if_shape_has_low_n_trials"]
            params = {}
            params["prune_if_shape_has_low_n_trials"] = [5]

            DS.clean_preprocess_data(methods, params)

        if SUBSAMPLE:
            # OPTIONAL
            D.subsampleTrials(1, 1)

        #### GEnerate DS
        from pythonlib.dataset.dataset_strokes import DatStrokes
        DS = DatStrokes(D)

        ##### Perform clustering
        ClustDict, ParamsDict, ParamsGeneral, dfdat = DS.features_wrapper_generate_all_features(WHICH_LEVEL, 
                                                                                                which_basis_set=WHICH_BASIS_SET)
        plt.close("all")

        # For each trial, extracting clustering score, etc.
        DS.clustergood_assign_data_to_cluster(ClustDict, ParamsDict, 
                    ParamsGeneral, dfdat,
                    which_features = "beh_motor_sim",
                    trial_summary_score_ver="clust_sim_max")
        plt.close("all")

        ##################### PLOTS
        WHICH_FEATURE = "beh_motor_sim" 
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
        fig, _, _, _ = Cl._plot_heatmap_data(cc, labels_row=Cl.LabelsCols, labels_col=Cl.LabelsCols);
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

