""" Daily plots for single prim expts
"""

from pythonlib.tools.plottools import savefig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.snstools import rotateLabel
import os
import seaborn as sns

def preprocess_dataset(D, PLOT=True):
    """
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes, preprocess_dataset_to_datstrokes

    # D = D.copy()
    # D.Dat = D.Dat[D.Dat["task_kind"] == "prims_single"].reset_index(drop=True)

    assert "seqc_0_shape" in D.Dat.columns

    SAVEDIR = D.make_savedir_for_analysis_figures("singleprims")
    D.taskclass_gridsize_assign_column()
    D.seqcontext_preprocess()
    D.sketchpad_fixation_append_as_string()

    DS = preprocess_dataset_to_datstrokes(D, "singleprim")
    DS.dataset_append_column("epoch") 
    DS.dataset_append_column("origin_string") 

    if PLOT:
        ### PLOt drawings of all conjucntions
        plot_drawings_grid_conjunctions(DS, SAVEDIR)
        # DS.plotshape_multshapes_egstrokes_grouped_in_subplots(key_to_extract_stroke_variations_in_single_subplot="gridsize", n_examples=1);

        # Plot (row, col, location in suplot) = (size, shape, loc)
        sdir = f"{SAVEDIR}/shape_size_loc_drawings"
        os.makedirs(sdir, exist_ok=True)
        niter = 5
        for i in range(niter):
            fig_beh, fig_task = DS.plotshape_row_col_size_loc()
            savefig(fig_beh, f"{sdir}/iter_{i}-beh.pdf")      
            savefig(fig_task, f"{sdir}/iter_{i}-task.pdf")      
            plt.close("all")
            
        ### PRint and plot conjucntions for neural analy
        from neuralmonkey.metadat.analy.anova_params import conjunctions_print_plot_all
        # which_level="trial"
        # ANALY_VER = "seqcontext"
        # animal = D.animals()[0]
        conjunctions_print_plot_all([D], SAVEDIR, ANALY_VER="singleprim")
        plt.close("all")
        assert "seqc_0_shape" in D.Dat.columns

        # PLOT COnjunctions... (this fails if incluyde in conjunctions_print_plot_all, becuase it doenst have loc on clust.)
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        sdir = f"{SAVEDIR}/conjunctions_2"
        os.makedirs(sdir, exist_ok=True)
        for task_kind in ["prims_single", "prims_on_grid"]:
            dfthis = DS.Dat[DS.Dat["task_kind"]==task_kind].reset_index(drop=True)

            if len(dfthis)>5:
                # If variation in fixation onset location and loc_on_clust(align to touch onset)
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1=("shape", "gridloc"), var2="origin_string", 
                                                                  vars_others=["stroke_index"])
                path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc_origin_string-task_kind_{task_kind}.pdf"
                savefig(fig, path)

                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", 
                                                                  vars_others=["gridsize"])
                path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc_gridsize-task_kind_{task_kind}.pdf"
                savefig(fig, path)

                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1=("shape", "loc_on_clust"), var2="origin_string", 
                                                                  vars_others=["stroke_index"])
                path = f"{sdir}/STROKELEVEL-conjunctions_shape_loc_on_clust_origin_string-task_kind_{task_kind}.pdf"
                savefig(fig, path)

                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="loc_on_clust", var2="shape", 
                                                                  vars_others=["stroke_index", "origin_string"])
                path = f"{sdir}/STROKELEVEL-conjunctions_loc_on_clust-shape-origin_string-task_kind_{task_kind}.pdf"
                savefig(fig, path)
        plt.close("all")

        ### Plot scores (beh)
        sdir = f"{SAVEDIR}/beh_eval_scores"
        os.makedirs(sdir, exist_ok=True)

        list_block = D.Dat["block"].unique().tolist()
        for bk in list_block:
            df = D.Dat[(D.Dat["block"]==bk)]

            from pythonlib.tools.pandastools import stringify_values
            df = stringify_values(df)

            if len(df)>15:
                assert "seqc_0_shape" in df.columns

                for yval in ["rew_total", "beh_multiplier", "aborted"]:

                    if yval=="aborted":
                        hue = None
                    else:
                        hue = "aborted"
                    print("Plotting for (block, yval): ", bk, yval)

                    for x in ["gridsize", "seqc_0_shape", "seqc_0_loc"]:
                        fig = sns.catplot(data=df, x=x, y=yval, hue=hue, row="session", aspect=3, height=2, jitter=True, alpha=0.2)
                        rotateLabel(fig)
                        savefig(fig, f"{sdir}/block_{bk}-x={x}-yval={yval}-1.pdf")

                        fig = sns.catplot(data=df, x=x, y=yval, hue=hue, row="session", aspect=3, height=2, kind="bar")
                        rotateLabel(fig)
                        savefig(fig, f"{sdir}/block_{bk}-x={x}-yval={yval}-2.pdf")

                    # fig = sns.catplot(data=df, x="gridsize", y=yval, hue="aborted", row="session", aspect=3, height=2, kind="bar")
                    # rotateLabel(fig)
                    # savefig(fig, f"{sdir}/block_{bk}-yval_{yval}-2.pdf")      

                    # fig = sns.catplot(data=df, x="seqc_0_shape", y=yval, hue="gridsize", row="session", aspect=3, height=2, kind="bar")
                    # rotateLabel(fig)
                    # savefig(fig, f"{sdir}/block_{bk}-shape_size-yval_{yval}-2.pdf")      

                    try:
                        fig = sns.catplot(data=df, x="seqc_0_shape", y=yval, hue="seqc_0_loc", row="session", aspect=3, height=2, kind="bar")
                        rotateLabel(fig)
                        savefig(fig, f"{sdir}/block_{bk}-shape_loc-yval={yval}.pdf")      
                    except Exception as err:
                        # Hack
                        pass

                # fig = sns.catplot(data=df, x="gridsize", y="aborted", row="session", aspect=3, height=2, kind="bar")
                # rotateLabel(fig)
                # savefig(fig, f"{sdir}/block_{bk}-gridsize_aborted.pdf")      
                    try:
                        fig = sns.catplot(data=df, x="seqc_0_shape", y=yval, hue="gridsize", row="session", 
                                        aspect=3, height=2, kind="bar")
                        rotateLabel(fig)
                        savefig(fig, f"{sdir}/block_{bk}-shape_size-yval={yval}.pdf")      
                    except Exception as err:
                        pass

                    try:
                        fig = sns.catplot(data=df, x="seqc_0_loc", y=yval, hue="gridsize", row="session", 
                                        aspect=3, height=2, kind="bar")
                        rotateLabel(fig)
                        savefig(fig, f"{sdir}/block_{bk}-loc_size-yval={yval}.pdf")      
                    except Exception as err:
                        pass
                
                # try:
                #     fig = sns.catplot(data=df, x="seqc_0_shape", y="aborted", hue="seqc_0_loc", row="session", 
                #                     aspect=3, height=2, kind="bar")
                #     rotateLabel(fig)
                #     savefig(fig, f"{sdir}/block_{bk}-shape_loc_aborted.pdf")      
                # except Exception as err:
                #     # Hack
                #     pass
                
                    plt.close("all")

    return DS, SAVEDIR


def plot_drawings_grid_conjunctions(DS, SAVEDIR):
    """ Plot drawings in grid.

    :param DS: _description_
    :param SAVEDIR: _description_
    """

    # pull out strokes into a column called "strokes"
    list_stroke = []
    for ind in range(len(DS.Dat)):
        list_stroke.append([DS.Dat.iloc[ind]["Stroke"]()])
    DS.Dat["strokes_beh"] = list_stroke

    ### Plot grid showing example conjunctions of size, shape, and location
    sdir = f"{SAVEDIR}/drawings_grid_conjunctions"
    os.makedirs(sdir, exist_ok=True)
    niter = 3

    # SEPARATE for each epoch
    DS.dataset_append_column("epoch") 
    list_epoch = DS.Dat["epoch"].unique().tolist()

    # Separate for each fixation onset location
    niter = 3
    list_origin_string = DS.Dat["origin_string"].unique().tolist()
    for ep in list_epoch:
        for origin_string in list_origin_string:
            ds = DS.copy()
            ds.Dat = ds.Dat[
                (ds.Dat["epoch"]==ep) & (ds.Dat["origin_string"]==origin_string)
                ].reset_index(drop=True)

            if len(ds.Dat)>8:
                sdir = f"{SAVEDIR}/drawings_grid_conjunctions/epoch={ep}--origin_string={origin_string}"
                os.makedirs(sdir, exist_ok=True)

                for i in range(niter):

                    print("singleprims.plot_drawings_grid_conjunctions()", ep, i)

                    fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="shape", n_examples_per_sublot=3)
                    path = f"{sdir}/gridsize-vs-shape-{i}.pdf"
                    savefig(fig, path)



                    fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridloc", colvar="shape", n_examples_per_sublot=3)
                    path = f"{sdir}/gridloc-vs-shape-{i}.pdf"
                    savefig(fig, path)



                    fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="gridloc", n_examples_per_sublot=3)
                    path = f"{sdir}/gridsize-vs-gridloc-{i}.pdf"
                    savefig(fig, path)


                    if "loc_on_clust" in DS.Dat.columns:
                        fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="loc_on_clust", colvar="shape", n_examples_per_sublot=3)
                        path = f"{sdir}/loc_on_clust-vs-shape-{i}.pdf"
                        savefig(fig, path)

                        
                    if i==0:
                        fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="shape", n_examples_per_sublot=1, plot_task=True)
                        path = f"{sdir}/gridsize-vs-shape-{i}-TASK.pdf"
                        savefig(fig, path)

                        fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridloc", colvar="shape", n_examples_per_sublot=1, plot_task=True)
                        path = f"{sdir}/gridloc-vs-shape-{i}-TASK.pdf"
                        savefig(fig, path)

                        fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="gridloc", n_examples_per_sublot=1, plot_task=True)
                        path = f"{sdir}/gridsize-vs-gridloc-{i}-TASK.pdf"
                        savefig(fig, path)

                        if "loc_on_clust" in DS.Dat.columns:
                            fig, _ = ds.plotshape_row_col_vs_othervar(rowvar="loc_on_clust", colvar="shape", n_examples_per_sublot=1, plot_task=True)
                            path = f"{sdir}/loc_on_clust-vs-shape-{i}-TASK.pdf"
                            savefig(fig, path)

                    plt.close("all")    

                    # Also this, beucase it plots ALL cases, wheras above has max.
                    figholder = ds.plotshape_multshapes_egstrokes_grouped_in_subplots(key_subplots="shape", 
                                                                          key_to_extract_stroke_variations_in_single_subplot="gridloc",
                                                                          n_examples=3)
                    for j, (fig, _) in enumerate(figholder):
                        path = f"{sdir}/shapes_ALL-vs-gridloc-sub{j}-iter{i}.pdf"
                        savefig(fig, path)

                        if i==0:
                            figholder = ds.plotshape_multshapes_egstrokes_grouped_in_subplots(key_subplots="shape", 
                                                                                key_to_extract_stroke_variations_in_single_subplot="gridloc",
                                                                                n_examples=1, ver_behtask="task")

                    plt.close("all")    

    sdir = f"{SAVEDIR}/drawings_grid_conjunctions/all_epoch-origin_string"
    os.makedirs(sdir, exist_ok=True)
    if len(list_origin_string)>0:
        for i in range(niter):
            fig, _ = DS.plotshape_row_col_vs_othervar(rowvar="loc_on_clust", colvar="origin_string", n_examples_per_sublot=5)
            path = f"{sdir}/loc_on_clust-vs-origin_string-{i}.pdf"
            savefig(fig, path)