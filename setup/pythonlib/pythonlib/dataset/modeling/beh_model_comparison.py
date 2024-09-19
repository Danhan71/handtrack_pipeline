""" plots/processing for comparing beh and models. 
Currently optimized for draw nn models, but shoud be flexible nough with minor
mods to apply to toher momdels (e.g., bpl)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows, filterPandas, aggregGeneral, summarize_feature
import os
from pythonlib.tools.expttools import makeTimeStamp, findPath
import seaborn as sns


assert False, "use beh_model_holder instead. this Obsolete, there is a nice class"

class ColNames(object):
    """ helper to get column names in D, given sets of model classes and rules
    , and different manipulatios of the scores
    """
    def __init__(self, model_classes, model_rules):
        """
        model_classes, model_rules, Takes cross-product?
        Note: The order of model_rules matters, mod2 minus mod1...
        """
        self.Classes = model_classes
        self.Rules = model_rules
        
        assert isinstance(model_rules, list)
        assert isinstance(model_classes, list)

    def colnames_score(self, mclass=None, mrule=None):    
        if mclass is None:
            return [col for mclass in self.Classes for col in self.colnames_score(mclass)]
        if mrule is None:
            return [self.colnames_score(mclass, mrule) for mrule in self.Rules]
        else:
            return f"behmodpost_{mrule}_{mclass}"
        
    def colnames_minus(self, mclass=None):
        if mclass is None:
            return [self.colnames_minus(mclass) for mclass in self.Classes]
        return f"mod2_minus_mod1_{mclass}"

    def colnames_minus_usingnames(self, mclass=None):
        """ same but uses names of rules instead of just mod2 - mod1
        """
        assert len(self.Rules)==2
        if mclass is None:
            return [self.colnames_minus_usingnames(mclass) for mclass in self.Classes]

        return f"{self.Rules[1]}_minus_{self.Rules[0]}_{mclass}"

        
    def colnames_alignment_trials(self, mclass=None):
        if mclass is None:
            return [self.colnames_alignment_trials(mclass) for mclass in self.Classes]
        return f"alignment_trials_{mclass}"
        
    def colnames_alignment_tasks(self, mclass=None):
        if mclass is None:
            return [self.colnames_alignment_tasks(mclass) for mclass in self.Classes]
        return f"alignment_tasks_{mclass}"
    
    def get_model_rule_other(self, rulethis):
        """ returns list of other modelrules
        in order that was saved
        """
        assert rulethis in self.Rules
        return [r for r in self.Rules if r!=rulethis]
        
    def get_all_cols(self, exclude=None):
        """ return list of col names, excluding if flexible.
        """
        if exclude is None:
            exclude = []
        out = []
        if "score" not in exclude:
            out.extend(self.colnames_score())
        if "minus" not in exclude:
            out.extend(self.colnames_minus())
        if "alignment_trials" not in exclude:
            out.extend(self.colnames_alignment_trials())
        if "alignment_tasks" not in exclude:
            out.extend(self.colnames_alignment_tasks())
        return out
        

def plots_cross_prior_and_model_anynum(BM, monkey_prior_col_name="epoch", monkey_prior_list=None,
    list_classes=None, model_score_name_list =None):
    """
    Summary plot of test dataset against models (scores), when num models is >2, this still works.
    NOTE: modified 12/19/22 to work with beh_model_holder, not with multiple... (which should be 
    changed to be a wrapper for beh_model_holder)
    INPUT:
    - monkey_prior_col_name
    --- e..g, "epoch"
    - GROUPING_LEVELS [aka monkey_prior_list]

    """

    assert(False, "use plot_score_cross_prior_model in beh_model_holder")

    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.plottools import plotScatter45
    ALPHA = 0.2

    ### ALIGNMENT - rank, compute all
    BM.analy_compute_alignment_wrapper()

    # Get Single dataset
    Dat, DatWide, DatFlat, DatThisAgg, DatFlatAgg = BM.extract_concatenated_aggregated_dataset(
        monkey_prior_col_name, monkey_prior_list, list_classes, model_score_name_list)[:5]

    # 1) Plot score fr all combo of dataset and model
    fig = sns.catplot(data=DatFlat, x=monkey_prior_col_name, y="score", hue="model", aspect=3, kind="bar")

    # 2) same, agg over trials
    fig = sns.catplot(data=DatFlatAgg, x=monkey_prior_col_name, y="score", hue="model", aspect=3, kind="bar")


    # if column exists for binary_tuple then plot
    # to create 'binary_rule_tuple' column: run dataset.modeling.discrete.add_binary_rule_tuple_col
    if 'binary_rule_tuple' in Dat.columns:
        fig,ax = plt.subplots(figsize=(8,4))
        sns.histplot(data=Dat,x='epoch',hue='binary_rule_tuple',ax=ax,multiple="dodge",shrink=0.8)
        fig,ax = plt.subplots(figsize=(8,4))
        sns.histplot(data=Dat,x='binary_rule_tuple',hue='epoch',ax=ax,multiple="dodge",shrink=0.8)


    # For each trial, alignment, as rank out of all model scores.
    for colthis in BM.colnames_extract_alignment():
        # colthis = "alignment_rank_chunks" # was this, for gridlinecircle.
        sns.catplot(data=Dat, x="epoch", y=colthis)
        sns.catplot(data=Dat, x="epoch", y=colthis, kind="boxen")
        sns.catplot(data=Dat, x="epoch", y=colthis, kind="swarm")

    ############### STUFF PULLED IN FROM  plots_cross_prior_and_model
    # there is only for 2, so uses mod minus mod. here apply those plots for each pair of models...
    if model_score_name_list is None:
        model_score_name_list = BM.colnames_extract_score()
    BM.plot_score_scatter_compare_models(model_score_name_list[0], model_score_name_list[1])        
    BM.plot_score_scatter_compare_epochs(model_score_name_list[0], epoch1, epoch2)        

    return DatWide, DatFlat, DatThisAgg, DatFlatAgg
    # Plot alignment, but aggregated

def plots_cross_prior_and_model(DatThisTest, monkey_prior_col_name, monkey_prior_list, 
                                model_score_name_list, savedir=None, return_DatThisAggPaired=False):
    """ All plots, where take cross product of prior model (monkey) and model (anythign else)
    Assumes that there are only 2 priors and 2 models, for now.
    NOTES:
    - column_dsets = "monkey_prior" # names the dartasets.
    - column_dsets = "epoch" # names the dartasets.

    """

    assert False, "this is old, only workse for cases with 2 models (uses mod minus mod). All relevant things moved to plots_cross_prior_and_model_anynum"

    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.plottools import plotScatter45
    ALPHA = 0.2

    print(model_score_name_list, monkey_prior_list)
    assert len(model_score_name_list)==2, "not coded for otherwise"
    assert len(monkey_prior_list)==2, "not coded for otherwise"

    
    # --- subtract model1 from model2 score
    DatThisTest["mod2_minus_mod1"] = DatThisTest[model_score_name_list[1]] - DatThisTest[model_score_name_list[0]]
    print("mod2_minus_mod1:", model_score_name_list[1], "-", model_score_name_list[0])
    
    # aggregate over tasks
    # tasklist = set(DatThisTest["character"])
    DatThisAgg = aggregGeneral(DatThisTest, group = ["character", monkey_prior_col_name], 
                               values=[model_score_name_list[0], model_score_name_list[1], "mod2_minus_mod1"])

    # Pivot
    DatThisAggPaired = pivot_table(DatThisAgg, index=["character"], columns = [monkey_prior_col_name], 
                                   values=[model_score_name_list[0], model_score_name_list[1], "mod2_minus_mod1"])
    # cleanup remove any rows with nans
    DatThisAggPaired = DatThisAggPaired.dropna().reset_index(drop=True)


    ################################### ALL TRIALS
    # plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True)
    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisTest[DatThisTest[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # Scatter
        ax = axes.flatten()[i]
        plotScatter45(x1, x2, ax=ax, alpha=ALPHA, means=True)

        ax.set_xlabel(model_score_name_list[0])
        ax.set_ylabel(model_score_name_list[1])
        ax.set_title(lev)
    if savedir:
        fig.savefig(f"{savedir}/all_trials_origscores_scatter.pdf")


    # == plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True, figsize=(10,4))

    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisTest[DatThisTest[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # histograms
        ax = axes.flatten()[i]
        ax.hist(x1, label="model1", density=True, histtype="step")
        ax.hist(x2, label="model2", density=True, histtype="step")

        ax.set_title(lev)
        ax.legend()
    if savedir:
        fig.savefig(f"{savedir}/all_trials_origscores_hist.pdf")


    #  PLOT OVERVIEW, ALL TASKS
    fig = sns.catplot(data=DatThisTest, x=monkey_prior_col_name, y="mod2_minus_mod1", aspect=3)
    plt.axhline(0, color="k", alpha=0.5)
    if savedir:
        fig.savefig(f"{savedir}/all_trials_catplot_summary_mod2minus1.pdf")


    fig = sns.displot(data=DatThisTest, hue=monkey_prior_col_name, x="mod2_minus_mod1", 
                aspect=3, kind="hist", stat="probability", common_norm=False)
    plt.axvline(0, color="r")
    if savedir:
        fig.savefig(f"{savedir}/all_trials_summary_mod2minus1.pdf")



    ########### ONE PLOT PER TASK
    fig = sns.displot(data=DatThisAgg, hue=monkey_prior_col_name, x="mod2_minus_mod1", 
                aspect=2, kind="hist", stat="probability", common_norm=False, bins=20,
               element="bars", fill=True)
    plt.axvline(0, color="r")
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_summary_mod2minus1.pdf")

    # === 
    fig = sns.catplot(data=DatThisAgg, x=monkey_prior_col_name, y="mod2_minus_mod1", aspect=3)
    plt.axhline(0, color="k", alpha=0.5)
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_catplot_summary_mod2minus1.pdf")

    ALPHA = ALPHA*2
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True)
    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisAgg[DatThisAgg[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # Scatter
        ax = axes.flatten()[i]
        plotScatter45(x1, x2, ax=ax, alpha=ALPHA, means=True)

        ax.set_xlabel(model_score_name_list[0])
        ax.set_ylabel(model_score_name_list[1])
        ax.set_title(lev)
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_origscores_scatter.pdf")

    # == plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True, figsize=(10,4))

    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisAgg[DatThisAgg[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # histograms
        ax = axes.flatten()[i]
        ax.hist(x1, label="model1", density=True, histtype="step")
        ax.hist(x2, label="model2", density=True, histtype="step")

        ax.set_title(lev)
        ax.legend()
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_origscores_hist.pdf")

    ###############################################
    col1 = monkey_prior_list[0]
    col2 = monkey_prior_list[1]
    dfthis = DatThisAggPaired["mod2_minus_mod1"]
    x = dfthis[col1]
    y = dfthis[col2]
    tasknames = [v.split("-")[0] for v in DatThisAggPaired["character"].values]

    fig, ax = plt.subplots()
    plotScatter45(x, y, ax, dotted_lines="plus", means=True, labels = None)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("mod2 - mod1")
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_scatter_mod2minus1.pdf")

    fig, ax = plt.subplots(figsize=(15,15))
    plotScatter45(x, y, ax, dotted_lines="plus", means=False, labels = tasknames)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("mod2 - mod1")
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_scatter_mod2minus1_largetext.pdf")

    if return_DatThisAggPaired:
        return DatThisAggPaired


def plots_cross_prior_and_model_combined(D, GROUPING, GROUPING_LEVELS, list_mclass, list_mrule,
        savedir_this = "/tmp"):
    """ combines plots for positive control (monkey-monkey) and model
    model-monkey - e..g, a single bar plot comparing alignemnt. And any number 
    of alternative model classes (in progress)
    
    NOTES:
    - initallly developed in ("See notebook drawnn --> test_on_monkey_tasks...")
    - STILL MESSY CODE.
    """

    assert False, 'Obsolete --> all incorporated in to plots_cross_prior_and_model_anynum'
    # Note: even before this, it was mostly redundant with plots_cross_prior_and_model

    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.dataset.analy import score_alignment
    import pandas as pd
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import plotScatter45


    # Col Names getter
    CN = ColNames(list_mclass, list_mrule)


    # 1) For each model class, compute alignment (at level of trials)
    COL_NAMES_MINUS = []
    COL_NAMES_ALIGNMENT  = []
    for mclass in list_mclass:
    #     mclass = "planner"
        SCORE_COL_NAMES_THIS = CN.colnames_score(mclass)
        # SCORE_COL_NAMES_THIS = [f"behmodpost_{c}_{mclass}" for c in MODEL_NAMES]
        colname_m2_m1, newcol = score_alignment(D, GROUPING, GROUPING_LEVELS, SCORE_COL_NAMES_THIS, suffix=mclass)
        COL_NAMES_MINUS.append(colname_m2_m1)
        COL_NAMES_ALIGNMENT.append(newcol)



    # DatThis = D.Dat
    DatThis = D.filterPandas({"monkey_train_or_test":["test"]}, "dataframe")
    COLS_KEEP = [col for col in CN.get_all_cols() if col in DatThis.columns]
    print(COLS_KEEP)
    DatThisAggPairedAll = pivot_table(DatThis, index=["character"], columns = GROUPING,
                                   values=COLS_KEEP)
    DatThisAggPairedAll = DatThisAggPairedAll.dropna().reset_index(drop=True)

    def _get_samemod_minus_diffmod(DatThisAggPairedAll, datagrp, modelclass):
        """ return score for same model minus for diff model, assuming
        datagrp is the grp for data, and there is a corresponding model
        OUT:
        - np array, same len as DatThisAggPairedAll
        e.g.;
            datagrp = "bent"
            modelclass = "mkvsmk"
        """
        rulesame = datagrp
        tmp = CN.get_model_rule_other(rulesame)
        assert len(tmp)<2, "hvae not coded for if multiple rules for this model"
        rulediff = tmp[0]

        scoresame = CN.colnames_score(modelclass, rulesame)
        scorediff = CN.colnames_score(modelclass, rulediff)
        
        print(rulesame, " : (same)", scoresame, " - (diff)",  scorediff)
        
        vals = (DatThisAggPairedAll[scoresame][datagrp] - DatThisAggPairedAll[scorediff][datagrp]).values
        return vals

    # compute alignment for each model class
    for modelclass in list_mclass:
        assert len(GROUPING_LEVELS)==2, "not done"
        list_vals = []
        for datagrp in GROUPING_LEVELS:
            vals = _get_samemod_minus_diffmod(DatThisAggPairedAll, datagrp, modelclass)
            list_vals.append(vals)

        colname_alignment = CN.colnames_alignment_tasks(modelclass)
        DatThisAggPairedAll[colname_alignment] = np.mean(np.stack(list_vals, axis=1), axis=1)
        print("** added this alignment column: ", colname_alignment)


    # ===== BAR PLOTS, SHOWING ALL RAW SCORES
    DatThisAggPairedAllFlat = pd.melt(DatThisAggPairedAll, id_vars = ["character"])
    DatThisAggPairedAllFlat = DatThisAggPairedAllFlat.rename(columns={None:"model"})
    # DatThisAggPairedAllFlat["model"].value_counts()

    # PLOT
    col_scores = CN.colnames_score()
    df = DatThisAggPairedAllFlat[DatThisAggPairedAllFlat["model"].isin(col_scores)]
    fig = sns.catplot(data=df, x=GROUPING, y="value", hue="model", kind="bar", ci=68, aspect=1.5)
    #     fig = sns.catplot(data=df, x="monkey_prior", y="value", hue="model", kind="point", ci=68, aspect=1.5)
    # sns.catplot(data=df, x="model", y="value", hue="monkey_prior", kind="bar", ci=68)
    fig.savefig(f"{savedir_this}/pairedtasks_rawscores_allmodelspriors_bars.pdf")

    col_minus = CN.colnames_minus()
    df = DatThisAggPairedAllFlat[DatThisAggPairedAllFlat["model"].isin(col_minus)]
    fig = sns.catplot(data=df, x=GROUPING, y="value", hue="model", kind="bar", ci=68)
    rotateLabel(fig)
    fig.savefig(f"{savedir_this}/pairedtasks_modeldiffscores_allmodelspriors_1.pdf")
    fig = sns.catplot(data=df, x="model", y="value", hue=GROUPING, kind="bar", ci=68)
    rotateLabel(fig)
    fig.savefig(f"{savedir_this}/pairedtasks_modeldiffscores_allmodelspriors_2.pdf")


    # compute alignement score.
    colnames = CN.colnames_alignment_tasks()
    df = DatThisAggPairedAllFlat[DatThisAggPairedAllFlat["model"].isin(colnames)]
    fig = sns.catplot(data=df, x="model", y="value", kind="bar", ci=68)
    rotateLabel(fig)
    fig.savefig(f"{savedir_this}/pairedtasks_alignmentscore_1.pdf")
    fig = sns.catplot(data=df, x="model", y="value")
    plt.axhline(0, color="k", alpha=0.5)
    rotateLabel(fig)
    fig.savefig(f"{savedir_this}/pairedtasks_alignmentscore_2.pdf")
    fig = sns.catplot(data=df, x="model", y="value", kind="point", ci=68)
    plt.axhline(0, color="k", alpha=0.5)
    rotateLabel(fig)
    fig.savefig(f"{savedir_this}/pairedtasks_alignmentscore_3.pdf")

    # SCATTER PLOT
    colnames = CN.colnames_minus()
    SHARE_AX = False

    col1 = GROUPING_LEVELS[0]
    col2 = GROUPING_LEVELS[1]
    #     df = DatThisAggPairedAll[value]
    tasknames = [v.split("-")[0] for v in DatThisAggPairedAll["character"].values]
    ncol = 3
    nrow = int(np.ceil(len(colnames)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*4,nrow*4), sharex=SHARE_AX, sharey=SHARE_AX)
    for value, ax in zip(colnames, axes.flatten()):
    #         value = "mod2_minus_mod1_monkeymodel"
        dfthis = DatThisAggPairedAll[value]
        x = dfthis[col1]
        y = dfthis[col2]

        plotScatter45(x, y, ax, dotted_lines="plus", means=True, labels = None)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(value)
        
    fig.savefig(f"{savedir_this}/pairedtasks_modeldiffscores_scatters.pdf")


