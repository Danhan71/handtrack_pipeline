""" To study learning of rules/grammars.
Here assumes there is a single ground-truth sequence for each grammar, which is
saved in the ObjectClass (matlab task definition). Does not deal with model-based
analysis, e.g., parsing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel
from pythonlib.tools.plottools import savefig
import pandas as pd
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from matplotlib import rcParams
from .learning import print_useful_things, plot_performance_all, plot_performance_timecourse, plot_performance_static_summary, plot_counts_heatmap, plot_performance_each_char, plot_performance_trial_by_trial
from .learning import preprocess_dataset as learn_preprocess
from pythonlib.tools.exceptions import NotEnoughDataException


rcParams.update({'figure.autolayout': True})

## OLD, before changed it to make sure it only works with  matlab rules (not new parses)
def preprocess_dataset_recomputeparses(D, DEBUG=False,
                                       exclude_because_online_abort=False,
                                       ONLY_ACTUAL_RULE=True,
                                       replace_agent_with_epoch=True,
                                       include_alternative_rules=False):
    """ Preprocess Dataset, extracting score by looking at parsese of rules for each epoch,
    and asking if beh is compatible with any of them.
    NOTE: dataset length will be multiplied by however many rules there are...
    PARAMS:
    - include_alternative_rules, bool, if True, then includes all rules related to the rules in this experiment,
    which can be time consuming and also fail if tasks are incompaible with rules which assert that tasks are...
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_long
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto
    
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # get taskfeatures (e.g., repeat circle, separated by line)
    D.taskfeatures_category_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    # - get rules autoamticlaly.
    if include_alternative_rules:
        # Then include alternative hypotheses. This can lead to many rules. Can also fail if it is a rule that is incompatibvle
        # with data (e.g., sequence of shapes, but this sequence doesnt include all of the sahpes in beh)
        list_rules = rules_related_rulestrings_extract_auto(D)
    else:
        list_rules = D.grammarparses_rulestrings_exist_in_dataset()
    bm = generate_scored_beh_model_data_long(D, list_rules = list_rules, DEBUG=DEBUG,
                                             ONLY_ACTUAL_RULE=ONLY_ACTUAL_RULE)

    if ONLY_ACTUAL_RULE:
        # Then confirm that max one trial per trialcode.
        from pythonlib.tools.pandastools import grouping_get_inner_items
        groupdict = grouping_get_inner_items(bm.DatLong, "trialcode")
        if max([len(x) for x in groupdict.values()])>1:
            print(bm.DatLong["trialcode"].value_counts())
            print(groupdict)
            assert False, "prob multiple agents per trialcode?"

    if replace_agent_with_epoch:
        # This so that plots use epoch, not agent (which is rulestring).
        # This is the standard if do matlab version, but not for parses version
        bm.DatLong["agent_rule_orig"] = bm.DatLong["agent_rule"]
        bm.DatLong["agent_rule"] = bm.DatLong["epoch"]
        bm._initialize_agent()

    if exclude_because_online_abort:
        # remove the rows from bm that have good sequence, but online abort.
        bm.DatLong = bm.DatLong[~bm.DatLong["exclude_because_online_abort"]].reset_index(drop=True)

    return bm

def preprocess_dataset_matlabrule(D, exclude_because_online_abort=False):
    """ Preprocess Dataset using matlab rules (NOT all parses)
    Each trial is success/failure based on ObjectClass
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_matlabrule
        
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # get taskfeatures (e.g., repeat circle, separated by line)
    D.taskfeatures_category_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    bm = generate_scored_beh_model_data_matlabrule(D)

    if exclude_because_online_abort:
        # remove the rows from bm that have good sequence, but online abort.
        bm.DatLong = bm.DatLong[~bm.DatLong["exclude_because_online_abort"]].reset_index(drop=True)

    return bm

def pipeline_generate_and_plot_all(D,
                                   # which_rules="matlab", # 11/28/23 - beause parses are required for somet hings, like stepwise.py
                                   which_rules="recompute_parses",
                reset_grammar_dat=False, doplots=True, remove_repeated_trials=True,
                save_suffix=None, run_inner_loop_only=False):
    """ Entire pipeline to extract data and plot, for 
    a single dataset
    PARAMS:
    - which_rules, str, either to use ObjectClass matlab rule, or to regenreate
    parsesa nd ask if beh is compativle iwth any of the "same-rule" parses.
    RETURNS:
    - bmh, with removed trials that are "exclude_because_online_abort"
    - SDIR
    OR (None, None) if not enough data 
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto

    # Remove baselione
    D.preprocessGood(params=["remove_baseline"])

    # Get shape code seuqence
    D.sequence_tasksequencer_shapeseq_assign()

    ################# AD HOC SPLITTING OF DATASETS
    # if D.animals()==["Diego"] and D.Dat["date"].unique().tolist()==['231025'] and not run_inner_loop_only:
    if False and not run_inner_loop_only:
        # Then split sess
        list_sess = sorted(D.Dat["session"].unique().tolist())
        for sess in list_sess:
            Dcopy = D.copy()
            Dcopy.Dat = Dcopy.Dat[Dcopy.Dat["session"]==sess].reset_index(drop=True)
            bmh, SDIR = pipeline_generate_and_plot_all(Dcopy, which_rules, 
                reset_grammar_dat, doplots, remove_repeated_trials, 
                save_suffix=f"sess_{sess}", run_inner_loop_only=True)
    else:
        ################## Create save directiory
        SDIR = D.make_savedir_for_analysis_figures("grammar")
        if save_suffix is not None:
            SDIR = f"{SDIR}/{save_suffix}"

        savedir= f"{SDIR}/summary"
        os.makedirs(savedir, exist_ok=True) 

        if reset_grammar_dat:
            D.GrammarDict = {}

        # 1) Get learning metaparams
        list_blocksets_with_contiguous_probes = learn_preprocess(D, remove_repeated_trials=remove_repeated_trials)

        # Epochsets
        # print(D.Dat)
        # print(len(D.Dat))
        # print(D.Dat["epoch"])
        # assert False
        D.sequence_strokes_compute_01_sameness_status() # strokes01_sameness

        # grammar_recompute_parses = False # just use the matlab ground truth
        if which_rules=="matlab":
            # use the ground truth objectclass
            bmh  = preprocess_dataset_matlabrule(D)
        elif which_rules=="recompute_parses":
            print("******** len Dat:", len(D.Dat))
            bmh  = preprocess_dataset_recomputeparses(D, ONLY_ACTUAL_RULE=True)
            # assert False, "aggregate bmh.DatLong so that there is only one ind per trialcode. this should work since success_binary_quick should be identical for all instance for a given trialcode. confirm this"
            # Problem sovled with ONLY_ACTUAL_RULE
        else:
            print(which_rules)
            assert False

        if len(D.Dat)==0:
            print("NO DATA!!")
            return None, None

        if doplots:
            ## STEPWISE action plots (e..g, classify seuqence errors)
            plot_stepwise_actions(D)
            # from pythonlib.grammar.stepwise import preprocess_plot_actions
            # preprocess_plot_actions(D)

            ####### 1) COmpare beh to all hypotheses (rules, discrete)
            # Also make plots for rule-based analysis
            savedir= f"{SDIR}/discrete_rules"
            os.makedirs(savedir, exist_ok=True) 

            # combine in single plot (all taskgroups)
            sdir = f"{savedir}/score_epoch_x_rule_splitby"
            os.makedirs(sdir, exist_ok=True)

            LIST_SPLIT_BY = ["taskconfig_shp_code", "taskgroup", "probe", "strokes01_sameness"]
            if "epochset" in bmh.DatLong.columns:
                LIST_SPLIT_BY.append("epochset")
            if "taskfeat_cat" in bmh.DatLong.columns:
                LIST_SPLIT_BY.append("taskfeat_cat")          
          
            # Use only no-sup data for these
            # dfthis = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
            # dfthis = dfthis[dfthis["exclude_because_online_abort"]==False]
            bmh.DatLong = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
            bmh.DatLong = bmh.DatLong[bmh.DatLong["exclude_because_online_abort"]==False].reset_index(drop=True)

            for split_by in LIST_SPLIT_BY:
                
                try:
                    # Old plots
                    bmh.plot_score_cross_prior_model_splitby(df=bmh.DatLong, split_by=split_by,
                        sdir=sdir, suffix="trialdat") 

                    # fig1, fig2 = bmh.plot_score_cross_prior_model_splitby(df=dfthis, split_by=split_by) 
                    # savefig(fig1, f"{sdir}/splitby_{split_by}-trialdat-1.pdf") 
                    # savefig(fig2, f"{sdir}/splitby_{split_by}-trialdat-2.pdf")

                    # agg version of old plots
                    bmh.plot_score_cross_prior_model_splitby_agg(split_by=split_by,
                        sdir=sdir, suffix="aggdat") 
                    # savefig(fig1, f"{sdir}/splitby_{split_by}-aggdat-1.pdf") 
                    # savefig(fig2, f"{sdir}/splitby_{split_by}-aggdat-2.pdf")
                    
                    # New plots
                    bmh.plot_score_cross_prior_model_splitby_v2(df=bmh.DatLong, split_by=split_by, savedir=sdir)

                    plt.close("all")

                    # except Exception as err:
                    #     pass

                    # Plot timecourse, one plot for each epoch
                    list_levels = D.Dat[split_by].unique().tolist()
                    for lev in list_levels:
                        df = D.Dat[(D.Dat["exclude_because_online_abort"]==False) & (D.Dat[split_by]==lev)]
                        fig=sns.relplot(data=df, x="tvalfake", col="epoch", col_wrap=2, y="success_binary_quick", 
                                    hue="session",
                                    height=3, aspect=3, alpha=0.25)
                        savefig(fig, f"{sdir}/timecourse-splitby_{split_by}-lev_{lev}.pdf")   

                        plt.close("all") 
                except Exception as err:
                    print("[grammar] Skipping for split_by=", split_by)

            # Do permutation test of whether score is significant across epochs
            # (e.g., if using microstim to perturb baehavior)
            if not np.all(bmh.DatLong["epoch"] == bmh.DatLong["epoch_orig"]):
                # Then epoch is different from epoch_orig
                bmh.stats_score_permutation_test(split_plots_by="epoch_orig", 
                    savedir=sdir)        
                bmh.stats_score_permutation_test(split_plots_by="epoch_orig", 
                    savedir=sdir, suffix="flat")        
            try:
                bmh.stats_score_permutation_test(split_plots_by=None, savedir=sdir)
            except NotEnoughDataException as err:
                pass
            except Exception as err:
                raise err
            bmh.stats_score_permutation_test(split_plots_by=None, savedir=sdir, suffix="flat")

            #### Separate p-vals for each epochset.
            # make grouping var
            if "epochset" in bmh.DatLong.columns:
                from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
                _, bmh.DatLong = grouping_append_and_return_inner_items(bmh.DatLong, ["epochset", "epoch_orig"], new_col_name="epochset_epoch", return_df=True)
                bmh.stats_score_permutation_test(split_plots_by = "epochset_epoch", savedir=sdir)

            ### MICROSTIM PLOTS
            if "microstim_epoch_code" in bmh.DatLong.columns:
                CODE_OFF = "off"
                sdir = f"{savedir}/MICROSTIM-score_epoch_x_rule_splitby"
                os.makedirs(sdir, exist_ok=True)

                for split_by in LIST_SPLIT_BY:
                    bmh.plot_score_cross_prior_model_splitby(df=bmh.DatLong, split_by=split_by,
                        sdir=sdir, suffix="trialdat", var="microstim_epoch_code") 
                    if False:
                        # afgg doesnt have the variables.
                        bmh.plot_score_cross_prior_model_splitby_agg(split_by=split_by,
                            sdir=sdir, suffix="aggdat", var="microstim_epoch_code") 
                
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig")
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by=None)

                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix="flat")
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix="flat")

                # compare each one to off
                list_code = bmh.DatLong["microstim_epoch_code"].unique().tolist()
                for code in list_code:
                    if code==CODE_OFF:
                        continue
                    dfthis = bmh.DatLong[bmh.DatLong["microstim_epoch_code"].isin([CODE_OFF, code])].reset_index(drop=True)

                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix=f"code_{code}")
                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix=f"code_{code}")

                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix=f"code_{code}-flat")
                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix=f"code_{code}-flat")

            ############# time-dependence
            sdir = f"{savedir}/by_time"
            os.makedirs(sdir, exist_ok=True)
            plot_binned_by_time(D, sdir)
            if "microstim_epoch_code" in bmh.DatLong.columns:
                plot_trial_by_trial(D, sdir)

            ######### 2) Plot summary
            # dfGramScore = bmh.DatLong  
            # dfGramScore = dfGramScore[dfGramScore["exclude_because_online_abort"]==False]

            from pythonlib.tools.pandastools import stringify_values
            DF = stringify_values(bmh.DatLong)

            if not checkIfDirExistsAndHasFiles(f"{SDIR}/summary")[1]:
                plot_performance_all(DF, list_blocksets_with_contiguous_probes, SDIR)
                plot_performance_timecourse(DF, list_blocksets_with_contiguous_probes, SDIR)
                plot_performance_static_summary(DF, list_blocksets_with_contiguous_probes, SDIR, False)
                plot_performance_static_summary(DF, list_blocksets_with_contiguous_probes, SDIR, True)
                plot_counts_heatmap(DF, SDIR)
                plot_performance_trial_by_trial(DF, D, SDIR)
                plot_performance_each_char(DF, D, SDIR)
                # 1) print all the taskgroups
                D.taskgroup_char_ntrials_print_save(SDIR)

                # plot counts, only success triuals
                df = DF[DF["success_binary_quick"]==True].reset_index(drop=True)
                plot_counts_heatmap(df, SDIR, suffix="SUCCESS")
            else:
                print("[SKIPPING, since SDIR exists and has contents: ", SDIR)

            ######## CONJUNCTIONS PLOTS
            # DS, dataset_pruned_for_trial_analysis, params_anova, params_anova_extraction = conjunctions_preprocess(D)
            DS, D, params_anova = conjunctions_preprocess(D)
            if DS is not None:
                savedir = f"{SDIR}"
                conjunctions_plot(D, DS, savedir, params_anova)

        return bmh, SDIR

def plot_stepwise_actions(D):
    """ All plots related to probs of actions at each stroke step.
    """
    from pythonlib.grammar.stepwise import preprocess_plot_actions

    # 1) All data
    preprocess_plot_actions(D, suffix="alldata")

    # 2) Separate for each epochset
    list_epochset = D.Dat["epochset"].unique().tolist()
    nmin = 10
    for es in list_epochset:
        Dc = D.copy()
        Dc.Dat = Dc.Dat[Dc.Dat["epochset"]==es].reset_index(drop=True)
        if len(Dc.Dat)>nmin:
            preprocess_plot_actions(Dc, suffix=f"epochset-{es}")

    # 3) split by epochset_char
    D.epochset_extract_common_epoch_sets("character", epochset_col_name="epochsetchar")
    list_epochset = D.Dat["epochsetchar"].unique().tolist()
    nmin = 10
    for es in list_epochset:
        Dc = D.copy()
        Dc.Dat = Dc.Dat[Dc.Dat["epochsetchar"]==es].reset_index(drop=True)
        if len(Dc.Dat)>nmin:
            preprocess_plot_actions(Dc, suffix=f"epochsetchar-{es}")

    # 3) Split into first and last half of data
    D1, D2 = D.splitdataset_by_trial()
    preprocess_plot_actions(D1, suffix=f"splitbytime_half1")
    preprocess_plot_actions(D2, suffix=f"splitbytime_half2")

def plot_binned_by_time(D, sdir):
    """ bin trials by their times. change over session?"""

    LIST_SPLIT_BY = ["probe", "epochset"]

    Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline", "no_supervision"])
    # print(len(Dc.Dat))

    ############### SPLIT BY TIME (TIME BINS)
    # bin trials and assing to dataframe
    from pythonlib.tools.nptools import bin_values
    list_sess = Dc.Dat["session"].unique().tolist()
    for sess in list_sess:
        dfthissess = Dc.Dat[Dc.Dat["session"]==sess].reset_index(drop=True)
        # sort by trial
        dfthissess = dfthissess.sort_values(by="trial").reset_index(drop=True)
        for nbins in [2,4,8]:
            vals = dfthissess["trial"].tolist()
            timebins = bin_values(vals, nbins)
            dfthissess["trial_binned"] = timebins

            y = "success_binary_quick"
            for split_by in LIST_SPLIT_BY:
                fig = sns.catplot(data=dfthissess, x="trial_binned", y=y, hue="epoch", row=split_by, kind="point", ci=68)
                path = f"{sdir}/binned_time-nbins_{nbins}-sess_{sess}-splitby_{split_by}.pdf"
                savefig(fig, path)
                print(path)
                plt.close("all")



def plot_trial_by_trial(D, sdir):
    """ to lookk closely at whether trials post stim have lasting effets
    """

    Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline", "no_supervision"])
    dfthis = Dc.Dat
    print(len(dfthis))

    # Plot timecourse and visualize
    # sns.relplot(data=dfthis, x="trial", y="success_binary_quick", hue="microstim_epoch_code",
    #            aspect=8, alpha=0.5, style="block", row="session")

    # sns.relplot(data=dfthis, x="trial", hue="success_binary_quick", y="microstim_epoch_code",
    #            aspect=8, alpha=0.5, style="block", row="session")

    fig = sns.relplot(data=dfthis, x="trial", hue="microstim_epoch_code", y="block",
               aspect=10, alpha=0.5, style="success_binary_quick", row="session")
    savefig(fig, f"{sdir}/TIMECOURSE.pdf")   

    ####### ALINGED TO STIM TRIALS.
    for n in [1,2,3,4]:
        # y = "success_seq_nmatch"
        y = "success_binary_quick"
        EXCLUDE_IF_ABORT=False
        PRINT = False

        # run a sliding window over trials
        list_sess = dfthis["session"].unique().tolist()
        for sess in list_sess:
            dfthissess = dfthis[dfthis["session"]==sess].reset_index(drop=True)

            # sort by trial
            dfthissess = dfthissess.sort_values(by="trial").reset_index(drop=True)
            
            list_dfwindows = []
            for i_center in range(n, len(dfthissess)-n):
                dfwind = dfthissess.iloc[i_center-n:i_center+n+1].reset_index(drop=True)
                
                ## CRITERIA FOR EXCLUSION
                if EXCLUDE_IF_ABORT:
                    if np.any(dfwind["exclude_because_online_abort"]):
                        continue
                
                if not np.all(dfwind["microstim_epoch_code"][:n]=="off"):
        #             print(dfwind["microstim_epoch_code"][:n])
                    continue
                
                if not np.all(dfwind["microstim_epoch_code"][n+1:]=="off"):
                    continue
                    
                assert np.all(np.diff(dfwind["trial"])>0)
                
                list_dfwindows.append(dfwind) # get n flanking each side.

                if PRINT:
                    print(i_center, "----", dfwind["trial"].tolist())

            DICT_LISTS_VALUES = {}

            for dfwind in list_dfwindows:

                # only keep if middle index is stim
                STIM_CODE = dfwind.iloc[n]["microstim_epoch_code"]

                # get timecoures of variable.
                succs = 1*dfwind[y].values

                if STIM_CODE not in DICT_LISTS_VALUES.keys():
                    DICT_LISTS_VALUES[STIM_CODE] = [succs]
                else:
                    DICT_LISTS_VALUES[STIM_CODE].append(succs)        

                if False:
                    if STIM_CODE=="off":
                        ax = axes.flatten()[0]
                    elif STIM_CODE=="TTL3-fgon":
                        ax = axes.flatten()[1]
                    elif STIM_CODE=="TTL4-fgon":
                        ax = axes.flatten()[2]
                    else:
                        print(STIM_CODE)
                        assert False

                    ax.plot(xs, succs, "ok", alpha=0.005)

            for k,v in DICT_LISTS_VALUES.items():
                DICT_LISTS_VALUES[k] = np.stack(v, axis=0) # (ndat, ntimes)

            # plot
            nstimcodes = len(DICT_LISTS_VALUES.keys())
            if nstimcodes==0:
                continue
            elif nstimcodes==1:
                nstimcodes=2 # for unsqueezing
            fig, axes = plt.subplots(1, nstimcodes, figsize=(6,3))

            xs = np.arange(2*n+1)

            for ax, (stim_code, values_mat) in zip(axes.flatten(), DICT_LISTS_VALUES.items()):
                values_mean = np.mean(values_mat, axis=0)    
                ax.plot(xs, values_mean, "-ok")
                ax.set_title(stim_code)
                
                ax.set_ylabel(f"n={values_mat.shape[0]}")
                ax.set_xlabel(f"trials, aligned at middle trial")

                if y=="success_seq_nmatch":
                    ax.set_ylim(0,6)
                elif y=="success_binary_quick":
                    ax.set_ylim(0,1)
                else:
                    assert False

            savefig(fig, f"{sdir}/ALIGNED_TRIALS-n_{n}-sess_{sess}.pdf")   
            plt.close("all")

def conjunctions_preprocess(D):
    """
    Online sequenbce, plot all conjucntions for grammar neural analyses
    NOTE: returns None if D is empty after preprocessing.
    """
    from neuralmonkey.metadat.analy.anova_params import dataset_apply_params, params_getter_dataset_preprocess
    from neuralmonkey.classes.snippets import datasetstrokes_extract

    # remove baseline
    # D.grammarmatlab_successbinary_score()

    # First remove baseline
    # print(1, D.Dat["epoch"].unique())    
    D.preprocessGood(params=["remove_baseline", "one_to_one_beh_task_strokes"])
    # print(2, D.Dat["epoch"].unique())
    # assert False
    
    if len(D.Dat)==0:
        return (None for _ in range(4))

    # Second get parses.
    D.grammarparses_successbinary_score_wrapper()
    D.preprocessGood(params=["correct_sequencing_binary_score"])

    if len(D.Dat)==0:
        return (None for _ in range(4))

    # Assign chunks info to tokens
    for ind in range(len(D.Dat)):
        D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)

    # Clean and extract dataset as would in neural analy
    # ListD = [D]
    animal = D.animals(force_single=True)[0]
    tmp = D.Dat["date"].unique()
    assert len(tmp)==1
    DATE = int(tmp[0])
    # which_level = "stroke"
    ANALY_VER = "seqcontext"
    # anova_interaction = False

    # Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params_anova, params_anova_extraction = \
    #     dataset_apply_params(ListD, None, ANALY_VER, animal, DATE)
    D, DS, params_anova = dataset_apply_params(D, None, ANALY_VER, animal, DATE)

    # # list_features = ["chunk_rank", "chunk_within_rank", "chunk_within_rank", "chunk_n_in_chunk"]
    # list_features = []
    # DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, list_features=list_features)

    # # Legacy code -- get params anova.
    # params_anova = params_getter_dataset_preprocess(ANALY_VER, animal, DATE)

    return DS, D, params_anova


def conjunctions_plot(D, DS, savedir, params_anova):
    """ Make all plots and printed text for conjuctuions relevant for grammar, at level of
    strokes during drawing
    PARAMS:
    - D, DS, savedir, params_anova, see output of conjunctions_preprocess
    """

    globals_nmin = 5

    from neuralmonkey.metadat.analy.anova_params import _conjunctions_print_plot_all
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    
    # LIST_VAR = params_anova["LIST_VAR"]
    # LIST_VARS_CONJUNCTION = params_anova["LIST_VARS_CONJUNCTION"]           

    DS.context_define_local_context_motif(1,1)
    DS.context_chunks_assign_columns()

    ########## Rank within chunk
    sdir = f"{savedir}/conjunctions_strokes/chunk_within_rank"
    LIST_VAR = [
        "chunk_within_rank",
        "chunk_within_rank_fromlast",
        ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next", "chunk_rank", "epoch"],
        ["CTXT_prev_this_next", "chunk_rank", "epoch"],
    ]
    DF = DS.Dat
    _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

    ########## N prims in upcoming chunk
    sdir = f"{savedir}/conjunctions_strokes/chunk_n_in_chunk"
    LIST_VAR = [
        "chunk_n_in_chunk",
        "chunk_n_in_chunk",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next", "chunk_diff_from_prev", "chunk_n_in_chunk_prev", "epoch"],
        ["CTXT_prev_this_next", "chunk_diff_from_prev", "epoch"],
    ]
    DF = DS.Dat[DS.Dat["chunk_diff_from_prev"]==True].reset_index(drop=True)
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

    ########## Rule switching, chunk switch, whether dissociate rule switch from shape switch
    sdir = f"{savedir}/conjunctions_strokes/chunk_diff_from_prev"
    LIST_VAR = [
        "chunk_diff_from_prev",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

    ######## epoch encoding, context same
    sdir = f"{savedir}/conjunctions_strokes/epoch_context1"
    LIST_VAR = [
        "epoch",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    # DS.context_define_local_context_motif(1,1)
    # DS.context_chunks_assign_columns()
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

    ######## epoch encoding, context same
    sdir = f"{savedir}/conjunctions_strokes/epoch_context2"
    LIST_VAR = [
        "epoch",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    DS.context_define_local_context_motif(2,1)
    DS.context_chunks_assign_columns()
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)
    # Restore to defaults
    DS.context_define_local_context_motif(1,1)
    DS.context_chunks_assign_columns()

    ######## encoding of shape (otherwise context matched), 
    sdir = f"{savedir}/conjunctions_strokes/shape_byepoch"
    LIST_VAR = [
        "shape",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_next", "gridloc", "epoch"],
    ]
    DF, dictdf = extract_with_levels_of_conjunction_vars(DS.Dat, var="epoch", vars_others=["CTXT_prev_next", "gridloc"],
                                                         n_min_across_all_levs_var=2,
                                                         lenient_allow_data_if_has_n_levels=2)
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)
