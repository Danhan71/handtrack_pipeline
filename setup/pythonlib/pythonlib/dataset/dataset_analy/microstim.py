"""Analhysis of beh effects of microstim
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pythonlib.tools.plottools import savefig
import seaborn as sns
import pandas as pd
import numpy as np
from pythonlib.tools.exceptions import NotEnoughDataException

def plot_all_wrapper(D):
    """ 
    """

    # primitiveness
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
    Dcopy = D.copy()
    preprocess_plot_pipeline(Dcopy, microstim_version=True)

    # single prims.
    from pythonlib.dataset.dataset_analy.singleprims import preprocess_dataset
    Dcopy = D.copy()
    preprocess_dataset(Dcopy, PLOT=True)

    # motor timing
    if False:
        # all folded into primitiveness now
        Dcopy = D.copy()
        plot_motortiming(Dcopy)

    # Grammar
    date = D.dates(True)[0]
    is_grammar = (sum(D.Dat["epoch_orig"] != date)>0) and (np.any(D.Dat["task_kind"]=="prims_on_grid"))

    # Example values:
    # print(D.Dat["epoch"].unique())
    # print(D.Dat["epoch_orig"].unique())
    # print(D.Dat["task_kind"].unique())
        # ['240611|off' '240611|TTL3-fg']
        # ['240611']
        # ['prims_single' 'prims_on_grid']
    if is_grammar:
        try:
            from pythonlib.dataset.dataset_analy.grammar import pipeline_generate_and_plot_all
            Dcopy = D.copy()
            pipeline_generate_and_plot_all(Dcopy)
        except NotEnoughDataException as err:
            # skip, since rules are not defined
            pass
        except Exception as err:
            raise err

def plot_overview_behcode_timings(D, sdir, STIM_DUR = 0.5):
    """ Wuick plot , for each stim trial, of 
    of timings of behe vents, strokes, and stim onsets.
    Based on beh codes extracted.
    Only plots for trials with detercted stim, based on beh codes.
    PARAMS:
    - STIM_DUR, num in sec, duiratin of stim, for use in plotting
    """  
    
    from pythonlib.tools.plottools import plot_beh_codes

    D.ml2_extract_raw()
    PRINT=False

    for ind in range(len(D.Dat)):

        # get times of stim
        # codes_keep = [141, 142, 151, 152] # 152 ignore. is TTL off
        codes_keep = [141, 142, 151] 
        codes, times = D.ml2_utils_getTrialsBehCodes(ind, codes_keep, PRINT=PRINT)
        
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_stroke = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]
        
        if len(codes)>0:
                    
            assert ms_fix["on"]==1
            assert ms_stroke["on"]==1
            
            fig, ax = plt.subplots(figsize=(12,4))

            # Plot times of stim
            plot_beh_codes(codes, times, ax=ax, color="r", yval=2)
            # Plot offsets
            times_offset = [t+STIM_DUR for t in times]
            plot_beh_codes(codes, times_offset, ax=ax, color="m", yval=2)

            # get times of strokes and trial events
            codes, times = D.ml2_utils_getTrialsBehCodes(ind, keep_only_codes_standard=True, PRINT=PRINT)
            plot_beh_codes(codes, times, ax=ax, color="k", yval=0)

            # get times of strokes
            strokes = D.Dat.iloc[ind]["strokes_beh"]

            ons = [s[0,2] for s in strokes]
            offs = [s[-1,2] for s in strokes]

            plot_beh_codes(["on" for _ in range(len(ons))], ons, ax=ax, color="b", yval=1)
            plot_beh_codes(["off" for _ in range(len(offs))], offs, ax=ax, color="k", yval=1)

            ax.set_ylim([-3, 5])
            ax.set_ylabel("0:trial events; 1:strokes: 2: stim")
            tc = D.Dat.iloc[ind]["trialcode"]
            tr = D.Dat.iloc[ind]["trial"]
            ax.set_title(f"tc_{tc}|trial_{tr}")
            print(f"{sdir}/trial_{tr}.png")
            fig.savefig(f"{sdir}/trial_{tr}.png")
            # assert False
            plt.close("all")    
        else:
            assert ms_fix["on"]==0
            assert ms_stroke["on"]==0
            

def preprocess_assign_stim_code(D, map_ttl_region, code_ignore_within_trial_time=False):
    """
    Give each trial a string code for its stim params, which can vary across 
    expts. Code indicates both the map_ttl_region, and which time windows
    within trial are being stimmed (as a string code).
    PARAMS:
    - map_ttl_region, dict mapping from int ttl values to string , usually
    breian regions, which will be the code.
    - code_ignore_within_trial_time, then will code simply by ttl, not by when
    in trial it stimmed.
    EXAMPLE:
    - 
    """

    # if HACK:
    #   map_ttl_region = {
    #       3:"M1",
    #       4:"pSMA"
    #   }
    # else:
    #   assert False, "code this input"

    print(map_ttl_region)
    list_stim_code = []
    for ind in range(len(D.Dat)):
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_str = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]

        ## Parse when stim occured on this trial, abstractly.
        def _parse_stim_strokes(ms_str):
            """ Return a list of stims that are assigned to this trial, such as
            e./.g, [('go_cue',), ('on', 1), ('on', 2), ('on', 3), ('on', 4), ('off', 1), ('off', 2), ('off', 3)]
            """
            _list_stims =[]
            for x in ms_str["stimlist"]:
                if x[0]=="go_cue":
                    _list_stims.append(("go_cue",))
                elif x[0] in ["on", "off"]:
                    indstroke = int(x[1])
                    _list_stims.append((x[0], indstroke))
                else:
                    print(x[0])
                    print(x)
                    assert False, "dont know this"
            return _list_stims

        # Collect a string code for which time windows wihtin trial.
        stim_window_code = ""
        if ms_fix["on"]==1:
            stim_window_code+="f"
        if ms_str["on"]==1:
            # could be go_cue, stroke onsets, or stroke offsets
            _list_stims = _parse_stim_strokes(ms_str)
            _list_stim_kinds_unique = sorted(list(set([x[0] for x in _list_stims]))) # ['on', 'go_cue', 'off']
            # print(_list_stims)
            # print(_list_stim_kinds_unique)

            for _stim in _list_stim_kinds_unique:
                if _stim=="go_cue":
                    stim_window_code+="g"
                elif _stim=="on":
                    stim_window_code+="n"
                elif _stim=="off":
                    stim_window_code+="o"
                else:
                    print(_stim)
                    assert False

        # print(" --------------- ", ind)
        # print(ms_fix)
        # print(ms_str)
        # print(stim_window_code)

        # FIxation
        stim_code = []
        ttl_codes = [] #collect across all stims.
        stim_code.append(ms_fix["on"]==1)
        if ms_fix["on"]==1:
            ttls = [int(x[0]) for x in ms_fix["stimlist"] if int(x[0]) in map_ttl_region.keys()]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])

        # Strokes
        stim_code.append(ms_str["on"]==1)
        if ms_str["on"]==1:
            ttls = [int(x[2]) for x in ms_str["stimlist"]  if int(x[2]) in map_ttl_region.keys()]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])
        
        stim_code = sorted(stim_code) # so it is unqiue
        # print(stim_code)

        # SHORTHANDS
        if all([x==False for x in stim_code]):
            # Stim off
            sc = "off"
        else:
            # Code it by its ttl
            # ttls = stim_code[1::2]
            assert len(set(ttl_codes))==1
            sc = map_ttl_region[ttl_codes[0]]

            # and append the code indicating WHEN in trial it is stimmed
            if not code_ignore_within_trial_time:
                sc = f"{sc}-{stim_window_code}"

        list_stim_code.append(sc)

    # place stim code back into data
    D.Dat["microstim_epoch_code"] = list_stim_code

    print("New column: microstim_epoch_code")
    

def plot_motortiming(D, PLOT=True, microstim_version=True):
    """
    Plots of timing (gaps and strokes) effects of stimulation, controlling for
    stroke and seuqence context (and stroke index).
    RETURNS:
    - DS, DFTHIS
    And makes plots.
    """
    from pythonlib.dataset.dataset_analy.motortiming import gapstroke_timing_compare_by_variable
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import assign_epochsets_group_by_matching_levels_of_var
    from pythonlib.stats.lme import lme_categorical_fit_plot
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping

    PLOT_BY_STROKE_INDEX_SET = False

    # Prep dataset
    D.grouping_get_inner_items("block", "epoch") 
    SAVEDIR_BASE = D.make_savedir_for_analysis_figures_BETTER("microstim/motortiming")
    
    if False:
        for ind in range(len(D.Dat)):
            block = D.Dat.iloc[ind]["block"]
            rule = D.blockparams_extract_single_combined_task_and_block(ind)["GENERAL"]["block_transition_rule"]
            print(ind, block, rule)

    # Extract Strokes data
    VARS_CONTEXT = ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"]
    params_preprocess = ["remove_baseline", "no_supervision", "only_blocks_with_n_min_trials"]
    VAR = "epoch"
    DS, DFTHIS_ORIG = gapstroke_timing_compare_by_variable(D, VAR, VARS_CONTEXT, 
        params_preprocess, PLOT=False, microstim_version=microstim_version)
    # Take controlled data --> plot a single distribution, one for each (context, index) combo
    DFTHIS_ORIG = append_col_with_grp_index(DFTHIS_ORIG, ["stroke_index", "context"], "strk_idx_ctxt", use_strings=False)

    if len(DFTHIS_ORIG)>0 and PLOT:

        for context_var in ["context", "strk_idx_ctxt"]:

            DFTHIS = DFTHIS_ORIG.copy()

            SAVEDIR = f"{SAVEDIR_BASE}/contextvar_{context_var}"
            os.makedirs(SAVEDIR, exist_ok=True)

            # Remove strk_idx_ctxt that have too few data, or else makes plot harder to interpret.
            n_min_each_conj_outer_inner = 3
            # print("**** len DS (3)", len(DFTHIS))
            DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                            var_outer_trials=context_var, 
                                                                                            var_inner="epoch",
                                                                                            epochset_col_name="epochset", 
                                                                                            PRINT=True, n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)
            #print("**** len DS (4)", len(DFTHIS))
            #assert False
            # Plot as function of storke index
            if PLOT_BY_STROKE_INDEX_SET:
                n_min_each_conj_outer_inner = 3
                DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                                var_outer_trials="locshape_pre_this", 
                                                                                                var_inner="stroke_index",
                                                                                                epochset_col_name="stroke_index_set", 
                                                                                                PRINT=True,
                                                                                               n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)

            print("SAVING FIGURES AT: ", SAVEDIR)
            for y in ["gap_from_prev_dur", "gap_from_prev_dist", "gap_from_prev_vel", "time_duration", "distcum", "velocity"]:
                
                fig = sns.catplot(data=DFTHIS, x=context_var, y=y, hue="epoch", kind="point", aspect=3, row="epoch_orig")
                rotateLabel(fig)    
                savefig(fig, f"{SAVEDIR}/{y}-vs-{context_var}.pdf")

                fig = sns.catplot(data=DFTHIS, x=context_var, y=y, hue="epoch", row="epochset", kind="point", col="epoch_orig")
                rotateLabel(fig)    
                savefig(fig, f"{SAVEDIR}/{y}-vs-{context_var}-grp_by_epochset.pdf")

                plt.close("all")
                
                if PLOT_BY_STROKE_INDEX_SET:
                    # Too large, and sparse, I dont use anwyay
                    fig = sns.catplot(data=DFTHIS, x="stroke_index", y=y, hue="epoch", row="locshape_pre_this", col="stroke_index_set", kind="point")
                    rotateLabel(fig)    
                    savefig(fig, f"{SAVEDIR}/{y}-vs-stroke_index-grp_by_stroke_index_set.pdf")

                plt.close("all")

                ####### PLOTS OF CONTRAST ACROSS LEVELS.
                INDEX = [context_var, "epoch_orig", "block"]
                if "microstim_epoch_code" in DFTHIS.columns:
                    fixed_treat = "microstim_epoch_code"
                    lev_treat_default = "off"
                else:
                    fixed_treat = "epoch"
                    lev_treat_default = None

                if True:
                    # Linear mixed effects
                    # This doesnt make sense, since there is only one datapt per group
                    RES, fig, ax = lme_categorical_fit_plot(DFTHIS, y=y, fixed_treat=fixed_treat, 
                            lev_treat_default=lev_treat_default, 
                            rand_grp_list=INDEX, PLOT=True)
                    savefig(fig, f"{SAVEDIR}/LME-{fixed_treat}-{y}.pdf")

                # Plot normalized to the default level.
                _, _, _, _, fig = datamod_normalize_row_after_grouping(DFTHIS, 
                                                                      fixed_treat, 
                                                                      INDEX, 
                                                                      y,
                                                                      lev_treat_default,
                                                                      PLOT=True
                                                                     )
                savefig(fig, f"{SAVEDIR}/NORM-{fixed_treat}-{y}.pdf")

                # ## ALso seaprately for each epoch-orig
                # for epoch_orig in list_epoch_orig:

                #     _, _, _, _, fig = datamod_normalize_row_after_grouping(DFTHIS, 
                #                                                           fixed_treat, 
                #                                                           INDEX, 
                #                                                           y,
                #                                                           lev_treat_default,
                #                                                           PLOT=True
                #                                                          )
                #     savefig(fig, f"{SAVEDIR}/NORM-{fixed_treat}-{y}.pdf")



    return DS, DFTHIS