"""
Tools to evaluate step by step the beh, including in combiantion with rules (e.g,, at each step how does your actiuon
relate to what each rule would do.

Could build from this to do RL model.

Coded for microstim effects on seuqencing, in order to classify sequence errors.


"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

def preprocess_plot_actions_all(D):
    """
    :param D:
    :return:
    """

    ## 1) Plot all
    preprocess_plot_actions(D)

    # - separate for first stroke vs. non-first strokes


    # 2) 1 plot per epojhsettt
    # Separate for each epochset
    list_epochset = D.Dat["epochset"].unique().tolist()
    nmin = 10
    for es in list_epochset:
        Dc = D.copy()
        Dc.Dat = Dc.Dat[Dc.Dat["epochset"]==es].reset_index(drop=True)
        if len(Dc.Dat)>nmin:
            preprocess_plot_actions(Dc, suffix=f"epochset-{es}")

    # 3) Separate for first and second half of day
    D.trialcode_tuple_extract_assign()
    # find the middle trial
    list_tc = sorted(D.Dat["trialcode_tuple"].tolist())
    n = len(list_tc)
    tc_mid = list_tc[int(n/2)]

    # First half
    Dc = D.copy()
    Dc.Dat = Dc.Dat[Dc.Dat["trialcode_tuple"]<=tc_mid].reset_index(drop=True)
    preprocess_plot_actions(Dc, suffix=f"splitbytime_half1")

    # Second half
    Dc = D.copy()
    Dc.Dat = Dc.Dat[Dc.Dat["trialcode_tuple"]>tc_mid].reset_index(drop=True)
    preprocess_plot_actions(Dc, suffix=f"splitbytime_half2")

def preprocess_plot_actions(D, suffix=None, saveon=True, cleanup_actions=True):
    """ Returns None, None, None, None if find mult parses exist
    """
    from pythonlib.tools.pandastools import expand_categorical_variable_to_binary_variables
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import grouping_print_n_samples, convert_to_2d_dataframe
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.snstools import rotateLabel

    # Make save dir
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("stepwise")
    sdir = f"{SAVEDIR}/actions_and_trials"

    if suffix is not None:
        sdir = f"{sdir}-{suffix}"

    os.makedirs(sdir, exist_ok=True)

    # Only keep nonsupervision
    Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline", "no_supervision"])

    ### 1) Extract action-level data.
    df_actions, Params = extract_each_stroke_vs_rules(Dc)
    if df_actions is None and Params is None:
        # Then skip it.
        return None, None, None, None

    ### Cleanup actions:
    if cleanup_actions:
        # 1. Remove cases with only one option left
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            """ Return n remaining taskinds"""
            return len(x["taskinds_features"])
        df_actions = applyFunctionToAllRows(df_actions, F, "n_remain_taskinds")
        df_actions = df_actions[df_actions["n_remain_taskinds"]>1].reset_index(drop=True)

        # 2. cases where failure has already occured in that trial. i.e, only consider first failures.
        df_actions["already_failed"].value_counts()
        df_actions = df_actions[df_actions["already_failed"]==False].reset_index(drop=True)

    ### Get trial-level actions.
    _, df_actions_trial = dfactions_convert_to_trial_level(df_actions, Params)

    if saveon:
        ## PRinta nd plot summary of distribution of task states.
        grouping_print_n_samples(df_actions, ["epoch", "task_state_code"],
                                 save_convert_keys_to_str=True,
                                 savepath = f"{sdir}/actions_groupings.txt")

        _, fig, _, _ = convert_to_2d_dataframe(df_actions, "epoch", "task_state_code", True)
        savefig(fig, f"{sdir}/actions-counts-task_state_code.pdf")

        ## 2) Make GOOD PLOTS
        for var in ["beh_label_code", "choice_code"]:
            df_actions_expanded = expand_categorical_variable_to_binary_variables(df_actions, var)

            fig = sns.catplot(data=df_actions_expanded, x=var, y="value", hue="task_state_crct_ti_uniq", row="epoch",
                        kind="point", ci=68, aspect=1.5)
            rotateLabel(fig)
            savefig(fig, f"{sdir}/actions-task_state_crct_ti_uniq-{var}.pdf")

            fig = sns.catplot(data=df_actions_expanded, x=var, y="value", row="task_state_crct_ti_uniq", hue="epoch",
                        kind="point", ci=68, aspect=1.5)
            rotateLabel(fig)
            savefig(fig, f"{sdir}/actions-task_state_crct_ti_uniq-{var}-2.pdf")


            fig = sns.catplot(data=df_actions_expanded, x=var, y="value", hue="task_state_code", row="epoch",
                        kind="point", ci=68, aspect=1.5)
            rotateLabel(fig)
            savefig(fig, f"{sdir}/actions-task_state_code-{var}.pdf")

            # Plot, restricting to just the strongest cases (separated states)
            df_actions_this = df_actions[df_actions["task_state_code"]==((), (0,), (1,))].reset_index(drop=True)
            if len(df_actions_this)>0:
                df_actions_this_expanded = expand_categorical_variable_to_binary_variables(df_actions_this, var)
                fig = sns.catplot(data=df_actions_this_expanded, x=var, y="value", hue="epoch", kind="point", aspect=1.5, ci=68,
                            row="epoch_orig")
                rotateLabel(fig)
                savefig(fig, f"{sdir}/actions-task_state_code-only_taskstatecode_separated-{var}.pdf")

            plt.close("all")

            if "microstim_epoch_code" in df_actions_expanded.columns:
                fig = sns.catplot(data=df_actions_expanded, x=var, y="value", hue="microstim_epoch_code", row="epoch_orig",
                            kind="point", ci=68, aspect=1.5)
                rotateLabel(fig)
                savefig(fig, f"{sdir}/actions-microstim_epoch_code-{var}.pdf")

                # First
                df_actions_expanded_this = df_actions_expanded[df_actions_expanded["idx_beh"]==0]
                fig = sns.catplot(data=df_actions_expanded_this, x=var, y="value", hue="microstim_epoch_code", row="epoch_orig",
                    kind="point", ci=68, aspect=1.5)
                rotateLabel(fig)
                savefig(fig, f"{sdir}/actions-microstim_epoch_code-{var}-FIRST_ACTION.pdf")

                df_actions_expanded_this = df_actions_expanded[df_actions_expanded["idx_beh"]>0]
                fig = sns.catplot(data=df_actions_expanded_this, x=var, y="value", hue="microstim_epoch_code", row="epoch_orig",
                    kind="point", ci=68, aspect=1.5)
                rotateLabel(fig)
                savefig(fig, f"{sdir}/actions-microstim_epoch_code-{var}-NOT_FIRST_ACTION.pdf")

            plt.close("all")

        df_actions_trial_expand = expand_categorical_variable_to_binary_variables(df_actions_trial,
                                                                                  "trial_sequence_outcome")
        fig = sns.catplot(data=df_actions_trial_expand, x="trial_sequence_outcome", y="value", kind="point", hue="epoch", ci=68,
                          row="epoch_orig")
        rotateLabel(fig)
        savefig(fig, f"{sdir}/actionstrial-outcome-{var}.pdf")

        plt.close("all")

    # Save df_actions
    pd.to_pickle(df_actions, f"{sdir}/df_actions.pkl")
    pd.to_pickle(Dc.Dat, f"{sdir}/df_dat.pkl")
    pd.to_pickle(Params, f"{sdir}/Params.pkl")

    return df_actions, df_actions_trial, Dc, Params

def extract_each_stroke_vs_rules(D, DEBUG=False):
    """
    For each stroke, (i) consider all the possible strokes it could have been, and evaluate each of those,
    thereby finding what feature predicts a stroke will be chosen. (ii) Also relate those features to what would
    be done under specific rules that are in D.
    RETURNS:
        - df_actions, len of num strokes across all trials.
        - Params
    -or- None, None, if any rule has multiple correct parses for a trial.. then it doesnt kknow what ot do.
    """

    HACK_ADD_LEFT_RULE=True # Pancho, when doign RIGHT, he seems to sometimes revert to LEFT. in this case,
    # want to also evaluate if using LEFT>

    FORCE_SINGLE_PARSE_PER_RULE = False
    if FORCE_SINGLE_PARSE_PER_RULE==False:
        print("NOTE: State code will not be accurate. it can say that a rule's ti overlaps closest. But this is just of the many ti for the rule...")

    list_res = []

    # To ask about choices of "correct shape, wrong location", determine whether any rules care about
    # shapes
    list_rules_using_shapes = D.grammarparses_rules_involving_shapes()
    if len(list_rules_using_shapes)==0:
        # Then no relevant shapes
        RULE_FOR_SHAPE = None
    elif len(list_rules_using_shapes)==1:
        RULE_FOR_SHAPE = list_rules_using_shapes[0]
    else:
        print(list_rules_using_shapes)
        print("Multiple rules that care about hsapes... not sure what to do.")
        # assert False, "Multiple rules that care about hsapes... not sure what to do"
        return None, None

    # RULE_FOR_SHAPE = "ss-rankdir-AnBmCk2" # which rule to use to define if got the corect shape.
    # RULE_FOR_SHAPE = "dir-null-L" # which rule to use to define if got the corect shape.

    rulestrings_check = sorted(D.grammarparses_rulestrings_exist_in_dataset()) # get rules that are used int his day
    # rulestrings_check = rulestrings_check[:-1]

    if HACK_ADD_LEFT_RULE:
        if rulestrings_check[0]=="dir-null-R":
            rulestrings_check = rulestrings_check[:1] + ["dir-null-L"] + rulestrings_check[1:]

    map_ruleidx_rulestring = {i:rule for i, rule in enumerate(rulestrings_check)}
    map_rulestr_ruleidx = {rule:i for i, rule in enumerate(rulestrings_check)}

    for ind in range(len(D.Dat)):
        g = D.grammarparses_grammardict_return(ind, False)
        rulestring, _ = D.grammarparses_ruledict_rulestring_extract(ind) # rulesring for this trial

        beh = D.grammarparses_extract_beh_taskstroke_inds(ind)

        if False: # Dont need this. it was only used for getting unordered "task"
            parses = D.grammarparses_parses_extract_trial(ind)
            if len(parses)>1:
                print("EXITING stepwise, since not yet coded for mjultiple parses. conceptualyl unclear.")
                return None, None
            # assert len(parses)==1, "this only coded for cases with single determinstic sequence otherwise is complex"
            task = list(parses[0])

        task = list(range(len(D.Dat.iloc[ind]["strokes_task"])))

        # Get Tokens for this trial
        tokens = D.taskclass_tokens_extract_wrapper(ind, "task")
        map_taskind_tok = {i:t for i, t in enumerate(tokens)}

        def _taskinds_state(idx):
            """ Given that you are on this beh stroke index (idx),
            what ist he state of the task?
            e.g., if i is 2, this means as you are starting stroke 3, so here
            returns the state during gap between strokes 2 and 3.
            RETURNS:
                - taskinds_already_gotten, list of ints.
                - taskinds_remain, list of ints, is entire task, even if beh storpped short.
            """
            taskinds_already_gotten = beh[:idx]
            taskinds_remain = [i for i in task if i not in taskinds_already_gotten]

            return taskinds_already_gotten, taskinds_remain

        # track if you have already failed this trial
        already_failed = False
        for idx_stroke, taskind_beh in enumerate(beh):

            if DEBUG:
                print("STroke id:", idx_stroke)

            # Individula toks for specific strokes
            # t_chosen = map_taskind_tok[taskind_beh]

            # t_correct = map_taskind_tok[taskind_task]

            ## for each remaining token, classify it in a way that is independent of the rule.

            # 1) Get the task state
            taskinds_already_gotten, taskinds_remain = _taskinds_state(idx_stroke)

            # 2) Get the remainig toekns, ordered according to each rule.
            def _remaining_inds_in_order_of_parse(parse):
                """ Get parse but only indices that are not yet gotten
                """
                return [p for p in parse if p in taskinds_remain]

            # 3) for each rule, find the correct taskinds sequencig, according to rule
            if FORCE_SINGLE_PARSE_PER_RULE:
                taskinds_remain_each_rule = {}
                for i, rule in enumerate(rulestrings_check):
                    parses_this_rule = g.parses_extract_generated(rule)

                    # for now, assume there is only a single parse per rule
                    parses_this_rule = [list(p) for p in parses_this_rule]
                    if len(parses_this_rule)>1:
                        print("EXITING stepwise, since not yet coded for mjultiple parses. conceptualyl unclear.")
                        return None, None

                    # assert len(parses_this_rule)==1, "not yet coded for mjultiple parses. conceptualyl unclear."
                    parse = parses_this_rule[0]

                    # Get remaining inds
                    taskinds_parse_remain = _remaining_inds_in_order_of_parse(parse)
                    taskinds_remain_each_rule[rule] = taskinds_parse_remain
            else:
                def _remaining_ind_this_rule_all_parses(rule):
                    """ Fopr each parses for this rule, return
                    all possible future trajctories.
                    RETURNS:
                        - list_parse_remain = list of list, where inner
                        list is future taskinds for one parse. outer list is len
                        num parses
                    """
                    parses_this_rule = g.parses_extract_generated(rule)
                    parses_this_rule = [list(p) for p in parses_this_rule]
                    list_parse_remain = []
                    for parse in parses_this_rule:
                        # Get remaining inds
                        taskinds_parse_remain = _remaining_inds_in_order_of_parse(parse)
                        list_parse_remain.append(taskinds_parse_remain)
                    return list_parse_remain

            ### Collect features overall the remaining taskinds.
            taskinds_features = []
            # VAR_LOC = "loc_concrete" # problem: doesnt work for fixation. and doesnt break ties.
            VAR_LOC = "center"
            for ti in taskinds_remain:
                # give this a dict holding its features
                features = {}
                features["taskind"] = ti

                # 1) distnace from previous beh stroke
                tok_this = map_taskind_tok[ti]
                loc_this = np.array(tok_this[VAR_LOC])
                if idx_stroke>0:
                    tok_beh_prev = map_taskind_tok[beh[idx_stroke-1]] # token of stroke before this stroke
                    loc_beh_prev = np.array(tok_beh_prev[VAR_LOC])
                else:
                    # Get distance from the fixation point
                    assert VAR_LOC=="center", "now cant compare to fixation. how to deal with first strokes?"
                    loc_beh_prev = D.sketchpad_fixation_button_position(ind)
                    # print(tok_this["center"], loc_beh_prev)
                    # assert False, "how get pixel coords?"
                dist = np.linalg.norm(loc_this - loc_beh_prev)

                # get the x an y translations
                y_trans = loc_this[1] - loc_beh_prev[1]
                x_trans = loc_this[0] - loc_beh_prev[0]

                features["dist_prev_beh"] = dist
                features["trans_prev_beh"] = (x_trans, y_trans)

                # 2) is it the correct next stroke, given each rule
                for rule in rulestrings_check:
                    if FORCE_SINGLE_PARSE_PER_RULE:
                        features[f"correct_rule_{rule}"] = ti==taskinds_remain_each_rule[rule][0]
                    else:
                        # Consider rule correct if _any_ of its parses are aligned with next stroke.
                        list_parse_remain = _remaining_ind_this_rule_all_parses(rule)
                        features[f"correct_rule_{rule}"] = False # initialize
                        for taskinds_remain in list_parse_remain:
                            if ti==taskinds_remain[0]:
                                # then found a match
                                features[f"correct_rule_{rule}"] = True
                                break
                taskinds_features.append(features)

            # Get correct ti for each rule, in list
            map_rulestr_correctti = {}
            map_rulestr_correctti_list = {}
            for rule in rulestrings_check:
                tmp = [features["taskind"] for features in taskinds_features if features[f"correct_rule_{rule}"]==True]
                # assert len(tmp)==1, "can only be one correct parse..."
                # map_rulestr_correctti[rule] = tmp[0]
                map_rulestr_correctti_list[rule] = tmp

            ### Features that require comparing across remaining taskinds
            # 1) which ti is closest in space to prev beh stroke.
            # ti_closest_to_prev_beh = sorted(taskinds_features, key=lambda x:x["dist_prev_beh"])[0]["taskind"] # int
            dist_closest = sorted(taskinds_features, key=lambda x:x["dist_prev_beh"])[0]["dist_prev_beh"]
            for features in taskinds_features:
                features["closest"] = np.isclose(features["dist_prev_beh"], dist_closest) # better, since ties are both called close.
                # features["closest"]=features["taskind"]==ti_closest_to_prev_beh

            ### Classify the state of the task.
            ti_closest_list = [features["taskind"] for features in taskinds_features if features["closest"]==True]

            ## Get the state code.
            # Each slot holds the rules that exist. first slot is "closest",
            # and the rest are arbitrary order, filled up one by one. a slot
            # can have multiple rules if they share same "correct taskind"
            nslots = len(rulestrings_check)+1 # [closest, rule0, rule1, ..]
            state_code = [[] for _ in range(nslots)]

            for i, rule in enumerate(rulestrings_check):
                # ti_rule_this = map_rulestr_correctti[rule]
                ADDED = False

                if any([x in ti_closest_list for x in map_rulestr_correctti_list[rule]]):
                # if map_rulestr_correctti[rule] in ti_closest_list:
                    # put in slot 0, all those that are close.
                    state_code[0].append(i)
                    ADDED = True
                else:
                    # Try slots until reach an empty one.
                    for slot in state_code[1:]:
                        if len(slot)==0:
                            slot.append(i)
                            ADDED = True
                            break
                        else:
                            rs_other = map_ruleidx_rulestring[slot[0]] # any index would igve same result.

                            if any([x in map_rulestr_correctti_list[rs_other] for x in map_rulestr_correctti_list[rule]]):
                            # if map_rulestr_correctti[rs_other]==map_rulestr_correctti[rule]:
                                # Then they are the same ti. put them in same slot.
                                slot.append(i)
                                ADDED = True
                                break
                            else:
                                # keep trying.
                                continue
                assert ADDED==True, "bug in code."
            state_code = tuple([tuple(sc) for sc in state_code])

            if DEBUG:
                print(ti_closest_list, map_rulestr_correctti, state_code)

            ### [Semantic state] is the correct item for the correct rule
            # different from all other rules (including close).
            if FORCE_SINGLE_PARSE_PER_RULE:
                ti_correct_rule = map_rulestr_correctti[rulestring]
                ti_incorrect_list = [map_rulestr_correctti[r] for r in rulestrings_check if not r == rulestring] # ti for other rules.
                state_crct_ti_uniq = ti_correct_rule not in (ti_closest_list + ti_incorrect_list)
            else:
                ti_correct_rule_list = map_rulestr_correctti_list[rulestring]
                ti_incorrect_list = [map_rulestr_correctti_list[r] for r in rulestrings_check if not r == rulestring]
                ti_incorrect_list = [xx for x in ti_incorrect_list for xx in x] # flatten.
                state_crct_ti_uniq = all([x not in (ti_closest_list + ti_incorrect_list) for x in ti_correct_rule_list])

            ### Label beh labels.
            # Did he pick the correct rule?
            # - even if the correc trule is the closest, still call it correct
            a = [taskind_beh in ti_closest_list]
            # b = [taskind_beh == map_rulestr_correctti[rs] for rs in rulestrings_check]
            b = [taskind_beh in map_rulestr_correctti_list[rs] for rs in rulestrings_check]


            # 1) Did he choose the same shape?
            # same shape, different location
            if RULE_FOR_SHAPE is None:
                # then no rule cares about shape ...
                c = []
            else:
                if FORCE_SINGLE_PARSE_PER_RULE:
                    ti = map_rulestr_correctti[RULE_FOR_SHAPE]
                    shape_correct = map_taskind_tok[ti]["shape"]
                else:
                    ti_list = map_rulestr_correctti_list[RULE_FOR_SHAPE]
                    shape_correct = [map_taskind_tok[ti]["shape"] for ti in ti_list]
                    assert len(set(shape_correct))==1, "bug. next ti should have have same sahpe, if this is a shape rule"
                    shape_correct = shape_correct[0]

                shape_chosen = map_taskind_tok[taskind_beh]["shape"]
                c = [shape_chosen == shape_correct]

            # Concat to code
            choice_code = tuple(a + b + c)
            # convert to string
            choice_code = "".join([str(int(ch)) for ch in choice_code])

            # ### Summarize choice into a single semantically meaningful class
            # LABEL THE behavior semantically summary. Each
            # trial gets a single mutually exclusive label.
            if c==[True]:
                # same shape, diff loc
                # label_code = (-2,)
                label_code = ("s",)
            elif a==[True]:
                # call this matching "closets"
                # label_code = (-1,) # close
                label_code = ("c",) # close
            elif all([_b==False for _b in b]):
                # Then doesnt match any rule...
                # Call this a random error
                label_code = ("x",)
            else:
                # Call this beh basd on all the rules it matches.
                label_code = [i for i, _b in enumerate(b) if _b==True]

                # tmp = []
                # for i, rule in map_ruleidx_rulestring.items():
                #     ti_rule_this = map_rulestr_correctti[rule]
                # # for i in range(len(rulestrings_check)):
                # #     ti_rule_this = map_rulestr_correctti[map_ruleidx_rulestring[i]]
                #     if taskind_beh == ti_rule_this:
                #         tmp.append(i)
                # assert tmp==label_code
                label_code = tuple(label_code)
            label_code = "".join([str(l) for l in label_code])
            assert len(label_code)>0, "sanity check, did not capture case above."

            ### Save
            resthis = {
                "ind_dat": ind,
                "trialcode":D.Dat.iloc[ind]["trialcode"],
                "epoch":D.Dat.iloc[ind]["epoch"],
                "character":D.Dat.iloc[ind]["character"],
                "epoch_orig":D.Dat.iloc[ind]["epoch_orig"],
                # "microstim_epoch_code":D.Dat.iloc[ind]["microstim_epoch_code"],
                "idx_beh":idx_stroke,
                "task_state_code":state_code,
                "task_state_crct_ti_uniq":state_crct_ti_uniq, # True if the correct ti is different from ti for all other rules (including closest)
                "beh_label_code":label_code,
                "correct_rulestr":rulestring,
                "correct_ruleidx":map_rulestr_ruleidx[rulestring],
                "choice_code":choice_code,
                # "choice_correct":taskind_beh == map_rulestr_correctti[rulestring],
                "choice_correct":taskind_beh in map_rulestr_correctti_list[rulestring],
                "already_failed":already_failed,
                "taskind_beh":taskind_beh,
                # "taskind_correct":map_rulestr_correctti[rulestring],
                "taskind_list_correct":map_rulestr_correctti_list[rulestring],
                "taskinds_features":taskinds_features
            }

            if "microstim_epoch_code" in D.Dat.columns:
                resthis["microstim_epoch_code"] = D.Dat.iloc[ind]["microstim_epoch_code"]

            list_res.append(resthis)
            if resthis["choice_correct"]==False:
                # Track failures.
                already_failed = True

    # Convert to df
    dfactions = pd.DataFrame(list_res)

    # Colect some params
    Params = {
        "RULE_FOR_SHAPE":RULE_FOR_SHAPE,
        "list_rules_using_shapes":list_rules_using_shapes,
        "rulestrings_check":rulestrings_check,
        "map_ruleidx_rulestring":map_ruleidx_rulestring,
        "map_rulestr_ruleidx":map_rulestr_ruleidx,
    }

    return dfactions, Params

def dfactions_convert_to_trial_level(dfactions, Params):
    """ Instead of representign each action, represent each trial,
    using the failed action for failure trials, and simply
    as ''success'' for good trials
    RETURNS:
        - [modifies dfactions] with new column summarizign the trial
        - dfactions_trial, one row per trial
    """
    from pythonlib.tools.pandastools import aggregGeneral

    # convert df_actions to classify each trial to success or failure (including reason for failure).

    # list_ind_dat = df_actions["ind_dat"].unique().tolist()
    # for ind_dat in range()

    list_g =[]
    list_g_one_per_trial = []
    for g in dfactions.groupby("ind_dat"):
        if all(g[1]["choice_correct"].tolist()):
            # Then all strokes were correct.
            # if True:

            # call it ok.
            trial_sequence_code = "ok"
            g[1]["trial_sequence_outcome"] = trial_sequence_code

            # else:
            #     # call it
            #     trial_sequence_code = str(g[1].iloc[-1]["choice_code"])
            #     g[1]["trial_sequence_outcome"] = trial_sequence_code

            # take the last stroke (arbitrary choice)
            list_g_one_per_trial.append(g[1].iloc[-1,:])
        else:
            # classify the kind of error.
            # Tke the row that has the first error
            bools = g[1]["choice_correct"]==False
            idx_beh_take = min(g[1][bools]["idx_beh"].tolist()) # the first failure.
            datrow = g[1][g[1]["idx_beh"] == idx_beh_take]
            assert len(datrow)==1
            # print(datrow)
            # print(len(datrow))
            # assert False

            # # failure is the last stroke.
            # assert g[1]["idx_beh"].tolist()[-1]==max(g[1]["idx_beh"]), "bad assumption of taking the last ind"

            choice_code = datrow["choice_code"].item()
            correct_ruleidx = datrow["correct_ruleidx"].item()
            # assert choice_code[correct_ruleidx+1] == False, "sanity check"
            if not choice_code[correct_ruleidx+1] == "0":
                print(datrow)
                print(choice_code)
                print(correct_ruleidx)
                assert False, "sanity check"

            if False:
                g[1]["trial_sequence_outcome"] = [choice_code for _ in range(len(g[1]))]
            else:
                # string is easier to plot
                g[1]["trial_sequence_outcome"] = str(choice_code)

            list_g_one_per_trial.append(datrow)

        list_g.append(g[1])

    dfactions = pd.concat(list_g).reset_index(drop=True)

    # get one row per trial
    group = ["trialcode", "epoch", "epoch_orig", "trial_sequence_outcome"]
    if "microstim_epoch_code" in dfactions.columns:
        group.append("microstim_epoch_code")

    dfactions_trial = aggregGeneral(dfactions, group)

    return dfactions, dfactions_trial


