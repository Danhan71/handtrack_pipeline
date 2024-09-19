""" methods to preprocess from different kinds of dataframes to 
a single kind of BMH
"""

from .beh_model_holder import BehModelHolder
import pandas as pd
import numpy as np

# def dataframe_reshape_long_to_wide(df):
#     """ Convert from long-form (each row is a trial x model) to 
#     wide form, where each trial has one row, and each score is a separate
#     column. Is clever about which variables to use, sholud work generalyl across
#     all kinds of inputs.
#     """


def generate_diagnostic_model_data(D, LIST_MODELS=None, LIST_MOTIFS=None,
    COLS_TO_KEEP = ("taskcat_by_rule")):
    """ From Dataset, genberate DiagnosticModel, and also convert to BM.
    See grammar.diagnostic_model
    - Diagnostic means scores are counting num instances of given motifs (seuqnece).
    PARAMS:
    - LIST_MODELS, list of rulestrings (sdaf-asdfa-asdf)
    - LIST_MODELS, list of dicts, each a set of params for findings motifs (for scoring)
    """
    
    from pythonlib.grammar.diagnostic_model import DiagnosticModel
    
    DM = DiagnosticModel()
    DM.preprocess_dataset_extract_scores(D, LIST_MODELS=LIST_MODELS, 
        LIST_MOTIFS=LIST_MOTIFS, COLS_TO_KEEP=COLS_TO_KEEP)
    BM = BehModelHolder(DM.Dat, input_ver="long_form")
    return BM, DM



# for BOOLEAN rules only—adds column for each rule with True/False values.
# @noReturn; modifies existing D in-place, adding column 'binary_rule_tuple'
def _add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
    for i in range(len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))



# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data_long(D, list_rules, binary_rule=False, 
    how_define_correct_order="epoch", DEBUG=False, return_as_bmh_object=True,
                                        ONLY_ACTUAL_RULE=False, USE_DATASET_DF=True):
    """High-level extraction for each trial its parses under each given rule, including
    rules that are not conssteitnw with a trials' ecpoh (i.e., get all rules, regardless
    of epoch) [column = "score"], and also and evaluate whether this trials' beahvhiro 
    was correct/incorrect (compoaring to only those rules taht are aligned with
    this trials' epoch --> column = "success_binary_quick").
    DOES NOT use the matlab objectclass sequence.
    PARAMS:
    - list_rules, list of rules trings  (a-b-c). see grammar module for geetting these auto.
    - dict_map_epoch_to_rulenames, dict mapping from epoch to rule. Useful to know for each
    trials whats its ground truth rule. If None, assumes identity.
    - binary_rule, bool, whether to get for eahc trial a binary code for bollean rules
    - how_define_correct_order, str, in {'matlab', 'epoch'}, whether to definde each trials' 
    correct sequence using the saved sequen inematlab or recomputed based on rules
    - ONLY_ACTUAL_RULE, bool, if True, then only includes agents/rules which match the
    rulestring of the actual trial.
    - USE_DATASET_DF, then merges D.Dat into output Dataframe.
    RETURNS:
    - df, with important columns score and success_binary_quick
    --- one row for conjunction of trial x rule (in list_rules).
    """
    from pythonlib.dataset.modeling.discrete import _rules_consistent_rulestrings_extract_auto
    
    D.preprocessGood(params=["remove_baseline"])

    ################## Generate behclass
    D.behclass_preprocess_wrapper()

    results = [] #dataframe to return

    # Collect these things to put back into D.Dat
    LIST_success_binary = []
    LIST_beh_sequence_wrong = []
    LIST_beh_too_short = []
    LIST_exclude_because_online_abort = []
    for ind in range(len(D.Dat)):
        if ind%100==0:
            print("trial #", ind)
        if DEBUG:
            print("---- Trial ", ind)

        # rename value in 'epoch' to rulename, if applicable
        epoch_i = D.Dat.iloc[ind]['epoch_orig']

        # generate all parses for this trial
        # gramdict = D.grammarmatlab_extract_beh_and_task(ind)
        # taskstroke_inds_beh_order = gramdict["taskstroke_inds_beh_order"]
        taskstroke_inds_beh_order = D.grammarparses_extract_beh_taskstroke_inds(ind)
        # parsesdict = D._grammarparses_parses_extract(ind, list_rules)       
        D._grammarparses_parses_extract(ind, list_rules) # need to extract.
        GD = D.grammarparses_grammardict_return(ind)       

        # Score this trial as correct/incorrect
        if how_define_correct_order=="matlab":
            assert False, "use ...matlabrule"
            # correct order specified in matlab code
            order_correct = gramdict["taskstroke_inds_correct_order"]
            assert order_correct is not None, "not defined in matlab ObjectClass.. you must recompute, useing grammar_recompute_parses"
            list_order_correct = [order_correct] # only a single correct order
        elif how_define_correct_order=="epoch":
            # 1) get all the acceptable rules
            from pythonlib.dataset.modeling.discrete import _rules_consistent_rulestrings_extract_auto
            list_rule_strings = _rules_consistent_rulestrings_extract_auto([epoch_i], return_as_dict=False)[0] # list of rulestrings
            list_order_correct = []
            if DEBUG:
                print("Current epoch: ", epoch_i)
                print("...collecting all correct parses for rulestrings aliggned with this epoch...")
            for rule_string in list_rule_strings:
                parses = GD.parses_extract_generated(rule_string)
                # parses = parsesdict[rule_string] # list of tuples of ints
                list_order_correct.extend(parses)
                if DEBUG:
                    print(f"Correct sequences for rulestring {rule_string}: {parses}")
            # make sure all parses are lists
            list_order_correct = [list(par) for par in list_order_correct]
        else:
            assert False
        success_binary = taskstroke_inds_beh_order in list_order_correct

        ########## CONSIDER EXCEPTIONs, i..e, to say didn't actally fail the sequence...
        # note cases where did not complete the trial, but sequence was correct so far (exclude those)
        # NOTE: if multipel correct possible parses, considers each criterion one by one, instread of
        # considering only if any possible parses passes all criteria.. This is OK?
        def beh_good_ignore_length(order_beh, order_correct):
            """ True if beh is good up unitl the time beh fails.
            e.g., if beh is [0, 3, 2], and correct is [0, 3, 2, 1], then
            this is True
            """
            for x, y in zip(order_beh, order_correct):
                if x!=y:
                    return False
            return True

        list_beh_sequence_wrong = []
        list_beh_too_short = []
        list_beh_got_first_stroke = []
        for order_correct in list_order_correct:
            beh_sequence_wrong = not beh_good_ignore_length(taskstroke_inds_beh_order, order_correct)
            beh_too_short = len(taskstroke_inds_beh_order) < len(order_correct)
            beh_got_first_stroke = False # whether the first beh stroke was correct.
            if len(taskstroke_inds_beh_order)>0:
                if taskstroke_inds_beh_order[0]==order_correct[0]:
                    beh_got_first_stroke = True

            list_beh_sequence_wrong.append(beh_sequence_wrong)
            list_beh_too_short.append(beh_too_short)
            list_beh_got_first_stroke.append(beh_got_first_stroke)

        beh_too_short = all(list_beh_too_short)
        beh_got_first_stroke = any(list_beh_got_first_stroke)
        if any([not x for x in list_beh_sequence_wrong]):
            # Then for at least one parse, the beh is correct up until beh 
            # seq ends, implying failure due to online abort (stroke quality)
            beh_sequence_wrong = False
        else:
            # all are wrong..
            beh_sequence_wrong = True

        # exclude cases where beh was too short, but order was correct
        exclude_because_online_abort = beh_too_short and not beh_sequence_wrong

        if DEBUG:
        # if success_binary==False and beh_sequence_wrong==False:
            print("taskstroke_inds_beh_order:", taskstroke_inds_beh_order)
            print("list_order_correct:", list_order_correct)
            print("success_binary, beh_too_short, beh_got_first_stroke, beh_sequence_wrong:")
            print(success_binary, beh_too_short, beh_got_first_stroke, beh_sequence_wrong)
        
        if "which_probe_blockset" in D.Dat.columns:
            which_probe_blockset = D.Dat.iloc[ind]["which_probe_blockset"]
        else:
            which_probe_blockset = np.nan

        ########## [END] CONSIDER EXCEPTIONs
        rule_actual, _ = D.grammarparses_ruledict_rulestring_extract(ind)
        for rule in list_rules:
            # parses = parsesdict[rule]
            # parses = GD.parses_extract_generated(rule)
            # dfGramScore.at[i, f"behmodpost_{rule}_default"] = taskstroke_inds_beh_order in parses
            # results[-1][f"behmodpost_{rule}_{modelclass}"] = taskstroke_inds_beh_order in parses

            if ONLY_ACTUAL_RULE:
                if not rule==rule_actual:
                    continue

            results.append({
                "agent_kind":"model",
                "agent_rule":rule, 
                "score_name":"binsucc",
                "score":GD._score_beh_in_parses(taskstroke_inds_beh_order, rule),
                # "score":tuple(taskstroke_inds_beh_order) in parses,
                "probe":D.Dat.iloc[ind]["probe"],
                "epoch_superv":D.Dat.iloc[ind]["epoch_superv"],
                "success_binary_quick":success_binary, # success if match any of the rules that are aligned with this epoch.
                "beh_sequence_wrong":beh_sequence_wrong,
                "beh_too_short":beh_too_short,
                "beh_got_first_stroke":beh_got_first_stroke,
                "exclude_because_online_abort":exclude_because_online_abort,
                "epoch":D.Dat.iloc[ind]['epoch'],
                "epoch_orig":D.Dat.iloc[ind]['epoch_orig'],
                "block": D.Dat.iloc[ind]["block"],
                # "datind":ind,
                "trialcode":D.Dat.iloc[ind]["trialcode"],
                "taskgroup":D.Dat.iloc[ind]["taskgroup"],
                "character":D.Dat.iloc[ind]["character"],
                "epochset":D.Dat.iloc[ind]["epochset"] if "epochset" in D.Dat.columns else None,
                "taskfeat_cat":D.Dat.iloc[ind]["taskfeat_cat"] if "taskfeat_cat" in D.Dat.columns else None,
                "superv_SEQUENCE_SUP":D.Dat.iloc[ind]["superv_SEQUENCE_SUP"],
                "which_probe_blockset":which_probe_blockset,
                "parsesdict":GD.parses_extract_generated(rule)
            })

            if "microstim_epoch_code" in D.Dat.columns:
                results[-1]["microstim_epoch_code"] = D.Dat.iloc[ind]["microstim_epoch_code"]

        LIST_success_binary.append(success_binary)
        LIST_beh_sequence_wrong.append(beh_sequence_wrong)
        LIST_beh_too_short.append(beh_too_short)
        LIST_exclude_because_online_abort.append(exclude_because_online_abort)

    dfGramScore = pd.DataFrame(results)

    if USE_DATASET_DF:
        # merge the dataframes
        assert np.all(dfGramScore["trialcode"]==D.Dat["trialcode"])
        columns_keep = [x for x in dfGramScore.columns if (x not in D.Dat or x=="trialcode")]
        dfGramScore = dfGramScore.loc[:, columns_keep]
        dfGramScore = pd.merge(dfGramScore, D.Dat, on="trialcode", suffixes=(None, None)) # suffixes flag means it will throw error if olverapping columns

    # Append things to dfgramscore
    D.Dat["success_binary_quick"] = LIST_success_binary
    D.Dat["beh_sequence_wrong"] = LIST_beh_sequence_wrong
    D.Dat["beh_too_short"] = LIST_beh_too_short
    D.Dat["exclude_because_online_abort"] = LIST_exclude_because_online_abort

    ############# Convert to bm object -- useful method below for colnames
    if return_as_bmh_object:

        from .beh_model_holder import BehModelHolder
        # dict_modelclass_to_rules = {modelclass:list_rules} # leav this hard coded for now. since not working with mult model classes.
        bm = BehModelHolder(dfGramScore)

        # if recompute_success_using_acceptable_scorers:
        #     from pythonlib.tools.pandastools import applyFunctionToAllRows
        #     from pythonlib.dataset.modeling.discrete import _rules_consistent_rulestrings_extract_auto
        #     def F(x):
        #         rulethis = x["epoch"]
        #         if rulethis=="base":
        #             return True
        #         else:
        #             list_rule = _rules_consistent_rulestrings_extract_auto([rulethis], return_as_dict=False)[0]
        #             list_sname = [bm.Map_score_rule_agent_to_colname[("binsucc", rule, "model")] for rule in list_rule]
        #             # list_sname = [bm.Map_rule_to_colname[rule] for rule in list_rule]
        #             if len(list_sname)==0:
        #                 print(rulethis, list_rulestr, list_sname)
        #                 assert False, f"this rule has not acceptable rule scoreres: {rulethis}"
        #             successes = [x[name] for name in list_sname] # list of bool
        #             return any(successes)
        #     bm.DatWide = applyFunctionToAllRows(bm.DatWide, F, "success_binary_parses")

        # def _add_binary_rule_tuple_col(df, rule_cols):
        #     tuple_col_name = 'binary_rule_tuple'
        #     df[tuple_col_name] = ''

        #     # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
        #     for i in range(len(df)):
        #         df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))

        return bm
    else:
        return dfGramScore
        

    # from pythonlib.tools.pandastools import applyFunctionToAllRows
    # cols_to_copy = ["success_binary", "beh_sequence_wrong", "beh_too_short", "exclude_because_online_abort"]
    # for col in cols_to_copy:
    #     list_vals = []
    #     for i in range(len(D.Dat)):
    #         vals = dfGramScore[dfGramScore["datind"] == i][col].tolist()
    #         assert len(vals)==1
    #         list_vals.append(vals[0])
    #     if col in D.Dat.columns:
    #         assert list_vals==D.Dat[col].tolist()
    #     else:
    #         D.Dat[col] = list_vals
    #     print("Added this columnt to D.Dat: ", col)



# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data_matlabrule(D, binary_rule=False, 
        DEBUG=False, return_as_bmh_object=True, remove_repeated_trials=False,
        USE_DATASET_DF=True):
    """High-level extraction for each trial whether it was success given the matlab objectclass
    sequence. Does not try to extract new parses. binary success on each trial
    PARAMS:
    - binary_rule, bool, whether to get for eahc trial a binary code for bollean rules
    - USE_DATASET_DF, then merges D.Dat into output Dataframe.
    """
    from pythonlib.dataset.modeling.discrete import _rules_consistent_rulestrings_extract_auto

    # if "which_probe_blockset" not in D.Dat.columns:
    #     from pythonlib.dataset.dataset_analy.learning import preprocess_dataset as learn_preprocess
    #     learn_preprocess(D, remove_repeated_trials=remove_repeated_trials)

    if False:
        assert False, "merge this with generate_scored_beh_model_data_long"

    D.preprocessGood(params=["remove_baseline"])

    ################## Generate behclass
    D.behclass_preprocess_wrapper()

    results = [] #dataframe to return

    # Collect these things to put back into D.Dat
    LIST_success_binary = []
    LIST_beh_sequence_wrong = []
    LIST_beh_too_short = []
    LIST_exclude_because_online_abort = []
    for ind in range(len(D.Dat)):
        if ind%100==0:
            print("trial #", ind)
        
        # rename value in 'epoch' to rulename, if applicable
        epoch_i = D.Dat.iloc[ind]['epoch']

        # generate all parses for this trial
        gramdict = D.grammarmatlab_extract_beh_and_task(ind)
        taskstroke_inds_beh_order = gramdict["taskstroke_inds_beh_order"]

        # Score this trial as correct/incorrect
        # correct order specified in matlab code
        order_correct = gramdict["taskstroke_inds_correct_order"]
        assert order_correct is not None, "not defined in matlab ObjectClass.. you must recompute, useing grammar_recompute_parses"
        list_order_correct = [order_correct] # only a single correct order
        success_binary = taskstroke_inds_beh_order in list_order_correct

        ########## CONSIDER EXCEPTIONs, i..e, to say didn't actally fail the sequence...
        # note cases where did not complete the trial, but sequence was correct so far (exclude those)
        # NOTE: if multipel correct possible parses, considers each criterion one by one, instread of
        # considering only if any possible parses passes all criteria.. This is OK?
        def beh_good_ignore_length(order_beh, order_correct):
            """ True if beh is good up unitl the time beh fails.
            e.g., if beh is [0, 3, 2], and correct is [0, 3, 2, 1], then
            this is True
            """
            for x, y in zip(order_beh, order_correct):
                if x!=y:
                    return False
            return True

        list_beh_sequence_wrong = []
        list_beh_too_short = []
        list_beh_got_first_stroke = []
        for order_correct in list_order_correct:
            beh_sequence_wrong = not beh_good_ignore_length(taskstroke_inds_beh_order, order_correct)
            beh_too_short = len(taskstroke_inds_beh_order) < len(order_correct)
            beh_got_first_stroke = False # whether the first beh stroke was correct.
            if len(taskstroke_inds_beh_order)>0:
                if taskstroke_inds_beh_order[0]==order_correct[0]:
                    beh_got_first_stroke = True

            list_beh_sequence_wrong.append(beh_sequence_wrong)
            list_beh_too_short.append(beh_too_short)
            list_beh_got_first_stroke.append(beh_got_first_stroke)

        beh_too_short = all(list_beh_too_short)
        beh_got_first_stroke = any(list_beh_got_first_stroke)
        if any([not x for x in list_beh_sequence_wrong]):
            # Then for at least one parse, the beh is correct up until beh 
            # seq ends, implying failure due to online abort (stroke quality)
            beh_sequence_wrong = False
        else:
            # all are wrong..
            beh_sequence_wrong = True

        # exclude cases where beh was too short, but order was correct
        exclude_because_online_abort = beh_too_short and not beh_sequence_wrong

        if "which_probe_blockset" in D.Dat.columns:
            which_probe_blockset = D.Dat.iloc[ind]["which_probe_blockset"]
        else:
            which_probe_blockset = np.nan

        ######## Extract string edit distnace scores.
        from pythonlib.tools.string_edit_dist import nmatch_until_first_diff
        d_nmatch = nmatch_until_first_diff(taskstroke_inds_beh_order, order_correct)

        results.append({
            "agent_kind":"model",
            "agent_rule":epoch_i, 
            "score_name":"binsucc",
            "score":success_binary,
            "success_binary_quick":success_binary,            
            "dist_nmatch_until_diff":d_nmatch,            
            "probe":D.Dat.iloc[ind]["probe"],
            "epoch_superv":D.Dat.iloc[ind]["epoch_superv"],
            "beh_sequence_wrong":beh_sequence_wrong,
            "beh_too_short":beh_too_short,
            "beh_got_first_stroke":beh_got_first_stroke,
            "exclude_because_online_abort":exclude_because_online_abort,
            "epoch":D.Dat.iloc[ind]['epoch'],
            "epoch_orig":D.Dat.iloc[ind]['epoch_orig'],
            "block": D.Dat.iloc[ind]["block"],
            # "datind":ind,
            "trialcode":D.Dat.iloc[ind]["trialcode"],
            "taskgroup":D.Dat.iloc[ind]["taskgroup"],
            "epochset":D.Dat.iloc[ind]["epochset"] if "epochset" in D.Dat.columns else None,
            "taskfeat_cat":D.Dat.iloc[ind]["taskfeat_cat"] if "taskfeat_cat" in D.Dat.columns else None,
            "superv_SEQUENCE_SUP":D.Dat.iloc[ind]["superv_SEQUENCE_SUP"],
            "character":D.Dat.iloc[ind]["character"],
            "which_probe_blockset":which_probe_blockset
        })

        LIST_success_binary.append(success_binary)
        LIST_beh_sequence_wrong.append(beh_sequence_wrong)
        LIST_beh_too_short.append(beh_too_short)
        LIST_exclude_because_online_abort.append(exclude_because_online_abort)

    dfGramScore = pd.DataFrame(results)

    if USE_DATASET_DF:
        # merge the dataframes
        assert np.all(dfGramScore["trialcode"]==D.Dat["trialcode"])
        columns_keep = [x for x in dfGramScore.columns if (x not in D.Dat or x=="trialcode")]
        dfGramScore = dfGramScore.loc[:, columns_keep]
        dfGramScore = pd.merge(dfGramScore, D.Dat, on="trialcode", suffixes=(None, None)) # suffixes flag means it will throw error if olverapping columns

    # Append things to dfgramscore
    D.Dat["success_binary_quick"] = LIST_success_binary
    D.Dat["beh_sequence_wrong"] = LIST_beh_sequence_wrong
    D.Dat["beh_too_short"] = LIST_beh_too_short
    D.Dat["exclude_because_online_abort"] = LIST_exclude_because_online_abort

    ############# Convert to bm object -- useful method below for colnames
    if return_as_bmh_object:

        from .beh_model_holder import BehModelHolder
        # dict_modelclass_to_rules = {modelclass:list_rules} # leav this hard coded for now. since not working with mult model classes.
        bm = BehModelHolder(dfGramScore)

        return bm
    else:
        return dfGramScore