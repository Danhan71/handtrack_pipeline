"""
chunks, analysis.
Run this after gotten all perms (incluidng best-fit perms)
"""


from pythonlib.behmodel.score_dataset import score_dataset
# from pythonlib.behmodel.scorer.prior_functions import prior_feature_extractor
from pythonlib.behmodel.scorer.likeli_functions import likeli_dataset
from pythonlib.behmodel.scorer.poster_functions import poster_dataset
from pythonlib.behmodel.behmodel import BehModel
from pythonlib.behmodel.scorer.scorer import Scorer
from pythonlib.dataset.dataset import Dataset
from pythonlib.dataset.analy_dlist import concatDatasets
from pythonlib.dataset.analy import preprocessDat
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.dataset.beh_model_comparison import plots_cross_prior_and_model_combined, ColNames
from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper_params
from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
from pythonlib.dataset.analy import extract_strokes_monkey_vs_self




# Load datasets
SDIR = "/data2/analyses/main/model_comp/planner"
PRUNE_PERMS = True
PRUNE_TRIALS_BY_RULE=True
SUBSAMPLE = False

# animal = "Red"
# expt = "lines5"
# rule_list = ["bent"]

animal = "Pancho"
expt = "gridlinecircle"
# rule_list = ["baseline", "linetocircle", "circletoline", "lolli"]
# rule_list = ["linetocircle", "circletoline", "lolli"]
rule_list = ["lolli", "linetocircle", "circletoline"]

LOAD_ALL_PARSES = False
ONLY_SUMMARY_DATES = False
EXTRACT_BEH_ALIGNED_PARSES = False
FIXED = True

from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
import os

from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
dset = f"{animal}-{expt}"
ts = makeTimeStamp()
SDIR = f"/data2/analyses/main/model_comp/planner/pilot-{ts}/{dset}"
os.makedirs(SDIR, exist_ok=True)
print(SDIR)

# LOAD ALL DATASETS
for rule_dset in rule_list:
    D = Dataset([])
    D.load_dataset_helper(animal, expt, rule=rule_dset)
    
    if SUBSAMPLE:
        if FIXED:
            D.subsampleTrials(1,2)
        else:
            D.subsampleTrials(10, 2)
        
    D.load_tasks_helper()
    del D.Dat["Task"]

    if FIXED:
        D.filterPandas({"random_task":[False], "insummarydates":[ONLY_SUMMARY_DATES]}, "modify")
    else:
        D.filterPandas({"insummarydates":[ONLY_SUMMARY_DATES]}, "modify")
    
    if expt in ["lines5"]:
        pathbase = None
        name_ver = "trialcode"
        if LOAD_ALL_PARSES:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":f"fixed_{FIXED}"},
                {"quick":True, "ver":"nographmod", "savenote":f"fixed_{FIXED}"}]
            list_suffixes = ["graphmod", "nographmod"] # the name in dataset
        else:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":f"fixed_{FIXED}"}]
            list_suffixes = ["graphmod"]
    elif expt in ["gridlinecircle"]:
        pathbase = f"/data2/analyses/database/PARSES_GENERAL/{expt}"
        name_ver = "unique_task_name"
        if LOAD_ALL_PARSES:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":""},
                {"quick":True, "ver":"nographmod", "savenote":""}]
            list_suffixes = ["graphmod", "nographmod"] # the name in dataset
        else:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":""}]
            list_suffixes = ["graphmod"]
    else:
        assert False
    
    # gridlinecircle.
    # test are in block > N and not random
    if expt in ["gridlinecircle"]:
        print("MOVE THIS TO PREPROCESS.PY")
        key = "random_task"
        list_train = [True]
        list_test = [False]
        D.analy_reassign_monkeytraintest(key, list_train, list_test)
    else:
        assert False


    D.parser_load_presaved_parses(list_parse_params, list_suffixes, pathbase=pathbase, name_ver=name_ver,
                                 ensure_extracted_beh_aligned_parses=EXTRACT_BEH_ALIGNED_PARSES)
        
    # Preprocess
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt)

    if PRUNE_TRIALS_BY_RULE:
        D = D.parser_prunedataset_bychunkrules(rule_list)

    # Prune to just the best-fit perms
    if PRUNE_PERMS:
        for i in range(len(D.Dat)):
            P = D.parser_get_parser_helper(i)
            P.parses_remove_all_except_bestfit_perms(rule_list)

    if False:
        from pythonlib.dataset.analy_dlist import subsampleTrialsByFixedTask
        [D] = subsampleTrialsByFixedTask([D], 2)

    # Load other arbitrary models
    from pythonlib.behmodel.multiple_behmodel_handler import MultBehModelHandler
    mclass = "chunks"
    MBH = MultBehModelHandler()
    for rule in rule_list:
        MBH.load_untrained_models([D], mclass, [rule])

    # Summarize models
    MBH.print_summary_untrained_models()

    # 2) Apply models to test set
    MBH.apply_models_to_mult_new_dataset([D])

    ## Save extracted likelis, statedict
    savedict = {}
    for key, DH in MBH.DictTestDH.items():
        D = DH["D"]
        H = DH["H"]

        savedict[key] = [D, H.extract_state(get_dataset=False)]

    import pickle
    pathdir = f"{SDIR}/applied_to_test_data"
    import os
    os.makedirs(pathdir, exist_ok=True)
    # path = f"{pathdir}/MBH_statedict-dset_{rule_dset}-mclass_{mclass}-rule_{rule}.pkl"
    path = f"{pathdir}/MBH_statedict-dset_{rule_dset}-mclass_{mclass}.pkl"

    with open(path, "wb") as f:
        pickle.dump(savedict, f)

    print("Saved to:", path)
