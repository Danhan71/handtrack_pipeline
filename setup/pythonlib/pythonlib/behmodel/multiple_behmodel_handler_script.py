""" For loading pre-trained models,
applying models to new test dataset,
saving
"""

from pythonlib.behmodel.score_dataset import score_dataset
# from pythonlib.behmodel.scorer.prior_functions import prior_feature_extractor
from pythonlib.behmodel.scorer.likeli_functions import likeli_dataset
from pythonlib.behmodel.scorer.poster_functions import poster_dataset
from pythonlib.behmodel.behmodel import BehModel
from pythonlib.behmodel.scorer.scorer import Scorer
from pythonlib.dataset.dataset import Dataset
from pythonlib.dataset.dataset import concatDatasets
from pythonlib.dataset.analy import preprocessDat
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.dataset.beh_model_comparison import plots_cross_prior_and_model_combined, ColNames
from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper_params
from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
from pythonlib.dataset.analy import extract_strokes_monkey_vs_self


###### LOAD DATASETS
# Load datasets
SDIR = "/data2/analyses/main/model_comp/planner"

SUBSAMPLE = False
animal = "Red"
expt = "lines5"
rule_list = ["straight", "bent"]
# rule_list = ["bent"]
LOAD_ALL_PARSES = True
ONLY_SUMMARY_DATES = True

# LOAD ALL DATASETS
Dlist  = []
for rule in rule_list:
    D = Dataset([])
    D.load_dataset_helper(animal, expt, rule=rule)
    
    if SUBSAMPLE:
        D.subsampleTrials(1,2)
        
    D.load_tasks_helper()
    
    D.filterPandas({"random_task":[False], "insummarydates":[ONLY_SUMMARY_DATES]}, "modify")
    
    if LOAD_ALL_PARSES:
        list_parse_params = [
            {"quick":True, "ver":"graphmod", "savenote":"fixed_True"},
            {"quick":True, "ver":"nographmod", "savenote":"fixed_True"}]
        list_suffixes = ["graphmod", "nographmod"]
    else:
        list_parse_params = [
            {"quick":True, "ver":"graphmod", "savenote":"fixed_True"}]
        list_suffixes = ["graphmod"]
    D.parser_load_presaved_parses(list_parse_params, list_suffixes)
    Dlist.append(D)
        
                
# Preprocess
D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(Dlist[0], expt)


############ PREP TEST DATASET
### SPLIT DATASET
DictD = {}

for D, rule in zip(Dlist, rule_list):

    inds_train, inds_val, inds_test = D.splitTrainTestMonkey(expt, epoch=rule)
    
    DictD[(rule, "train")] = D.subsetDataset(inds_train)
    DictD[(rule, "test")] = D.subsetDataset(inds_test)

    print("**", (rule, "train"), "this many:", len(inds_train))
    print("**", (rule, "test"), "this many:", len(inds_test))

## PREPARE DATASET
# get test dataset (combine rules)
Dlist_test = []
for rule in rule_list:
    
    Dthis = DictD[(rule, "test")]
    
    # Clear scores, in case they are gotten before...
#     CN = ColNames(["lines5", "mkvsmk"], ["straight", "bent"])
#     cols_to_drop = CN.colnames_score()
#     cols_to_drop = [col for col in cols_to_drop if col in Dtest.Dat.columns]
#     Dthis.Dat = Dthis.Dat.drop(cols_to_drop, axis=1)

    Dlist_test.append(Dthis)

# Prepare strokes (other monkey) in dataset
extract_strokes_monkey_vs_self(Dlist_test, GROUPING, GROUPING_LEVELS)


############### LOAD PRETRAINED DATA
# LOAD
from pythonlib.behmodel.multiple_behmodel_handler import MultBehModelHandler


SDIR = "/data2/analyses/main/model_comp/planner/pilot-210727_090321/Red-lines5" # (many models)
# Instead, get classes and rules autoatmically.
# list_mclass =  ['mix_features_bd', 'mix_features_bnd', 'mix_features_nd', 'mix_features_tbd', 
# 	'mix_features_tbnd', 'mix_features_tnd', 'mix_features_td']
list_mclass =  ['mix_features_tnd', 'mix_features_td']
for mclass in list_mclass:
	list_mrule = ["straight", "bent"]
	MBH = MultBehModelHandler()
	MBH.load_pretrained_models(SDIR, list_mclass=[mclass], list_mrule=list_mrule)
	list_mclass = MBH.ListMclass

	if False:
		# Add positive control model
		MBH.load_untrained_models(Dlist, "mkvsmk", ["straight", "bent"])
		list_mclass = list(set(list_mclass + ["mkvsmk"]))

		MBH.load_untrained_models(Dlist, "random", ["straight", "bent"])
		list_mclass = list(set(list_mclass + ["random"]))


	################## APPLY MODEL TO DSET
	# 2) Apply models to test set
	MBH.apply_models_to_mult_new_dataset(Dlist_test)


	################### Save extracted likelis, etc.
	savedict = {}
	for key, DH in MBH.DictTestDH.items():
	    D = DH["D"]
	    H = DH["H"]
	    
	    savedict[key] = [D, H.extract_state(get_dataset=False)]
	    
	import pickle
	import os
	pathdir = f"{SDIR}/applied_to_test_data"
	os.makedirs(pathdir, exist_ok=True)
	path = f"{pathdir}/MBH_statedict-{mclass}.pkl"

	with open(path, "wb") as f:
	    pickle.dump(savedict, f)