""" stores and operates on datasets"""
import pandas as pd
import sys
if sys.version_info[1]<8:
    import pickle5 as pickle
else:
    import pickle
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows
# import torch
import os
from pythonlib.tools.expttools import makeTimeStamp, findPath
from .analy_dlist import mergeTwoDatasets, matchTwoDatasets
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES, PATH_ANALYSIS_OUTCOMES_SERVER
from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.plottools import savefig
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel
from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

base_dir = PATH_ANALYSIS_OUTCOMES
# base_dir = os.path.expanduser("~/data2/analyses")

def _checkPandasIndices(df):
    """ make sure indices are monotonic incresaing by 1, starting from 0
    """
    if len(df)==0:
        # empty is fine.
        return

    assert df.index[0]==0

    if len(df)==1:
        # is fine.
        return

    tmp =  np.unique(np.diff(df.index))
    assert len(tmp)==1
    assert np.unique(np.diff(df.index)) ==1 


def load_dataset_daily_helper(animal, date, rename_shapes_if_cluster_labels_exist=True):
    """Helper, just pass in animal and dat, load the
    single dataset for this day. 
    PARAMS;
    - date, YYMMDD, either str or int
    - rename_shapes_if_cluster_labels_exist, bool, if True, then allows renaming shapes
    by thier cluster labels (usually for char).
    RETURNS:
    - D, dataset.
    """
    date = str(date)
    from pythonlib.dataset.dataset_preprocess.general import extract_expt_metadat
    list_metadat = extract_expt_metadat(animal=animal, rule=str(date))
    if len(list_metadat)==0:
        print(animal, date)
        assert False, "couldnt find this metadat"
    elif len(list_metadat)>1:
        print(animal, date)
        print(list_metadat)
        assert False, "found >1 dataset -- not sure what to do"
    else:
        # Load
        expt = list_metadat[0][0]
        print("Loading this dataset", animal, expt, date)
        D = load_dataset_notdaily_helper(animal, expt, rulelist=[date],
                                         rename_shapes_if_cluster_labels_exist=rename_shapes_if_cluster_labels_exist)
        assert hasattr(D, "TokensStrokesBeh"), "how is this possible? It should have run tokens_generate_replacement_quick_from_beh..."    
        return D

def load_dataset_notdaily_helper(animal, expt, rulelist=None, return_rulelist=False,
                                 rename_shapes_if_cluster_labels_exist=True):
    """
    Helper to load a dataset, using most common methods. Works for both
    daily and main analysis (see PARAMS).
    PARAMS;
    - rulelist, either:
    --- None, (main analys) auto finds all rules
    --- list of YYMMDD str, for daily analyses
    RETURNS:
    - Dataset, with preprocessing and Tasks already extracted.
    - [if return_rulelist], auto extracted, if rulelist is None. otherwise returns the input.
    """

    assert expt is not None, "if you want to get daily, then use load_dataset_daily_helper"
    if rulelist is None:
        # Then find all the rules automatically
        from pythonlib.dataset.dataset_preprocess.general import get_rulelist
        rulelist = get_rulelist(animal, expt)
        assert len(rulelist)>0
    # elif isinstance(rulelist, list) and len(rulelist)==1 and isinstance(rulelist[0], str) and str(int(rulelist[0]))==rulelist[0]:
    #     # Then is dailyy, as rulelist is like ["220321"]
    #     date = rulelist[0]
    #     return load_dataset_daily_helper(animal, date)
    else:
        assert isinstance(rulelist, list)

    D = Dataset([])
    D.load_dataset_helper(animal, expt, ver="mult", rule=rulelist,
                          rename_shapes_if_cluster_labels_exist=rename_shapes_if_cluster_labels_exist)
    assert hasattr(D, "TokensStrokesBeh"), "how is this possible? It should have run tokens_generate_replacement_quick_from_beh..."
    
    if return_rulelist:
        return D, rulelist
    else:
        return D    

class Dataset(object):
    """ 
    """
    def __init__(self, inputs, append_list=None, reloading_saved_state=False,
            remove_dot_strokes=True, remove_online_abort=False):
        """
        Load saved datasets. 
        - inputs, is either:
        --- list of strings, where each string is
        --- Leave this empyt if want to load datsaet in other way
        full path dataset.
        --- list of tuples, where each tuple is (dat, metadat), where
        dat and metadat are both pd.Dataframes, where each wouyld be the object
        that would be loaded if list of paths.
        - append_list, dict, where each key:value pair will be 
        appendeded to datset as new column, value must be list 
        same length as path_list.
        - reloading_saved_state, then doesnt try to cleanup. Note, this doesnt mean
        reloading the probedat-->dataset, but is if reload after saving using Dataset.save...
        """

        # Cleanup params
        self.ParamsCleanup = {
            "remove_dot_strokes":remove_dot_strokes,
            "remove_online_abort":remove_online_abort
        }

        self._reloading_saved_state = reloading_saved_state

        if len(inputs)>0:
            self._main_loader(inputs, append_list)
        # else:
        #     print("Did not load data!!!")


        self._possible_data = ["Dat", "BPL", "SF"]

        # Save common coordinate systems
        # [[xmin, ymin], [xmax, ymax]]
        self._edges = {
        "bpl":np.array([[0, -104], [104, 0]])
        }

        # Initialize things
        self._ParserPathBase = None
        self.LockPreprocess = False # if true, then doesnt allow to modify dataset with...
        self._BehClassExtracted = False
        self.Log_preprocessGood = []
        # For swtiching to Dataset-stored toekns
        self.TokensStrokesBeh = None
        self.TokensStrokesBehUsingTaskStrokes = None
        self.TokensTask = None
        self.TokensVersion = "taskclass"

        self.ML2_FILEDATA = {}

    def initialize_dataset(self, ver, params):
        """ main wrapper for loading datasets of all kinds, including
        monkey, model, etc.
        - ver, {"monkey", "generic"}
        - params, dict of params
        """

        if ver=="monkey":
            animal = params["animal"]
            expt = params["expt"]
            if "ver" in params.keys():
                ver = params["ver"]
            else:
                ver = "single"
            self.load_dataset_helper(animal, expt, ver)
        elif ver=="generic":
            # then give me behaviro and tasks
            strokes_list = params["strokes_list"]
            task_list = params["task_list"]
            self.Dat = pd.DataFrame({
                "strokes_beh":strokes_list,
                "Task":task_list})
        else:
            print(ver)
            assert False

    def load_dataset_helper(self, animal, expt, ver="single", rule="",
                            rename_shapes_if_cluster_labels_exist=True):
        """ load a single dataset. 
        - ver, str
        --- "single", endures that there is one and only one.
        --- "mult", allows multiple. if want animal or expt to 
        - rule, some datasets defined by a "rule". To skip, pass in "" or None
        NOTE: for animal, expt, or rule, can pass in lists of strings. Must use ver="mult"
        MOST COMMON USAGE:
            D = Dataset([])
            D.load_dataset_helper(animal, expt, ver="mult", rule=rulelist), where rulelist is 
            list of strings.
        """

        assert isinstance(expt, str), "if multiple expeirments, then not sure what to do for preprocess below."
        expt_orig = expt

        if rule is None:
            rule = ""

        if isinstance(rule, str):
            pass
        elif isinstance(rule, list):
            for r in rule:
                assert isinstance(r, str)
        else:
            print(rule, type(rule))
            assert False, "wrong type"

        if ver=="single":
            pathlist = self.find_dataset(animal, expt, assert_only_one=True, rule=rule)
            self._main_loader(pathlist, None, animal_expt_rule=[(animal, expt, rule)])
        elif ver=="mult":
            pathlist = []
            aer_list =[]
            if isinstance(animal, str):
                animal = [animal]
            if isinstance(expt, str):
                expt = [expt]
            if isinstance(rule, str):
                rule = [rule]
            for a in animal:
                for e in expt:
                    for r in rule:
                        pathlist.extend(self.find_dataset(a, e, True, rule=r))
                        aer_list.append((a,e,r))
            self._main_loader(pathlist, None, animal_expt_rule=aer_list)
        else:
            assert False

        # By default, try to load tasks
        self.load_tasks_helper()

        if False:
            # By default, do preprocess (these run when load first time, but not if reload cached dataset in neuralmonkey)
            self, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(self, expt_orig)
        else:
            # 2/4/23 - Now run preprocess whenever load.
            GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = self._cleanup_preprocess_each_time_load_dataset(
                rename_shapes_if_cluster_labels_exist=rename_shapes_if_cluster_labels_exist)
            
        assert hasattr(self, "TokensStrokesBeh"), "how is this possible? It should have run tokens_generate_replacement_quick_from_beh..."

        # self._analy_preprocess_done = False
        # self, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(self, expt)
        # print(GROUPING_LEVELS)
        self.MetadatPreprocess = {
             "GROUPING":GROUPING,
             "GROUPING_LEVELS":GROUPING_LEVELS, 
             "FEATURE_NAMES":FEATURE_NAMES, 
             "SCORE_COL_NAMES":SCORE_COL_NAMES}

        # Cleanup


    def _main_loader(self, inputs, append_list, animal_expt_rule=None):
        """ MAIN loading function, use this for all loading purpose.
        Minimal functoins, so that athis can be used for loading data direclty (given inputed)_
        paths, or, with subsequence processing, auto loading datasets semanticalyl.
        - animal_exp MAINt_rule = [aer1, aer2, ..] length of inputs, where aer1 is like (a, e, r)
        -- only applies if input is paths
        """
        
        if self._reloading_saved_state:
            assert len(inputs)==1, "then must reload one saved state."
            self._reloading_saved_state_inputs = inputs

        if isinstance(inputs[0], str):
            # then loading datasets (given path)
            self._load_datasets(inputs, append_list, animal_expt_rule=animal_expt_rule)
        else:
            # then you passed in pandas dataframes directly.
            self._store_dataframes(inputs, append_list)

        if not self._reloading_saved_state:
            self._cleanup(
                remove_dot_strokes = self.ParamsCleanup["remove_dot_strokes"],
                remove_online_abort = self.ParamsCleanup["remove_online_abort"],
                remove_bad_strokes=True,
                smooth_strokes=True,
                )
            self._check_consistency()
        # Ignore this, since its not doing anything important, and will run later.
        # else:
        #     self._cleanup_preprocess_each_time_load_dataset()

    def _store_dataframes(self, inputs, append_list):
        assert append_list is None, "not coded yet!"

        dat_list = []
        metadats = {}

        for i, inthis in enumerate(inputs):
            dat = inthis[0]
            m = inthis[1]

            dat["which_metadat_idx"] = i
            dat_list.append(dat)
            print("Loaded dataset, size:")
            print(len(dat))

            metadats[i] = m
            print("Loaded metadat:")
            print(m)

        self.Dat = pd.concat(dat_list, axis=0)
        if self._reloading_saved_state:
            assert len(metadats)==1
            self.Metadats = metadats[0]
        else:
            self.Metadats = metadats

        # reset index
        print("----")
        print("Resetting index")
        self.Dat = self.Dat.reset_index(drop=True)



    def _load_datasets(self, path_list, append_list, animal_expt_rule=None):
        # INPUTS:
        # - animal_expt_rule, expect this to be list of 3-tuples: (animal, expt, rule) or None.
        # --- len(list) must be length of path_list

        assert append_list is None, "not coded yet!"
        if animal_expt_rule is not None:
            if len(animal_expt_rule) != len(path_list):
                print(path_list)
                print(animal_expt_rule)
                assert False, "should be list of (a,e,r)"
        else:
            animal_expt_rule = [([], [], []) for _ in range(len(path_list))] # just pass in empty things.

        dat_list = []
        metadats = {}
        blockparams_defaults = {} # each datsaet, its default blockparams, a dict. None otherwise
        # indexed by (date, sess, blocknum)

        for i, (path, aer) in enumerate(zip(path_list, animal_expt_rule)):
            
            print("----------------")
            print(f"Currently loading dataset pkl: {path}")
            # Open dataset

            try:            
                with open(f"{path}/dat.pkl", "rb") as f:
                    dat = pickle.load(f)
            except FileNotFoundError:
                with open(f"{path}/Dat.pkl", "rb") as f:
                    dat = pickle.load(f)
            dat["which_metadat_idx"] = i
            dat_list.append(dat)
            print(".. Done!")

            # Open metadat
            try:
                with open(f"{path}/metadat.pkl", "rb") as f:
                    m = pickle.load(f)
            except FileNotFoundError:
                with open(f"{path}/Metadats.pkl", "rb") as f:
                    m = pickle.load(f)
            metadats[i] = m
            print("Loaded metadat:")
            print(m)
            metadats[i]["path"] = path
            metadats[i]["animal"] = aer[0]
            metadats[i]["expt"] = aer[1]
            metadats[i]["rule"] = aer[2]

            # Try loading blockparamds dict
            # (only started saving this 8/13/22)
            path_bp = f"{path}/BlockParamsByDateSessBlock.pkl"
            if os.path.exists(path_bp):
                from pythonlib.tools.monkeylogictools import dict2list2
                
                # Then load
                print("Loading BlockParamsByDateSessBlock!")
                with open(path_bp, "rb") as f:
                    BlockParamsByDateSessBlock = pickle.load(f)

                # clean it up (the monkeylogic dicts have numbers as keys, should be lists)
                tmp = {}
                for k, v in BlockParamsByDateSessBlock.items():
                    # have to iterate becuase k is int, this would be converetd to list.
                    tmp[k] = dict2list2(v)
                BlockParamsByDateSessBlock = tmp

            else:
                print("[Skipping loading] Did not find BlockParamsByDateSessBlock")
                BlockParamsByDateSessBlock = None
            blockparams_defaults[i] = BlockParamsByDateSessBlock

        self.Dat = pd.concat(dat_list, axis=0)
        self.Metadats = metadats
        self.BlockParamsDefaults = blockparams_defaults

        # reset index
        print("----")
        print("Resetting index")
        self.Dat = self.Dat.reset_index(drop=True)

    def sliceDataset(self, inds):
        """ given inds, slices self.Dat.
        inds are same as row num, will ensure so
        RETURNS:
        - self.Dat, modified, index reset
        """
        assert False, "not coded, didnt think necesayr"

    def copy(self, dfdeep=True, just_df=False):
        """ returns a copy. does this by extracting 
        data, and unlocks perprocessing.
        PARAMS:
        - dfdeep, whether to copy self.Dat deep. pandas defualt is True. if
        False, will not be changed even if you subsample the original dataframe..
        It will change though if tyou change the index or data.
        NOTE: copies over all metadat, regardless of whether all
        metadats are used.
        - just_df, then quicker, just gets the df and main things, not all attributes.
        If True, then also gets thigns that start with cap or with _Cap.
        """
        import copy
        from pythonlib.tools.classtools import attributes_get_capitalized_or_underscore_capitalized

        # Get attriburtes start with captial, or _<capital>
        attributes_with_capital = attributes_get_capitalized_or_underscore_capitalized(self)

        Dnew = Dataset([])
        Dnew.Dat = self.Dat.copy(deep=dfdeep)

        if hasattr(self, "GrammarDict"):
            Dnew.GrammarDict = copy.copy(self.GrammarDict)
        else:
            Dnew.GrammarDict = {}

        if hasattr(self, "Metadats"):
            Dnew.Metadats = copy.deepcopy(self.Metadats)
        if hasattr(self, "BPL"):
            Dnew.BPL = copy.deepcopy(self.BPL)
        if hasattr(self, "SF"):
            Dnew.SF = self.SF.copy(deep=dfdeep)
        if hasattr(self, "Parses"):
            Dnew.Parses = copy.deepcopy(self.Parses)
        if hasattr(self, "BlockParamsDefaults"):
            Dnew.BlockParamsDefaults = copy.deepcopy(self.BlockParamsDefaults)

        if not just_df:
            for attr in attributes_with_capital:
                setattr(Dnew, attr, copy.deepcopy(getattr(self, attr)))

        # Unlock, since one goal of copy is speciifacl to do so.

        Dnew.LockPreprocess = False
        return Dnew

    def is_finalized(self):
        """ Was metadata flagged as finalized by me? Returns True or False
        - If there are multiple datasets loaded simultanesuoy, returns True
        only if all are finalized
        """
        outs = []
        for k, v in self.Metadats.items():
            if "metadat_probedat" not in v.keys():
                outs.append(False)
            else:
                outs.append(v["metadat_probedat"]["finalized"])
        return all(outs)

    def is_equal_to(self, other):
        """ return True if self is equal to other, by passing certain diagnostics.
        Is not fool-proof, but works for things that matter.
        Checks:
        - id string
        - trialcodes match
        - columns match.
        """

        if self.identifier_string() != other.identifier_string():
            return False
        if not all(self.Dat.columns == other.Dat.columns):
            return False
        if not all(self.Dat["trialcode"] == self.Dat["trialcode"]):
            return False

        return True



    def filterByTask(self, filtdict, kind="any_shape_in_range", return_ver="dataset"):
        """ uses method in TaskGeneral class. 
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # Make a new column, indicating whether passes filter
        def F(x):
            return x["Task"].filter_by_shapes(filtdict, kind)
        self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")

        # filter
        return self.filterPandas({"tmp":[True]}, return_ver=return_ver)



    def filterPandas(self, filtdict, return_ver = "indices"):
        """
        RETURNS:
        - if return_ver is:
        --- "indices", then returns inds.(self.Dat not modifed,) 
        --- "modify", then modifies self.Dat, and returns Non
        --- "dataframe", then returns new dataframe, doesnt modify self.Dat
        --- "dataset", then copies and returns new dataset, without affecitng sefl.
        """
        from pythonlib.tools.pandastools import filterPandas
        # traintest = ["test"]
        # random_task = [False]
        # filtdict = {"traintest":traintest, "random_task":random_task}

        _checkPandasIndices(self.Dat)
        # print(f"Original length: {len(self.Dat)}")
        if return_ver=="indices":
            return filterPandas(self.Dat, filtdict, return_indices=True)
        elif return_ver=="modify":
            print("self.Dat modified!!")
            self.Dat = filterPandas(self.Dat, filtdict, return_indices=False)
        elif return_ver=="dataframe":
            return filterPandas(self.Dat, filtdict, return_indices=False)
        elif return_ver=="dataset":
            Dnew = self.copy()
            Dnew.Dat = filterPandas(self.Dat, filtdict, return_indices=False)
            return Dnew
        else:
            print(return_ver)
            assert False

    def subsetDataset(self, inds):
        """ returns a copy of self, with self.Dat only keeping inds
        INdices will be reset
        """
        Dnew = self.copy()
        Dnew.Dat = self.Dat.iloc[inds].reset_index(drop=True)
        return Dnew


    def subsetDataframe(self, inds):
        """ prunes self.Dat (dataframe) to only use the given inds,
        This modifies Dataset in place so is different from subsetDataset. resets dataframe indices.
        """
        self.Dat = self.Dat.iloc[inds].reset_index(drop=True)
        print("self.Dat starting legnth: ", len(self.Dat))
        print("Modified self.Dat, keeping only the inputted inds")
        print("self.Dat final legnth: ", len(self.Dat))

    def findPandas(self, col, list_of_vals, reset_index=True):
        """ returns slice of self.Dat, where rows are matched one-to-one to list_of_vals.
        INPUTS:
        - col, str name of col
        - list_of_vals, list of values that pick out the rows.
        OUTPUT:
        - dfout, dataframe same length as list_of_vals
        """
        from pythonlib.tools.pandastools import findPandas
        return findPandas(self.Dat, col, list_of_vals, reset_index=reset_index)


    def index_by_trialcode(self, tc, assert_only_one_match=True):
        """ Return integere indiex into self.Dat which matches this tc
        RETURNS:
        - index into self.Dat. if multiple matches, returns the first
        """

        indices = self.Dat[self.Dat["trialcode"] == tc].index.tolist()
        if assert_only_one_match:
            if not len(indices)==1:
                print(indices)
                print(len(indices))
                assert False, "Trialcode probably doesnt exist..."
        return indices[0]

    def trialcode_tuple_extract_assign(self):
        """
        assign new column in self.Dat, trialcode_tuple, which is (date, sess, trial)
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            d = x["date"]
            s = x["session"]
            t = x["trial"]

            assert x["trialcode"] == f"{d}-{s}-{t}"
            return (d, s, t)
        self.Dat = applyFunctionToAllRows(self.Dat, F, "trialcode_tuple")

    ############### UTILS
    def _strokes_kinds(self):
        """ returns list with names (strings) for the kinds of strokes that
        we have data for, 
        e.g., ["strokes_beh", "strokes_task", ...]
        """
        allkinds = ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines", "parses"]
        return [k for k in allkinds if k in self.Dat.columns]


    def removeTrialsExistAcrossGroupingLevels(self, GROUPING, GROUPING_LEVELS):
        """ only keeps trials that exist across all grouping levels (conditions on character)
        """
        assert False, "deprecated - use prune_min_ntrials_across_higher_levels."
        _checkPandasIndices(self.Dat)

        tasklist = self.Dat["character"].unique().tolist()
        inds_to_remove = []
        for task in tasklist:
            dfthis = self.Dat[self.Dat["character"]==task]

            # check that all grouping levels exist
            grouping_levels_exist = dfthis[GROUPING].unique().tolist()

            if not all([lev in grouping_levels_exist for lev in GROUPING_LEVELS]):
                # Remove all trials for this characters, since not all desired grouping
                # levels exist for this task.
                inds_to_remove.extend(dfthis.index.tolist())

        print("removing these inds (lack trials in at least one grioupuibg level):")
        print(inds_to_remove, "N=", len(inds_to_remove))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0).reset_index(drop=True)


    def removeNans(self, columns=None):
        """ remove rows that have nans for the given columns
        INPUTS:
        - columns, list of column names. 
        --- if None, then uses all columns.
        --- if [], then doesnt do anything
        RETURNS:
        [modifies self.Dat]
        """

        print("--- Removing nans")
        print("start len:", len(self.Dat))

        print("- num names for each col")
        if columns is None:
            tmpcol = self.Dat.columns
        else:
            if len(columns)==0:
                print("not removing nans, since columns=[]")
                return
            tmpcol = columns
        for c in tmpcol:
            print(c, ':', np.sum(self.Dat[c].isna()))

        if columns is None:
            self.Dat = self.Dat.dropna()
        else:
            self.Dat = self.Dat.dropna(subset=columns)
        self.Dat = self.Dat.reset_index(drop=True)
        print("ending len:", len(self.Dat))


    def printNansPerCol(self, columns=None):
        """ PRINT, for each column how many nans
        """
        if columns is None:
            columns = self.Dat.columns
        # inds_bad = []
        for col in columns:
            # print(f"{col} : {sum(self.Dat[col].isna())}")
            print(f"{sum(self.Dat[col].isna())}, {col}")
            # inds_bad.extend(np.where(dfthis[col].isna())[0])
        print(f"Total num rows: {len(self.Dat)}")

    def removeOutlierRows(self, columns, prctile_min_max, df=None):
        """ remove rows that are outliers, based on percentiles for 
        columns of interest.
        INPUT:
        - columns, list of columns, if row is outline for _any_ column,
        will remove it entirely.
        - prctile_min_max, (2,) array, range=[0, 100], indicating min and max
        prctiles to use for calling somthing outlier. e.g., [1 99]
        RETURNS:
        -  [modifies self.Dat] if df is None. always returns df or self.Dat
        """

        REPLACE_DAT = False
        if df is None:
            df = self.Dat
            REPLACE_DAT = True

        print("--- Removing outliers")
        assert len(df)>0, "empty dat.."
        inds_bad = []
        for val in columns:
            limits = np.percentile(df[val], prctile_min_max)
            indsthis = (df[val]<limits[0]) | (df[val]>limits[1])
            inds_bad.extend(np.where(indsthis)[0])

        inds_bad = sorted(set(inds_bad))
        inds_good = [i for i in range(len(df)) if i not in inds_bad]
        print("starting len(self.Dat)", len(df))
        df = df.iloc[inds_good] 
        df = df.reset_index(drop=True)
        print("final len: ", len(df))

        if REPLACE_DAT:
            self.Dat = df

        return df

    def removeOutlierRowsTukey(self, col, niqr = 2, replace_with_nan=False):
        """ remove rows with outliers, based on iqr (tukey method)
        outlers are those either < 25th percentile - 1.5*iqr, or above...
        INPUTS:
        - replace_with_nan, then doesnt remove row, but instead replaces with nan.
        """

        from scipy.stats import iqr

        # Get the upper and lower limits
        x = self.Dat[col]
        lowerlim = np.percentile(x, [25])[0] - niqr*iqr(x)
        upperlim = np.percentile(x, [75])[0] + niqr*iqr(x)
        outliers = (x<lowerlim) | (x>upperlim)

        print("Lower and upper lim for outlier detection, feature=", col)
        print(lowerlim, upperlim)
        print("this many outliers / total")
        print(sum(outliers), "/", len(outliers))

        if False:
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            x.hist(ax=ax, bins=30)
            ax.plot(x, np.ones_like(x), "o")
            ax.axvline(lowerlim)
            ax.axvline(upperlim)

        # x[outliers] = np.nan
        # print(outliers)
        inds_outliers = outliers[outliers==True].index
        if replace_with_nan:
            print("Replacing outliers with nan")
            print(inds_outliers)
            self.Dat[col] = self.Dat[col].mask(outliers)
        else:
            print("Removing outliers")
            print(inds_outliers)
            self.Dat = self.Dat.drop(inds_outliers).reset_index(drop=True)

    ################# BLOCKPARAMS
    # (NOTE: default blockaprams, not hotkey updated)
    def blockparams_extract_single(self, ind):
        """ get the blockparams for this datapt
        Also cleans up the dict (monkeylogic issues)
        """
        idx_bp = self.Dat.iloc[ind]["which_metadat_idx"]
        date = self.Dat.iloc[ind]["date"]
        sess = self.Dat.iloc[ind]["session"]
        block = self.Dat.iloc[ind]["block"]
        trialcode = self.Dat.iloc[ind]["trialcode"]

        for _, bp in self.BlockParamsDefaults.items():
            if bp is None: 
                assert False, "you probably  need to regenerate the dataset (newer code saved Blockparams)"
        trialcodes_included = self.BlockParamsDefaults[idx_bp][(date, sess, block)]["trialcodes_included"]
        blockparams = self.BlockParamsDefaults[idx_bp][(date, sess, block)]["blockparams"]

        assert trialcode in trialcodes_included
        return blockparams

    def blockparams_extract_single_taskparams(self, ind):
        """ Extract the taskparams fro this trail, only after
        ~9/19/22, when moved task params from BlockPrams to TaskParams
        RETURNS:
        - TaskParams, [or BlockParams, if before this date], dict
        """
        import numpy as np

        fields_in_taskparams = ['probes', 'TaskSet', 'tasks', 'task', 'sketchpad', 'task_objectclass']

        BP = self.blockparams_extract_single(ind)
        taskml = self.Dat.iloc[ind]["Task"].extract_as("ml2")

        taskstruct = taskml.Task # matlab struct, as dict
        if "taskparams_index" in taskstruct.keys():
            # Then is new version, extract taskparams
            index = taskstruct["taskparams_index"]
            if isinstance(index, np.ndarray): # [[1.]]
                index = int(index[0][0])
            # if index>1:
            #     print("lenght: ", len(BP["TaskParams"]))
            #     assert False, "confirm length >1"
            return BP["TaskParams"][index-1] # index is 1-index
        else:
            # Then is old version, just extract blockparams in entirety

            # prune to just those fields that are in TaskParams.
            BP = {k:v for k,v in BP.items() if k in fields_in_taskparams}
            return BP

    def blockparams_extract_single_combined_task_and_block(self, ind):
        """ Extracts a single dict combining taskparams and blockparams.
        [obsolete: Makes sure that there are no overlapping keys. stopped
        doing this becuase blockparams_extract_single_taskparams actually extract
        bp if tp doesnt exist (before ~9/19/22), so then would definitely have 
        overlapping keys. and this is not an issue]
        """

        bp = self.blockparams_extract_single(ind)
        tp = self.blockparams_extract_single_taskparams(ind)

        combined_params = {}

        for key, val in bp.items():
            # assert key not in tp.keys() # check no overlapping keys
            combined_params[key] = val

        for key, val in tp.items():
            combined_params[key] = val

        # remove taskparams from blockparams (since contents of taskparams are now flattened into blockaprams)
        combined_params = {k:v for k, v in combined_params.items() if k!="TaskParams"}

        return combined_params


    ############### TASKS
    def load_tasks_helper(self, reinitialize_taskobjgeneral=True, 
            redo_cleanup=True, convert_coords_to_abstract=False, 
            unique_names_post_Sep17=True):
        """ To load tasks in TaskGeneral class format.
        Must have already asved them beforehand
        - Uses default path
        - reinitialize_taskobjgeneral, then reinitializes, which is uiseful if code for
        general taskclas updates.
        - redo_cleanup, useful since some cleanup items require that tasks are already loaded.
        RETURN:
        - self.Dat has new column called Task
        NOTE:
        - works even if multiple rules (i.e, multiple datasets loaded together, each correpsonding to a 
        metadat)
        """
        from pythonlib.tools.expttools import findPath
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if convert_coords_to_abstract==False:
            assert reinitialize_taskobjgeneral==True, "otherwise convert_coords_to_abstract does nothing"
        # assert len(self.Metadats)==1, "only works for single datasets"

        # a = self.animals()
        # e = self.expts()
        # if len(a)>1 or len(e)>1:
        #     assert False, "currently only works if single animal/ext dataset. can modify easily"

        # Pre-extract all tasks across all medatats (i.e., rules)
        TasksDict = {}
        for idx, M in self.Metadats.items():
            
            # Load tasks
            r = M["rule"]
            a = M["animal"]
            e = M["expt"]

            # Find path, load Tasks
            if len(r)>0:
                sdir = f"{base_dir}/database/TASKS_GENERAL/{a}-{e}-{r}-all"
            else:
                sdir = f"{base_dir}/database/TASKS_GENERAL/{a}-{e}-all"

            # Load the tasks
            pathlist = findPath(sdir, [], "Tasks", "pkl")
            if len(pathlist)==0:
                # Try server
                if len(r)>0:
                    sdir = f"{PATH_ANALYSIS_OUTCOMES_SERVER}/database/TASKS_GENERAL/{a}-{e}-{r}-all"
                else:
                    sdir = f"{PATH_ANALYSIS_OUTCOMES_SERVER}/database/TASKS_GENERAL/{a}-{e}-all"
                pathlist = findPath(sdir, [], "Tasks", "pkl")

            if len(pathlist)!=1:
                print(pathlist)
                assert False
            # assert len(pathlist)==1
            print("--- Loading tasks pkl file: ", pathlist[0])
            with open(pathlist[0], "rb") as f:
                Tasks = pickle.load(f)

            # store tasks
            TasksDict[idx] = Tasks

        # Align tasks with dataset
        def _get_task(trialcode, metadat_idx):
            Tasksthis = TasksDict[metadat_idx]
            tmp = [T["Task"] for T in Tasksthis if T["trialcode"]==trialcode]
            if len(tmp)==0:
                assert False, "no found"
            if len(tmp)>1:
                assert False, "too many"
            return tmp[0]

        def F(x):
            trialcode = x["trialcode"]
            metadat_idx = x["which_metadat_idx"]
            T = _get_task(trialcode, metadat_idx)

            # Reinitalize tasks
            if reinitialize_taskobjgeneral:
                from pythonlib.drawmodel.taskgeneral import TaskClass
                taskobj = T.Params["input_params"]
                Tnew = TaskClass()

                # A hack for a single expt (gridlinecircle2), for lollis.
                if "date" in M.keys() and M["date"] <= 210903:
                    old = True
                elif "date" not in M.keys():
                    old = True  
                else:
                    old = False
                auto_hack_if_detects_is_gridlinecircle_lolli = M["expt"]=="gridlinecircle" and old==True
                Tnew.initialize("ml2", taskobj, 
                    convert_coords_to_abstract=convert_coords_to_abstract,
                    auto_hack_if_detects_is_gridlinecircle_lolli=auto_hack_if_detects_is_gridlinecircle_lolli)
                T = Tnew
            return T

        self.Dat = applyFunctionToAllRows(self.Dat, F, "Task")      
        print("added new column self.Dat[Task]")  

        if redo_cleanup:
            self._cleanup_using_tasks(unique_names_post_Sep17=unique_names_post_Sep17)

    def _task_hash(self, Task, use_objects = False, 
        original_ver_before_sep21=True):
        if use_objects:
            assert False, "not coded"

        if original_ver_before_sep21:
            # Original (actually 2nd to original) ver, before 9/17/21. Is good, but
            # problem is random tasks have tasknum in name, which means might not compare
            # well across tasknums that are actually same task. (since randomly generated
            # tasknums can be arbitrary). Solution, increase hash length, and dont have 
            # taskstrings
            ndigs = 6
            return Task._get_number_hash(ndigs=ndigs, include_taskstrings=True)
        else:
            return Task.get_unique_name()
            # if random_task:
            #     return Task.get_number_hash(ndigs=10, include_taskstrings=False,
            #         include_taskcat_only=True, compact=True)

            # else:
            #     # Thiese two lines are identical. This is idenitcal to before Sep 17
            #     return Task.get_number_hash(ndigs=6, include_taskstrings=True)
            #     # return self.Dat.iloc[ind]["unique_task_name"]


    # def task_hash(self, ind, use_objects = False, 
    #     original_ver_before_sep21=True):
    #     """ Return task hash, differeing depending in smart way on onctext
    #     INPUT:
    #     - either ind or Task
    #     RETURNS: 
    #     - hashable identifier that should apply across all datsaets.
    #     --- if task is Fixed, then returns the unique_task_name
    #     --- if task is random, then returns a 10-digit hash.
    #     --- if use_objects, then overwrites others, and returns a FrozenSet with
    #     the objects (and affine features). This only works after ML2 using objects, so like
    #     July 2021 +. SO: only change is for random tasks. Fixed tasks have exact same name as 
    #     before.
    #     """

    #     Task = self.Dat.iloc[ind]["Task"]
    #     return self._task_hash(Task, use_objects = use_objects, 
    #         original_ver_before_sep21=original_ver_before_sep21)

    def save_generate_string_animal_dates(self):
        """ Generate a stroing useful for summarizing
        this datset, epsecialy if is acros aniamsl and dates,
        RETURNS:
        - string, like Diego_221103_221103-Pancho_220531_220630,
        with deates being first and last dates.
        """

        s = ""
        list_animals = sorted(self.animals())
        i=0
        for an in list_animals:
            dates = sorted(self.Dat[self.Dat["animal"]==an]["date"].unique())

            d1 = min(dates)
            d2 = max(dates)
            if i>0:
                pref = "-"
            else:
                pref = ""
            s += f"{pref}{an}_{d1}_{d2}"
            i+=1
        return s

    def save_task_for_dragmonkey(self, indtrial, 
        SDIR = "/data2/analyses/main/resaved_tasks_for_matlab"):
        """ Save the task in a format that can be loaded in dragmonkey (matlab)
        to generate tasks for experiments
        """
        from pythonlib.tools.matlabtools import convert_to_npobject
        from scipy.io import savemat

        # 1) Extract the task and save
        T = self.Dat.iloc[indtrial]["Task"]
        idstring = self.save_generate_string_identifier_wrapper(concise=False)
        if len(idstring)>35:
            # get concise
            idstring = self.save_generate_string_identifier_wrapper(concise=True)
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        fname = f"trial_{indtrial}-trialcode_{trialcode}"
        T.save_task_for_dragmonkey(subdirname=idstring, fname=fname)

        return SDIR, idstring

    def taskcharacter_print_score_per_char(self, scorename, sdir):
        """ Printa nd save text sorting characters by some score and
        each line being acharacter...
        PARAMS:
        - scorename, str, column in self.Dat, scalar values that will
         sort chars, e.g, "strokes_clust_score"
        """
        from pythonlib.tools.expttools import writeStringsToFile

        def _save_chars_score_to_text(list_char, list_score, suffix):
            # get sample sizes
            list_n = [sum(self.Dat["character"]==char) for char in list_char]
            # strings
            list_s = []
            for char, score, n in zip(list_char, list_score, list_n):
                list_s.append(f"{score:.3f}   (n = {n})   {char}")

            fname = f"{sdir}/char-sorted_by_{scorename}-{suffix}"
            writeStringsToFile(fname, list_s)

            # also write a list that you can easily copy and paste. 
            los_setnames = [" ".join([self.taskcharacter_fixed_set_los(c)[0] for c in list_char])]
            fname = f"{sdir}/char-sorted_by_{scorename}-{suffix}-los_setnames"
            writeStringsToFile(fname, los_setnames)

            los_setnums = [" ".join([self.taskcharacter_fixed_set_los(c)[1] for c in list_char])]
            fname = f"{sdir}/char-sorted_by_{scorename}-{suffix}-los_setnums"
            writeStringsToFile(fname, los_setnums)

            los_inds = [" ".join([self.taskcharacter_fixed_set_los(c)[2] for c in list_char])]
            fname = f"{sdir}/char-sorted_by_{scorename}-{suffix}-los_setinds"
            writeStringsToFile(fname, los_inds)

        # Extract chars and scores
        list_char, list_score, list_n = self.taskcharacter_find_plot_sorted_by_score(scorename)

        # 1) Save all characters
        _save_chars_score_to_text(list_char, list_score, "ALL")

        # 2) Split into each category
        for cat in self.Dat["task_stagecategory"].unique():
            # get all chars that fall in this category
            chars = self.Dat[self.Dat["task_stagecategory"]==cat]["character"].unique().tolist()

            tmp = [(c,s) for c,s in zip(list_char, list_score) if c in chars]
            chars_this = [x[0] for x in tmp]
            scores_this = [x[1] for x in tmp]

            _save_chars_score_to_text(chars_this, scores_this, f"__task_stagecategory_{cat}")

            figbeh, figtask, _, _ = self.taskcharacter_plot_examples(chars_this,
                                                                            titles=scores_this)
            savefig(figbeh, f"{sdir}/task_stagecategory_{cat}-drawings_sorted_byscore-beh.pdf")
            savefig(figtask, f"{sdir}/task_stagecategory_{cat}-drawings_sorted_byscore-task.pdf")
            plt.close("all")


    def taskcharacter_plot_examples(self, list_char, titles=None,
                                    nmax=60, plot_which="best_worst"):
        """ Plot a gridplot, one example for each char, in the
        order they are passed in.
        PARAMS:
        - nmax, if more than this then plots the first nmax/2 and last nmax/2
        """
        inds, list_char = self.taskcharacter_extract_examples(list_char)

        if len(inds)>nmax:
            idxs = list(range(len(list_char)))
            assert len(idxs)==len(inds)
            if plot_which=="best_worst":
                # take from both ends
                n = int(np.floor(nmax/2))
                idxs = idxs[:n] + idxs[-n:]
                # inds = inds[:n] + inds[-n:]
            elif plot_which=="best":
                # take best
                idxs = idxs[:nmax]
            elif plot_which=="worst":
                idxs = idxs[-nmax:]
            else:
                print(plot_which)
                assert False
            # print(inds)
            # print(len(inds), len(list_char))
            inds = [inds[i] for i in idxs]
            list_char = [list_char[i] for i in idxs]
            titles = [titles[i] for i in idxs]

        figbeh, _, _ = self.plotMultTrials2(inds, which_strokes="strokes_beh",
                                               titles=list_char)
        figtask, _, _ = self.plotMultTrials2(inds, which_strokes="strokes_task",
                                               titles=titles)
        return figbeh, figtask, inds, list_char

    def taskcharacter_extract_examples(self, list_char=None, n=1,
                                       var="character"):
        """ Extract an example index for each of the chars in list_char
        RETURNS:
            - lsit of indices into self.Dat
            - matching list of char names,
        """
        from pythonlib.tools.pandastools import extract_trials_spanning_variable
        if list_char is None:
            list_char = self.Dat[var].unique().tolist()

        inds, chars = extract_trials_spanning_variable(self.Dat, var, list_char,
                                                       n_examples=n)
        return inds, chars

    def taskcharacter_find_plot_sorted_by_score(self, scorename, plot=False,
                                                sdir=None, n_iter=3, nmax=60,
                                                path_prefix=None):
        """ Get list of characters sorted by their avreage score across trials.
        PARAMS:
        - scorename, str name of score to use. 
        - plot, whether to plot drawings, sinlge examples, in a grid, sorted
        by score.
        RETURNS"
        - list_char, list of characters, sorted in decresing order of score.
        - list_score, list of num, scores matching the char. 
        """

        list_char = self.Dat["character"].unique().tolist()
        list_score = []
        list_n = []
        for char in list_char:
            sc = np.mean(self.Dat[self.Dat["character"]==char][scorename])
            list_score.append(sc)
            list_n.append(sum(self.Dat["character"]==char))

        list_char, list_score, list_n = self._taskcharacter_find_plot_sorted_by_score(
            list_char, list_score, plot, sdir, n_iter, nmax, path_prefix)

        return list_char, list_score, list_n
        # # sort
        # tmp = [(ch, sc, n) for ch, sc, n in zip(list_char, list_score, list_n)]
        # tmp = sorted(tmp, key=lambda x:-x[1])
        # list_char = [x[0] for x in tmp]
        # list_score = [x[1] for x in tmp]
        # list_n = [x[2] for x in tmp]
        #
        # if plot:
        #     assert sdir is not None
        #     # Plot
        #     # -- get one trial for each char
        #     from pythonlib.tools.pandastools import extract_trials_spanning_variable
        #     for i in range(n_iter):
        #         print("Plotting...", i)
        #         figbeh, figtask, _, _ = self.taskcharacter_plot_examples(list_char,
        #                                                                         titles=list_score,
        #                                                                  nmax=nmax)
        #         if path_prefix is not None:
        #             print("saving to:", f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-beh.pdf")
        #             savefig(figbeh, f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-beh.pdf")
        #             savefig(figtask, f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-task.pdf")
        #         else:
        #             print("saving to:", f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
        #             savefig(figbeh, f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
        #             savefig(figtask, f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")
        #         plt.close("all")
        #

    def _taskcharacter_find_plot_sorted_by_score(self, list_char, list_score, plot=False,
                                                sdir=None, n_iter=3, nmax=60,
                                                path_prefix=None, plot_which="best_worst"):
        """ Get list of characters sorted by their avreage score across trials.
        PARAMS:
        - scorename, str name of score to use.
        - plot, whether to plot drawings, sinlge examples, in a grid, sorted
        by score.
        RETURNS"
        - list_char, list of characters, sorted in decresing order of score.
        - list_score, list of num, scores matching the char.
        """

        list_n = []
        for char in list_char:
            list_n.append(sum(self.Dat["character"]==char))

        # sort
        tmp = [(ch, sc, n) for ch, sc, n in zip(list_char, list_score, list_n)]
        tmp = sorted(tmp, key=lambda x:-x[1])
        list_char = [x[0] for x in tmp]
        list_score = [x[1] for x in tmp]
        list_n = [x[2] for x in tmp]

        if plot:
            assert sdir is not None
            # Plot
            # -- get one trial for each char
            from pythonlib.tools.pandastools import extract_trials_spanning_variable
            for i in range(n_iter):
                print("Plotting...", i)
                figbeh, figtask, _, _ = self.taskcharacter_plot_examples(list_char,
                                                                                titles=list_score,
                                                                         nmax=nmax, plot_which=plot_which)
                if path_prefix is not None:
                    print("saving to:", f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-beh.pdf")
                    savefig(figbeh, f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-beh.pdf")
                    savefig(figtask, f"{sdir}/{path_prefix}-drawings_sorted_byscore-iter{i}-task.pdf")
                else:
                    print("saving to:", f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
                    savefig(figbeh, f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
                    savefig(figtask, f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")
                plt.close("all")

        return list_char, list_score, list_n

    def taskgroup_reassign_ignoring_whether_is_probe(self, CLASSIFY_PROBE_DETAILED=True, PRINT=False):
        """ By default taskgroiups only categorize propbe tasks. 
        Run this to replace "taskgroup" with recoputation of taskgroups, considering all tasks,.
        taskgrops will not have "-P" as suffix. Uses same methods as deefault, but just use all tasks.
        PARAMS:
        - CLASSIFY_PROBE_DETAILED, if True, then appends suffixes indicating ways in which probes are (e..g, extrpolation).
        RETURNS:
        - replaces self.Dat["taskgroup"] in place.
        """
        from pythonlib.dataset.dataset_preprocess.probes import compute_features_each_probe, taskgroups_assign_each_probe
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if PRINT:
            print("- Beginning taskgroups:")
            print(self.Dat["taskgroup"].value_counts())
        map_char_to_tg = taskgroups_assign_each_probe(self, False, CLASSIFY_PROBE_DETAILED=CLASSIFY_PROBE_DETAILED)
        def F(x):
            return map_char_to_tg[x["character"]]
        self.Dat = applyFunctionToAllRows(self.Dat, F, "taskgroup")
        if PRINT:
            print("- Ending taskgroups (after reassign):")
            print(self.Dat["taskgroup"].value_counts())

    def taskgroup_char_ntrials_print_save(self, sdir=None, fname="taskgroup_char_trial"):
        """ Print all the existing tasks in each taskgroup, and thier n trials,
        and save to text file, if sdir is not None, will call file:
        f"{sdir}/{fname}.txt"
        """
        from pythonlib.tools.expttools import writeStringsToFile

        # 1) collect n trials and strings lines.
        groupdict = self.grouping_get_inner_items("taskgroup", "character")
        list_str = []
        for key, vals in groupdict.items():
            print(key)
            list_str.append(key)
            for char in vals:
                n = sum(self.Dat["character"]==char)
                print(f" - {char}  -  {n} trials")
                list_str.append(f" - {char}  -  {n} trials")

        if sdir is not None:
            fname = f"{sdir}/{fname}.txt"
            writeStringsToFile(fname, list_str)
            print("Saved to: ", fname)


    def taskcharacter_fixed_set_los(self, char):
        """
        e.g., if char="singleprims-34-11-846132", returns
        ['singleprims', '34', '11', '846132']
        """
        from pythonlib.tools.stringtools import decompose_string
        return decompose_string(char)

    def taskcharacter_find(self, setname, setnum, index):
        """ Get all indices that have this fixed task name
        <setname>-<setnum>-<index>-<hash>
        PARAMS:
        - setname, str
        - setnum, int
        - index, int
        RETURNS;
        - inds, indices into self.Dat
        - characters, list of string names
        """
        assert not isinstance(index, list)
        keyword = f"{setname}-{setnum}-{index}"
        
        inds = [i for i, char in enumerate(self.Dat["character"]) if keyword in char]
        characters = self.Dat.iloc[inds]["character"]

        return inds, characters

    ########### WORKING WITH RAW ML2 (monkeylogic) data
    def ml2_extract_raw(self):
        """ Extract fd for each datapt in self.Dat, and store in 
        self.ML2_FILEDATA[(animal, date, expt, sess)]
        """
        # Extract filedata for all trials
        from neuralmonkey.utils.monkeylogic import loadSingleDataQuick

        if len(self.ML2_FILEDATA)==0:
            self.ML2_FILEDATA_ads_to_key = {}
            for ind in range(len(self.Dat)):    
                animal = self.Dat.iloc[ind]["animal"]
                date = self.Dat.iloc[ind]["date"]
                sess = self.Dat.iloc[ind]["session"]
                # tc = self.Dat.iloc[ind]["trialcode"]
                # expt = self.Dat.iloc[ind]["expt"] # NO! this is not ml2 expt. its datraset expt.
                # try all the expts
                mdidx = self.Dat.iloc[ind]["which_metadat_idx"]
                # exptnames_ml2 = self.Metadats[mdidx]["exptnames"]
                exptnames_ml2 = self.Metadats[mdidx]["metadat_probedat"]["exptnames"]
                for expt in exptnames_ml2:
                    key = (animal, date, expt, sess)
                    if key not in self.ML2_FILEDATA.keys():
                        print("Loading fd for: ", key)

                        try:
                            fd = loadSingleDataQuick(animal, date, expt, sess)
                        except AssertionError as err:
                            # Then didnt find this data. is fine. will error later if this
                            # is not becuyase of that.
                            continue

                        self.ML2_FILEDATA[key] = fd  

                        # Map from trial to key
                        self.ML2_FILEDATA_ads_to_key[(animal, date, sess)] = key

        print("Found these data pkl files:")
        for k in self.ML2_FILEDATA.keys():
            print(k)

    def _ml2_extract_fd_trial(self, ind):
        """ Return fd and trial for this ind in self.Dat
        RETURNS:
        - fd, trial_ml2
        """      
        if False:  
            animal = self.Dat.iloc[ind]["animal"]
            date = self.Dat.iloc[ind]["date"]
            expt = self.Dat.iloc[ind]["expt"]
            sess = self.Dat.iloc[ind]["session"]
            key = (animal, date, expt, sess)
        else:
            animal = self.Dat.iloc[ind]["animal"]
            date = self.Dat.iloc[ind]["date"]
            sess = self.Dat.iloc[ind]["session"]
            key = self.ML2_FILEDATA_ads_to_key[(animal, date, sess)]
        fd = self.ML2_FILEDATA[key]
        trial_ml2 = self.Dat.iloc[ind]["trial"]
        return fd, trial_ml2


    def ml2_utils_getTrialsBehCodes(self, ind, codes_keep=None,
        keep_only_codes_standard=False, PRINT = False):
        """
        Get behcodes
        RETURNS:
        - behcodes_num, list of ints
        - behcodes_time, list of times, in sec, matching the nums.
        """
        import neuralmonkey.utils.monkeylogic as ml

        fd, trial_ml2 = self._ml2_extract_fd_trial(ind)
        tmp = ml.getTrialsBehCodes(fd, trial_ml2)
        behcodes_num = tmp["num"]
        behcodes_time = tmp["time"]

        assert np.all(np.diff(behcodes_time)>0)

        if keep_only_codes_standard:
            assert codes_keep is None
            codes_keep = [
                11, # fix cue on
                16, # touch fix,
                132, # rulecue 2
            #     91, 92, 21, # guide (default)
                91, # guide (default)
            #     93, 94, 22, # guide (flipped)
                22, # guide (flipped)
                71, # go
            ]

        if codes_keep is not None:
            behcodes_num_tmp = []
            behcodes_time_tmp = []
            for num, time in zip(behcodes_num, behcodes_time):
                if num in codes_keep:
                    behcodes_num_tmp.append(num)
                    behcodes_time_tmp.append(time)
            behcodes_num = behcodes_num_tmp
            behcodes_time = behcodes_time_tmp

        if PRINT:
            for num, time in zip(behcodes_num, behcodes_time):
                print(num, " -- ", time)

        return behcodes_num, behcodes_time

    # def ml2_behcode_get_occurances_ordered(self, ind, behcodes_num, behcodes_time, 
    #         codes_print=None, PRINT=False):
    #     """Return list of occurances of behcodes, keeping only those in codes_print, returning
    #     in chronological order. 
    #     RETURNS:
    #     - list_nums, list of ints, the 
    #     """

    #     if codes_print is None:
    #         # Hard codede times of behcodes that are relevant for determining that order has 
    #         # correctly been flipped.
    #         codes_print = [
    #             11, # fix cue on
    #             16, # touch fix,
    #             132, # rulecue 2
    #         #     91, 92, 21, # guide (default)
    #             91, # guide (default)
    #         #     93, 94, 22, # guide (flipped)
    #             22, # guide (flipped)
    #             71, # go
    #         ]

    #     behcodes_num, behcodes_time = self.ml2_utils_getTrialsBehCodes(ind)
    #     assert np.all(np.diff(behcodes_time)>0)

    #     list_nums = []
    #     for num, time in zip(behcodes_num, behcodes_time):
    #         if num in codes_print:
    #             if PRINT:
    #                 print(num, " -- ", time)
    #             list_nums.append(num)

    #     return list_nums

    # def ml2_run_utils_fn(self, fn_str, ind):
    #     """ Execute a call to function called fn_str in drawmonkey
    #     repo (utils). It's signature must be taking (fd, trial, ...)
    #     """
    #     import neuralmonkey.utils.monkeylogic as ml

    #     tmp = ml.getTrialsBehCodes(fd, trial_ml2)

    ############ WORKING WITH TASKS
    def nstrokes_task_extract(self):
        """ For each row, append column that is "strokes_task" the 
        ground truth n strokes
        """
        list_n = []
        for ind in range(len(self.Dat)):
            n = len(self.Dat.iloc[ind]["strokes_task"])
            list_n.append(n)
        self.Dat["nstrokes_task"] = list_n
        print("New column: nstrokes_task")
    
    def taskclass_is_new_version(self, ind):
        """ Returns True if this trial; uses the new "PLanclass"
        versipn of tasks
        PARAMS:
        - ind, index into self.Dat
        RETURNS:
        - bool.
        """
        T = self.Dat.iloc[ind]["Task"]
        return T.get_is_new_version()

    def taskclass_extract_ml2(self, ind):
        """ Extract drawmodel (ml2) taskclass, which is lower level 
        than TaskGeneral class
        """
        T = self.Dat.iloc[ind]["Task"]
        TT = T.Params["input_params"]
        return TT
    
    def taskclass_extract_low_level_plan(self, ind):
        """ REturn the plan inoputed to generate this task, i.e,, 
        {prim, rel, prim, rel..} in matlab
        """

        # Find the plan
        T = self.Dat.iloc[ind]["Task"]
        Tt = T.extract_monkeylogic_ml2_task()
        plan = Tt.planclass_inputed_plan_extract(0)
        return plan

    def taskclass_extract_planclass(self, ind):
        """
        Return Task.Plan, custom matlab class, here
        is a dict.
        """
        T = self.Dat.iloc[ind]["Task"]
        return T.PlanDat

    def taskclass_extract_objectclass(self, ind):
        """ Return the Objectclass for this trial ind
        """
        TT = self.taskclass_extract_ml2(ind)
        TT.objectclass_extract_all()
        return TT.ObjectClass

    def taskclass_extract_tasksequencer_params(self, ind):
        """ 
        REturtn tasksequencer params used in ml2 to conditions the sequencing for this task.
        RETURNS:
        - sequence_ver, str name of type of sequ, e.g, 'directionv2'
        - seuqence_params, list of params for this ver,e .g, [['left'], 'default', array(1.)]
        """
        T = self.Dat.iloc[ind]["Task"]
        return T.ml2_tasksequencer_params_extract()

    def taskclass_extract_tstruct_index(self, ind):
        """ Return the tstruct index, where the tstruct is 
        the cell array of task sets defined in trainGetterv2 in matlab 
        code. Only works after 10/11/22, when I started saving this in 
        matlab dragmonkey.
        RETURNS:
        - [FILL IN]
        """
        assert False, "Check outputs type, see if working"
        TT = self.taskclass_extract_ml2(ind)
        if "tstruct_index" in TT.get_tasknew()["Task"]["info"].keys():
            return TT.get_tasknew()["Task"]["info"]["tstruct_index"]
        else:
            return None

    def taskclass_extract_los_info(self, ind):
        """
        REturn the "load_old_set" information, i.e,, the fixed tasks info
        RETURNS:
        - (setname, setid, taskind_within_set)
        """

        T = self.Dat.iloc[ind]["Task"]
        out = T.get_los_id()
        if out is None:
            return None, None, None
        else:
            return out

    def taskclass_find_repeat_trials(self, print_trials=False):
        """ Find trials that are immediate repeats of the same task (character).
        ie..,, if do char1, char2, char2, char3, then the third trial (char2) will
        be returned. Only considered to be repeats if have identical epoch, supervision, 
        and block num (and character, of course).
        PARAMS:
        - print_trials, bool, to print info for each trial.
        RETURNS:
        - inds, indices of repeated trials
        """

        # Append column with epoch and block for each trial
        self.grouping_append_col(["epoch_superv", "block"], "dummy")

        ct = 0
        char_previous = ""
        prms_previous = ""
        inds_repeated_not_first = []
        for i in range(len(self.Dat)):
            
            char = self.Dat.iloc[i]["character"]
            err = self.Dat.iloc[i]["ErrorCode"]
            rand = self.Dat.iloc[i]["random_task"]
            prms = self.Dat.iloc[i]["dummy"]

            # if char == char_previous:
            if char == char_previous and prms==prms_previous:
                # Then this is repeat
                ct +=1
            else:
                ct = 0
            char_previous = char
            prms_previous = prms
            
            if ct>0 and rand==False:
                # Then this is a repeat (trial 2 +). exclude it
                inds_repeated_not_first.append(i)
            
            if print_trials:
                print(i, "--", ct, "--", char, "--", prms, "--", err, "--", rand)
        return inds_repeated_not_first        

    def taskclass_extract_los_info_append_col(self):
        """ For each trial, extract its los info and append as 
        new col in D.Dat, called "los_info"
        """
        tmp = []
        for ind in range(len(self.Dat)):
            tmp.append(self.taskclass_extract_los_info(ind))
        self.Dat["los_info"] = tmp
        print("Appended column: los_info")

        # i

        # tmp_sorted = sorted(set(tmp))
        # for x in tmp_sorted:
        #     print(x)        

    def taskclass_gridsize_assign_column(self):
        """ Extract gridsize (string) and assign to new column in self.Dat
        caled "gridsize"
        """
        # Extract gridsize
        list_gridsize = []
        for ind in range(len(self.Dat)):
            T = self.Dat.iloc[ind]["Task"]
            gridsize = T.PlanDat["TaskGridClass"]["Gridname"]
            list_gridsize.append(gridsize)
        self.Dat["gridsize"] = list_gridsize
        

    def taskclass_get_grid_xy_over_all_tasks(self):
        """ return gridxy that is union over all tasks.
        RETURNS:
        - gridx, np array of scalars, sorted increasing
        - gridy, np array of scalars, sorted increasing
        """
        # Get grid across all tasks.
        # Generate new grid based on all tasks in dataset
        xs = np.array([])
        ys = np.array([])
        for ind in range(len(self.Dat)):
            T = self.Dat.iloc[ind]["Task"]
            xgrid, ygrid, grid_ver = T.get_grid_xy()
            if grid_ver=="on_grid":
                xs = np.append(xs, xgrid)
                ys = np.append(ys, ygrid)
                
        xgrid = np.sort(np.unique(xs.round(decimals=3)))
        ygrid = np.sort(np.unique(ys.round(decimals=3)))

        try:
            assert len(xgrid)<12, "this is weird, numerical precision?"
            assert len(ygrid)<12, "this is weird, numerical precision?"
            assert np.all(np.diff(xgrid)>0.01), "weird, numerical precision?"
            assert np.all(np.diff(ygrid)>0.01)    
        except AssertionError as err:
            print(xgrid)
            print(ygrid)
            print(xs)
            print(ys)
            print(np.diff(xgrid))
            print(np.diff(ygrid))
            raise err

        return xgrid, ygrid    

    def taskclass_get_grid_ver(self, ind):
        """ return string naem of grid ("on_rel" or "on_grid"),
        forcing that if this is "character",
        then it cant be on grid.
        """
        if self.Dat.iloc[ind]["task_kind"] == "character":
            # Then is not grid
            return "on_rel"
        else:
            T = self.Dat.iloc[ind]["Task"]
            return T.get_grid_ver()

    def taskclass_tokens_sanitycheck_gridloc_identical(self):
        """ Check that each tasks gridloc:loc mapping is the same
        This can fail if different datasets with different TSC, since
        gridloc is defined relative to all tasks within a dataset
        RETURNS:
        - throws error if fails
        """

        # VERY HACKY, skip days with "align to onset"
        if self.animals(True)[0]=="Pancho" and int(self.dates(True)[0])==220719:
            print("SKIPPING taskclass_tokens_sanitycheck_gridloc_identical!")
            print(".. because this day aligned to onset, so gridloc:loc mapping is not same across tasks")
            return

        # Sanity check that all grids are aligned across tasks
        map_gridloc_loc_x = {}
        map_gridloc_loc_y = {}
        for ind in range(len(self.Dat)):
            tokens = self.taskclass_tokens_extract_wrapper(ind, "task")
            # print(ind, len(tokens))
            for tok in tokens:
                if tok["gridloc"] is not None and not tok["gridloc"]==("IGN", "IGN"):
                    x = tok["Prim"].extract_as("params")["cnr_x"]
                    y = tok["Prim"].extract_as("params")["cnr_y"]
                    # chars are None
                    xgrid = tok["gridloc"][0]
                    ygrid = tok["gridloc"][1]
                    
                    if xgrid in map_gridloc_loc_x.keys():
                        if not np.isclose(map_gridloc_loc_x[xgrid], x):
                            print(map_gridloc_loc_x)
                            print(xgrid)
                            print(x)
                            assert False, "maye this is becuase tasks are aligned at onset? same gridloc, but different center"
                    else:
                        map_gridloc_loc_x[xgrid] = x
                        
                    if ygrid in map_gridloc_loc_y.keys():
                        assert np.isclose(map_gridloc_loc_y[ygrid], y)
                    else:
                        map_gridloc_loc_y[ygrid] = y
            
        print("Success! all gridloc identical!")     
        print("These are the x and y mappings, gridloc:loc")
        print("x...", map_gridloc_loc_x)
        print("y...", map_gridloc_loc_y)   

    def tokens_preprocess_wrapper_good(self, PLOT=False, label_as_novel_if_shape_semantic_fails=False):
        """
        Wrapper for preprocessing --> Given just-generated tokens (using tokens_append_to_dataframe_column) here do all
        processing steps, e.g, derived features, like storke onset, and binned/clustered fgeatures, like onset binned.
        :return: Modifies tokens and appends columns with "seqc_" to dataset.

        NOTE: for extracting semantic shape, this calls it "NOVEL" for any trial that has an extra tform in ML2
        Plan class, which means it doesnt have well-defined shape. Problem is this currently checks entire trial, so if any
        single prim is rotated, then this might call all of them NOVEL. Solution: incorporate code to detect "tforms_each_prim"
        """

        # - and get touch onset binned.
        _, fig_final = self.tokens_cluster_touch_onset_loc_across_all_data(PLOT_FINAL=True)
        sdir = self.make_savedir_for_analysis_figures_BETTER("preprocess_general")
        fig_final.savefig(f"{sdir}/tokens_cluster_touch_onset_loc_across_all_data.pdf")

        # - and get touch offset binned.
        _, fig_final = self.tokens_cluster_touch_offset_loc_across_all_data(PLOT_FINAL=True)
        fig_final.savefig(f"{sdir}/tokens_cluster_touch_offset_loc_across_all_data.pdf")

        # - Get sequence context for all tokens
        for ind in range(len(self.Dat)):    
            # Determine if any shapes were transformed rotations
            prims_extra_params = self.taskclass_extract_prims_extra_params_tforms(ind)
            if False: # Ignroe this for now -- this raises err for novel motif char flex, which I doint care about, since those are actually single prims uisng multple strokes.
                # SHOULD: check that this is PIG, and then those are the cases where I actully care about trakcing this.
                try:
                    if (prims_extra_params is not None) and (len(prims_extra_params)>0) and "tforms_each_prim" in prims_extra_params.keys():
                        print(prims_extra_params)
                        print(prims_extra_params[0]["tforms_each_prim"])
                        assert False, "add code to apply the below step (assigning shape semantic to be NOVEL) to only the specific prims that were tformed, using the information in tforms_each_prim"
                except Exception as err:
                    print("HERE", prims_extra_params)
                    raise err
            tforms_extra_exist = self.taskclass_check_prims_extra_params_tforms_exist_single(ind)

            # Beh strokes
            Tk_behdata = self.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", return_as_tokensclass=True)
            Tk_behdata.features_extract_wrapper(["loc_on", "angle"])
            Tk_behdata.sequence_context_relations_calc() # Get sequence, will be needed for datseg

            Tk_behtaskdata = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True)
            # append extra tforms (e.g.,  novel prims).

            # assert len(Tk_behtaskdata.Tokens)==len(tforms_extra)
            # for tk, tf in zip(Tk_behtaskdata.Tokens, tforms_extra):
            #     tk["tforms_extra"] = tf
            for tk in Tk_behtaskdata.Tokens:
                tk["tforms_extra_exist"] = tforms_extra_exist
            Tk_behtaskdata.features_extract_wrapper(["shape_semantic"], label_as_novel_if_shape_semantic_fails=label_as_novel_if_shape_semantic_fails)

            # Task strokes (ignore beh)
            Tk_taskdata = self.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)
            # append extra tforms (e.g.,  novel prims).
            # assert len(Tk_taskdata.Tokens)==len(tforms_extra)
            # for tk, tf in zip(Tk_taskdata.Tokens, tforms_extra):
            #     tk["tforms_extra"] = tf
            for tk in Tk_taskdata.Tokens:
                tk["tforms_extra_exist"] = tforms_extra_exist
            Tk_taskdata.features_extract_wrapper(["shape_semantic"], label_as_novel_if_shape_semantic_fails=label_as_novel_if_shape_semantic_fails)
            Tk_taskdata.sequence_context_relations_calc() # Get sequence, will be needed for datseg

        # (2) Compute all binned data, using beh data
        nbins = 3 # 2 or 3...
        self.tokens_bin_feature_across_all_data("loc_on", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)
        self.tokens_bin_feature_across_all_data("angle", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)

        self.tokens_bin_feature_across_all_data("center", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)
        self.tokens_bin_feature_across_all_data("center", "beh_using_task_data", nbins=nbins, PLOT=PLOT)
        self.tokens_bin_feature_across_all_data("center", "task", nbins=nbins, PLOT=PLOT)

        # Get locon_bin_in_loc
        if False:
            # This fails in lemur - not needed anwya...
            self.tokens_sequence_bin_location_within_gridloc()

            # In lemur this is not possible, since dont have binned locations from above.
            # Replace loc, for char, with loc within gridloc.
            # And then get shape_loc conjunctions
            self.tokens_gridloc_replace_with_recomputed_loc_chars()

        # (3) IMAGE PARSE
        self.shapesemantic_classify_novel_shape()

        self.taskclass_shapes_loc_configuration_assign_column()
        # 1. specific
        self.taskclass_shapes_loc_configuration_assign_column(version="char", shape_ver="shape_semantic", suffix="SHSEM", plot_examples=PLOT)
        # 2. more lenient
        self.taskclass_shapes_loc_configuration_assign_column(version="char", shape_ver="shape_semantic_cat", suffix="SHSEMCAT", plot_examples=PLOT)

        # (4) LAST: Extract new seq context variables, based on variables in tokens.
        self.seqcontext_preprocess(plot_examples=PLOT, force_run=True)


    def tokens_append_to_dataframe_column(self, force_regenerate=False):
        """
        Extract tokens and store in self.Dat (simply "expose"). Useful for nerual stuff.
        You ned to have first arleady extracted strokesbeh tokens --
        :return:
        - Appends to self.Dat (Tokens Object)
            self.Dat["Tkbeh_stkbeh"] = list_Tk_beh_1
            self.Dat["Tkbeh_stktask"] = list_Tk_beh_2
            self.Dat["Tktask"] = list_Tk_task
        """

        # Must run this first to get strokesbeh tokens
        self.tokens_generate_replacement_from_raw_helper(force_regenerate=force_regenerate)

        list_Tk_beh_1 = []
        list_Tk_beh_2 = []
        list_Tk_task = []
        for ind in range(len(self.Dat)):
            list_Tk_beh_1.append(self.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", return_as_tokensclass=True))
            list_Tk_beh_2.append(self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True))
            list_Tk_task.append(self.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True))
        self.Dat["Tkbeh_stkbeh"] = list_Tk_beh_1
        self.Dat["Tkbeh_stktask"] = list_Tk_beh_2
        self.Dat["Tktask"] = list_Tk_task

    def tokens_extract_variables_as_dataframe(self, list_var, tk_ver="beh_using_beh_data",
                                              list_var_dataset=None):
        """
        Generate a dataframe where each row is a set of variables extracted from token.
        Useful for doing stuff with tokens.
        Place variables back into self.Dat using tokens_assign_dataframe_back_to_self
        :param list_var: list of str, variable sto extract (new columns). First looks for it in
        tokens. if not find, then looks in self.Dat
        :return: dataframe
        """

        assert isinstance(list_var, (list, tuple))
        
        res = []
        for i, row in self.Dat.iterrows():
            Tk = self.taskclass_tokens_extract_wrapper(i, tk_ver, return_as_tokensclass=True)
            for j, tok in enumerate(Tk.Tokens):
                # print("---")
                # print(tok["ind_behstrokes"], j)
                # print(len(Tk.Tokens))
                # assert len(tok["ind_behstrokes"])==1
                # assert tok["ind_behstrokes"][0] == j, "just sanityc chekc that this old varaible makes sense"
                res.append({
                    "trialcode":row["trialcode"],
                    "ind_taskstroke_orig":tok["ind_taskstroke_orig"],
                    "idx":i,
                    "idx_tok":j,
                    "stroke_index":j,
                })

                # print(tok.keys())
                # assert False
                for var in list_var:
                    if var in tok.keys():
                        res[-1][var] = tok[var]
                    else:
                        res[-1][var] = row[var]

                if list_var_dataset is not None:
                    for var in list_var_dataset:
                        res[-1][var] = row[var]

        df = pd.DataFrame(res)

        # Store the token ver
        df["token_ver"] = tk_ver

        return df

    def tokens_assign_dataframe_back_to_self_mult(self, df, list_var_extract=None, tk_ver="beh_using_beh_data"):
        """Helper to assign multiple or all (if list_var_extract is None) variables back to tokens
        :param df: _description_
        :param list_var_extract: _description_, defaults to None
        :param tk_ver: _description_, defaults to "beh_using_beh_data"
        """
        if list_var_extract is None:
            # Take all vars
            list_var_extract = [var for var in df.columns if var not in ["trialcode", "ind_taskstroke_orig",
                                                                         "idx", "idx_tok", "stroke_index"]]
        for var_extract in list_var_extract:
            print("Returning this var to tokens...:", var_extract)
            self.tokens_assign_dataframe_back_to_self(df, var_extract, tk_ver)

    def tokens_assign_dataframe_back_to_self(self, df, var_extract, tk_ver="beh_using_beh_data",
                                             var_name_assign=None):
        """
        Given a dataframe genreated using tokens_extract_as_dataframe, assign a column back
        to self.Dat, making sure to get all rows and toekns that exist in self.Dat (fails
        if cant find one).
        :param df:
        :param var_extract: variable from df to extract
        :param var_name_assign: name of column in self.Dat to overwrite. if None, then uses
        var_extract
        :return: modifies self.Dat
        """

        if var_name_assign is None:
            var_name_assign = var_extract

        assert np.all(df["token_ver"] == tk_ver)

        for i, row in self.Dat.iterrows():
            Tk = self.taskclass_tokens_extract_wrapper(i, tk_ver, return_as_tokensclass=True)
            for j, tok in enumerate(Tk.Tokens):
                dfthis = df[(df["idx"] == i) & (df["idx_tok"] == j)]
                assert len(dfthis)==1
                # tmp = df[(df["idx"] == i) & (df["idx_tok"] == j)][var_extract].values
                # assert len(tmp)==1
                assert tok["ind_taskstroke_orig"] == dfthis["ind_taskstroke_orig"].values[0]
                assert dfthis["trialcode"].values[0] == row["trialcode"]
                tok[var_name_assign] = dfthis[var_extract].values[0]

    def tokens_gridloc_replace_with_recomputed_loc_chars(self, PRINT=False):
        """
        BEcuase for "character" task_kind the gridloc is not defined, here
        replace gridloc in tokens with recomputed gridloc based
        on stroke onset, binned within chars only, doing this only for chars.

        :return: modifies tokens
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from pythonlib.tools.pandastools import grouping_print_n_samples

        assert False, "NOTE: stopped doing this. isntead, use locaiton clusters"

        # Extract flattened tokens
        vars = ["task_kind", "gridloc", "locon_bin_in_loc"]
        df = self.tokens_extract_variables_as_dataframe(vars)
        if PRINT:
            print(" ----- BEFORE:")
            grouping_print_n_samples(df, vars)

        # Replace gridloc, for chars
        inds = df["task_kind"] == "character"
        df.loc[inds, "gridloc"] = df.loc[inds, "locon_bin_in_loc"]
        if PRINT:
            print(" ----- AFTER:")
            grouping_print_n_samples(df, vars)

        # put back into dataset
        self.tokens_assign_dataframe_back_to_self(df, "gridloc")

    def tokens_sequence_bin_location_within_gridloc(self, nbins=2):
        """
        Bin touch onset location within categorical values of gridloc.
        (which generlaly means within char, and within (SP + PIG).
        :return: Modifies tokens (beh using beh)
        """
        from pythonlib.tools.pandastools import bin_values_conditioned_on_class

        if False:
            # PROBLEM: this just acted on seqc variables, which dont carry over to
            # DS.

            # bin locations within each gridloc
            assert "seqc_0_locon" in self.Dat.columns, "extract usingn seqcontext_preprocess"
            for i in range(10):
                if f"seqc_{i}_locon" in self.Dat.columns:
                    print("Binning ", f"seqc_{i}_locon", "within ", f"seqc_{i}_loc")
                    self.Dat = bin_values_conditioned_on_class(self.Dat, f"seqc_{i}_locon", [f"seqc_{i}_loc"], nbins,
                                                               var_bin_ndim=2, new_col_name=f"seqc_{i}_locon_bin_in_loc")
        else:
            # BETTER - this modifies tokens directly.

            # Extract dataframe
            var_bin = "loc_on"
            vars_condition = ["gridloc"]
            list_var = [var_bin] + vars_condition
            df = self.tokens_extract_variables_as_dataframe(list_var, "beh_using_beh_data")

            # Bin values
            new_col_name = "locon_bin_in_loc"
            df = bin_values_conditioned_on_class(df, var_bin, vars_condition, nbins,
                                                       var_bin_ndim=2, new_col_name=new_col_name)

            # Put back into tokens
            self.tokens_assign_dataframe_back_to_self(df, new_col_name, 'beh_using_beh_data')

    def tokens_cluster_touch_onset_loc_across_all_data(self, token_ver="beh_using_beh_data",
                                                       PLOT_SIL=False, PLOT_FINAL=False):
        """
        Collect onset locations (beh stroke) acrioss tok in Tokens, then clluster these in 2d space
        (kmeans) so that each has a class label. Then store this label in tokens as "loc_on_clust".

        Works even across different task_kinds (e.g., PIG and CHAR).

        Automaitlcaly determeins n_clust by using n_clust that maximizes silhouette score.

        :param token_ver:
        :param PLOT_SIL:
        :param PLOT_FINAL:
        :return: TokensBeh adds two new columns: loc_on and loc_on_clust
        """
        from pythonlib.tools.statstools import cluster_kmeans_with_silhouette_score
        n_clusters_min_max=[6, 26]
        # Get required data from tokens.
        for ind in range(len(self.Dat)):

            # Beh strokes
            Tk_behdata = self.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", return_as_tokensclass=True)
            Tk_behdata.features_extract_wrapper(["loc_on"])

        # Extract dataframe
        if False:
            df = self.tokens_extract_variables_as_dataframe(["Stroke"], token_ver)
            X = np.stack([row["Stroke"]()[0,:2] for _, row in df.iterrows()])
        else:
            df = self.tokens_extract_variables_as_dataframe(["loc_on"], token_ver)
            X = np.stack(df["loc_on"])

        cluster_labels, fig, fig_final = cluster_kmeans_with_silhouette_score(X, n_clusters=None, n_clusters_min_max=n_clusters_min_max,
                                                              PLOT_SIL=PLOT_SIL, PLOT_FINAL=PLOT_FINAL,
                                                              return_figs=True)
        df["loc_on_clust"] = cluster_labels

        # Put back into tokens
        self.tokens_assign_dataframe_back_to_self(df, "loc_on_clust", token_ver)

        return fig, fig_final

    def tokens_cluster_touch_offset_loc_across_all_data(self, token_ver="beh_using_beh_data",
                                                       PLOT_SIL=False, PLOT_FINAL=False):
        """
        Collect offset locations (beh stroke) acrioss tok in Tokens, then clluster these in 2d space
        (kmeans) so that each has a class label. Then store this label in tokens as "loc_off_clust".

        Works even across different task_kinds (e.g., PIG and CHAR).

        Automaitlcaly determeins n_clust by using n_clust that maximizes silhouette score.


        :param token_ver:
        :param PLOT_SIL:
        :param PLOT_FINAL:
        :return: TokensBeh adds two new columns: loc_off and loc_off_clust
        """
        from pythonlib.tools.statstools import cluster_kmeans_with_silhouette_score

        n_clusters_min_max=[4, 26]
        # Get required data from tokens.
        for ind in range(len(self.Dat)):

            # Beh strokes
            Tk_behdata = self.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", return_as_tokensclass=True)
            Tk_behdata.features_extract_wrapper(["loc_off"])

        # Extract dataframe
        df = self.tokens_extract_variables_as_dataframe(["loc_off"], token_ver)
        X = np.stack(df["loc_off"])

        cluster_labels, fig, fig_final = cluster_kmeans_with_silhouette_score(X, n_clusters=None, n_clusters_min_max=n_clusters_min_max,
                                                              PLOT_SIL=PLOT_SIL, PLOT_FINAL=PLOT_FINAL,
                                                              return_figs=True)
        df["loc_off_clust"] = cluster_labels

        # Put back into tokens
        self.tokens_assign_dataframe_back_to_self(df, "loc_off_clust", token_ver)

        return fig, fig_final


    def tokens_bin_feature_across_all_data(self, feature, ver_token, nbins=3, PLOT=False):
        """
        Helper, for a given feature in Tokens, extract all across all data (trials x strokes), convert
        to binned data (vby value, not rank) and then store in tokens with colname = f"{feature}_binned"
        :param feature:
        :param nbins:
        :return:
        """
        from pythonlib.tools.nptools import bin_values
        from pythonlib.tools.vectools import bin_angle_by_direction

        # Collect values across all trials.
        values =[]
        list_i_j = []
        for i, row in self.Dat.iterrows():
            Tk = self.taskclass_tokens_extract_wrapper(i, ver_token, return_as_tokensclass=True)

            if feature not in Tk.Tokens[0].keys():
                # Collect this feature from all tokens
                Tk.features_extract_wrapper([feature])

            for j, tok in enumerate(Tk.Tokens):
                values.append(tok[feature])
                list_i_j.append((i,j))

        # Bin the values
        # - params - what is size of feature
        tmp = self.taskclass_tokens_extract_wrapper(0, ver_token, return_as_tokensclass=True).Tokens[0][feature]

        if isinstance(tmp, (float)):
            ndims = 1
        elif isinstance(tmp, np.ndarray) and len(tmp.shape)==0:
            ndims = 1
        else:
            ndims = len(self.taskclass_tokens_extract_wrapper(0, ver_token, return_as_tokensclass=True).Tokens[0][feature])
        print(f"ndims for feature {feature} = {ndims}")
        if ndims ==2:
            values_arr = np.stack(values, axis=0)
            assert values_arr.shape[0] == len(values), f"problem with stacking? shape = {values_arr.shape}"
            xs = values_arr[:,0]
            ys = values_arr[:,1]
            # xs_binned = bin_values_by_rank(xs, nbins=2)
            # ys_binned = bin_values_by_rank(ys, nbins=2)
            xs_binned = bin_values(xs, nbins=nbins)
            ys_binned = bin_values(ys, nbins=nbins)
            # Convert to list of 2-tuples
            values_binned = [(x, y) for x, y in zip(xs_binned, ys_binned)] # list of 2-typles of 2 strings.
            # values_binned = np.stack([xs_binned, ys_binned], axis=1)
        elif ndims==1:
            if "angle" in feature:
                values_binned = bin_angle_by_direction(values, num_angle_bins=nbins, PLOT=PLOT)
            else:
                values_binned = bin_values(values, nbins=nbins)
        else:
            print(ndims)
            assert False, 'Code it'

        # Store binned values in tokens
        colname = f"{feature}_binned"
        assert len(values_binned) == len(list_i_j)
        assert len(values_binned) == len(values)
        for i_j, val_binned in zip(list_i_j, values_binned):
            i, j = i_j
            Tk = self.taskclass_tokens_extract_wrapper(i, ver_token, return_as_tokensclass=True)
            Tk.Tokens[j][colname] = val_binned
        print("New colname in tokens:", colname)

        if PLOT:
            import seaborn as sns
            fig, ax = plt.subplots()
            if ndims==2:
                df = pd.DataFrame({"x":values_arr[:,0], "y": values_arr[:,1], "bin":[tuple(v) for v in values_binned]})
                sns.scatterplot(data=df, x="x", y="y", hue="bin", alpha=0.5, ax=ax)
            elif ndims==1:
                # print(values)
                # print(values_binned)
                sns.scatterplot(x=values, y=np.ones((len(values))), hue=values_binned, ax=ax)
            else:
                assert False

    def tokens_generate_replacement_from_raw_helper(self, force_regenerate=False):
        """
        Runt his only once - replace Tokens with new version. advartnageS:
        - works for Chars, using pre-saved cluster labels.
        - seprates into 3 kinds of tokens (just 1 for char): (i) beh using beh strokes
        (ii) beh using task strokes (iii) task.
        :return:
        """
        if "charclust_shape_seq" in self.Dat.columns:
            self.tokens_generate_replacement_from_raw(shape_sequence_col="charclust_shape_seq",
                                                      force_regenerate=force_regenerate)
        else:
            self.tokens_generate_replacement_from_raw(shape_sequence_col="TASK_SHAPES",
                                                      force_regenerate=force_regenerate)

    def tokens_generate_replacement_quick_from_beh(self):
        """
        QWUick and dirtly, always ruin this in preprocess, to generate beh datsegs.
        Overwrites self.TokensStrokesBeh with tokens using beh strokes, nothing do to
        with task, except using the shape from task.
        """
        from pythonlib.drawmodel.tokens import generate_tokens_from_raw
        from pythonlib.drawmodel.tokens import Tokens

        # Only run this if Tokens are not already regenrated.
        TokensStrokesBeh = {} # len beh, using strokes beh
        for ind in range(len(self.Dat)):
            tc = self.Dat.iloc[ind]["trialcode"]

            # (1) Strokes beh, what sahpes to call each beh stroke
            tokens_beh_old = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")
            shape_seq = [tok["shape"] for tok in tokens_beh_old]
            strokes = self.Dat.iloc[ind]["strokes_beh"]
            if not len(strokes)==len(shape_seq):
                # Keep the tokens that match the strokes..
                print(len(strokes))
                print(tokens_beh_old)
                for tok in tokens_beh_old:
                    print("----")
                    print(tok)
                assert False, "did you prune data? eg substrokes? fix this"

            # (2) Strokes beh, what locations to call each beh stroke
            if self.taskclass_get_grid_ver(ind)=="on_grid":
                assert self.Dat.iloc[ind]["task_kind"] in ["prims_single", "prims_on_grid"], "downstream code assumes this"
                # Then keep locations...
                gridlocs = [t["gridloc"] for t in tokens_beh_old]
                gridlocs_local = [t["gridloc_local"] for t in tokens_beh_old]
            else:
                # For char, this should be recomputed later, by binning across all data
                assert self.Dat.iloc[ind]["task_kind"] in ["character"], "downstream code assumes this"
                # Fill with Nones.
                gridlocs = None
                gridlocs_local = None

            # Generate tokens: strokesbeh_using_beh
            Tk = generate_tokens_from_raw(strokes, shape_seq, gridlocs, gridlocs_local)
            assert Tk is not None
            TokensStrokesBeh[tc] = Tk

        # Switch so that always uses these tokens
        self.TokensStrokesBeh = TokensStrokesBeh

    def tokens_generate_replacement_from_raw(self, shape_sequence_col="charclust_shape_seq",
                                             skip_if_labels_not_found=False, force_regenerate=False):
        """
        Generate Tokens based on actual beh strokes for each trial,
        and store in self.TokensStrokesBeh, and switch TokensVersion to
        "regenerated_from_raw". And do this differentyl for SP/PIG vs. CHAR, where
        the latter completely throws out task-related information.
        This will then be used as replacement
        for all tokens stuff (e.g., in constructing DatasetStrokes), useful
        for tasks where strokes do not nicely align wiht task strokes (which
        formed to default Tokens), such as chars. The idea is that these data can be used
        in taskclass_tokens_extract_wrapper. The shapes must have already been loaded
        and stored in tuples in self.Dat[<charclust_shape_seq>].
        :param shape_sequence_col:
        :return:
        - updates self.TokensStrokesBeh, dict from trialcode --> tokens (strokes, beh)
        [if char, then the following two are identical to above]
        - updates self.TokensStrokesBehUsingTaskStrokes, dict from trialcode --> tokens (strokes, task)
        - updates self.TokensTask, dict from trialcode --> tokens (task)
        """
        from pythonlib.drawmodel.tokens import generate_tokens_from_raw
        from pythonlib.drawmodel.tokens import Tokens

        assert skip_if_labels_not_found==False, "not coded"

        if not hasattr(self, "TokensVersion"):
            self.TokensVersion = "taskclass"

        # Only run this if Tokens are not already regenrated.
        
        A = (not self.TokensVersion=="regenerated_from_raw") or (len(self.TokensStrokesBeh)==0) or (len(self.TokensTask)==0) or (len(self.TokensStrokesBehUsingTaskStrokes)==0) or (self.TokensStrokesBeh is None) or (self.TokensTask is None) or (self.TokensStrokesBehUsingTaskStrokes is None)
        if A or force_regenerate:
            TokensStrokesBeh = {} # len beh, using strokes beh
            TokensStrokesBehUsingTaskStrokes = {} # len beh, using best-aligned task strok
            TokensTask = {} # len task, using task strokes.
            for ind in range(len(self.Dat)):
                tc = self.Dat.iloc[ind]["trialcode"]
                tokens_beh_old = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")

                # (1) Strokes beh, what sahpes to call each beh stroke
                if shape_sequence_col == "TASK_SHAPES":
                    shape_seq = [tok["shape"] for tok in tokens_beh_old]
                else:
                    shape_seq = self.Dat.iloc[ind][shape_sequence_col] #
                strokes = self.Dat.iloc[ind]["strokes_beh"]
                assert len(strokes)==len(shape_seq)

                # (2) Strokes beh, what locations to call each beh stroke
                if self.taskclass_get_grid_ver(ind)=="on_grid":
                    assert self.Dat.iloc[ind]["task_kind"] in ["prims_single", "prims_on_grid"], "downstream code assumes this"
                    # Then keep locations...
                    gridlocs = [t["gridloc"] for t in tokens_beh_old]
                    gridlocs_local = [t["gridloc_local"] for t in tokens_beh_old]

                else:
                    # For char, this should be recomputed later, by binning across all data
                    assert self.Dat.iloc[ind]["task_kind"] in ["character"], "downstream code assumes this"
                    # Fill with Nones.
                    gridlocs = None
                    gridlocs_local = None

                # Generate tokens: strokesbeh_using_beh
                Tk = generate_tokens_from_raw(strokes, shape_seq, gridlocs, gridlocs_local)
                TokensStrokesBeh[tc] = Tk

                #### HOW TO DEFINE TASK STROKE TOKENS
                if self.taskclass_get_grid_ver(ind)=="on_grid":
                    assert self.Dat.iloc[ind]["task_kind"] in ["prims_single", "prims_on_grid"], "downstream code assumes this"
                    # Use the standard.
                    # (2) Also save task tokens, so that can safely delete task tokens and still use it here.
                    Task = self.Dat.iloc[ind]["Task"]
                    TokensTask[tc] = Task.tokens_generate(assert_computed=True, return_as_tokensclass=True)

                    # (3) Also save the old "beh strokes, aligned to task"
                    TokensStrokesBehUsingTaskStrokes[tc] = Tokens(tokens_beh_old)
                else:
                    # For char, should ignore entirely any task tokens, since they are meaninglesss. This avoids erros later.
                    # Make them identical to Beh
                    assert self.Dat.iloc[ind]["task_kind"] in ["character"], "downstream code assumes this"
                    TokensTask[tc] = TokensStrokesBeh[tc]
                    TokensStrokesBehUsingTaskStrokes[tc] = TokensStrokesBeh[tc]

            # Switch so that always uses these tokens
            # MAKE sure that if "regenerated_from_raw", then these three tokens must not be empty
            self.TokensVersion = "regenerated_from_raw"
            assert TokensStrokesBeh is not None
            assert TokensTask is not None
            assert TokensStrokesBehUsingTaskStrokes is not None
            assert len(TokensStrokesBeh)>0
            assert len(TokensTask)>0
            assert len(TokensStrokesBehUsingTaskStrokes)>0
            
            self.TokensStrokesBeh = TokensStrokesBeh
            self.TokensTask = TokensTask
            self.TokensStrokesBehUsingTaskStrokes = TokensStrokesBehUsingTaskStrokes

            # # Save, so that can clear derived keys that are extra from these..
            self.TokensStrokesBeh_OriginalKeys = list(list(self.TokensStrokesBeh.values())[0].Tokens[0].keys())
            self.TokensTask_OriginalKeys = list(list(self.TokensTask.values())[0].Tokens[0].keys())
            self.TokensStrokesBehUsingTaskStrokes_OriginalKeys = list(list(self.TokensStrokesBehUsingTaskStrokes.values())[0].Tokens[0].keys())

    def tokens_generate_replacement_clear_derived_keys(self):
        """
        For all Tokens in self that have been regenerated from raw, prune so they only use
        the original keys, not the derived keys from binning or clustering data -- useful if
        you want to concat datasets and then ensure that they only use clusters combining datasets.
        :return: Modifies tokens (e.g., self.TokensStrokesBeh)
        """

        def _clear_keys(tokens_ver):
            if hasattr(self, tokens_ver):
                TokensDict = getattr(self, tokens_ver)
                TokensKeys = getattr(self, f"{tokens_ver}_OriginalKeys")
                dict_Tk = {}
                for tc, Tk in TokensDict.items():
                    # Regenreate Tk.Tokens, including only orig keys
                    tmp = []
                    for tok in Tk.Tokens:
                        tmp.append({k:v for k, v in tok.items() if k in TokensKeys})
                    Tk.Tokens = tmp
                    # Save Tk.
                    dict_Tk[tc] = Tk
                setattr(self, tokens_ver, dict_Tk)
                print("Cleared tokesn to original ones, for:", tokens_ver)
            else:
                print("SKIPPED clearing tokesn to original ones (did not find in self), for:", tokens_ver)

        for tokens_ver in ["TokensStrokesBeh", "TokensTask", "TokensStrokesBehUsingTaskStrokes"]:
            _clear_keys(tokens_ver)
        assert self.TokensStrokesBeh is not None
        assert self.TokensTask is not None
        assert self.TokensStrokesBehUsingTaskStrokes is not None

    def taskclass_tokens_extract_wrapper(self, ind, which_order,
                                         plot=False, return_as_tokensclass=False):
        """ [GOOD] The helper to rEturn tokens (lsit of dict) for this trial, in any order,
        and with aligned indices into beh strokes.
        PARAMS:
        - which_order, in
        --- 'task', order of ind_taskstrokes
        --- 'beh_using_beh_data', matching each beh stroke, holding data of strokes_beh (tok["Prim"])
        --- 'beh', matching each beh stroke, holding data on the best-matching task stroke
        --- 'beh_firsttouch', ordered by behstroke first touch, but length <=ntask.
        RETURNS:
        - list of dict, a reordered tokens
        """

        if not hasattr(self, "TokensVersion"):
            self.TokensVersion = "taskclass"

        if self.TokensVersion == "regenerated_from_raw":
        # if use_replacement_strokesbeh_tokens:
            # Completely ignore original taskclass based tokens

            tc = self.Dat.iloc[ind]["trialcode"]
            if which_order=="beh_using_beh_data":
                # Beh length, beh strokes
                assert self.TokensStrokesBeh is not None, "run tokens_generate_replacement_from_raw_helper()"
                Tk = self.TokensStrokesBeh[tc]
                assert Tk is not None
            elif which_order == "beh_using_task_data":
                # beh lenght, task strokes (DEFAULT)
                try:
                    Tk = self.TokensStrokesBehUsingTaskStrokes[tc]
                except Exception as err:
                    print(self.TokensStrokesBehUsingTaskStrokes)
                    print(self.TokensStrokesBeh)
                    print(self.TokensTask)
                    print("run tokens_generate_replacement_from_raw_helper(). Was this messed up when concatting?")
                    raise err
            elif which_order == "task":
                assert self.TokensTask is not None, "run tokens_generate_replacement_from_raw_helper()"
                Tk = self.TokensTask[tc]
            else:
                print(which_order)
                assert False, "only beh and task make sense for 'regenerated_from_raw'..."

            if return_as_tokensclass:
                return Tk
            else:
                return Tk.Tokens
        elif self.TokensVersion == "taskclass":
            # Original, default, etc...
            # if not self.behclass_check_if_tokens_extracted():
            self.behclass_preprocess_wrapper()

            # mapper from taskstrokeinds to beh
            mapper_taskstroke_to_beh = {}
            this = self.behclass_extract_beh_and_task(ind)[3]

            for x in this:
                ind_task = x[2]["ind_taskstroke_orig"]
                mapper_taskstroke_to_beh[ind_task] = x[0] # indidces into storkes_beh
            # some tasktrokes were missed
            n_task_strokes = len(self.Dat.iloc[ind]["strokes_task"])
            for i in range(n_task_strokes):
                if i not in mapper_taskstroke_to_beh.keys():
                    mapper_taskstroke_to_beh[i] = []

            Beh = self.Dat.iloc[ind]["BehClass"]
            if which_order=="beh_using_beh_data":
                assert self.TokensStrokesBeh is not None, "run self.tokens_generate_quick_from_beh"
                tc = self.Dat.iloc[ind]["trialcode"]
                Tk = self.TokensStrokesBeh[tc]
                assert Tk is not None
                assert Tk.Tokens is not None
                assert Tk.Tokens[0] is not None
                # REturn here, since it doesnt have valid taskstroke_orig indices.
                if return_as_tokensclass:
                    return Tk
                else:
                    return Tk.Tokens
            elif which_order=="task":
                Task = self.Dat.iloc[ind]["Task"]
                tokens = Task.tokens_generate(assert_computed=True)
            elif which_order=="beh_using_task_data":
                tokens = Beh.alignsim_extract_datsegs_both_beh_task()[1]
                # tokens = self.behclass_extract_beh_and_task(ind)[1]
            elif which_order=="beh_firsttouch":
                tokens = Beh.alignsim_extract_datsegs_both_beh_task()[2]
                # tokens = self.behclass_extract_beh_and_task(ind)[2]
            else:
                print(which_order)
                assert False, "not coded yet"

            # Also get the corresponding beahvior
            for t in tokens:
                t["ind_behstrokes"] = mapper_taskstroke_to_beh[t["ind_taskstroke_orig"]]
                # print(t["ind_taskstroke_orig"],  t["ind_behstrokes"])

            if plot:
                self.grammarmatlab_extract_beh_and_task(ind, ploton=True)

            if return_as_tokensclass:
                from pythonlib.drawmodel.tokens import Tokens
                return Tokens(tokens)
            else:
                return tokens
        else:
            assert False

    # def taskclass_shapes_loc_configuration_extract_helper(self, ind, force_all_task_kind_same_loc_coord=False):
    #     """
    #     Extract the "task config", in a way that works for any task, i.e,., SP, PIG, CHAR.
    #     :param ind:
    #     :param force_all_task_kind_same_loc_coord, bool, if True, then uses center_binned, based on recomputing
    #     binning across all trials. If False, then does that for Char, but for others uses the task gridloc.
    #     :return:
    #     """
    #
    #     task_kind = self.Dat.iloc[ind]["task_kind"]
    #     if task_kind in ["singleprim", "prims_on_grid"]:
    #         # Then easy, use the task, ingoring behavior.
    #         shape_token = "task"
    #         shape_ver = "shape"
    #         if force_all_task_kind_same_loc_coord:
    #             # use binned center of
    #             loc_token = "beh_using_task_data"
    #             loc_version = "center_binned"
    #         else:
    #             loc_token = "task"
    #             loc_version = "gridloc"
    #     elif task_kind in ["character"]:
    #         # Ignore the task entirely. Instead, use the shape labels assigned to each
    #         # beh stroke, and the binned location.
    #         shape_token = "beh_using_beh_data"
    #         shape_ver = "shape"
    #         loc_token = "beh_using_beh_data"
    #         loc_version = "center_binned"


    def taskclass_shapes_loc_configuration_extract(self, ind, shape_token = "task", loc_token="task",
                                                   shape_version="shape", loc_version="gridloc",
                                                   use_recomputed_prim_labels=False):
        """ Extract the shapes or location config (global) for this task. 
        Ignores behavior.
        PARAMS;:
        - use_recomputed_prim_labels, bool[False], if True, then recomputes for each prim the
        (shape, scale, angle). This is important for cases where extra transforms are applied
        to prims in matlab, (usually only for chars).
        RETURNS:
        - dict, 
        --- "shape":tuple of str, eahc a shape
        --- "loc":tuple of tuples, eahc holding two ints, a gridloc
        --- "shape_loc": tuple of (shape, loc) tuples.
        The lists are sorted, so a given task will always return 
        the same thing. This means that shape and loc are NOT aligned, but each iten within shape_loc is algined.
        """

        # To ensure that tokens for shape and loc are aligned, thesea re the only compatible combinations
        if shape_token in ["task"]:
            assert loc_token in ["task"]
        elif shape_token in ["beh_using_beh_data", "beh_using_task_data"]:
            assert loc_token in ["beh_using_beh_data", "beh_using_task_data"]
        else:
            print(shape_token, loc_token)
            assert False, "does this combo aligned?"

        tokens_shape = self.taskclass_tokens_extract_wrapper(ind, shape_token)
        tokens_loc = self.taskclass_tokens_extract_wrapper(ind, loc_token)

        assert len(tokens_shape)==len(tokens_loc), f"you chose incompatible token versions, {shape_token}, {loc_token}"

        if loc_version=="pixel":
            LOC = "center"
        else:
            LOC = loc_version

        list_shapes = tuple(sort_mixed_type([t[shape_version] for t in tokens_shape]))
        list_loc = tuple(sort_mixed_type([t[LOC] for t in tokens_loc]))
        list_shape_loc = tuple(sort_mixed_type([(ts[shape_version], tl[LOC]) for ts, tl in zip(tokens_shape, tokens_loc)]))

        return {
            "shape":list_shapes,
            "loc":list_loc,
            "shape_loc":list_shape_loc
        }

    def taskclass_shapes_loc_configuration_assign_column(self, version="task", shape_ver="shape",
                                                         plot_examples=False, suffix=None):
        """ Assigns three new columns indicating the tasks shape, loc, and 
        shape_loc configurations
        PARAMS:
        - version, str, which toekns to use to define task config:
        --- "task", then uses task, ignoring entirely the beh
        --- "char", then optimized for characters, same as "task" but using recomputed location bins, instaed of
        gridloc.
        RETURNS:
        - modifies self.Dat, with columsn:
        --- "taskconfig_loc"
        --- "taskconfig_shp"
        --- "taskconfig_shploc"
        """

        if version=="task":
            # This is general, works for SP, PIG, and CHAR, since char now has correct task tokens.
            shape_token = "task"
            shape_ver = shape_ver
            loc_token = "task"
            loc_version = "gridloc"
        elif version=="char":
            # Same, except use recomputed binned locations (centers) insteadf of gridloc, which is not
            # defined for chars
            # shape_token = "beh_using_beh_data"
            # shape_ver = "shape_semantic"
            # loc_token = "beh_using_beh_data"
            # loc_version = "center_binned"
            shape_token = "task"
            shape_ver = shape_ver
            loc_token = "task"
            # loc_version = "center_binned"
            loc_version = "gridloc"
        else:
            print(version)
            assert False


        list_loc =[]
        list_sh = []
        list_shloc =[]
        for ind in range(len(self.Dat)):
            this = self.taskclass_shapes_loc_configuration_extract(ind, shape_token,  loc_token, shape_ver, loc_version)
            list_loc.append(this["loc"])
            list_sh.append(this["shape"])
            list_shloc.append(this["shape_loc"])

        # Append
        if suffix is not None:
            suffix = f"_{suffix}"
        else:
            suffix = ""

        self.Dat[f"taskconfig_loc{suffix}"] = list_loc
        self.Dat[f"taskconfig_shp{suffix}"] = list_sh
        self.Dat[f"taskconfig_shploc{suffix}"] = list_shloc

        if plot_examples:
            # Print/plot showing taskshape config
            import random
            n = 5
            inds = random.sample(range(len(self.Dat)), n)
            fig, axes, idxs = self.plotMultTrials2(inds)
            fig, axes, idxs = self.plotMultTrials2(inds, "strokes_task")
            # D.Dat.loc[inds, ["taskconfig_loc", "taskconfig_shp", "taskconfig_shploc"]]

            print(self.Dat.loc[inds, [f"taskconfig_shploc{suffix}"]].values)

    def taskclass_shapes_extract_unique_alltrials(self):
        """ REturn list of (sorted) shapes across all trial sin dataset,
        unqiue. 
        """
        shapes = []
        for ind in range(len(self.Dat)):
            shapes.extend(self.taskclass_shapes_extract(ind))

        return sorted(list(set(shapes)))

    def taskclass_extract_prims_extra_params_tforms(self, ind):
        """
        Return dict holding extra params applied to prims, including tforms,
        and rotations (e.g., novel prims)
        RETURNS:
        - prims_extra_params, dict
            {'tforms': {},
            'tforms_each_prim_p': [{},
                [['th', array(-0.16839016)],
                ['sx', array(1.01434708)],
                ['sy', array(1.01434708)]],
                {}]
            }        
        """
        T = self.Dat.iloc[ind]["Task"]
        return T.extra_tform_params_extract()
    
        # P = self.taskclass_extract_planclass(ind)
        # if "PrimsExtraParams" not in P.keys():
        #     prims_extra_params = []
        # else:
        #     prims_extra_params = P["PrimsExtraParams"]
        
        # # if len(tforms)>0:
        # #     nstrokes = len(self.Dat.iloc[ind]["strokes_beh"])
        # #     if not len(tforms)==nstrokes:
        # #         print(tforms)
        # #         print(len(tforms), nstrokes)
        # #         assert False
        
        # return prims_extra_params

    def taskclass_check_prims_extra_params_tforms_exist_single(self, ind):
        """ check if, for this trial, if additiaonl tforms to prims in matlab code. 
        if so, then cannot represnt them as line-10-0-1... etc.
        """
        # tforms = self.taskclass_extract_prims_extra_params_tforms(ind)
        T = self.Dat.iloc[ind]["Task"]
        return T.check_prims_extra_params_exist()
    
    def taskclass_check_prims_extra_params_tforms_exist(self):
        """ check if, out of all trials, any of them applyh
        additiaonl tforms to prims in matlab code. if so, then
        cannot represnt them as line-10-0-1... etc.
        """

        # Check just once and cacjhe it
        if not hasattr(self, "_TaskclassCheckPrimsExtraParams"):
            self._TaskclassCheckPrimsExtraParams=False
            for ind in range(len(self.Dat)):
                if self.taskclass_check_prims_extra_params_tforms_exist_single(ind):
                # T = self.Dat.iloc[ind]["Task"]
                # if T.check_prims_extra_params_exist():
                    self._TaskclassCheckPrimsExtraParams=True
                    break
        return self._TaskclassCheckPrimsExtraParams

    def taskclass_shapes_extract(self, ind):
        """ Return list of shape strings used in 
        this task, in order of indices in taskstrokes (HAS NOTHING
        TO DO WITH BEHAVIOR)
        (i.e. get shape sequence)
        """
        tokens = self.taskclass_tokens_extract_wrapper(ind, "task")
        shapes = [d["shape_oriented"] for d in tokens]
        return shapes

    def taskfeatures_category_by(self, method="shape_repeat", params=None,
        colname="taskfeat_cat", PRINT=False):
        """ Give each trial a label for the catgegory of its tak,
        based on specific kinds of features. often will depend just on the 
        task image.
        PARAMS;
        - method, str, what features to use        
        - params, flexible, for method
        - colname, str, what to call the new col
        RETURNS:
        - assigns new column, default name "taskfeat_cat"
        """

        if method=="shape_repeat":
            # For each shape that exists today, determine whether it eitehrh (i) doesnt exists in this trial,
            # (ii) has at least 2 isntances separated in space by another shape. or 
            # (iii) exists, but is not separated.
            # - reutrns values like VNS-lNS-lNS, for 3 shapes V l l.
            from pythonlib.drawmodel.task_features import shapes_has_separated_cases_of_shape

            # Test each shape in the order it needs to be done given the rule.
            list_shapes_today = self.taskclass_shapes_extract_unique_alltrials() # get list of shapes that exist today.

            # Go thru each trial and classify
            list_trial_code = []
            for ind in range(len(self.Dat)):
                T = self.Dat.iloc[ind]["Task"]
                Tok = self.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)
                shapes_this_trial = [t["shape"] for t in Tok.Tokens]

                dict_task_category = {}
                for i, shape in enumerate(list_shapes_today):

                    if shape not in shapes_this_trial:
                        # shape doesnt exist
                        # cat = "not_exist"
                        cat = "NE"
                    elif shapes_has_separated_cases_of_shape(T, shape_same=shape):
                        # shape exists, with separted gap
                        cat = "S"
                    else:
                        # exists, not separted
                        cat = "NS"

                    dict_task_category[shape] = cat
                
                # Give this trial a code
                trial_code = "-".join([f"{shape[:1]}{dict_task_category[shape]}" for shape in list_shapes_today])
                
                # Collect
                list_trial_code.append(trial_code)

                if PRINT:
                    print(ind, dict_task_category)
        else:
            print(method)
            assert False

        print(f"Assinging to column: self.Dat[{colname}]")
        self.Dat[colname] = list_trial_code           


    def objectclass_get_active_chunk(self, ind):
        """ Get the active chunk thbat was used online, 
        from ObjectClass
        RETURNS:
        - ChunksClass representing active chunk. OR
        --- None, if no ChunksListClass is defined (older datasets)
        """

        TT = self.taskclass_extract_ml2(ind)
        return TT.objectclass_extract_active_chunk()
        # O = self.taskclass_extract_objectclass(ind)

        # # Can only work if ChunksListClass is present
        # if O["ChunksListClass"] is None:
        #     return None
        # else:
        #     # which are active chunk
        #     model = O["ChunkState"]["active_chunk_model"]
        #     index = O["ChunkState"]["active_chunk_ind"]

        #     # find it
        #     CLC = O["ChunksListClass"]
        #     C = CLC.find_chunk(model, index)
        #     return C

    def objectclass_summarize_rule_failures(self, ploton=False, sdir=None):
        """ Returns a dict summarizing which rules were failed
        for each trial (1 means failed, 0 means success)
        PARAMS;
        - sdir, if not None, saves figure and text file summary/
        RETURNS:
        - DictRuleFailures, dict, each item has indtrial and each rule as keys.
        - dffails, same but dataframe
        """
        # Across all trials, collect rule failure stats
        import pandas as pd

        if sdir is not None:
            ploton=True

        DictRuleFailures = []
        for ind in range(len(self.Dat)):
            O = self.taskclass_extract_objectclass(ind)
            
            this = {}
            this["indtrial"] = ind
            this["trialcode"] = self.Dat.iloc[ind]["trialcode"]
            for i, x in enumerate(O["RuleList"]):
                rulename = x[0]
                nfails = O["RuleFailureTracker"][i]
                
                # collect
                this[rulename] = int(nfails>0)
            
            DictRuleFailures.append(this)

        dffails = pd.DataFrame(DictRuleFailures)

        if ploton:
            import seaborn as sns
            cols_keep = [col for col in dffails.columns if not col in ["trialcode", "indtrial"]]
            dfthis = dffails.loc[:, cols_keep]
            fig, ax = plt.subplots(1,1, figsize=(20, 3))
            sns.heatmap(data=dfthis.T, vmin=0, vmax=1, ax=ax)           
            if sdir is not None:
                fig.savefig(f"{sdir}/rule_failures_objectclass.pdf")
    
        if sdir is not None:
            # save a text file
            dffails.to_csv(f"{sdir}/rule_failures_objectclass.txt", sep=' ')

        return DictRuleFailures, dffails

    def objectclass_wrapper_extract_sequence_chunk_color(self, ind, plot_summary=False):
        """ Helps to extract all relevant info this trial regarding online 
        supervision of chunking and sequencing, looking mainly into ObjectClass
        PARAMS;
        - ind, index into self.Dat
        - plot_summary, bool, to print and plot summary
        """

        # 1) Was color supervision on?
        if "INSTRUCTION_COLOR" not in self.Dat.columns:
            self.supervision_check_is_instruction_using_color_assign()
            # assert False, "you prob need to reassign grouping in preprocess. see preprocess for grammardircolor (note, this is currently default in preproces...)"
            
        color_on = self.Dat.iloc[ind]["INSTRUCTION_COLOR"] 
        color_method = self.supervision_extract_params(ind)["COLOR_METHOD"]

        # 2) objectclass info
        O = self.taskclass_extract_objectclass(ind)
        ChunkActive = self.objectclass_get_active_chunk(ind)

        # 3) Color of strokes actually presented (during task)
        if color_on:
            stroke_colors = O["Features_Active"]["color"]
        else:
            # default colors
            bp = self.blockparams_extract_single(ind)
            col = bp["sizes"]["InkColor_untouched"]
            nstrokes = O["actual_nstrokes"]
            # for k, v in O.items():
            #     print(k, len(v))
            # nstrokes = len(O["StrokesFinalNorm"])
            # nstrokes = len(O["Features_Active"]["shapes"])
            stroke_colors = [col for _ in range(nstrokes)]

            # sanity check that did not fade. if faded, then InkColor_untouched is not accurate. ahve to look into bb.
            for k, v in bp["fade"].items():
                # e.g, k, v is 'guide': array([], shape=(0, 0), dtype=float64),
                if k=="guidelight":
                    assert(np.all(v==1.))
                else:
                    # print("here")
                    # print(bp["fade"])
                    # print(k)
                    # print(v)
                    # print(v.shape)
                    if len(v.shape)>0 and len(v)>0:
                        # ok if v.shape is ()
                        print(bp["fade"])
                        print(k)
                        print(v)
                        print(v.shape)
                        assert False

        # TT.ObjectClass["ChunksListClass"].print_summary()
        if plot_summary:
            # 1) plot the beh and task, numbered.
            self.plotSingleTrial(ind, task_add_num=True, number_from_zero=True);

            print("\n--- Was color sueprvision on?:")
            print(color_on)

            # 2) print colors
            print("\n--- Colors (orig stroke order):")
            for col in stroke_colors:
                print(' ', col)

            print('\n--- Active chunk:')
            if ChunkActive is None:
                print("SKIPPED: ChunkActive is None")
            else:
                ChunkActive.print_summary()

        out = {
            "color_supervision_on": color_on,
            "color_supervision_method": color_method,
            "active_chunk": ChunkActive,
            "stroke_colors_orig_order": stroke_colors,
        }
        return out

    ############### SHAPE SEMANTIC (e.g., novel shape)
    def shapesemantic_cluster_taskclass_strokes_to_rename_shapes(self, THRESH_DIST = 0.96):
        """
        Label strokes using the ground truth shape image pixels.
        Does this by centering task stroke. 
        Diff sizes will be call different shapes, and so wuold if the pts are slightly off.
        Assign each stroke a cluster index (0,1,2,..) where strokes within cluster are identical 
        in image pixel distance (hausdorff_max_max).

        NOTE: this is useful for novel shapes, is better than hash, which runs into numerical 
        impresision errors.
        
        PARAMS:
        - THRESH_DIST, minim sim score, above which pairs are called the same sahep. Empriicalyl, 
        0.95 is similar, but 0.92 is not (Pancho, 240523, novel prims)
        RETURNS:
        - Appends to "task" tokens column shape_clust_idx, which is index (0,1,2,..) that is globally
        indexing unique shapes across entire dataset.
        - map_datidx_to_clust_idxs, dict: stroke index after flatten:clust idx
        - list_list_clustidx, list of list of idx, matching D.Dat structure.
        """
        
        from pythonlib.drawmodel.strokedists import distStrokWrapperMult
        from pythonlib.dataset.dataset_strokes import DatStrokes

        CLUSTER_MERGE_METHOD = "intersect"

        savedir_preprocess = self.make_savedir_for_analysis_figures_BETTER("preprocess_general")
        savedir_preprocess = f"{savedir_preprocess}/cluster_strokes_to_rename_shapes"
        os.makedirs(savedir_preprocess, exist_ok=True)

        # Extract all stroke tokens
        if False: # cant do this if tokens not defined yet
            token_ver = "task"
            DF = self.tokens_extract_variables_as_dataframe(["shape", "Prim"], token_ver)
            strokes = [p.Stroke() for p in DF["Prim"]]
        else:
            # Extract strokes here
            LIST_STROKES_TASK = self.Dat["strokes_task"].values.tolist()
            strokes = []
            map_idx_to_orig_trial_stroke = {}
            map_orig_trial_stroke_to_idx = {}
            idx = 0
            for i, _strokes in enumerate(LIST_STROKES_TASK):
                for j, _strok in enumerate(_strokes):
                    strokes.append(_strok)
                    map_idx_to_orig_trial_stroke[idx] = (i, j)
                    map_orig_trial_stroke_to_idx[(i, j)] = idx
                    idx+=1

        # Get pairwise distances between each stroke.
        # simmat = distStrokWrapperMult(strokes, strokes, distancever="hausdorff", align_to_center=True, ploton=True)
        simmat = distStrokWrapperMult(strokes, strokes, distancever="hausdorff_norm_dist", align_to_center=True, ploton=True)
        # simmat = distStrokWrapperMult(strokes, strokes, distancever="hausdorff_max", align_to_center=True, ploton=True)
        # simmat = distStrokWrapperMult(strokes, strokes, distancever="hausdorff_max", align_to_onset=True, ploton=True)

        # Plot histogram of similarityies
        fig, axes = plt.subplots(1,2, figsize=(10,5))

        ax = axes.flatten()[0]
        ax.hist(simmat.reshape(-1), bins=100)
        ax.axvline(THRESH_DIST)
        ax.set_xlabel("all pairwise similairities (vline=thresh)")

        ax = axes.flatten()[1]
        ax.hist(simmat.reshape(-1), bins=200)
        ax.set_xlim([0.9, 1])
        ax.set_title("zooming in")
        ax.axvline(THRESH_DIST)

        savefig(fig, f"{savedir_preprocess}/pairwise_sims_histogram.pdf")

        # # Cluster them
        # n = len(strokes)
        # list_clusters = []
        # for idx_stroke in range(n):
        #     inds = np.argwhere(simmat[idx_stroke, :]>THRESH_DIST)[:,0].tolist()
        #     assert idx_stroke in inds

        #     # Look for the cluster holding this
        #     _found = False

        #     # 1. Check if clusters already added
        #     for cl in list_clusters:
                
        #         if CLUSTER_MERGE_METHOD=="intersect":
        #             # Method 1: if these clusters intersect, then merge
        #             if any([i in cl for i in inds]):
        #                 # they intersect. merge them
        #                 for i in inds:
        #                     if i not in cl:
        #                         print(f"[Clustering shapes] Adding index {i} to cluster {cl} (means that there was partial intersection...):")
        #                         cl.append(i)
        #                 _found = True
        #                 break
        #         elif CLUSTER_MERGE_METHOD=="identity":
        #             # Method 2: must match the cluster exaclty. if not then add it to end,
        #             # and becuase of this, it will fail sanity check later.
        #             if cl == inds:
        #                 # inds have already been added
        #                 _found = True
        #                 break
        #         else:
        #             print(CLUSTER_MERGE_METHOD)
        #             assert False

        #     if not _found:
        #         # Then add it
        #         list_clusters.append(inds)

        # Cluster them
        # This method adds strokes one by one if the stroke's matches are in a cluster,
        # in a greedy fasion. This is guaranteed to add each stroke to one and only one cluster.
        n = len(strokes)
        list_clusters = []
        for idx_stroke in range(n):
            inds = np.argwhere(simmat[idx_stroke, :]>THRESH_DIST)[:,0].tolist()
            assert idx_stroke in inds

            # Look for the cluster holding this
            _found = False
            for cl in list_clusters:
                if any([i in cl for i in inds]):
                    # they intersect. Add this stroke to this cluster
                    cl.append(idx_stroke)
                    _found = True
                    break
            if not _found:
                # Then add it
                list_clusters.append([idx_stroke])

        # Sanity check - every ind is in one and only one cluster.
        tmp = []
        for cl in list_clusters:
            tmp.extend(cl)
        tmp = sorted(tmp)
        if not sorted(tmp) == list(range(len(strokes))):
            if THRESH_DIST < 0.99 and CLUSTER_MERGE_METHOD=="identity":
                # Then try again, this time using 0.99
                # This is becuase if less, then it is techncialyl possible for an index to be
                # in multiple shape groups (a simialr to b simiar to c, but a not to c)
                # You must confirm in preprocess_general plots that shape groups are distinct.
                print("** FAILED, so trying shapesemantic_cluster_taskclass_strokes_to_rename_shapes() again, using THRESH:", 0.99)
                return self.shapesemantic_cluster_taskclass_strokes_to_rename_shapes(THRESH_DIST = 0.99)
            else:
                # Throw error
                print("THRESH_DIST:", THRESH_DIST)
                print(len(tmp))
                print(len(strokes))
                print(tmp)
                for i, val in enumerate(tmp):
                    if i!=val:
                        print("First index that doesnt match:", i, val)
                        break
                assert False

        # Now assign each index (stroke) to its cluster
        map_datidx_to_clust_idxs = {}
        map_cluster_to_inds = {}
        for idx_clust, inds_in_clust in enumerate(list_clusters):
            map_cluster_to_inds[idx_clust] = inds_in_clust
            for ind in inds_in_clust:
                map_datidx_to_clust_idxs[ind] = idx_clust

        ########## MORE SANITY CHECKS
        if False: # doesnt make much sense
            # Sanity checks
            n_same = np.sum(simmat>0.99)
            n_tot = simmat.shape[0]*simmat.shape[1]

            # Expect, usually, much more that are identical, vs. those same...
            nclose = np.sum((simmat>0.9) & (simmat<0.99))
            nidentical = np.sum((simmat>0.99))
            assert nidentical>nclose
            assert nclose/nidentical<0.1

            # Usulaly, expeect not that many shapoes.
            n_shapes = len(map_cluster_to_inds)
            assert n_shapes<150, "weird, I usually dont have so many unique shapes..."   

        ########### PLOTS

        if False: # This fails, since DatStrokes looks for  beh token,which isnt readyu
            # Plot diagnostic, each cluster, plot 20 example shapes (task).
            # Finally, plot each class, showing examples
            DS = DatStrokes(self)
            # Use this to check that no clusters have different actual tasks.
            figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(key_subplots="shape", n_examples=20, ver_behtask="task")
            for i, (fig, axes) in enumerate(figholder):
                savefig(fig, f"{savedir_preprocess}/taskimage_examples_each_shape_cluster-sub{i}.pdf")

        # Assign back to tokens
        if False: # Fails, since tokens are not yet exist.
            assert len(map_datidx_to_clust_idxs) == len(DF)
            DF["shape_clust_idx"] = [map_datidx_to_clust_idxs[i] for i in range(len(DF))]
            self.tokens_assign_dataframe_back_to_self_mult(DF, ["shape_clust_idx"], tk_ver=token_ver)

            # Extract list of list of inds
            list_list_clustidx = []
            for ind in range(len(self.Dat)):
                tokens = self.taskclass_tokens_extract_wrapper(ind, token_ver)
                list_list_clustidx.append([tk["shape_clust_idx"] for tk in tokens])
                for j, tk in enumerate(tokens):
                    assert j==tk["ind_taskstroke_orig"], "just sanity check"
        else:
            # Extract list of idxs for each row of D.Dat
            list_list_clustidx = []
            for i, _strokes in enumerate(LIST_STROKES_TASK):
                list_clust_idx = []
                for j, _strok in enumerate(_strokes):
                    idx_flat = map_orig_trial_stroke_to_idx[(i, j)]
                    idx_shape_clust = map_datidx_to_clust_idxs[idx_flat]
                    list_clust_idx.append(idx_shape_clust)
                list_list_clustidx.append(list_clust_idx)

        return map_datidx_to_clust_idxs, list_list_clustidx

    def shapesemantic_taskclass_map_between_shape_and_shape_orig(self):
        """
        REturn dict mapping between shape, as is currently named, and original shape,
        which is like circle-4-2-3, whch can be different when you have chajged the
        name due to novel shapes
        """

        token_ver = "task"
        df = self.tokens_extract_variables_as_dataframe(["shape", "Prim"], token_ver)

        # Extract the original shape, which should have been overwritten in rprepovessing, but is useufl as a category
        # to anchor the variations.
        list_shape_orig = []
        map_shape_to_shape_orig = {}
        map_shape_orig_to_shape = {}
        for shape, P in zip(df["shape"], df["Prim"]):
            map_shape_to_shape_orig[shape] = P.shape_oriented()
            map_shape_orig_to_shape[P.shape_oriented()] = shape

        return map_shape_to_shape_orig, map_shape_orig_to_shape

    def shapesemantic_taskclass_cont_morph_extract_params(self, ind):
        """ Get params for this trial realted to it as cotniuoisn morph, which is
        an interpolation between two other tasks, e..,g, some frac interpolation, or
        morph between line (first half) and V (second half)
        RETURNS:
        - info_base_prims, lsit of 2 dicts, one for each base prim, e.g.,
        [{'los': ('singleprims', 113, 11), 'shape': ('V', 2, 2, 0)},
            {'los': ('singleprims', 113, 10), 'shape': ('arcdeep', 4, 2, 0)}]
        OR None, if this trial was not a morph.
        """

        Tt = self.taskclass_extract_ml2(ind)

        if len(Tt.get_tasknew()["Info"])>0 and "MorphSavedInfo" in Tt.get_tasknew()["Info"].keys():
            MorphSavedInfo = Tt.get_tasknew()["Info"]["MorphSavedInfo"]

            # This is a morph between two base prims. For each of the base prims, get info
            info_base_prims = []

            for idx in [1,2]:

                # LOS
                los_ver = MorphSavedInfo[f"{idx}"]["Task_info"]["load_old_set_ver"]
                los_num = int(MorphSavedInfo[f"{idx}"]["Task_info"]["load_old_set_setnum"])
                los_ind = int(MorphSavedInfo[f"{idx}"]["Task_info"]["load_old_set_indthis"])
                los = (los_ver, los_num, los_ind)

                # (sh, lev, rot, refl)
                assert len(MorphSavedInfo["1"]["Objects"]["Features_Active"]["prot_shape"])==1, "assuming only one object per task"

                sh = (MorphSavedInfo[f"{idx}"]["Objects"]["Features_Active"]["prot_shape"]["1"], 
                int(MorphSavedInfo[f"{idx}"]["Objects"]["Features_Active"]["prot_level"]["1"]), 
                int(MorphSavedInfo[f"{idx}"]["Objects"]["Features_Active"]["prot_rot"]["1"]), 
                int(MorphSavedInfo[f"{idx}"]["Objects"]["Features_Active"]["prot_refl"]["1"]))

                # Params for the morph
                frac = Tt.get_tasknew()["Info"]["MorphParams"]["2"][0][0]
                flip_task1 = int(Tt.get_tasknew()["Info"]["MorphParams"]["3"])==1
                flip_task2 = int(Tt.get_tasknew()["Info"]["MorphParams"]["4"])==1
                frac_func = Tt.get_tasknew()["Info"]["MorphParams"]["5"]

                # Combine
                info_base_prims.append({
                    "los":los,
                    "shape":sh,
                    "frac":frac,
                    "flip_task1":flip_task1, 
                    "flip_task2":flip_task2,
                    "frac_func":frac_func
                })
        else:
            info_base_prims = None

        return info_base_prims
        
    def shapesemantic_classify_novel_shape(self, DS=None):
        """
        Append columns idnicating wehther shapes are novel, eacjh stroke
        a shape, using the semantic lable of the shape.
        NOTE: for any rows that are "character" task_kind, skips, since might
        run into error if this is using cluster labels.. (?) either way is not
        relevant.
        :return: None, but appends to self.Dat two columns:
        'shape_is_novel_list" (tuple)
        "shape_is_novel_all" (bool)
        """

        # Get learned shapes
        if DS is None:
            from pythonlib.dataset.dataset_strokes import DatStrokes
            DS = DatStrokes(self)
        map_shape_to_shapesemantic = DS.shapesemantic_stroke_shape_cluster_database()
        labels_learned = list(map_shape_to_shapesemantic.values())

        # Extract from D
        labels_all_trials = []
        list_novels = []
        list_novels_all_strokes = []
        for i, row in self.Dat.iterrows():
            Tk = self.taskclass_tokens_extract_wrapper(i, "task", return_as_tokensclass=True)
            if "shape_semantic" not in Tk.Tokens[0].keys():
                Tk.features_extract_wrapper(["shape_semantic"])
            tokens_task = Tk.Tokens

            labels = []
            novels = []
            for tok in tokens_task:
                lab = tok["shape_semantic"]
                # try:
                #     lab = tok["Prim"].label_classify_prim_using_stroke_semantic()
                # except Exception as err:
                #     print(tok)
                #     print(row["trialcode"])
                #     self.plotSingleTrial(i)
                #     assert False, "why..."
                labels.append(lab)
                novels.append(lab not in labels_learned)

            labels_all_trials.append(labels)
            list_novels.append(tuple(novels))
            list_novels_all_strokes.append(all(novels))

        # Append to self.Dat
        self.Dat["shape_is_novel_list"] = list_novels
        self.Dat["shape_is_novel_all"] = list_novels_all_strokes
        self.Dat["shape_semantic_labels"] = labels_all_trials

        # FAILS -- This looks at beh strokes..
        # if DS is None:
        #     from pythonlib.dataset.dataset_strokes import DatStrokes
        #     DS = DatStrokes(self)
        #
        # list_novels = []
        # list_novels_all_strokes = []
        # for i, row in self.Dat.iterrows():
        #     if row["task_kind"] == "character":
        #         # then this is not novel shape
        #         novels = [False for _ in range(len(row["strokes_beh"]))]
        #     else:
        #         tc = row["trialcode"]
        #         novels = DS.dataset_extract_strokeslength_list(tc, "shape_is_novel")
        #
        #     list_novels.append(tuple(novels))
        #     list_novels_all_strokes.append(all(novels))
        #     # assert all(novels) or not any(novels)
        #
        # self.Dat["shape_is_novel_list"] = list_novels
        # self.Dat["shape_is_novel_all"] = list_novels_all_strokes


    ############### MICROSTIM
    def microstim_assign_catch_trial_objectclass(self, PRINT=False):
        """ catch trial means there were no objclass rules
        on this trial. STIM and non-stim CATCH
        Appends new col to self.Dat["catch_trial"]
        """

        list_catch = []
        for ind in range(len(self.Dat)):
            nrules = len(self.blockparams_extract_single_combined_task_and_block(ind)["task_objectclass"]["RuleList"])
            # append True if there are no rules
            list_catch.append(nrules==0)
        self.Dat["catch_trial"]=list_catch
        print("New column catch_trial")
        if PRINT:
            # should be correlated
            self.grouping_print_n_samples(["catch_trial", "epochset"])

    ############# ASSERTIONS
    def _check_is_single_dataset(self):
        """ True if single, False, if you 
        contcatenated multipel datasets"""

        if len(self.Metadats)>1:
            return False
        else:
            return True

    def _check_character_fixed_los_not_multiple_hashes(self):
        """ Checks that each unique character that is a fixed task
        has only one hash, and vice versa
        Fails assertion if not true"""

        groupdict = self.grouping_get_inner_items("los_info", "character")
        for los, charlist in groupdict.items():
            if los[0] is not None:
                assert len(charlist)==1

        groupdict = self.grouping_get_inner_items("character", "los_info")
        for char, los in groupdict.items():
            if char is not None:
                assert len(los)==1

    def _check_consistency(self):
        """ sanity checks, should run this every time after fully load 
        a Dataset object and enter data
        """

        # No repeated unique trials
        assert len(self.Dat["trialcode"].unique().tolist()) == len(self.Dat), "unique trials occur in multipel rows.., did you concat dset to isetlf?"


        _checkPandasIndices(self.Dat)



    ############# CLEANUP
    def _cleanup_preprocess_each_time_load_dataset(self, rename_shapes_if_cluster_labels_exist=True):
        # e..g, if loading saved dataset using neuralmonkey
        print("=== CLEANING UP self.Dat (_cleanup_reloading_saved_state) ===== ")
        if not hasattr(self, "_BehClassExtracted"):
            self._BehClassExtracted = None

        # SInce this is a pre-saved dataset, some failure modes to address
        # - taskclass
        # for i in range(len(self.Dat)):
        #     T = self.Dat.iloc[i]["Task"]
        #     if not hasattr(T, "_BehClassExtracted"):
        #         T._TokensLocked = None


        # 2/4/24 - Decided to just rerun entire preprocessDat, since that is actualyl very quick,
        # except the step of behclass_preprocess_wrapper, but that would be done here anyway (the
        # purpose of the line self._BehClassExtracted = None
        from pythonlib.dataset.dataset_preprocess.general import preprocessDat
        expt = self.expts(force_single=True)[0]
        self._analy_preprocess_done=False
        self, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(self, expt,
                                                    rename_shapes_if_cluster_labels_exist=rename_shapes_if_cluster_labels_exist)

        # print("1.5 dfafasf", self.TokensVersion)
        assert hasattr(self, "TokensStrokesBeh"), "how is this possible? It should have run tokens_generate_replacement_quick_from_beh..."    

        return GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES

    def cleanup_wrapper(self, ver):
        """ To organize notes on cleanup inup
        REcipes for cleaning up , since htey are all over the place.

        HIERARCHY of steps:
        concat
        -- cleanup()

        load_dataset_helper
            _main_loader
                -- cleanup()
                    _cleanup_cut_strokes_whose_onset_is_at_fixation
                    _cleanup_remove_strokes_that_are_actually_fixation_or_done
                -- _check_consistency
        -- load_tasks_helper
            -- _cleanup_using_tasks
        -- _cleanup_preprocess_each_time_load_dataset
        """

        if ver == "substrokes":
            # Since _cleanup_preprocess_each_time_load_dataset fails, becuase
            # the old toeksn will not match teh new strokes, which may be pruned..
            # just skip that step for now
            self._cleanup(remove_dot_strokes=False, remove_online_abort=False,
                          remove_bad_strokes=False, smooth_strokes=False)
            self._check_consistency()
            # self._cleanup_preprocess_each_time_load_dataset()
        elif ver == "no_pruning_strokes":
            # Do everything, except no pruning of strokes.
            self._cleanup(remove_dot_strokes=False, remove_online_abort=False,
                          remove_bad_strokes=False, smooth_strokes=False)
            self._check_consistency()
            self._cleanup_preprocess_each_time_load_dataset()
        else:
            print(ver)
            assert False


    def _cleanup(self, remove_dot_strokes, remove_online_abort,
                 remove_bad_strokes, smooth_strokes):
        """ automaitcalyl clean up using default params
        Removes rows from self.Dat, and resets indices.
        - This should be run BEFORE preprocessDat, or else this might overwrite changes 
        from that.
        """
        print("=== CLEANING UP self.Dat ===== ")
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # dfthis = self.Dat[self.Dat["random_task"]==False]
        # dfthis = self.Dat
        # tasklist = dfthis["unique_task_name"]
        # print([t for t in tasklist if "set-53" in t])
        # # print(sorted(dfthis["unique_task_name"]))
        # assert False

        ####### Remove online aborts
        # print("ORIGINAL: online abort values")
        # print(self.Dat["online_abort"].value_counts())
        # idx_good = self.Dat["online_abort"].isin([None])
        # self.Dat = self.Dat[idx_good]
        # print(f"kept {sum(idx_good)} out of {len(idx_good)}")
        # print("removed all cases with online abort not None")

        # reset 
        self.Dat = self.Dat.reset_index(drop=True)
        # print(sum(self.Dat["trialcode"] == "220608-1-132"))
        # print(sum((self.Dat["trial"] == 132) & (self.Dat["session"] == 1) & (self.Dat["date"] == "220608")))
        # print(sum((self.Dat["trial"] == 132) & (self.Dat["session"] == 1)))
        # print(sum(self.Dat["trial"] == 132))
        # assert False

        # Remove motor events that are derived. should should be revomputed.
        keys_keep = []
        LIST_THIS = []
        for ind in range(len(self.Dat)):
            this = self.Dat.iloc[ind]["motortiming"]
            this = {k:v for k,v in this.items() if k in keys_keep}
            LIST_THIS.append(this)
        self.Dat["motortiming"] = LIST_THIS

        keys_keep = ["go_cue", "raise", "done_touch", "done_triggered"]
        LIST_THIS = []
        for ind in range(len(self.Dat)):
            this = self.Dat.iloc[ind]["motorevents"]
            this = {k:v for k,v in this.items() if k in keys_keep}
            LIST_THIS.append(this)
        self.Dat["motorevents"] = LIST_THIS

        # Make sure strokes are all in format (N,3)
        def F(x):
            strokes_beh = x["strokes_beh"]
            # strokes_task = x["strokes_task"]
            for i, strok in enumerate(strokes_beh):
                if len(strok.shape)==1:
                    assert strok.shape==(3,), "is maybe (2,),without time var?"
                    strokes_beh[i] = strok.reshape(1,3)
                assert strok.shape[1]==3
            return strokes_beh
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_beh")

        # Make sure expts is the correct name, becuase in sme cases
        # the automaticlaly extracted name is from the filenmae, which may be
        # incorrect
        if "metadat_probedat" in self.Metadats[0].keys():
            # Since only starting saving this later...
            def F(x):
                idx = x["which_metadat_idx"]
                return self.Metadats[idx]["metadat_probedat"]["expt"]
            self.Dat = applyFunctionToAllRows(self.Dat, F, "expt")


        ####### Remove columns of useless info.
        cols_to_remove = ["probe", "feedback_ver_prms", "feedback_ver",
            "constraints_to_skip", "prototype", "saved_setnum", "tasknum", 
            "resynthesized", "resynthesized_path", "resynthesized_trial",
            "resynthesized_setnum", "resynthesized_setname", "modelscore", 
            "modelcomp", "hausdorff_positive", "circleness", "kind"]
        cols_to_remove.extend(["los_setinds", "los_setindthis", "los_setname", "los_setnum"]) # additional things to remove, these should use taskclass
        for col in cols_to_remove:
            if col in self.Dat.columns:
                del self.Dat[col]
        print("Deleted unused columns from self.Dat")


        ####### Construct useful params
        # For convenience, add column for whether task is train or test (from
        # monkey's perspective)
        def F(x):
            if x["taskgroup"] in ["train_fixed", "train_random", "T1"]:
                return "train"
            elif x["taskgroup"] in ["G2", "G3", "G4", "test_fixed", "test_random"]:
                return "test"
            elif x["taskgroup"] in ["undefined"]:
                # Not sure yet. Modify this later in postprocessing
                return "undefined"
            else:
                # Just use whatever is there
                return x["taskgroup"]
        print("applying monkey train test names")
        self.Dat = applyFunctionToAllRows(self.Dat, F, "monkey_train_or_test")


        ### reset tvals to the new earliest data
        self._get_standard_time()

        # Sort so is in increasing by date
        self.Dat = self.Dat.sort_values("tvalfake", axis=0).reset_index(drop=True)

        # Remove trial day since this might not be accurate anymore (if mixing datasets)
        if "trial_day" in self.Dat.columns:
            self.Dat = self.Dat.drop("trial_day", axis=1)


        # Replace epoch with rule, if that exists
        # 8/17/22 - now done in preprocessDat, because here would overwrite preprocessDat.
        # def F(x):
        #     idx = x["which_metadat_idx"]
        #     if self.Metadats[idx]["rule"]:
        #         return self.Metadats[idx]["rule"]
        #     else:
        #         return idx+1
        # self.Dat = applyFunctionToAllRows(self.Dat, F, "epoch")

        # Add new column where abort is now True or False (since None was hjard to wrok with)
        def F(x):
            if x["online_abort"] is None:
                return False
            else:
                return True
        self.Dat = applyFunctionToAllRows(self.Dat, F, "aborted")
        self.Dat["online_abort_ver"] = self.Dat["online_abort"] # so not confused in previous code for Non
        self.Dat = self.Dat.drop("online_abort_ver", axis=1)

        # print(self.Dat["aborted"].value_counts())
        # assert False
        # print(remove_online_abort)
        # print(sum(self.Dat["trialcode"] == "220608-1-132"))
        if remove_online_abort:
            print("HERE")
            # Remove any trials that were online abort.
            self.Dat = self.Dat[self.Dat["aborted"]==False]
            print("Removed online aborts")

        ################ STROKES
        self.cleanup_strokes_all_wrapper(remove_dot_strokes, remove_bad_strokes,
                                         smooth_strokes)

        ################# OTHER STUFF
        if "Task" in self.Dat.columns:
            self._cleanup_using_tasks() 

        # assign a "character" name to each task.
        self._cleanup_rename_characters()

        # dates for summary , make sure assigned
        # for MD in self.Metadats.values():
        #     print(MD[metadat_probedat])
        list_dates = [MD["metadat_probedat"]["dates_for_summary"] for MD in self.Metadats.values()]
        list_dates = [str(d) for dates in list_dates for d in dates]
        # update "is in summary date column"
        def F(x):
            return str(x["date"]) in list_dates
        self.Dat = applyFunctionToAllRows(self.Dat, F, "insummarydates")
        print("Updated columns: insummarydates, using Metadats")

        # Assigning a new column: date_sess, which is date-sess
        def F(x):
            return f"{x['date']}-{x['session']}"
        self.Dat = applyFunctionToAllRows(self.Dat, F, 'date_sess')

        # # fix a problem, sholdnt throw out epoch name
        # self.supervision_epochs_extract_orig()

        #########
        self.Dat = self.Dat.reset_index(drop=True)

        ### confirm that trialcodes are all unique (this is assumed for subsequent stuff)
        assert len(self.Dat["trialcode"])==len(self.Dat["trialcode"].unique().tolist()), "not sure why"

    def _cleanup_using_tasks(self, unique_names_post_Sep17=False):
        """ cleanups that require "Tasks" in columns,
        i.e,, extract presaved tasks
        - unique_names_post_Sep17, imprioving naming for random tasks. see hash function./
        """

        assert "Task" in self.Dat.columns

        # Replace unique name with new one, if tasks have been loaded
        if unique_names_post_Sep17: 
            def F(x):
                return self._task_hash(Task=x["Task"], original_ver_before_sep21=False)
        else:
            def F(x):
                return self._task_hash(Task=x["Task"], original_ver_before_sep21=True)
            # Equivalent to line above.
            # def F(x):
            #     # return x["Task"].Params["input_params"].info_generate_unique_name()
        self.Dat = applyFunctionToAllRows(self.Dat, F, "unique_task_name")

        # task cartegories should include setnum
        def F(x):
            return x["Task"].get_task_category()
        self.Dat = applyFunctionToAllRows(self.Dat, F, "task_stagecategory")

        # rename charactesrs
        self._cleanup_rename_characters()

        # replace strokes_task with strokes from Task.
        def F(x):
            if str(x["date"])==str(210828) and x["expt"]=="gridlinecircle":
                # Then lollis were chunked into single strokes. but should
                # really be multipel strokes. 
                x["Task"]._split_strokes_large_jump()
            return x["Task"].Strokes    
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_task")

        # Task kind (e..g, prims_on_grid, character, etc)
        def F(x):
            return x["Task"].get_task_kind()
        self.Dat = applyFunctionToAllRows(self.Dat, F, "task_kind")

        ################# RECOMPUTE MOTOR STUFF.


    def _cleanup_remove_strokes_that_are_actually_fixation_or_done(self,
            PLOT_EXAMPLE_BAD_TRIALS=False):
        """ remove strokes that are actually "dots" at fixation or done. find cases
        that are close to either (less than MAX_DIST) and are closer that that point
        than to any other image strokes
        RETURNS:
        - updates self.Dat["strokes_beh"]
        """

        MAX_DIST = 100 # Emprically seems good.
        from pythonlib.dataset.dataset_strokes import DatStrokes
        ds = DatStrokes()

        for location_to_test in ["origin", "donepos"]:
            
            list_stroks_remove = []
            LIST_STROKES_BEH = []
            list_inds_replaced = []
            print("-- CHECKING ", location_to_test)
            print("--- idat, trialcode, strok inds to remove, len strokes beofre remofe, len strokes after:")
            for idat in range(len(self.Dat)):

                strokes_beh = self.Dat.iloc[idat]["strokes_beh"]
                loc_fix = self.Dat.iloc[idat][location_to_test]
                trialcode = self.Dat.iloc[idat]["trialcode"]

                if loc_fix is None and location_to_test == "donepos":
                    # Then this has no done button.. single prims (e.g).
                    LIST_STROKES_BEH.append(strokes_beh)
                    continue

                if loc_fix is None:
                    print(location_to_test)
                    print(trialcode)
                    assert False, "probably doesnt have donie button? (single prim?)"

                inds_remove = []
                for _is, strok in enumerate(strokes_beh):

                    max_dist_strok_to_fix = np.max(np.linalg.norm(strok[:,[0,1]] - loc_fix, axis=1))

                    if max_dist_strok_to_fix<MAX_DIST:

                        # check that it is far from all strokes
                        list_dists = []
                        list_strok_task = self.Dat.iloc[idat]["strokes_task"]
                        for st in list_strok_task:
                            d = ds._dist_strok_pair(strok, st)
                            list_dists.append(d)

                        closer_to_fix_than_to_image = all([d>max_dist_strok_to_fix for d in list_dists])

                        if closer_to_fix_than_to_image:

                            # Then throw out this strok
                            list_stroks_remove.append([idat, _is])
                            inds_remove.append(_is)

            #                 print(idat, _is, max_dist_strok_to_fix, ' -- ', list_dists, closer_to_fix_than_to_image)

                if len(inds_remove)==0:
                    # keep
                    LIST_STROKES_BEH.append(strokes_beh)
                else:
                    # prune
                    tmp = [strok for i, strok in enumerate(strokes_beh) if i not in inds_remove]
                    print(idat, trialcode, inds_remove, len(strokes_beh), len(tmp))
                    LIST_STROKES_BEH.append(tmp)
                    list_inds_replaced.append(idat)

            assert len(LIST_STROKES_BEH)==len(self.Dat), "Sanity cehck"

            if len(list_inds_replaced)>0:
                if PLOT_EXAMPLE_BAD_TRIALS:
                    self.plotMultTrials(list_inds_replaced[:20])

                self.Dat["strokes_beh"] = LIST_STROKES_BEH

                if PLOT_EXAMPLE_BAD_TRIALS:
                    self.plotMultTrials(list_inds_replaced[:20])

    def _cleanup_cut_strokes_whose_onset_is_at_fixation(self, 
        PLOT_DISTROS=False, PLOT_EXAMPLE_BAD_TRIALS=False, DEBUG=False):
        """
        Cleanup cases where ther first stroke reach from go is so quick that the fixation
        hold is counted as part of the initial stroke., This occuers for Luca
        on 230512.
        DOes so in conservative way. Teh onset must be close to fixastion in space. 
        There must be high velocity peak, and the peak must be close in space to onset.
        RETURNS:
        - (updates self.Dat["strokes_beh"])
        - list_inds_replaced, list ints, indices into self.Dat, which were modded.
        """

        from pythonlib.tools.nptools import find_peaks_troughs

        # In prep, get distribtions of some metrics across all data
        # to be able to find outliers.
        list_dist_to_fix = []
        for idat in range(len(self.Dat)):
            strokes_beh = self.Dat.iloc[idat]["strokes_beh"]

            # === distnace from stroke onset to fixation
            loc_on = strokes_beh[0][0,[0,1]]
            loc_fix = self.Dat.iloc[idat]["origin"]

            dist_to_fix = np.linalg.norm(loc_fix - loc_on)
            list_dist_to_fix.append(dist_to_fix)
        min_dist_allowed = np.percentile(list_dist_to_fix, [12])
        min_dist_allowed = np.min(np.append(min_dist_allowed, 50))

        # get distribution of strok speeds across all data
        list_strokesspeed = self.extractStrokeVels(list(range(len(self.Dat))))

        speeds_flat = []
        for S in list_strokesspeed:
            if S is not None:
                for s in S:
                    speeds_flat.append(s)
        speeds_flat = np.concatenate(speeds_flat, axis=0)
        max_speed_allowed = np.percentile(speeds_flat, [99])[0]
        max_speed_allowed = np.max([max_speed_allowed, 1000])


        if PLOT_DISTROS:
            fig, ax = plt.subplots()
            ax.hist(speeds_flat, bins=50)
            ax.set_title(f"max allowed: {max_speed_allowed}")

            fig, ax = plt.subplots()
            ax.hist(list_dist_to_fix, bins=50)
            ax.set_title(f"min allowed: {min_dist_allowed}")

        # Each trial, see if recompute first stroke onset.
        LIST_STROKES_BEH = []
        list_inds_replaced = []
        list_raise_time = []
        print("* UDPATEING onset of first stroke [too close to fixation] (trial, new onset index):")
        for idat in range(len(self.Dat)):
            strokes_beh = self.Dat.iloc[idat]["strokes_beh"]
            raise_time = self.Dat.iloc[idat]["motorevents"]["raise"]

            # === distnace from stroke onset to fixation
            loc_on = strokes_beh[0][0,[0,1]]
            loc_fix = self.Dat.iloc[idat]["origin"]

            dist_to_fix = np.linalg.norm(loc_fix - loc_on)
            if DEBUG:
                print(idat, dist_to_fix, loc_on, loc_fix)
            if dist_to_fix>min_dist_allowed:
                LIST_STROKES_BEH.append(strokes_beh)
                continue

            # === New onset will be at minimum in stroke speed.
            strok_speed = self.extractStrokeVels([idat])[0][0][:,0] # (N,)
            if np.max(strok_speed) < max_speed_allowed:
                LIST_STROKES_BEH.append(strokes_beh)
                continue

            # FIND NEW STROKE ONSET
            inds_peak, inds_trough, _ = find_peaks_troughs(strok_speed, DEBUG)

            # first peak that is greater than max 
            inds_high = np.where(strok_speed>max_speed_allowed)[0].tolist()
            tmp = [i for i in inds_peak if i in inds_high]
            if not len(tmp)==1:
                LIST_STROKES_BEH.append(strokes_beh)
                continue

            ind_peak_above_max_vel = tmp[0]

            # first trough after this
            idx_new_stroke_on = [i for i in inds_trough if i>ind_peak_above_max_vel][0]

            # Find the time of raise, the trough preceding this peek
            idx_new_raise = [i for i in inds_trough if i<ind_peak_above_max_vel][-1]

            # sanity checks
            # - only small fraction of stroke
            if idx_new_stroke_on/len(strok_speed)>0.8:
                # cutting off >0.8 of stroke. dont do this.
                LIST_STROKES_BEH.append(strokes_beh)
                continue
                # FAIL = True

            # location of peak in vel should be close to onsetl
            if np.linalg.norm(strokes_beh[0][ind_peak_above_max_vel, [0,1]] - loc_fix)>200:
                # Then ad-hoc ditsance is breached.
                LIST_STROKES_BEH.append(strokes_beh)
                continue

            #################### new time of raise
            raise_time = strokes_beh[0][idx_new_raise+1,2]
            list_raise_time.append(raise_time)
            # print(self.Dat.iloc[idat]["motorevents"])
            # print(raise_time)
            # print(idx_new_raise)

            ##########  Create new stroke onset
            strokes_beh = [strok.copy() for strok in strokes_beh]
            # strok = strokes_beh[0][idx_new_stroke_on:, :]
            strokes_beh[0] = strokes_beh[0][idx_new_stroke_on:, :]
            LIST_STROKES_BEH.append(strokes_beh)

            print(idat, " -- ", idx_new_stroke_on)

            # print(idat, idx_new_stroke_on)
            list_inds_replaced.append(idat)

        if len(list_inds_replaced)>0:
            if PLOT_EXAMPLE_BAD_TRIALS:
                self.plotMultTrials(list_inds_replaced[:20])

            self.Dat["strokes_beh"] = LIST_STROKES_BEH

            if PLOT_EXAMPLE_BAD_TRIALS:
                self.plotMultTrials(list_inds_replaced[:20])

            # Update motor events
            assert len(list_inds_replaced)==len(list_raise_time)
            for idat, rt in zip(list_inds_replaced, list_raise_time):
                me = self.Dat.iloc[idat]["motorevents"]
                me["raise"] = rt

        return list_inds_replaced

    def _cleanup_rename_characters(self):
        """ makes column called "character" which is the unique name for fixed tasks 
        and the category name for random tasks."""

        def F(x):
            if x["random_task"]:
                # print(x["task_stagecategory"])
                # For random tasks, character is just the task category
                return x["task_stagecategory"]
            else:
                # For fixed tasks, it is the unique task name
                # print(x["unique_task_name"])
                return x["unique_task_name"]
        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="character")

    def _cleanup_remove_trials_with_empty_strokes(self):
        """ Modifies self.Dat, to Remove rows that now have no strokes
        i..e, empty list
        """
        inds_bad = [i for i, strokes in enumerate(self.Dat["strokes_beh"]) if len(strokes)==0]
        if len(inds_bad)>0:
            print("REmoving this many rows becuase they have len 0 strokes:")
            print(len(inds_bad))
            print("These indices: ", inds_bad)
        self.Dat = self.Dat.drop(index=inds_bad).reset_index(drop=True)

    def cleanup_strokes_all_wrapper(self, remove_dot_strokes=True, remove_bad_strokes=True,
                                    smooth_strokes=True):
        """ Holds all things relatd to cleaning up strokes.
        IMportant, becuase there are oreder of operations that matter. e..g,
        want to smooth BEFORE pruning, and so on.
        """

        ###### remove strokes that are empty or just one dot
        dist_min = 5 # This should be greater than 0, becuaes of numerical imprecision, even dot can be >0...
        if remove_dot_strokes:
            strokeslist = self.Dat["strokes_beh"].values
            for i, strokes in enumerate(strokeslist):
                # First, remove empty strokes
                strokes = [s for s in strokes if len(s)>1]

                # Second, remove dist=0 strokes
                from pythonlib.drawmodel.features import strokeDistances
                list_d = strokeDistances(strokes)
                strokes = [s for s,d in zip(strokes, list_d) if d>dist_min]

                strokeslist[i] = strokes
            self.Dat["strokes_beh"] = strokeslist
        self._cleanup_remove_trials_with_empty_strokes()

        # Cleanup strokes concatted with fixation touch.
        if remove_bad_strokes:
            self._cleanup_cut_strokes_whose_onset_is_at_fixation(PLOT_DISTROS=True,
                PLOT_EXAMPLE_BAD_TRIALS=True)
            self._cleanup_remove_trials_with_empty_strokes()

            self._cleanup_remove_strokes_that_are_actually_fixation_or_done(
                PLOT_EXAMPLE_BAD_TRIALS=True)
            self._cleanup_remove_trials_with_empty_strokes()

        if smooth_strokes:
            self.Dat["strokes_beh"] = self.strokes_smooth_preprocess()
            self._cleanup_remove_trials_with_empty_strokes()

        # RE-extract stroke motor events and timing
        # ORiginal items:
            # {'nstrokes': 4,
            #  'time_go2raise': 1.0335998000000073,
            #  'time_raise2firsttouch': -2.4320000000000004,
            #  'dist_raise2firsttouch': 14.599874703419573,
            #  'sdur': 5.280000000000002,
            #  'isi': 1.0879999999999992,
            #  'time_touchdone': 0.4158021000000254,
            #  'dist_touchdone': 479.12381428563003,
            #  'dist_strokes': 1886.1317695259509,
            #  'dist_gaps': 1292.682034468351}

            #  {'go_cue': 6.022400199999993,
            #  'raise': 5.912,
            #  'ons': [4.624, 7.424, 8.744, 10.04],
            #  'offs': [7.056, 8.408, 9.656, 10.992],
            #  'done_touch': 11.376,
            #  'done_triggered': 11.407802100000026}
        # currently just doing ons and offs
        for ind in range(len(self.Dat)):
            ons, offs = self.strokes_onsets_offsets(ind)

            # Place into ME
            me = self.Dat.iloc[ind]["motorevents"]
            me["ons"] = ons
            me["offs"] = offs

    # i.e. tvals from probedat might not be accurate if here combining multiple datsets
    def _get_standard_time(self, first_date=None):
        """ get real-world time for each trial, but using a 
        decimalized version that is not actual time, but is
        good for plotting timecourses. relative to first_date, so that
        for example, noon on first day is 1.5
        INPUTS:
        - first_date, if None, then uses the earlisest date over trials. Otherwise
        pass in int or string (format YYMMDD)
        RETURNS:
        - modifies self.Dat, removes tval col and adds new col called "tvalfake"
        """
        from pythonlib.tools.datetools import standardizeTime
        
        # Remove old tvals.
        if "tval" in self.Dat.columns:
            self.Dat = self.Dat.drop("tval", axis=1)

        # Add new tvals.
        if first_date is None:
            first_date = min(self.Dat["date"])

        first_date = str(first_date) + "-000000"
        def F(x):
            dt = x["datetime"]
            # return standardizeTime(dt, first_date, daystart=0.417, dayend=0.792)
            return standardizeTime(dt, first_date)
        self.Dat= applyFunctionToAllRows(self.Dat, F, "tvalfake")

        # tvalday is just the day (plus 0.6) useful for plotting
        def F(x):
            return np.floor(x["tvalfake"])+0.6
        self.Dat= applyFunctionToAllRows(self.Dat, F, "tvalday")



    def find_dataset(self, animal, expt, assert_only_one=True, rule="", take_most_recent=True):
        """ helper to locate path for presaved (from Probedat) dataset
        can then use this toload  it
        - assert_only_one, then asserts that one and only one path found.
        PARAMS:
        - take_most_recent, bool, applies before assert_only_one applies.h
        RETURNS:
        - list of strings, paths.
        """
        from pythonlib.tools.expttools import findPath

        # Collects across these dirs
        SDIR_LIST = [f"{base_dir}/database", f"{base_dir}/database/BEH",
            f"{PATH_ANALYSIS_OUTCOMES_SERVER}/database", f"{PATH_ANALYSIS_OUTCOMES_SERVER}/database/BEH"
            ]

        def _find(SDIR):
            pathlist = findPath(SDIR, [[f"{animal}-", f"{expt}-", f"{rule}-"]], "dat", ".pkl", True)
            return pathlist

        pathlist = []
        for SDIR in SDIR_LIST:
            pathlist.extend(_find(SDIR))

        if len(pathlist)==0:
            print("HErE")
            print(SDIR_LIST)
            print(animal, expt, rule)
            assert False

        # pathlist = findPath(SDIR, [[animal, expt]], "dat", ".pkl", True)
            
        if take_most_recent:
            # look at filename for the date.
            from pythonlib.tools.expttools import get_filename_most_recent_by_date
            pathlist = [get_filename_most_recent_by_date(pathlist)]

        if assert_only_one:
            assert len(pathlist)==1
            
        return pathlist

    def preprocessGoodCheckLog(self, params_allowed_done):
        """
        Fails if finds that self has logged a param which is not included in 
        params_allowed_done (list of str)
        """
        # if not all([p in self.Log_preprocessGood for p in params_allowed_done]):
        if not all([p in params_allowed_done for p in self.Log_preprocessGood]):
            print("Allowed params: ", params_allowed_done)
            print("Actual done params: ", self.Log_preprocessGood)
            assert False

    def preprocessGood(self, 
            # ver="modeling", 
            ver = None,
            params=None, apply_to_recenter="all",
            frac_touched_min = None, ft_decim_min=None, shortness_min=None,
            nmin_trials = None,
            DRY_RUN=False):
        """ save common preprocess pipelines by name
        returns a ordered list, which allows for saving preprocess pipeline.
        - ver, string. uses this, unless params given, in which case uses parasm
        - params, list, order determines what steps to take. if None, then uses
        ver, otherwise usesparmas
        """

        if DRY_RUN:
            print("** DRY RUN!!")
            # Make a copy and then run
            D = self.copy()
            D.preprocessGood(ver, params, apply_to_recenter, frac_touched_min, False)

        else:

            print("*** RUNNING D.preprocessGood using these params:")
            print(params)

            assert self.LockPreprocess==False, "need to unlock, or make a copy then unlock."

            if ver is None and params is None:
                assert False, "must pass in one"

            if len(self.Dat)==0:
                return

            ######## MODE 1 - PRESET PARAMS
            if params is None:
                assert isinstance(ver, str), "you made mistake? this actually is params?"
                
                if ver=="modeling":
                    # recenter tasks (so they are all similar spatial coords)
                    self.recenter(method="each_beh_center", apply_to=apply_to_recenter)

                    # interpolate beh (to reduce number of pts)
                    self.interpolateStrokes()

                    # subsample traisl in a stratified manner to amke sure good represnetaiton
                    # of all variety of tasks.
                    self.subsampleTrials()

                    # Recompute task edges (i..e, bounding box)
                    self.recomputeSketchpadEdges()

                    params = ["recenter", "interp", "subsample", "spad_edges"]
                elif ver=="strokes":
                    # interpolate beh (to reduce number of pts)
                    self.interpolateStrokes()

                    # subsample traisl in a stratified manner to amke sure good represnetaiton
                    # of all variety of tasks.
                    self.subsampleTrials()

                    params = ["interp", "subsample"]

                else:
                    print(ver)
                    assert False, "not coded"

            ######## MODE 2 - LIST OF PARAMS
            else:

                if frac_touched_min is not None and "frac_touched_min" not in params:
                    params.append("frac_touched_min")
                if ft_decim_min is not None and "ft_decim_min" not in params:
                    params.append("ft_decim_min")
                if shortness_min is not None and "shortness_min" not in params:
                    params.append("shortness_min")

                # log what preprocesses done.
                self.Log_preprocessGood.extend(params)

                # Do each preprocess step in params.
                for p in params:
                    print(f"-- Len of D, before applying this param: {p}, ... {len(self.Dat)}")
                    if p=="recenter":
                        self.recenter(method="each_beh_center", apply_to = apply_to_recenter) 
                    elif p=="interp":
                        self.interpolateStrokes()
                    elif p=="interp_spatial_int":
                        self.interpolateStrokesSpatial(strokes_ver = "strokes_beh", 
                            pts_or_interval="int")
                        self.interpolateStrokesSpatial(strokes_ver = "strokes_task", 
                            pts_or_interval="int")
                    elif p=="subsample":
                        self.subsampleTrials()
                    elif p=="spad_edges":
                        self.recomputeSketchpadEdges()
                    elif p=="rescale_to_1":
                        self.rescaleStrokes()
                    elif p=="no_supervision":
                        # Remove trials with online sequence supervision.
                        # Color can be optionally considered a sequence supervision (e.g.,
                        # for charstorkeseq) wheras in other expts it is not (e.g., colsup).
                        if False:
                            LIST_NO_SUPERV = ["off|0||0", "off|1|solid|0", "off|1|rank|0"]
                            self.Dat = self.Dat[self.Dat["supervision_stage_concise"].isin(LIST_NO_SUPERV)]
                        else:
                            # Better, since goes directly to what param matters.
                            # self.Dat = self.Dat[self.Dat["superv_SEQUENCE_SUP"]=="off"] # OLD VERSION, wasnt general enbough
                            # self.Dat = self.Dat[self.Dat["supervision_online"]==False]

                            # dont throw out cases that are "solid_sequence_mask", since these are using supervision
                            # as an actual testing epoch, not just as training.
                            self.Dat = self.Dat[(self.Dat["supervision_online"]==False) | (self.Dat["superv_COLOR_METHOD"]=="solid_sequence_mask")]

                    elif p=="remove_online_abort":
                        # Remove trials with online abort
                        self.Dat = self.Dat[self.Dat["aborted"]==False]
                    elif p=="correct_sequencing_binary_score":
                        # correct sequence matching at lesat one of the grammar parses.
                        # strict, only if complete entire trial, and is correct sequnece. i..e
                        # if correct so far, but online abort, then exclude.
                        assert "success_binary_quick" in self.Dat.columns, "first run self.grammarparses_successbinary_score(False). or grammarmatlab_successbinary_score Cant do here, since might overwrite diesred states."
                        inds_keep = []
                        for i in range(len(self.Dat)):
                            res = self.grammarparsesmatlab_score_wrapper(i)
                            if res in ["sequence_correct"]:
                                inds_keep.append(i)
                        self.subsetDataframe(inds_keep)
                    elif p=="wrong_sequencing_binary_score":
                        # incorrect sequencing. actually seuqenceing wrong, so does not include if failure is just due to onmien abort.
                        assert "success_binary_quick" in self.Dat.columns, "first run self.grammarparses_successbinary_score(False). Cant do here, since might overwrite diesred states."
                        inds_keep = []
                        for i in range(len(self.Dat)):
                            res = self.grammarparsesmatlab_score_wrapper(i)
                            if res in ["sequence_incorrect_online_abort", "sequence_incorrect_but_no_abort"]:
                                inds_keep.append(i)
                        self.subsetDataframe(inds_keep)
                    elif p=="one_to_one_beh_task_strokes_allow_unfinished":
                        """ each task stroke is maatched to max one beh stroke. is ok
                        if didnt get all task strokes
                        """
                        list_good = []
                        for i in range(len(self.Dat)):
                            list_good.append(self.sequence_compute_each_task_stroke_only_one_beh(i))
                        self.Dat = self.Dat[list_good]
                    elif p=="one_to_one_beh_task_strokes":
                        # must use exactyl 
                        list_good = []
                        for i in range(len(self.Dat)):
                            list_good.append(self.sequence_compute_one_to_one_beh_to_task(i))
                        self.Dat = self.Dat[list_good]
                    elif p=="beh_strokes_at_least_one":
                        # then only keep trials where at least one beh stroke was made
                        list_good = []
                        for i in range(len(self.Dat)):
                            list_good.append(len(self.Dat.iloc[i]["strokes_beh"])>0)
                        self.Dat = self.Dat[list_good]
                    elif p=="correct_sequencing":
                        # Only if beh sequence is consistent with at least one acceptable rule 
                        # based on the epoch.
                        assert False, "this old version uses the matlab rule. change this to use success_binary_parses"
                        bm = self.grammarmatlab_wrapper_extract()
                        self.Dat = self.Dat[(bm.Dat["success_binary_quick"]==True)].reset_index(drop=True)
                    elif p in ["frac_touched_ok", "frac_touched_min"]:
                        assert frac_touched_min is not None
                        # To see what is good value, try:
                        # plot_trials_after_slicing_within_range_values(self, colname, minval, 
                        # maxval, plot_hist=True):
                        if "frac_touched" in self.Dat.columns:
                            self.Dat = self.Dat[self.Dat["frac_touched"]>=frac_touched_min]
                        else:
                            print("SKIPPING, frac_touched doesnt exist in self.Dat")
                    elif p == "ft_decim_min":
                        assert ft_decim_min is not None
                        self.Dat = self.Dat[self.Dat["ft_decim"]>=ft_decim_min]

                    elif p == "shortness_min":
                        assert shortness_min is not None
                        self.Dat = self.Dat[self.Dat["shortness"]>=shortness_min]

                    elif p=="fixed_tasks_only":
                        self.Dat = self.Dat[self.Dat["random_task"]==False]
                    elif p=="remove_repeated_trials":
                        # remove trials that repeat the same task immediately after each otehr. only keep the first 
                        # iter in a set of repeated trials.
                        inds_repeated = self.taskclass_find_repeat_trials()
                        inds_keep = [i for i in range(len(self.Dat)) if i not in inds_repeated]
                        self.subsetDataframe(inds_keep)
                    elif p=="only_dates_with_probes":
                        # Only keep dates that have at least one probe task. Useful for looking
                        # at generalziation.
                        # [OPTIONAL] Only keep days with probe tasks
                        grpdict = self.grouping_get_inner_items("date", "probe")
                        dates_good = [date for date, probes in grpdict.items() if len(probes)>1]
                        print("Dates with probe tasks: ", dates_good)
                        self.filterPandas({"date":dates_good}, "modify")
                    elif p=="probes_only":
                        # only probe trials
                        self.Dat = self.Dat[self.Dat["probe"]==1]
                    elif p=="sanity_gridloc_identical":
                        # Sanity check that all gridloc are relative the same grid (across trials).
                        self.taskclass_tokens_sanitycheck_gridloc_identical()
                    elif p=="taskgroup_reassign_simple_neural":
                        # Reassign values to column "taskgroup" so values are simple, they ignore probe
                        # (since this is present in column "probes") and ignore suffixes (e..g,
                        # E or I, without n strokes..)
                        self.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
                    elif p=="only_blocks_with_probes":
                        # only keep trials from blcoks that have both probes and non-probe tasks.
                        dict_block_probes = self.grouping_get_inner_items("block", "probe")
                        blocks_with_probes = [bk for bk, probes in dict_block_probes.items() if sorted(probes)==[0,1]]
                        self.Dat = self.Dat[self.Dat["block"].isin(blocks_with_probes)]
                    elif p=="remove_baseline":
                        # Remove trials that are baseline epoch.
                        self.Dat = self.Dat[~self.Dat["epoch"].isin(["base", "baseline"])]
                        self.Dat = self.Dat[~self.Dat["epoch_orig"].isin(["base", "baseline"])]
                    elif p=="only_blocks_with_n_min_trials":
                        # Keep only blocks that have at least nmin num trials.
                        print("Keeping only_blocks_with_n_min_trials")
                        if nmin_trials is None:
                            nmin_trials = 20
                        grpdict = self.grouping_get_inner_items("block")
                        blocks_keep = []
                        for bk, inds in grpdict.items():
                            print("block: ", bk, "ntrials: ", len(inds))
                            if len(inds)>=nmin_trials:
                                print("keeping bk: ", bk)
                                blocks_keep.append(bk)
                        self.Dat = self.Dat[self.Dat["block"].isin(blocks_keep)]
                    else:
                        print(p)
                        assert False, "dotn know this"
                    self.Dat = self.Dat.reset_index(drop=True)
                    print(f"after: {len(self.Dat)}")
                
            return params


    #################### VARIOUS UTILS
    def assign_insert_value_using_trialcode_key(self, column, map_value_to_trialcodes):
        """ 
        Given a mapping from value to trailcode, or vice versa (todo), update values for
        all columns that are present within the mapper.
        PARAMS;
        - column, to modify
        - map_value_to_trialcodes, map from value (desired value to inset into self.Dat[column])
        to list of trialcodes to take that value.
        RETURNS;
        - modifies self.Dat
        """ 
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        for list_tc in map_value_to_trialcodes.values():
            assert isinstance(list_tc, list), "should be list of trialcodes."

        # Convert to map from tc to value
        map_trialcode_value = {}
        for value, list_tc in map_value_to_trialcodes.items():
            for tc in list_tc:
                assert tc not in map_trialcode_value.keys()
                map_trialcode_value[tc] = value
                
        def F(x):
            tc = x["trialcode"]
            if tc in map_trialcode_value.keys():
                return map_trialcode_value[tc]
            else:
                # Then no change
                return x[column]
        self.Dat = applyFunctionToAllRows(self.Dat, F, column)

    ####################
    def animals(self, force_single=False):
        """ returns list of animals in this datsaet.
        - force_single, checks that only one animal.
        """
        x = sorted(list(set(self.Dat["animal"])))
        if force_single:
            assert len(x)==1, " multipel aniaml"
        return x

    def dates(self, force_single=False):
        x = sorted(list(set(self.Dat["date"])))
        if force_single:
            assert len(x)==1, " multipel dates"
        return x

    def expts(self, force_single=False):
        """ returns list of expts
        """
        x =sorted(list(set(self.Dat["expt"])))
        if force_single:
            assert len(x)==1, " multipel expt"
        return x

    def rules(self, force_single=False):
        x = sorted(list(set([M["rule"] for M in self.Metadats.values()])))
        if force_single:
            assert len(x)==1, " multipel rules"
        return x

    def trial_tuple(self, indtrial, concise=False):        
        """ identifier unique over all data (a, e, r, trialcode)
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        PARAMS:
        - concise, bool (False), if true, uses only (a, trialcode), which
        is still a unique id for this trial.
        """
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        if concise:
            tp = (
                self.animals(force_single=True)[0],
                trialcode
                )
        else:            
            tp = (
                self.animals(force_single=True)[0],
                self.expts(force_single=True)[0],
                self.rules(force_single=True)[0],
                trialcode
                )

        return tp
        

    def save_generate_string_identifier_wrapper(self, concise=True, date_first=False):
        """ Wrapper for making string that identifie sthis dataset
        PARAMS:
        - concise, bool, if true, then is shorter, just the animals and first
        and last date. else, if animal_expt_date
        """
        
        if concise:
            return self.save_generate_string_animal_dates()
        else:
            return self.identifier_string(date_first=date_first) 

    def identifier_string(self, date_first=False):
        """ string, useful for saving
        """    
        a = "_".join(self.animals())
        e = "_".join(self.expts())
        r = "_".join(self.rules())

        if date_first:
            # since, for daily dataset, rules are actuall dates.
            return f"{a}_{r}_{e}"
        else:
            return f"{a}_{e}_{r}"

    def get_sample_rate_alltrials(self):
        """ 
        Return sample rate (samp/sec) that is uinque acorss all trials.
        if multipel trials have different samp rate, this throws error
        """

        if not hasattr(self, "Fs"):
            # check umique value all trials.
            list_fs = []
            for ind in range(len(self.Dat)):
                ind_md = self.Dat.iloc[ind]["which_metadat_idx"]
                fs = self.Metadats[ind_md]["filedata_params"]["sample_rate"]
                list_fs.append(fs)
            assert len(list(set(fs)))==1
            self.Fs = list(set(fs))[0]

        return self.Fs

    def get_sample_rate(self, ind):
        """ 
        ind is index in self.Dat
        """
        ind_md = self.Dat.iloc[ind]["which_metadat_idx"]
        fs = self.Metadats[ind_md]["filedata_params"]["sample_rate"]
        return fs
            
    def get_motor_stats(self, ind):
        """ 
        Simple - returns dict with all the motortiming and motorevent stats
        """
        out = {}
        for k, v in self.Dat.iloc[ind]["motorevents"].items():
            out[k]=v
        for k, v in self.Dat.iloc[ind]["motortiming"].items():
            out[k]=v

        # get stroke and gap lengths for each one
        from pythonlib.drawmodel.features import strokeDistances, gapDistances
        strokes_beh = self.Dat.iloc[ind]["strokes_beh"]
        out["dists_stroke"] = strokeDistances(strokes_beh)
        out["dists_gap"] = gapDistances(strokes_beh)

        return out

    # def motor_get_stats_append_columns(self):
    #     """
    #     """
    #     for i in range(len(self.Dat)):
    #         self.get_motor_stats()


    ################# SPATIAL OPERATIONS
    def recenter(self, method, apply_to="both"):
        """ translate all coordiantes to recenter, based on 
        user defined method.
        - method, str,
        --- "monkey_origin", then where monkey finger began at 
        task onset (i..e, origin aka fix) is new (0,0)
        --- "screen_center", this is default, center of screen
        --- "each_stim_center", finds center of each stim, then
        uses this as (0,0). can differ for each trial.
        --- "each_beh_center", similar, but uses center based on touch 
        coordinates for behavior.
        - apply_to, to apply to "monkey", "stim" or "both", or "all"
        [NOTE, only "both" is currently working, since I have to think 
        about in what scenario the others would make sense]
        --- can also pass in as list of strings, with strokes names, eg
        ["strokes_beh", "strokes_task"] is identical to "both"
        """

        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import getCenter, translateStrokes


        def _get_F(which_strokes):
            # which_strokes, either strokes_beh or strokes_task

            def F(x):
                if method=="monkey_origin":
                    xydelt = -x["origin"]
                elif method == "each_stim_center":
                    stim_center = np.array(getCenter(x["strokes_task"]))
                    xydelt = -stim_center
                elif method == "each_beh_center":
                    beh_center = np.array(getCenter(x["strokes_beh"]))
                    xydelt = -beh_center
                elif method == "screen_center":
                    assert False, "this is default, so no need to recenter"
                else:
                    print(method)
                    assert False, "not yet coded"

                strokes = x[which_strokes]

                if which_strokes=="parses":
                    # then this is list of strokes
                    return [translateStrokes(s, xydelt) for s in strokes]
                else:
                    return translateStrokes(strokes, xydelt)

            return F

        if isinstance(apply_to, list):
            which_strokes_list = apply_to
        else:
            if apply_to=="both":
                which_strokes_list = ["strokes_beh", "strokes_task"]
            elif apply_to=="all":
                which_strokes_list = self._strokes_kinds()
            else:
                print(apply_to)
                assert False, "not coded"

        for i, which_strokes in enumerate(which_strokes_list):
            dummy_name = f"tmp{i}"
            F = _get_F(which_strokes)
            self.Dat = applyFunctionToAllRows(self.Dat, F, dummy_name)

        # replace original with dummy names
        # dont do in above, since would overwrite what I am using for compting centers.
        for i, which_strokes in enumerate(which_strokes_list):
            dummy_name = f"tmp{i}"
            self.Dat[which_strokes] = self.Dat[dummy_name]
            del self.Dat[dummy_name]


    def interpolateStrokesSpatial(self, strokes_ver ="strokes_beh", pts_or_interval = "pts",
        npts=70, interval=10):
        """ interpolate in space, to npts pts
        (uniformly sampled)
        INPUTS:
        - pts_or_interval, which method
        --- "int", then keeps uniform spatial interavl; interval, pixels
        --- "pts", this num pts.
        """
        from ..tools.stroketools import strokesInterpolate2

        strokes_list = self.Dat[strokes_ver].values
        if pts_or_interval=="pts":
            N = ["npts", npts]
        elif pts_or_interval=="int":
            N = ["interval", interval]
        else:
            assert False

        for i, strokes in enumerate(strokes_list):
            strokes_list[i] = strokesInterpolate2(strokes, 
                N=N, base="space")

        self.Dat[strokes_ver] = strokes_list


    def interpolateStrokes(self, fs_new=20):
        """ interpolate strokes, still uniform time periods, 
        but generally subsampling. Autoamtically computes
        current sampling rate from time series. Only applies to 
        strokes_beh.
        - fs_new, new sampling rate (samp/sec)
        NOTE: modifies self.Dat in place.
        """
        from ..tools.stroketools import strokesInterpolate2

        fs_old = self.getFs()

        strokes_list = self.Dat["strokes_beh"].values
        for i, strokes in enumerate(strokes_list):
            strokes_list[i] = strokesInterpolate2(strokes, 
                N = ["fsnew", fs_new, fs_old])

        self.Dat["strokes_beh"] = strokes_list

    def rescaleStrokes(self, rescale_ver="stretch_to_1"):
        """ spatial rescaling of strokes
        Autoatmicalyl applies same rescaling to strokes_beh and strokes_task 
        by first concatenating, then applying, then unconcatnating
        """
        from pythonlib.tools.stroketools import rescaleStrokes
        assert rescale_ver=="stretch_to_1", "not coded"
        
        strokes_beh_list = self.Dat["strokes_beh"]
        strokes_task_list = self.Dat["strokes_task"]

        strokes_beh_list_out = []
        strokes_task_list_out = []
        for strokes_beh, strokes_task in zip(strokes_beh_list, strokes_task_list):

            strokes_combined = strokes_beh + strokes_task
            strokes_combined = rescaleStrokes(strokes_combined, ver="stretch_to_1")

            strokes_beh_list_out.append(strokes_combined[:len(strokes_beh)])
            strokes_task_list_out.append(strokes_combined[len(strokes_beh):])

        self.Dat["strokes_beh"] = strokes_beh_list_out
        self.Dat["strokes_task"] = strokes_task_list_out


    def sketchpad_compute_diagonal_using_all_strokes(self):
        """ First get the new sketchpad that uses all of task strokes
        across trials (bounding box) then return the diagonal.
        Think of this as the largest relevant distance in this dataset...
        """
        edges = self.recomputeSketchpadEdgesAll(strokes_ver="strokes_task")
        corner1 = edges[0,:] # (xmin, ymin)
        corner2 = edges[1,:]
        return np.linalg.norm(corner2 - corner1)

    def recomputeSketchpadEdgesAll(self, strokes_ver="strokes_beh"):
        """ 
        Gets smallest bounding box over all tasks.
        Will be affected if there are outliers (without "All", isntead
        will exlcude outliers)
        RETURNS:
        - in format [[-x, -y], [+x, +y]]. does not save in self, ie edges[:,0] is
        xlim.
        """
        from pythonlib.drawmodel.image import get_sketchpad_edges_from_strokes
        strokes_list = list(self.Dat[strokes_ver].values)
        edges = get_sketchpad_edges_from_strokes(strokes_list)
        return edges


    def recomputeSketchpadEdges(self):
        """ recompute sketchpad size, since tasks are now recentered.
        - will take bounding box of all behavior (using percentiles, to
        ignore outliers). 
        RETURNS:
        - Modifies self.Metadat, replacing all metadats to have same sketchpad.
        """
        import matplotlib.pyplot as plt
        from ..tools.stroketools import getMinMaxVals

        # 1. Find, in practice, edges of behavior
        # For each trial, get its min, max, for x and y
        def F(x):
            # out = np.array([
            #     np.min([np.min(xx[:,0]) for xx in x["strokes_beh"]]),
            #     np.max([np.max(xx[:,0]) for xx in x["strokes_beh"]]),
            #     np.min([np.min(xx[:,1]) for xx in x["strokes_beh"]]),
            #     np.max([np.max(xx[:,1]) for xx in x["strokes_beh"]])
            # ])

            # return out
            return getMinMaxVals(x["strokes_beh"])

        from pythonlib.tools.pandastools import applyFunctionToAllRows
        DAT2 = applyFunctionToAllRows(self.Dat, F, "beh_edges")

        # 2. get edges that contain all tasks (or 99% of tasks for each dim)
        beh_edges = np.stack(DAT2["beh_edges"].values)

        # sketchpad_edges = np.array([
        #     [np.min(beh_edges[:,0]), np.min(beh_edges[2,:])], 
        #     [np.max(beh_edges[:,1]), np.max(beh_edges[3,:])]]) # [-x, -y, +x +y]
        sketchpad_edges = np.array([
            [np.percentile(beh_edges[:,0], [0.5])[0], np.percentile(beh_edges[:,2], [0.5])[0]], 
            [np.percentile(beh_edges[:,1], [99.5])[0], np.percentile(beh_edges[:,3], [99.5])[0]]]) # [-x, -y, +x +y]

        # 3. Replace sketchpad in metadat (use same for all tasks)
        for k, v in self.Metadats.items():
            v["sketchpad_edges"] = sketchpad_edges
            self.Metadats[k] = v

        print("Replaced self.Metadats with updated sketchpad...")
        print("[-x, -y; +x +y]")
        print(sketchpad_edges)
        # print(self.Metadats)        

        # # ==== PLOT 
        # beh_edges = np.stack(DAT2["beh_edges"].values)

        # for i in range(beh_edges.shape[1]):
        #     print([np.min(beh_edges[:, i]), np.max(beh_edges[:, i])])


        # ## PLOT DISTRIBITUSIONS OF EDGES (BEH) ACROSS ALL TASKS.
        # plt.figure()
        # plt.hist(beh_edges[:,0], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,1], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,2], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,3], 100)


    ############# HELPERS
    def getFs(self):
        """ compute actual sampling rate for strokes_beh
        SUccesfulyl returns an fs across tasks only if:
        - is clearly periodic (regular samples)
        - same fs across all tirals
        """


        strokes_list = self.Dat["strokes_beh"].values
        periods_all = []
        for i, strokes in enumerate(strokes_list):
            for S in strokes:
                pers  = np.diff(S[:,2])
                if np.any(pers<0):
                    print("trial", i)
                periods_all.append(np.diff(S[:,2]))

        periods_all = np.concatenate(periods_all)
        print(periods_all.shape)

        delt = np.max(periods_all) - np.min(periods_all)
        print("max minus min periods over all data: ", delt)
        if ~np.isclose(delt, 0):
            plt.figure()
            plt.hist(periods_all)
            print(np.min(periods_all))
            print(np.max(periods_all))            
            assert False, "why some periods are different from others?"
        else:
            print("good! all periods are identical (across samples and trials:")
            print(periods_all[0])
            print("returning fs (1/per)")

        return 1/np.mean(periods_all)


    def subsampleTrials(self, n_rand_to_keep=400, n_fixed_to_keep=50, random_seed=3):
        """ subsample trials to make sure get good variety.
        Can run this before doing train test spkit
        - n_rand_to_keep, num random trials to keep for each 
        task_category, for random tasks. Generally try to keep this within
        order of magnitude of fixed tasks, but higher since these random.
        - n_fixed_to_keep, for each fixed task (given by unique_task_name), 
        how many to subsample to?
        """
        import random
        random.seed(random_seed)
        _checkPandasIndices(self.Dat)


        def get_inds_good(df, task_category_kind, n_sub):
            """ take n_sub num subsamples for each level in 
            task_category_kind. is stratified, so will take
            n separately for each expt/epoch/task combination.
            This ensure will sample across exts/epochs where behavior
            can be wquite different.
            - df, must not use reset_index() before passing in here.
            - task_category_kind, whether to take task_stagecategory (usually for
            random tasks) or unique_task_name (suitable only for fixed tasks)
            - n_sub, how mnay to take? if less than this, then will just take all.
            """

            assert task_category_kind in ["task_stagecategory", "unique_task_name"]
            tasks = set(df[task_category_kind])
            expts = set(df["expt"])
            epochs = set(df["epoch"])
            inds_good = []
            for ex in expts:
                for ep in epochs:
                    for t in tasks:
                        inds = list(df[
                            (df[task_category_kind]==t) & (df["expt"]==ex) & (df["epoch"]==ep)
                            ].index)
                        if len(inds)==0:
                            continue
                        print(f"for this task, expt, epoch, task, this many trials exist: {t, ex, ep, len(inds)}")
                        if len(inds)>n_sub:
                            inds = random.sample(inds, n_sub)
                        inds_good.extend(inds)
            return list(set(inds_good))


        # === 1) For random tasks, subsample so that they do not overwhelm the other tasks.
        print(f"RANDOM TASKS, before subsampling to {n_rand_to_keep}")
        df = self.Dat[self.Dat["random_task"]==True]
        print(df["task_stagecategory"].value_counts())

        inds_good = get_inds_good(df, "task_stagecategory", n_rand_to_keep)

        # convert to inds to remove
        inds_all = list(df.index)
        inds_to_remove = [i for i in inds_all if i not in inds_good]

        print("Removing this many inds")
        print(len(inds_to_remove))
        print("Original size of self.Dat")
        print(len(self.Dat))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0)
        print("New size of self.Dat")
        print(len(self.Dat))

        # reset
        self.Dat = self.Dat.reset_index(drop=True)


        # === 2) For fixed tasks, only keep subsample of number, for each unique task
        print("==== FIXED TASKS")
        df = self.Dat[self.Dat["random_task"]==False]
        print(df["unique_task_name"].value_counts())
        print(df["unique_task_name"].value_counts().values)


        inds_good = get_inds_good(df, "unique_task_name", n_fixed_to_keep)

        # convert to inds to remove
        inds_all = list(df.index)
        inds_to_remove = [i for i in inds_all if i not in inds_good]

        print("Removing this many inds")
        print(len(inds_to_remove))
        print("Original size of self.Dat")
        print(len(self.Dat))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0)
        print("New size of self.Dat")
        print(len(self.Dat))

        # reset
        self.Dat = self.Dat.reset_index(drop=True)


    def splitTrainTestMonkey(self, expt, val=0., epoch=None, seed=2):
        """ returns train and test tasks based on whether for monkey they
        were train (got reinforcing feedback, might have practiced many times) or
        test (no reinforcing feedback, not many chances)
        INPUT:
        - expt, str, the experiemnt name. will restrict tasks to just this expt. NOTE: 
        this is not necesaritly rthe same as each pkl files expt.
        - epoch, int, the epoch number, 1, 2, ... Leave None is fine, unless there are
        multiple epohcs, in which case will throw error.
        - val, how many subsample of train to pull out as validation set. default is none, but 
        good amount, if ytou need, is like 0.05?
        RETURNS:
        - inds_train, inds_test, indices into self.Dat, as lists of ints.
        """
        import random
        random.seed(seed)

        _checkPandasIndices(self.Dat)

        # First, get just for this epoch
        if epoch is None:
            df = self.Dat[self.Dat["expt"] == expt]
            epochlist = set(df["epoch"])
            assert len(epochlist)==1, "there are multipel epochs, you must tell me which one you want"
        else:
            df = self.Dat[(self.Dat["expt"] == expt) & (self.Dat["epoch"]==epoch)]

        # Second, split into train and test
        inds_train = df.index[df["monkey_train_or_test"]=="train"].to_list()
        inds_test = df.index[df["monkey_train_or_test"]=="test"].to_list()

        if val>0.:
            nval = int(np.ceil(val*len(inds_train)))
            random.shuffle(inds_train)
            inds_val, inds_train = np.split(inds_train, [nval])
        else:
            inds_val = []

        print("Got this many train, val, test inds")
        print(len(inds_train), len(inds_val), len(inds_test))
        return inds_train, inds_val, inds_test

    def splitdataset_by_trial(self):
        """ Split dataset into two halves, and return copies of
        datasets for first and last half
        RETURNS:
            - D1, D2, copies of dataset with first and last halves.
        """

        self.trialcode_tuple_extract_assign()

        # find the middle trial
        list_tc = sorted(self.Dat["trialcode_tuple"].tolist())
        n = len(list_tc)
        tc_mid = list_tc[int(n/2)]

        D1 = self.copy()
        D1.Dat = D1.Dat[D1.Dat["trialcode_tuple"]<=tc_mid].reset_index(drop=True)

        # Second half
        D2 = self.copy()
        D2.Dat = D2.Dat[D2.Dat["trialcode_tuple"]>tc_mid].reset_index(drop=True)

        return D1, D2


    def splitTrainTest(self, train_frac = 0.9, val_frac = 0.05, test_frac = 0.05,
        grouby = ("character", "expt", "epoch")):
        """ Extract datasets (splits). does useful things, like making sure
        there is good variety across tasks, etc.
        By default, combines tasks across difficulties, inclulding monkey 
        "extrapolation" (G2, G3, G4) with "training". In this case, splits done by 
        random splits (but stratified, see below, grouby)
        INPUT:
        - train, val, and test fracs, self explanaatory
        - grouby, cross produce of these levels will be used to group data, before taking
        fractions. This ensures that train, test and val are evenly distributed over
        all grouping levels (or combos thereof). 
        RETURNS:
        - inds_train, inds_val, inds_test
        NOTES:
        - val and test will usualyl have slihgly mroe than input fraction, because of
        stratification method (if low sample size , then can try to force getting a test or val)
        - shuffles before returning,
        """
        print("TODO: Alternatively, can split similar to how monkey experienced")
        from pythonlib.tools.pandastools import filterPandas
        import random

        _checkPandasIndices(self.Dat)

        train_frac = 0.9
        val_frac = 0.05
        test_frac = 0.05
        assert np.isclose(train_frac + val_frac + test_frac, 1.)

        # Collect indices for train, val, and test
        inds_train = []
        inds_val = []
        inds_test = []
        for g in self.Dat.groupby(grouby):
            inds = list(g[1].index)
            random.shuffle(inds)
            n = len(inds)
            if n==1:
                inds_train.append(inds[0])
            elif n==2:
                inds_train.append(inds[0]) # use one for trainig
                if random.random()>0.5: # use the other for etier val or train
                    inds_val.append(inds[1])
                else:
                    inds_test.append(inds[1])
            elif n==3:
                inds_train.append(inds[0]) # spread evenly, one each.
                inds_val.append(inds[1])
                inds_test.append(inds[2])
            else:
                n_val = int(np.round(np.max([val_frac*n, 1])))
                n_test = int(np.round(np.max([test_frac*n, 1])))
                n_train = int(n - n_val - n_test)
                a, b, c = np.split(inds, [n_train, n_train+n_val])
                inds_train.extend(a)
                inds_val.extend(b)
                inds_test.extend(c)

        print("got this many (train, val, test)")
        print(len(inds_train), len(inds_val), len(inds_test))

        # shuflfe
        random.shuffle(inds_train)
        random.shuffle(inds_val)
        random.shuffle(inds_test)

        return inds_train, inds_val, inds_test




    ############# EXTRACT THINGS
    def _sf_get_path(self, strokes_ver, idx_metadat):
        """ for single dataset. tell me which datset with idx_metadat
        """
        sdir = f"{self.Metadats[idx_metadat]['path']}/stroke_feats"
        os.makedirs(sdir, exist_ok=True)
        path_sf = f"{sdir}/sf-{strokes_ver}.pkl"
        path_params = f"{sdir}/params-{strokes_ver}.pkl"
        return path_sf, path_params


    def _sf_get_path_combined(self):
        """ for SF that has been reloaded, across multiple datasets.
        will get automatic name for this combo. If already exist, then will
        not update. otherwise comes up with new.
        will be sved in self.SFparams["sdir"] """
        # save and make save folder for this combined dataset
        from pythonlib.tools.expttools import makeTimeStamp

        if "sdir" not in self.SFparams:
            ts = makeTimeStamp()
            sdir = f"{base_dir}/database/combined_strokfeats"
            a = sorted(set(self.SF["animal"]))
            b = sorted(set(self.SF["expt"]))
            c = sorted(set([p["strokes_ver"] for p in self.SFparams["params_each_original_sf"]]))
            tmp = "_".join(a) + "-" + "_".join(b) + "-" + "_".join(c)

            self.SFparams["sdir"] = f"{sdir}/{tmp}-{ts}"
            
            os.makedirs(self.SFparams['sdir'])
        return self.SFparams["sdir"]


    def sf_extract_and_save(self, strokes_ver = "strokes_beh", saveon=True, assign_into_self=False):
        """ extract and save stroke features dataframe, in save dir
        where datsaet is saved. 
        Will by default run this for all rows in self.Dat, can post-hoc assign
        back to Datasets if want to do further preprocessing
        INPUT:
        - strokes_ver, {"strokes_beh", "strokes_parse", "strokes_beh_splines"}, 
        determines which storkles to use. will seave spearately depending on this 
        flag. Note: if strokes_parse, trhen will choose a random single parse. that means
        each run could give diff ansers.
        - assign_into_self, then also assigns into self.SF and self.SFparams
        RETURNS:
        - SF, dataframe. does not modify self.
        NOTE: 
        - by default only allows you to run if this is a single dataset.
        - will delete Dataset from column before saving.
        """
        import os
        
        assert self._check_is_single_dataset(), "not allowed to run this with multipe datasets..."

        params = {}
        params["strokes_ver"] = strokes_ver

        if params["strokes_ver"]=="strokes_parse":
            self.parsesLoadAndExtract() # extract parses, need to have done bfeore
        if params["strokes_ver"]=="strokes_beh_splines":
            self.strokesToSplines(strokes_ver='strokes_beh', make_new_col=True) # convert to splines (beh)
        SF = self.flattenToStrokdat(strokes_ver=strokes_ver)


        # save
        if saveon:
            # delete things you dont want to save
            del SF["Dataset"]

            # Dir
            path_sf, path_params = self._sf_get_path(strokes_ver, 0)
            # SF
            SF.to_pickle(path_sf)
            print("Saved SF to:")
            print(path_sf)
            # Save params
            with open(path_params, "wb") as f:
                pickle.dump(params, f)

        if assign_into_self:
            self.SF = SF
            self.SFparams = {}

        return SF

    def sf_load_preextracted(self, strokes_ver_list = ("strokes_beh",)):
        """ wrapper to load multiple SFs.
        - strokes_ver_list, list of strings. (see _sf_load_preextracted)
        RETURNS:
        self.SF, all concatenated. will keep record of both which dataset (idx_metadata)
        and which SF it came from
        """

        SF = []
        PARAMS = []
        for sver in strokes_ver_list:
            sf, prms = self._sf_load_preextracted(sver)

            SF.extend(sf)
            PARAMS.extend(prms)

        # concat
        SF = pd.concat(SF)
        SF = SF.reset_index(drop=True)

        # assign
        self.SF = SF
        self.SFparams = {
            "params_each_original_sf":PARAMS}

        print("SF put into self.SF")



    def _sf_load_preextracted(self, strokes_ver="strokes_beh"):
        """ Loads SF (and parasm) that have been presaved. 
        must have run self.sf_extract_and_save() first.
        RETURNS:
        - list of SF, list of params
        NOTE: fine to run this even if multiple Datasets.
        """

        SF  = []
        PARAMS = []
        for idx in range(len(self.Metadats)):

            # Load presaved for this subdataset
            path_sf, path_params = self._sf_get_path(strokes_ver, idx) 

            sf = pd.read_pickle(path_sf)
            with open(path_params, "rb") as f:
                params = pickle.load(f)

            sf["idx_metadat"] = idx
            sf["strokes_ver"] = strokes_ver
            params["idx_metadat"] = idx
            params["path_sf"] = path_sf
            params["path_params"] = path_params
            params["path_params"] = path_params

            # concat
            PARAMS.append(params)
            SF.append(sf)

        # concat
        # SF = pd.concat(SF)
        # SF = SF.reset_index(drop=True)

        # save into self
        return SF, PARAMS


    def sf_preprocess_stroks(self, align_to_onset = True, min_stroke_length_percentile = 2, 
        min_stroke_length = 50, max_stroke_length_percentile = 99.5, centerize=False, rescale_ver=None):
        """ preprocess, filter, storkes, after already extracted into 
        self.SF
        """
        from ..drawmodel.sf import preprocessStroks


        params = {
            "align_to_onset":align_to_onset,
            "min_stroke_length_percentile":min_stroke_length_percentile,
            "min_stroke_length":min_stroke_length,
            "max_stroke_length_percentile":max_stroke_length_percentile,
            "centerize":centerize,
            "rescale_ver":rescale_ver
        }

        self.SF = preprocessStroks(self.SF, params)

        # Note down done preprocessing in params
        self.SFparams["params_preprocessing"] = params


    def sf_save_combined_sf(self):
        """ must have already extracted SF. saves current state, inclding
        preprocessing, etc
        """
        sdir = self._sf_get_path_combined()

        path_SF = f"{sdir}/SF.pkl"
        self.SF.to_pickle(path_SF)

        path_SF = f"{sdir}/SFparams.pkl"
        with open(path_SF, "wb") as f:
            pickle.dump(self.SFparams, f)
        print("saved SF and SFparams to:")
        print(sdir)


    def sf_load_combined_sf(self, animals, expts, strokes, tstamp = "*", take_most_recent=True):
        """
        Load SF (already concatted across mult datasets, and possible prerpocessed) NOTE:
        do all preprocessing etc BEFORE use sf_save_combined_sf(), which saves objecst that iwll
        load here.
        - take_most_recent, then will allow if have multiple found paths by taking most recent.
        otherwise will raise error if get multiple.
        """
        sdir = f"{base_dir}/database/combined_strokfeats"
        a = sorted(animals)
        b = sorted(expts)
        c = sorted(strokes)
        tmp = "*".join(a) + "-" + "*".join(b) + "-" + "*".join(c) + "*".join(tstamp)

        pathlist = findPath(sdir, [[tmp]], "SF", "pkl", True)

        if len(pathlist) == 0:
            print(pathlist)
            assert False, "did not find presaved data"
        
        if len(pathlist)>1:
            if not take_most_recent:
                print(pathlist)
                assert False, "found to omany..."
            else:
                print("Taking most recent path")
                pathlist = [pathlist[-1]]


        sdir = pathlist[0]

        print("==== Loading:")
        path = f"{sdir}/SF.pkl"
        with open(path, "rb") as f:
            self.SF = pickle.load(f)
            print("** Loaded into self.SF:")
            print(path)

        path = f"{sdir}/SFparams.pkl"
        with open(path, "rb") as f:
            self.SFparams = pickle.load(f)
            print("** Loaded into self.SFparams:")
            print(path)

        # save this path
        self.SFparams["path_sf_combined"] = sdir



    def sf_embedding_bysimilarity(self, rescale_strokes_ver = "stretch_to_1", distancever = "euclidian_diffs",
        npts_space = 70, Nbasis = 300, saveon=True):
        """copmputs embedding of strokes in similarity space (ie defined by basis set of strokes). basis set
        picked randomly
        INPUT:
        - rescale_strokes_ver, str, how/whether to rescale strokes before process. will not change self.SF
        - distance_ver, str, distance to use between strok-strok.
        - npts_space, num pts to interpolate in space (spatially uniform)
        - Nbasis, num random sampled trials to use for basesi.
        - saveon, then saves in same dir as self.SF
        RETURNS:
        - similarity_matrix, Nsamp x Nbasis, range from 0 to 1
        - idxs_stroklist_basis, indices in self.SF used as basis set (len Nbasis)
        """
        from pythonlib.drawmodel.sf import computeSimMatrix

        # Get sim matrix
        similarity_matrix, idxs_stroklist_basis = computeSimMatrix(self.SF, rescale_strokes_ver, 
                                                                   distancever, npts_space, Nbasis)
        # params_embedding = {
        #     "align_to_onset":align_to_onset,
        # }

        dat = {
            "similarity_matrix":similarity_matrix,
        }

        params = {
            "rescale_strokes_ver":rescale_strokes_ver,
            "distancever":distancever,
            "npts_space":npts_space,
            "Nbasis":Nbasis,
            "idxs_stroklist_basis":idxs_stroklist_basis
        }

        if saveon:
            sdir = self._sf_get_path_combined()
            ts = makeTimeStamp()
            sdirthis = f"{sdir}/embeddings/similarity-{rescale_strokes_ver}-{distancever}-N{Nbasis}-{ts}"
            os.makedirs(sdirthis)
            params["path_embeddings_similarity"] = sdirthis

            # sim matrix
            path = f"{sdirthis}/dat.pkl"
            with open(path, "wb") as f:
                pickle.dump(dat, f)

            # params
            path = f"{sdirthis}/params.pkl"
            with open(path, "wb") as f:
                pickle.dump(params, f)
            print("Saved to:")
            print(sdirthis)


        return similarity_matrix, idxs_stroklist_basis, params



    def flattenToStrokdat(self, keep_all_cols = True, strokes_ver="strokes_beh",
        cols_to_exclude = ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines", "parses", "motor_program"]):
        """ flatten to pd dataframe where each row is one
        strok (np array)
        - keep_all_cols, then keeps all dataframe columns. otherwise
        will just have a reference (link) to dataset.
        - strokes_ver, can choose strokes_beh, strokes_task, strokes_parse...
        for the latter must have first exgtractd strokes parse.
        - cols_to_exclude, overwrites keep_all_cols.
        RETURNS:
        - Strokdat, pd dataframe. Includes reference to Probedat.
        NOTE: does not modify self.

        """
        _checkPandasIndices(self.Dat)

        if strokes_ver=="strokes_parse":
            if "strokes_parse" not in self.Dat.columns:
                print("extracting a single parse, since you haevent done that")
                self.parsesChooseSingle()
            # assert "strokes_parse" in self.Dat.columns, "need to first extract parases"
        if strokes_ver=="strokes_beh_splines":
            if "strokes_beh_splines" not in self.Dat.columns:
                print("Running conversion of strokes_beh to splines, saving as new col strokes_beh_splines")
                self.strokesToSplines()

        strokeslist = self.Dat[strokes_ver].values
        Strokdat = []
        columns = [col for col in self.Dat.columns if col not in cols_to_exclude]
        for i, strokes in enumerate(strokeslist):
            if i%500==0:
                print(i)

            row = self.Dat.iloc[i]

            for ii, strok in enumerate(strokes):
                Strokdat.append({
                    "stroknum":ii,
                    "strok":strok,
                    "row_in_Dataset":i,
                    "Dataset":self})

                if keep_all_cols:
                    for col in columns:
                        Strokdat[-1][col] = row[col]

        return pd.DataFrame(Strokdat)


    ############## SPATIAL
    def convertCoord(self, edges_out, strokes_ver="strokes_beh", edges_in=None):
        """ convert coordinate system.
        - edges_out and edges_in, in format [xmin, xmax; ymin, ymax], 2 x 2 array
        - edges_in, either give me, or if None wil compute automatically based on 
        smallest bounding box covering all tasks.
        RETURNS:
        - modified self.Dat[strokesver]
        - updates self._Sketchpad
        """
        from ..drawmodel.image import convCoordGeneral
        if edges_in is None:
            # compute edges
            edges_in = self.recomputeSketchpadEdgesAll().T
            print("Computed these for edges_in:")
            print(edges_in)

        strokes_list = self.Dat[strokes_ver]

        def F(strokes):
            return [convCoordGeneral(s, edges_in, edges_out) for s in strokes]

        strokes_list = [F(strokes) for strokes in strokes_list]

        self.Dat[strokes_ver] = strokes_list

        if not hasattr(self, "_Sketchpad"):
            self._Sketchpad = {}
        self._Sketchpad["edges"] = edges_out
        print("DONE converting coords")


    ############### IMAGES
    def strokes2image(self, strokes_ver, imageWH=105, inds=None, ploton=False):
        """
        convert strokes to image. 
        Automaticlaly finds boudning box to that all tasks in dataset are scaled same.
        INPUTS:
        - strokes_ver, which data to use.
        - imageWH, x and y dimension (square), 105 is standard for BPL.
        - inds, which indices in dataset to use. leave None to do all. must be None or
        list.
        - ploton, if true, then plots each image one by one. will not let you do this if inds 
        is too many.
        RETURNS:
        - img_list, list of np arrays holding images. formated so that plt.imshow() 
        prodices correct orietnation.
        """
        from pythonlib.drawmodel.image import strokes2image

        if inds is None:
            inds = range(len(self.Dat))
        else:
            assert isinstance(inds, list)

        # == coordinates of the sketchpad used by monkey
        # max width or hiegh (whichever one greater), num pixels for
        # half of page, so that sketchpad will be square with edges in 
        # both dimensions of : (-WH, WH)
        tmp = []
        for MD in self.Metadats.values():
            canvas_max_WH = np.max(np.abs(MD["sketchpad_edges"])) # smallest square that bounds all the stimuli
            tmp.append(canvas_max_WH)
        canvas_max_WH = np.max(tmp)
        print(f"canvas_max_WH atuomaticaly determiend to be: {canvas_max_WH}")

        # Which strokes dataset?
        strokes_list = list(self.Dat.iloc[inds][strokes_ver])

        if len(inds)>10 and ploton==True:
            print("Forcing turning off plots, too many trials")
            ploton=False

        image_list = [strokes2image(strokes, canvas_max_WH, imageWH, plot=ploton) for strokes in strokes_list]

        return image_list

    ############### MOTOR PROGRAMS (BPL)
    def _loadMotorPrograms(self, ver="strokes"):
        """ 
        Helper to load pre-saved motor programs.
        INPUTS:
        - ver, {"strokes", "parses"} which set of pre-saved programs
        to load.
        """
        from pythonlib.tools.expttools import findPath

        # Go thru each dataset, search for its presaved motor programs. Collect
        # into a list of dicts, one dict for each dataset.
        ndat = len(self.Metadats)
        MPdict_by_metadat = {}
        for i in range(ndat):

            # find paths
            if ver=="strokes":
                path = findPath(self.Metadats[i]["path"], [["infer_MPs_from_strokes"]],
                    "params",".pkl",True)
            elif ver=="parses":
                path = findPath(self.Metadats[i]["path"], [["MPs_for_parses"], ["infer_MPs_from_strokes"]],
                    "params",".pkl",True)
            else:
                print(ver)
                assert False, "not coded"
            
            # should be one and only one path
            if len(path)==0:
                assert False, "did not find"
            elif len(path)>1:
                assert False, "found > 1. you should deletethings."
            path = path[0]

            # lioad things from path
            _namelist = ["indices_list", "MPlist", "out_all", "params", "score_all"]
            MPdict = {}
            for name in _namelist:
                paththis = f"{path}/{name}.pkl"
                with open(paththis, "rb") as f:
                    MPdict[name] = pickle.load(f)

            # check consistency
            assert len(MPdict["MPlist"])==len(MPdict["indices_list"])
            assert len(MPdict["score_all"])==len(MPdict["indices_list"])
            assert len(MPdict["out_all"])==len(MPdict["out_all"])

            MPdict_by_metadat[i] = MPdict

        return MPdict_by_metadat

    def _extractMP(self, MPdict_by_metadat, idx_mdat, trialcode, expect_only_one_MP_per_trial=True,
        fail_if_None=True, parsenum=None):
        """
        Helper, to extract a single motor program for a signle trial (trialcode), and single
        metadat index (idx_mdat)
        INPUTS:
        - fail_if_None, then fialks if any trial doesnt find.
        NOTES:
        - expect_only_one_MP_per_trial, then each trial only got one parse. will return that MP 
        (instead of a list of MPs)
        - parsenum, {0, 1, ...}, parses, in order as in presaved parses. Leave as None if this is for
        strokes (so only one program).
        [SAME AS (not parses), but here each trial can have multiple paress. these are flattened in 
        MPdict_by_metadat. THis code pulls out a specific trialcode and parsenum"]
        RETURNS:
        - list of MPs. Usually is one, if there is only single parse. Unless 
        expect_only_one_MP_per_trial==True, then returns MP (not list). If doesnt fine, then will return None
        """

        # Get the desired dataset
        MPdict = MPdict_by_metadat[idx_mdat]
        trialcode_list = MPdict["indices_list"]
        MP_list = MPdict["MPlist"]

        if parsenum is None:
            # Pull out the desired trial        
            tmp = [m for m, t in zip(MP_list, trialcode_list) if t==trialcode]
        else:
            def _code(trialcode, parsenum):
                return f"tc_{trialcode}--p_{parsenum}"

            # Pull out the desired trial        
            tmp = [m for m, t in zip(MP_list, trialcode_list) if t==_code(trialcode, parsenum)]
        
        # === CHECK 
        if len(tmp)==0:
            if fail_if_None:
                print(f"Did not find MP for idx_metadat {idx_mdat}, trialcode {trialcode}")
                assert False
            return None
        elif len(tmp)>1:
            print(tmp)
            assert False, "not sure why, multiple inds have same trialcode? must be bug"

        if expect_only_one_MP_per_trial:
            assert len(tmp[0])==1
            tmp = tmp[0]

        return tmp[0]


    def bpl_reload_saved_state(self, path):
        """ reload BPL object, previously saved
        """

        # load BPL
        fthis = f"{path}/BPL.pkl"
        with open(fthis, "rb") as f:
            self.BPL = pickle.load(f)

        print(f"Reloaded BPL into self.BPL, from {fthis}")
    

    def bpl_extract_and_save_motorprograms(self, params_preprocess = ("recenter", "spad_edges",),
            sketchpad_edges =None, save_checkpoints = (100, "")):
        """ save for this dataset.
        does some auto preprocessing first too
        """
        from pythonlib.bpl.strokesToProgram import infer_MPs_from_strokes

        if sketchpad_edges is None:
            sketchpad_edges = np.array([[-260., 260.],[-260., 260.]])

        assert len(self.Metadats)==1, "advised to only do this if you are working wtih one dataset."
        save_checkpoints[1] = self.Metadats[0]["path"]

        # Preprocess
        self.preprocessGood(ver="", params=params_preprocess)

        if True:
            # maybe this better, keep it stable caross all epxts
            sketchpad_edges = sketchpad_edges
        else:
            sketchpad_edges = self.Metadats[0]["sketchpad_edges"].T

        ## returns params in lists.
        strokeslist = self.Dat["strokes_beh"].values
        trialcodelist = self.Dat["trialcode"].values
        MPlist, scores = infer_MPs_from_strokes(strokeslist, trialcodelist, params_preprocess,
                                                sketchpad_edges, save_checkpoints=save_checkpoints)

        return MPlist, scores

    def bpl_extract_and_save_motorprograms_parses(self, params_preprocess = ("recenter", "spad_edges",),
            sketchpad_edges =None, save_checkpoints = (100, ""), 
            parses_good=False):

        if sketchpad_edges is None:
            sketchpad_edges = np.array([[-260., 260.],[-260., 260.]])

        from pythonlib.bpl.strokesToProgram import infer_MPs_from_strokes

        assert len(self.Metadats)==1, "advised to only do this if you are working wtih one dataset."
        assert len(save_checkpoints[1])==0, "code wil overwrite..."
        if parses_good:
            save_checkpoints[1] = f"{self.Metadats[0]['path']}/parses_good/MPs_for_parses"
        else:
            save_checkpoints[1] = f"{self.Metadats[0]['path']}/MPs_for_parses"


        # Preprocess
        self.preprocessGood(ver="", params=params_preprocess, apply_to_recenter=["strokes_beh", "strokes_task", "parses"])

        if True:
            # maybe this better, keep it stable caross all epxts
            sketchpad_edges = sketchpad_edges
        else:
            sketchpad_edges = self.Metadats[0]["sketchpad_edges"].T

        # Strategy for extraction - flatten all MPs (so ntrials x nparses), but keep information about parse num and trialcode.
        strokeslist = []
        trialcode_parsenum_list =[]
        for row in self.Dat.iterrows():
            parselist = row[1]["parses"]
            trialcode = row[1]["trialcode"]

            for i, parse in enumerate(parselist):

                strokeslist.append(parse)
                trialcode_parsenum_list.append(f"tc_{trialcode}--p_{i}")


        print(len(strokeslist))
        print(len(trialcode_parsenum_list))
        # print(strokeslist[0])
        # print(trialcode_parsenum_list[0])
        # assert False

        MPlist, scores = infer_MPs_from_strokes(strokeslist, trialcode_parsenum_list, params_preprocess,
                                                sketchpad_edges, save_checkpoints=save_checkpoints)

        return MPlist, scores

    def bpl_load_motorprograms(self):
        """ 
        loads best parse (MP) presaved, BPL.
        (see strokesToProgram) for extraction.
        RETURNS:
        - self.Dat adds a new column "motor_program", with a single ctype (motor program)
        - self.BPL, which stores things related to BPL params.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # def _loadMotorPrograms():
        #     """ 
        #     """
        #     from pythonlib.tools.expttools import findPath
            
        #     # Go thru each dataset, search for its presaved motor programs. Collect
        #     # into a list of dicts, one dict for each dataset.
        #     ndat = len(self.Metadats)
        #     MPdict_by_metadat = {}
        #     for i in range(ndat):
                
        #         # find paths
        #         path = findPath(self.Metadats[i]["path"], [["infer_MPs_from_strokes"]],
        #             "params",".pkl",True)
                
        #         # should be one and only one path
        #         if len(path)==0:
        #             print(self.Metadats[i]["path"])
        #             assert False, "did not find"
        #         elif len(path)>1:
        #             assert False, "found > 1. you should deletethings."
        #         path = path[0]
                
        #         # lioad things from path
        #         _namelist = ["indices_list", "MPlist", "out_all", "params", "score_all"]
        #         MPdict = {}
        #         for name in _namelist:
        #             paththis = f"{path}/{name}.pkl"
        #             with open(paththis, "rb") as f:
        #                 MPdict[name] = pickle.load(f)

        #         # check consistency
        #         assert len(MPdict["MPlist"])==len(MPdict["indices_list"])
        #         assert len(MPdict["score_all"])==len(MPdict["indices_list"])
        #         assert len(MPdict["out_all"])==len(MPdict["out_all"])
                
        #         MPdict_by_metadat[i] = MPdict

        #     return MPdict_by_metadat
        
        # def _extractMP(MPdict_by_metadat, idx_mdat, trialcode, expect_only_one_MP_per_trial=True,
        #     fail_if_None=True):
        #     """
        #     Helper, to extract a single motor program for a signle trial (trialcode), and single
        #     metadat index (idx_mdat)
        #     INPUTS:
        #     - fail_if_None, then fialks if any trial doesnt find.
        #     NOTES:
        #     - expect_only_one_MP_per_trial, then each trial only got one parse. will return that MP 
        #     (instead of a list of MPs)
        #     RETURNS:
        #     - list of MPs. Usually is one, if there is only single parse. Unless 
        #     expect_only_one_MP_per_trial==True, then returns MP (not list). If doesnt fine, then will return None
        #     """

        #     # Get the desired dataset
        #     MPdict = MPdict_by_metadat[idx_mdat]
        #     trialcode_list = MPdict["indices_list"]
        #     MP_list = MPdict["MPlist"]

        #     # Pull out the desired trial        
        #     tmp = [m for m, t in zip(MP_list, trialcode_list) if t==trialcode]
        #     if len(tmp)==0:
        #         if fail_if_None:
        #             print(f"Did not find MP for idx_metadat {idx_mdat}, trialcode {trialcode}")
        #             assert False
        #         return None
        #     elif len(tmp)>1:
        #         print(tmp)
        #         assert False, "not sure why, multiple inds have same trialcode? must be bug"
            
        #     if expect_only_one_MP_per_trial:
        #         assert len(tmp[0])==1
        #         tmp = tmp[0]

        #     return tmp[0]

        # check that havent dione before
        assert not hasattr(self, "BPL"), "self.BPL exists. I woint overwrite."
        assert "motor_program" not in self.Dat.columns

        MPdict_by_metadat = self._loadMotorPrograms()

        # save things
        self.BPL = {}
        self.BPL["params_MP_extraction"] = {k:MP["params"] for k, MP in MPdict_by_metadat.items()}

        def F(x):
            idx_mdat = x["which_metadat_idx"]
            trialcode = x["trialcode"]
            MP = self._extractMP(MPdict_by_metadat, idx_mdat, trialcode)
            assert MP is not None, "did not save this MP..."
            return MP

        self.Dat = applyFunctionToAllRows(self.Dat, F, 'motor_program')


    def bpl_refit_libraries_to_MPs(self, gb = ("animal", "expt", "epoch", "monkey_train_or_test"), 
        params_to_update=('kappa', 'rel_type_mixture', 'prim_type_mixture', 'spatial_hist'),
        params_to_update_using_entire_dataset=None,
        increment_idx=True):
        """ generate libraries refit to MPs extracted for this datsets.
        Flexible, in that can slice dataset in different ways before applying as 
        training data.
        INPUT:
        - gb, groupby list, for each group will compute a new fitted library.
        - params_to_update, which params to update. either a list of strings, or a
        list of lists. latter case, will iterate over all sub-lists, and for each, recompute a model.
        - params_to_update_using_entire_dataset, similar idea, but these will not take gb levels but will
        use entire data. will check that there is no overlap between this and params_to_update
        - increment_idx, True, then doesnt overwrite. false then always overwrites. see RETURNS/.
        RETURNS:
        self.BPL["refits"][idx]["libraries"] = list of libraries (each a dict)
        self.BPL["refits"][idx]["libraries_grpby"] = gb
        self.BPL["refits"][idx]["libraries_params"] = params_to_update
        where idx incremenets at each run.
        """

        if params_to_update_using_entire_dataset is None:
            params_to_update_using_entire_dataset = []
        from pythonlib.bpl.refitting import libRefitted

        if isinstance(params_to_update[0], list):
            assert False, "not yet coded"

        assert len([p for p in params_to_update_using_entire_dataset if p in params_to_update])==0, "no overlaps allowed!"
        assert len([p for p in params_to_update if p in params_to_update_using_entire_dataset])==0, "no overlaps allowed!"

        # for saving
        if "refits" not in self.BPL:
            self.BPL["refits"] = []

        libraries_list = []
        params_this = params_to_update
        for g in self.Dat.groupby(gb):
            print("-- refitting libarry to this group")
            print(g[0])
            print(len(g[1]))
            print("updating these params")
            print(params_this)
            
            MPlist_this = list(g[1]["motor_program"].values)
            lib_update = libRefitted(MPlist_this, params_to_update=params_this)
            
            # any updates needed , using the entire dataset?
            if len(params_to_update_using_entire_dataset)>0:
                print("** updating using entire dataset!! these parasm:")
                print(params_to_update_using_entire_dataset)
                MPlist_entire_dataset = list(self.Dat["motor_program"].values)
                lib_update = libRefitted(MPlist_entire_dataset, params_to_update=params_to_update_using_entire_dataset,
                    lib_update=lib_update)

            # save
            tmp = {"lib":lib_update, "ntrial":len(g[1])}
            # for k, v in zip(gb, g[0]):
            #     tmp[k] = v
            tmp["index_grp"] = g[0]
            tmp["params_to_update"] = params_this
            tmp["params_to_update_using_entire_dataset"] = params_to_update_using_entire_dataset
            libraries_list.append(tmp)

        out = {}
        out["libraries"] = libraries_list
        out["libraries_grpby"] = gb
        out["libraries_params"] = params_to_update

        self.BPL["refits"].append(out)

    def bpl_index_to_col_name(self, index, ver="strokes"):
            if isinstance(index, (list, tuple)):
                tmp = [str(i) for i in index]
            else:
                tmp = [index]
            if ver=="strokes":
                return f"bpl-{'-'.join(tmp)}"
            elif ver=="parses":
                return f"bpl-parses-{'-'.join(tmp)}"
            else:
                print(ver)
                assert False


    def _bpl_extract_refitted_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None):
        """ Extract list of BPL libraries, they should have already been refitted,
        and are in self.BPL["refits"]
        INPUTS:
        - lib_refit_index, a particular grouping index. i.e.., self.BPL["refits"][lib_refit_index]
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        RETURNS:
        - LibListThis, list of libraries.
        """

        # Extract list of libraries
        libraries_list = self.BPL["refits"][lib_refit_index]["libraries"]
        # grp_by = self.BPL["refits"][lib_refit_index]["libraries_grpby"]
        # for L in libraries_list:
            # print(L["index_grp"])

        # 2) get libraries
        # LibListThis = [L for L in libraries_list if L["index_grp"] in dsets_to_keep] # old version, same for dste and libary
        if libraries_to_apply_inds is None:
            LibListThis = libraries_list
        elif isinstance(libraries_to_apply_inds[0], int):
            LibListThis = [libraries_list[i] for i in libraries_to_apply_inds]
        elif isinstance(libraries_to_apply_inds[0], tuple):
            LibListThis = [L for L in libraries_list if L["index_grp"] in libraries_to_apply_inds] # old version, same for dste and libary
            print(libraries_to_apply_inds)
            print([L["index_grp"] for L in libraries_list])
            assert len(LibListThis)==len(libraries_to_apply_inds)
        else:
            print(libraries_list)
            assert False
        return LibListThis


    def bpl_score_trials_by_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None,
        dsets_to_keep=None, scores_to_use = ("type",)):
        """
        Score each trial in self.Dat (their motor program) based on libraries, previuosl;yl
        extracted ands aved in self.BPL. returns a scalar score, log ll
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        - dats_to_keep not implmeneted, to get subset of data.
        RETURNS:
        - self.Dat modified with  noew columns, name "bpl_"<index_levels as a string>
        NOTES:
        dsets_to_keep = [
            ('Pancho', 'lines5', 1, 'test'),
            ('Pancho', 'lines5', 1, 'train'),
            ('Pancho', 'lines5', 2, 'test'),
            ('Pancho', 'lines5', 2, 'train')
        ]

        """
        from ..bpl.strokesToProgram import scoreMPs

        # Extract list of libraries
        # libraries_list = self.BPL["refits"][lib_refit_index]["libraries"]
        # for L in libraries_list:
        #     print(L["index_grp"])

        # 2) get libraries
        # # LibListThis = [L for L in libraries_list if L["index_grp"] in dsets_to_keep] # old version, same for dste and libary
        # if libraries_to_apply_inds is None:
        #     LibListThis = libraries_list
        # elif isinstance(libraries_to_apply_inds[0], int):
        #     LibListThis = [libraries_list[i] for i in libraries_to_apply_inds]
        # elif isinstance(libraries_to_apply_inds[0], tuple):
        #     LibListThis = [L for L in libraries_list if L["index_grp"] in libraries_to_apply_inds] # old version, same for dste and libary
        #     print(libraries_to_apply_inds)
        #     print([L["index_grp"] for L in libraries_list])
        #     assert len(LibListThis)==len(libraries_to_apply_inds)
        # else:
        #     print(LibListThis)
        #     assert False
        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)

        # [OPTIONAL - prune dataset, and models, before running]
        if dsets_to_keep is not None:
            assert False, "not done - need to first assign col with grpby levels, then filter"
            # first get column grouped
            def subsetDataset(dat_index):
                """
                - e..g, dat_index = ('Pancho', 'arc2', 1, 'test')
                RETURNS:
                - df, subset of D.Dat
                """
                i1 = self.Dat[grp_by[0]]==dat_index[0]
                i2 = self.Dat[grp_by[1]]==dat_index[1]
                i3 = self.Dat[grp_by[2]]==dat_index[2]
                i4 = self.Dat[grp_by[3]]==dat_index[3]
                df = self.Dat[i1 & i2 & i3 & i4]
                return df


            # 1) get Dataset
            tmp =[]
            for idx in dsets_to_keep:
                dfthis = subsetDataset(idx)
                tmp.append(dfthis)
                print(len(dfthis))

            DatThis = pd.concat(tmp)
            print("-- Final lenght")
            print(len(DatThis))
            print("-- Orig length:")
            print(len(self.Dat))


        # Add a column to self.Dat, for index grp
        grp_by = self.BPL["refits"][lib_refit_index]["libraries_grpby"]
        self.grouping_append_col(grp_by, "index_grp")
        print("-- SCORING, using this model:")
        
        if False:
            # === for each row, get score for each model
            def F(x, libthis):
                # Score a single row, given a library
                
                # data pt
                MP = x["motor_program"]

                # do score
            #     score = torch.tensor(scoreMPs([MP], lib=libthis, scores_to_use=scores_to_use))
                return torch.tensor(scoreMPs([MP], lib=libthis, scores_to_use=scores_to_use)[0]).numpy()
                
            for L in LibListThis:
                lib_index = L["index_grp"]
                libthis = L["lib"]
                print(lib_index)
                
                Fthis = lambda x: F(x, libthis)
                # newcol = f"bpl-{str(lib_index)}"
                newcol = self.bpl_index_to_col_name(lib_index)

                self.Dat = applyFunctionToAllRows(self.Dat, Fthis, newcol)
        else:
            # new way, extract as vector.
            for L in LibListThis:
                lib_index = L["index_grp"]
                libthis = L["lib"]
                print(lib_index)
                
                MPlist = list(self.Dat["motor_program"].values)
                scores = scoreMPs(MPlist, lib=libthis, scores_to_use=scores_to_use)
                scores = [s.numpy().squeeze() for s in scores]
                # newcol = f"bpl-{str(lib_index)}"
                newcol = self.bpl_index_to_col_name(lib_index)
                self.Dat[newcol] = scores

    def bpl_score_parses_by_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None,
        dsets_to_keep=None, scores_to_use = ("type",)):
        """
        Score each parse in self.Dat["parases"], each row can have multipe parese.
        Does this based on libraries, first converting the parses to motor programs,
        returns a scalar score, log ll
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        RETURNS:
        - self.Dat with new col: parses_scores_<index_levels as a string>
        """
        from ..bpl.strokesToProgram import scoreMPs

        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)

        # new way, extract as vector.
        for L in LibListThis:
            lib_index = L["index_grp"]
            libthis = L["lib"]
            print(lib_index)
            
            # Go thru all parses
            def F(x):
                MPlist = list(x["parses_motor_programs"])
                scores = scoreMPs(MPlist, lib=libthis, scores_to_use=scores_to_use)
                scores = [s.numpy().squeeze() for s in scores]
                return scores
            newcol = self.bpl_index_to_col_name(lib_index, ver="parses")
            self.Dat = applyFunctionToAllRows(self.Dat, F, newcol)


    def bpl_score_parses_factorized(self, lib_refit_index=0, libraries_to_apply_inds = None,
        weights = None):
        """
        RETURNS:
        - scores, factorized into features.
        """
        # weights = [1/3, 1/3, 1/3]
        from ..bpl.strokesToProgram import scoreMPs_factorized
        if weights is None:
            weights = {"k":1/3, "parts":1/3, "rel":1/3}
        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)
        _checkPandasIndices(self.Dat)

        # new way, extract as vector.
        out = []
        for L in LibListThis:
            lib_index = L["index_grp"]
            libthis = L["lib"]
            print(lib_index)
            
            # Go thru all parses
            for i, row in enumerate(self.Dat.iterrows()):
                MPlist = list(row[1]["parses_motor_programs"])
                scores_dict = scoreMPs_factorized(MPlist, lib=libthis, return_as_tensor=False)

                out.append({
                    "lib_index":lib_index,
                    "trialcode":row[1]["trialcode"],
                    "row_num":i,
                    "scores_features":scores_dict
                    })
        return out


    ################ PARSES GOOD [PARSER]
    def _get_parser_sdir(self, parse_params, pathbase=None, assume_only_one_parseset=True):
        """ Return cached pathbase (or caches, if this is run for first time)"""
        if self._ParserPathBase is None:
            if pathbase is None:
                assert len(self.Metadats)==1, "only run this if one dataset"
                self._ParserPathBase = f"{self.Metadats[0]['path']}/parser_good"
            else:
                # save
                self._ParserPathBase = pathbase

        pathlist = findPath(self._ParserPathBase, 
            [[f"ver_{parse_params['ver']}", f"quick_{parse_params['quick']}", f"{parse_params['savenote']}"]], 
            return_without_fname=True)
        if assume_only_one_parseset and len(pathlist)>1:
            
            assert False, "found >1 set, need to mnaually prune"
        pathdir = pathlist[0]
        return pathdir

    def parser_extract_parses_single(self, ind, ver,
            plot=False, quick=False,
            # nwalk_det = 10, max_nstroke=200, max_nwalk=50, N=1000, skip_parsing = False
            nwalk_det = 10, max_nstroke=350, max_nwalk=75, N=1200, skip_parsing = False
            ):
        """ extract Parser, along with parses, for this ind, using input params
        """
        from pythonlib.parser.parser import Parser

        if ver=="graphmod":
            # then modify graph. but do not enter the ground truth parses, since now 
            # graph will not be match original strokes.
            graph_mods = ["strokes_ends", "merge", "splitclose", "merge", 
                "merge_close_edges", "merge", 
                "loops_floating", "loops_cleanup", "loops_add"]
            graph_mods_params = [
                {"strokes":"StrokesInterp", "thresh":5},
                {"thresh":40},
                {"thresh":30}, # was 25 before 9/17/21
                {"thresh":40}, 
                {"thresh":25},
                {},
                {},
                {},
                {"THRESH":40}]
            do_input_by_hand=True
            do_input_by_hand_list = ["strokes"]
        elif ver=="nographmod":
            # then add input by hand.
            # Still enter stroke ends as new nodes.
            graph_mods = ["strokes_ends", "loops_floating", "loops_cleanup"]
            graph_mods_params = [
                {"strokes":"StrokesInterp", "thresh":5},
                {},
                {}]
            do_input_by_hand=True
            do_input_by_hand_list = ["strokes"]
        else:
            assert False, "not coded"

        Task = self.Dat.iloc[ind]["Task"]
        strokes_task = self.Dat.iloc[ind]["strokes_task"]

        # Initialize
        P = Parser()
        P.input_data(strokes_task)

        # Make skeleton
        didmod_all = P.make_graph_pipeline(graph_mods=graph_mods, 
            graph_mods_params=graph_mods_params)
        if skip_parsing and plot:
            P.plot_graph()

        if skip_parsing is False:
            ############ PARSES
            if do_input_by_hand:
                # if didmod_all is False:
                #     # Then still origianl pts in graph, can enter by hand.
                    
                for handver in do_input_by_hand_list:
                    if handver=="strokes":
                        # First, add by hand all the base parses.   
                        # Pass in parse based on strokes
                        strokes = P.StrokesInterp
                        # P.plot_graph()
                        # self.plotMultStrokes([P.StrokesInterp])
                        if ver=="nographmod":
                            # then looser
                            P.wrapper_input_parse(strokes, "strokes", 
                                apply_transform=False, note="strokes")
                        elif ver=="graphmod":
                            P.wrapper_input_parse(strokes, "strokes", 
                                apply_transform=False, note="strokes",
                                require_stroke_ends_on_nodes=False)
                        else:
                            assert False, "how to do?"

                    elif handver=="shapes":

                        ### PASS IN PARSE BASED ON OBJECTS
                        if all([x[0] is not None for x in Task.Shapes]):
                            # then you have shape info
                            assert False, "extract shapes, not yet coded"
                            # P.manually_input_parse_from_strokes(strokes, note="objects")
                            P.wrapper_input_parse(strokes, "strokes", note="objects")

                    elif handver=="chunks":
                        ### CHUNKS
                        Tml2 = self.Dat.iloc[ind]["Task"].Params["input_params"]
                        strokes_models = {}
                        chunk_models = Tml2.chunks_extract_models()
                        for mod, loc in chunk_models.items():
                            assert Tml2.chunks_extract(mod)==loc
                            los = Tml2.chunks_convert_to_strokes(loc, reorder=True)
                            strokes_models[mod] = los
                            for strokes in los:
                                # P.manually_input_parse_from_strokes(strokes, note="chunks")
                                P.wrapper_input_parse(strokes, "strokes", note="chunks")

                    else:
                        assert False
                # print(do_input_by_hand)
                # print(didmod_all)
                # P.summarize_parses()
                # assert False


            ## Walker to get parses.
            P.parse_pipeline(quick=quick, stroke_order_doesnt_matter=False, 
                direction_within_stroke_doesnt_matter=True,
                nwalk_det = nwalk_det, max_nstroke=max_nstroke, max_nwalk=max_nwalk, N=N)

            ############# FILTER PARSES

            # 1) Find parses where there are paths where repeat an edge (in the same path)
            # could repeat in same or diff direction
            P.filter_all_parses("any_path_has_redundant_edge")
            P.filter_all_parses("any_two_paths_have_common_edge")

            if plot:
                P.parses_fill_in_all_strokes()
                P.summarize_parses()

        return P

    def parser_extract_bestperms_wrapper(self, saveon = True, force_redo=False):
        """ RUN THIS wrapper. for each parser, for each base parse, extract the best-fitting permutation to beh.
        Is smart about checking if already done, and not overwriting old stuff. will
        not waste time urrnning if already done. 
        - goes thru all trials and runs all, along with appropriate logging, etc.
        REQUIRES:
        - extracted and loaded all parses into this dataset.
        NOTE: 
        This finsds the single best perm, works for
        parses that don't use chunks.
        """

        from pythonlib.tools.expttools import update_yaml_dict, load_yaml_config
        import os

        list_parse_params = self._Parser_list_parseparams # get this from loading.

        for parse_params in list_parse_params:
            graphmod = parse_params["ver"]

            # the actual cached save dir
            sdir = self._get_parser_sdir(parse_params)
            pathdict = f"{sdir}/dict_beh_aligned_permutations_log.yaml" # for logging
            
            for indtrial in range(len(self.Dat)):
                # Behavioral details for this trial.
                trialcode = self.Dat.iloc[indtrial]["trialcode"]
                behid = (self.identifier_string(), self.animals(force_single=True)[0], trialcode)
                taskname=self.Dat.iloc[indtrial]["unique_task_name"]
                fname = f"{sdir}/{taskname}-behalignedperms.pkl"

                # First check whether this already done
                if force_redo is False:
                    if True:
                        # New version, look for the file directly
                        if os.path.isfile(fname):
                            print("Skipping parser_extract_bestperms_wrapper, since alreayd done", fname)
                            continue
                    else:
                        # Old version, look for this file in the log. Not most accurate.
                        x = load_yaml_config(pathdict, make_if_no_exist=True)
                        if taskname in x.keys():
                            if behid in x[taskname]:
                                print("Skipping, since alreayd done", parse_params, indtrial, taskname)
                                continue

                # Run, get one best permutation for each base parse.
                P = self._parser_extract_bestperms(indtrial, graphmod)

                if saveon:
                    print("Saving parse with added beh-aligned perms", parse_params, indtrial, taskname)

                    # parser to save
                    # P = self.parser_list_of_parsers(indtrial, [f"parser_{graphmod}"])[0]

                    ## resave these parses...
                    with open(fname, "wb") as f:
                        pickle.dump(P, f)

                    ## log 
                    update_yaml_dict(pathdict, taskname, behid, allow_duplicates=False)

    def parser_get_parser_helper(self, indtrial, parse_params=None, load_from_disk=False):
        """ [GOOD] flexible getting of a single Parser instance
        IN:
        - parse_params
        --- if None, then there must only be one parser. otherwise fails
        OUT:
        - 
        NOTE:
        - fails if doesnt find only one.
        """
        
        if parse_params is None:
            parse_cols = [col for col in self.Dat.columns if "parser_" in col]
            assert len(parse_cols)==1
            col = parse_cols[0]
        else:
            graphmod = parse_params["ver"]
            col = f"parser_{graphmod}"
        list_P = self.parser_list_of_parsers(indtrial, [col])
        assert len(list_P)==1
        return list_P[0]

    def parser_prunedataset_bychunkrules(self, list_rule):
        """ Removes trials that dont have at least one self.ParsesBase that 
        for each rule in list_rule. Useful for model comparison, if a trial
        just is not parsable by a given model (e.g.,lolli model, but no possible
        lollis)
        INPUT:
        - list_rule, list of str, each trial must have all of these rules
        e.g.,         list_rule = ["lolli", "linetocircle", "circletoline"]
        RETURN:
        - D, a copy of self. Does not modify self.
        """

        # Get list of good trials
        list_inds_good = []
        for i in range(len(self.Dat)):
            P = self.parser_get_parser_helper(i)
            list_baseparse_rules = [p["rule"] for p in P.ParsesBase]
            x = [r in list_baseparse_rules for r in list_rule]
            
            if all(x):
                list_inds_good.append(i)

        # Return pruned dataset
        print("original length:", print(len(self.Dat)))
        Dcopy = self.subsetDataset(list_inds_good)
        print("new lenghth:", print(len(Dcopy.Dat)))
        return Dcopy
                    
            
    


    def parser_extract_chunkparses(self, indtrial, parse_params, saveon=True, 
        force_redo_best_fit=False, how_to_store="dict_in_baseparse", 
        reload_parser_each_time=False, DEBUG=False):
        """
        # Get chunked parses baed on rules, and get all permtuations, and get best-fitting 
        permutation to this trial's data.
        [MANYT HINGS, wrapper]. Must have already extracted parses, but otherwise dont have to have
        extracted the best-fit parses using the other code. Will not delete any previous parses. new parses
        are entered by updating old parses, not by appending. 
        - INPUT:
        - saveon, then overwrites file.
        - reload_parser_each_time, then reloads from disk this parser. Useful if doing in parallel multiple
        extractions over datasets, so that can make sure is loading the latest Parser.
        # TODO: split stuff operating on Tasks vs on Beh trial (top vs. bottom, see comments)
        """
        ################# MODIFY THESE
        list_rule = ["baseline", "linetocircle", "circletoline", "lolli"] # should not be ony the one for this expt
        # since want to score this beh across all models.
        expt = "gridlinecircle"
        # list_rule = self.rules(force_single=True) # otherwise does all rules multipel times...

        if DEBUG:
            # only run if it is this task
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]
            if not taskname=="mixture2-ss-5_1-064566":
                print("[DEBUG] SKIPPED, becuase is: ", taskname)
                print(taskname=="mixture2-ss-5_1-064566")
                return

        ############################## STUFF THAT DOESNT DEPENDS ON THE BEHAVIOR
        # 1) Extract things for this trial
        Task = self.Dat.iloc[indtrial]["Task"]
        objects = Task.Shapes
        if reload_parser_each_time:
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]
            sdir = self._get_parser_sdir(parse_params)
            fname = f"{sdir}/{taskname}-behalignedperms.pkl"
            print("** LOADING parser from disk", parse_params, indtrial, taskname, "from", fname)
            with open(fname, "rb") as f:
                P = pickle.load(f)
        else:
            P = self.parser_get_parser_helper(indtrial, parse_params)

        strokes_task = self.Dat.iloc[indtrial]["strokes_task"] # original, ignore chunking

        # 2) enter base parses for each rule into P.ParsesBase
        from pythonlib.drawmodel.chunks import find_chunks_wrapper
        # is_base_parse = True
        for rule in list_rule:
            tmp = P.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
            if len(tmp)>0:
                print("SKIPPING finding base parse, since already exists: ", rule)
                continue

            # Get chunk info for this rule
            list_chunks, list_hier, list_fixed_order = find_chunks_wrapper(Task, expt, 
                rule, strokes_task)
            
            # Enter each one as a new base parse
            for chunks, hier, fixed_order in zip(list_chunks, list_hier, list_fixed_order):
                print("ENTERING NEW BASE PARSE:, ", chunks, hier, fixed_order, rule)
                params = {
                    "chunks":chunks,
                    "hier":hier,
                    "fixed_order":fixed_order,
                    "objects_before_chunking":objects,
                    "rule":rule
                }

                P.wrapper_input_parse(strokes_task, "strokes_plus_chunkinfo", params=params, is_base_parse=True)

        # 3) Get all permutations for each baseparse. overlap them by appending to a list keeping track of link between perm and baseparse
        list_indparse = range(len(P.ParsesBase))
        P._parses_reset_perm_of_list()
        for indparse in list_indparse:
            # Permutations already gotten?
            parsedict = P.extract_parses_wrapper(indparse, "dict", is_base_parse=True)
            if "done_permutations" in parsedict.keys():
                if parsedict["done_permutations"]==True:
                    print("Skipping permutation extraction for base parse: ", indparse)
                    continue    
            if False:
                # dont do this, doesnt inset, but appends. need to remove reduntant, then will lsoe some maybe
                list_list_p = P.get_all_permutations(indparse, is_base_parse=True, direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)
                print(len(list_list_p))
                # dont need, since did updating method, not appending
                P.parses_remove_redundant(stroke_order_doesnt_matter=False, direction_within_stroke_doesnt_matter=True)
            else:
                P.get_hier_permutations(indparse, update_not_append=True)
            parsedict["done_permutations"]=True


        ############################## THIS DEPENDS ON THE BEHAVIOR
        # 4) Out of all perms, get the best-fitting to behvaior, and not that down
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        def _dist(strokes1, strokes2):
            #TODO: this distnace should be identical to the one used in likelihood. 
            #TODO: Deal with issue of repeated strokes.
            return distscalarStrokes(strokes1, strokes2, "dtw_segments")

        strokes_beh = self.Dat.iloc[indtrial]["strokes_beh"]
        trial_tuple = self.trial_tuple(indtrial) 
        # 3) for each, find best per`m
        # Method: search thru all perms
        # Cant use the lsa method since it assumes one to one map between strokes beh and task, but here there is 
        # complex hierarchy
        for rule in list_rule:
            list_indparse = P.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
            for indparse in list_indparse:

                # If best fit already gotten, skip
                if force_redo_best_fit is False:
                    if "best_fit_perms" in P.ParsesBase[indparse].keys():
                        if trial_tuple in P.ParsesBase[indparse]["best_fit_perms"].keys():
                            print(f"Skipping best-fit-parse-getter, (since already gotten) for {rule, indparse, trial_tuple}")
                            continue
                    
                print("finding best parse for rule ", rule , "ind parse", indparse)
                # find all perms that are of this 
                inds = P.findparses_bycommand("permutation_of_v2", {"parsekind": "base", "ind":indparse}, is_base_parse=False)
                if len(inds)==0:
                    print("need to first get perms for baseparsr", (rule, indparse))
                    assert False
                # across these inds, find the one that best fits the behavior.
                if False:
                    # Crashes - memoyry?
                    list_strokes_parse = [P.extract_parses_wrapper(i, "strokes", is_base_parse=False) for i in inds]
                    list_d = [_dist(strokes_beh, strokes_p) for strokes_p in list_strokes_parse]
                else:
                    list_d = [_dist(strokes_beh, P.extract_parses_wrapper(i, "strokes", is_base_parse=False)) for i in inds]

                
        #         D.plotMultStrokes([strokes_beh, list_strokes_parse[0]])
        #         assert False, "check units"
                
                if len(list_d)==0:
                    print(inds)
                    print(indparse)
                    assert False
                ind_parse_best = inds[list_d.index(min(list_d))]
                
        #         print(ind_parse_best, inds, list_d)
        #         D.plotMultStrokes([strokes_beh])
        #         P.plot_parses([ind_parse_best])
                
        #         assert False
                
                # save note into this parse
                # trialcode = self.Dat.iloc[indtrial]["trialcode"]
                # note_suffix_tuple = (
                #     self.animals(force_single=True)[0],
                #     self.expts(force_single=True)[0],
                #     self.rules(force_single=True)[0],
                #     trialcode
                #     )
                ind_par = ("base", indparse)
                keyvals_update = {
                    "bestperm_beh_list":trial_tuple,
                    "bestperm_of_list":ind_par,
                    "bestperm_rule_list":rule
                }
                print("Found best parse at perm: ", ind_parse_best, " trial_tuple: ", trial_tuple)
                if how_to_store=="dict_in_baseparse":
                    # better? save directly as a dict item
                    if "best_fit_perms" not in P.ParsesBase[indparse].keys():
                        P.ParsesBase[indparse]["best_fit_perms"] = {}
                    print(f"placing into dict: self.BaseParses[indtrial][{trial_tuple}]")
                    P.ParsesBase[indparse]["best_fit_perms"][trial_tuple] = P.extract_parses_wrapper(ind_parse_best, "dict")
                elif how_to_store=="insert_into_Parses":
                    # add metadat to self.Parses
                    pardict = P.update_existing_parse(ind_parse_best, keyvals_update, append_keyvals=True,
                                            is_base_parse=False)
                else:
                    False

        if saveon:
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]

            # parser to save
            # P = self.parser_list_of_parsers(indtrial, [f"parser_{graphmod}"])[0]
            sdir = self._get_parser_sdir(parse_params)

            ## resave these parses...
            fname = f"{sdir}/{taskname}-behalignedperms.pkl"
            print("** SAVING chunked parses", parse_params, indtrial, taskname, "to", fname)
            # if fname=="/data2/analyses/database/PARSES_GENERAL/gridlinecircle/ver_graphmod-quick_True/mixture2-ss-3_1-085968-behalignedperms.pkl":
            #     assert False
            with open(fname, "wb") as f:
                pickle.dump(P, f)



    def _parser_extract_bestperms(self, indtrial, ver, note_suffix=None, 
        how_to_store="insert_into_Parses", which_base_ver="Parses"):
        """
        For each baseparse, find the top permtuation that matches behavior on 
        this trial.
        INPUT
        - ind_trial, 0, 1, ...
        - ver, {'graphmod', 'nographmod'}
        - note_suffix, useful if want to tag this permutation as best based on a specific trial beh.
        NOTE:
        - can run this
        """
        from pythonlib.tools.stroketools import assignStrokenumFromTask

        # Get parser
        list_P = self.parser_list_of_parsers(indtrial, parser_names=[f"parser_{ver}"])
        assert len(list_P)==1
        P = list_P[0]

        assert P.Finalized==True, "easy fix if not - just finalize each parse one by one.."

        # Get all base parses
        if which_base_ver=="ParsesBase":
            assert False, "dont use this"
            # New version, seaprate atribute holding base parses
            inds_parses = list(range(len(P.ParsesBase)))
        elif which_base_ver=="Parses":
            inds_parses = P.findparses_bycommand("base_parses")
        else:
            assert False, ""
        
        # Get actual behavior    
        strokes_beh = self.Dat.iloc[indtrial]["strokes_beh"]
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        note_suffix = f"{self.animals(force_single=True)[0]}-{trialcode}" # uniquely id this beh
        note_suffix_tuple = (
            self.animals(force_single=True)[0],
            self.expts(force_single=True)[0],
            self.rules(force_single=True)[0],
            trialcode
            )
        for ind_par in inds_parses:

            # get parse in strokes
            strokes_baseparse = P.extract_parses_wrapper(ind_par, "strokes")
            list_p = P.extract_parses_wrapper(ind_par, "parser_stroke_class")

            if False:
                self.plotMultStrokes([strokes_beh, strokes_baseparse]);
                assert False, "confirm they overlap"

            # get task stroke inds in assigned order
            inds_baseparse_assigned = assignStrokenumFromTask(strokes_beh, 
                strokes_baseparse, ver="stroke_stroke_lsa_usealltask")

            # Get the permutation, to store as a new parse
            list_p_this = [list_p[i] for i in inds_baseparse_assigned]

            parsenew = {
                "strokes":None,
                "list_ps": list_p_this,
                "permutation_of": ind_par,
                "note":"bestperm" if note_suffix is None else f"bestperm-{note_suffix}",
                "bestperm_beh_list":[note_suffix_tuple],
                "bestperm_of_list":[ind_par]
            }
            
            if how_to_store=="insert_into_Parses":
                # Place into self.Parses. First tries to update. if cant find, then appends.
                # First, see if already exists
                keyvals_update = {
                    "bestperm_beh_list":note_suffix_tuple,
                    "bestperm_of_list":ind_par
                }
                success, ind = P.update_parse_with_input_parse(parsenew, keyvals_update, 
                    stroke_order_doesnt_matter=False,
                    direction_within_stroke_doesnt_matter=True,
                    append_keyvals=True
                    )
                if success:
                    print('Updated at ind', ind, ' with new best-fitting parse, for baseparse: ', ind_par)
                if not success:
                    # Then add it to parses
                    P.wrapper_input_parse(parsenew, ver="dict")
                    print('Added new best-fitting parse, for baseparse: ', ind_par)
            elif how_to_store=="dict_in_baseparse":
                assert False, "dont use this"
                # self.ParsesBase[this]["best_fit_perms"][trial_tuple]
                if "best_fit_perms" not in P.ParsesBase[ind_par].keys():
                    P.ParsesBase[ind_par]["best_fit_perms"] = {}
                P.ParsesBase[ind_par]["best_fit_perms"][note_suffix_tuple] = parsenew
            else:
                assert False
        P.parses_remove_redundant(stroke_order_doesnt_matter=False,
            direction_within_stroke_doesnt_matter=True)
        return P


    def parser_extract_and_save_parses(self, ver, quick=False, 
        saveon=True, savenote="", SDIR=None, save_using_trialcode=True):
        """ 
        INPUTS:
        - SDIR, if pass, then uses that. otherwise saves in this datasets folder
        - save_using_trialcode, if True, filename is trialcode. if false, filename is unqiue task id
        """
        # from pythonlib.tools.stroketools import getStrokePermutationsWrapper
        from pythonlib.tools.expttools import makeTimeStamp
        from pythonlib.parser.parser import Parser

        assert len(self.Metadats)==1, "only run this if one dataset"

        if SDIR is None:
            SDIR = f"{self._get_parser_sdir()}/parses/{makeTimeStamp()}"
        sdir = f"{SDIR}/ver_{ver}-quick_{quick}"
        if len(savenote)>0:
            sdir += savenote

        os.makedirs(sdir, exist_ok=True)
        print("Saving parses to : ", sdir) 
        
        # 1) Quickly save dict mapping trialcode --> unique task name
        if saveon:
            dict_trialcode_taskname = {}
            for i in range(len(self.Dat)):
                trialcode = self.Dat.iloc[i]["trialcode"]
                taskname = self.Dat.iloc[i]["unique_task_name"]
                dict_trialcode_taskname[trialcode]=taskname
            # save other things
            path = f"{sdir}/dict_trialcode_taskname-{self.identifier_string()}.pkl"
            with open(path, "wb") as f:
                pickle.dump(dict_trialcode_taskname, f)
            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(dict_trialcode_taskname, 
                f"{sdir}/dict_trialcode_taskname-{self.identifier_string()}.yaml")

        # 2) Get parses.
        taskname_list = []
        for i in range(len(self.Dat)):
            trialcode = self.Dat.iloc[i]["trialcode"]
            taskname = self.Dat.iloc[i]["unique_task_name"]

            if save_using_trialcode:
                path = f"{sdir}/trialcode_{trialcode}.pkl"
            else:
                path = f"{sdir}/{taskname}.pkl"

            if taskname in taskname_list:
                print("** Skipping: ", self.identifier_string(), taskname, path, "[ALREADY DONE 1]")
                continue
            
            taskname_list.append(taskname)

            if os.path.isfile(path):
                print("** Skipping: ", self.identifier_string(), taskname, path, "[ALREADY DONE 2]")
                continue

            print("** Making graph and parsing: ", self.identifier_string(), taskname, path)

            try:
                P = self.parser_extract_parses_single(i, ver,
                    plot=False, quick=quick)
            except AssertionError as err:
                print(self.identifier_string())
                print(i)
                print(self.Dat.iloc[i]["trialcode"])
                print(taskname)
                print(err)
                print("[EXCEPTION] ")
                raise err
        # for row in self.Dat.iterrows():
        #     trialcode = row[1]["trialcode"]
        #     strokes_task = row[1]["strokes_task"]

        #     P = Parser()
        #     P.input_data(strokes_task)
        #     P.parse_pipeline(quick=quick)

            # save
            if saveon:
                try:
                    # out = {"Planner":}
                    with open(path, "wb") as f:
                        pickle.dump(P, f)
                except:
                    import time
                    time.sleep(5) # sleep in case this error is becuase file accessed by other instance of phythong
                    # out = {"Planner":}
                    with open(path, "wb") as f:
                        pickle.dump(P, f)
        print("length of dataset", len(self.Dat))
        print("GOT THESE TASKS:", taskname_list)
        print("** DONE!! parser_extract_and_save_parses", self.identifier_string())



    def parser_load_presaved_parses(self, list_parseparams, list_suffixes=None,
        finalize=True, pathbase=None, name_ver = "unique_task_name", 
        ensure_extracted_beh_aligned_parses=False):
        """ warpper to load multiple parses, each into a diffefenrt coilumn in self.Dat, with names based on 
        list_suffixes.
        INPUT:
        -list_parseparams, list of dicts, one for each set of parses to load, into individula Parser
        objects
            e.g.,:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"},
                {"quick":True, "ver":"nographmod", "savenote":"fixed_True"}]
        - list_suffixes, names= ["graphmod", "nographmod"], expected in the name of the directory holding parses
        - pathbase, useful if there is common dir across expts, since they share tasks.
        - name_ver, convention that was used for naming the parses
        - ensure_extracted_beh_aligned_parses, check that all trials and parsers have already
        extracted the permutations aligned to behavior. if have not, then will do that extraction
        [NOTE: this might take time the first time!]
        """

        if list_suffixes is not None:
            assert len(list_suffixes)==len(list_parseparams)
        else:
            list_suffixes = [None for _ in range(len(list_parseparams))]

        for parse_params, suffix in zip(list_parseparams, list_suffixes):
            self._parser_load_presaved_parses(parse_params, suffix=suffix, finalize=finalize, 
                pathbase=pathbase, name_ver = name_ver)

        self._Parser_list_parseparams = list_parseparams

        if ensure_extracted_beh_aligned_parses:
            self.parser_extract_bestperms_wrapper()


    def _parser_load_presaved_parses(self, parse_params = None, 
        assume_only_one_parseset=True, suffix=None, finalize=True, pathbase=None,
        name_ver = "unique_task_name"):
        """ helper, to load a single set of parses, which are defined by their parse_params, which 
        indexes into a saved set of ata.
        - Fails if any tiral is faield to nbe found
        INPUT:
        - suffix, new columns will be called parser_{suffix}, useful if exrtracting multiple parsers, then merging
        """

        if parse_params is None:
            parse_params = {"quick":False, "ver":"graphmod", "savenote":""}

        pathdir = self._get_parser_sdir(parse_params, pathbase=pathbase)

        # pathlist = findPath(pathbase, [[f"ver_{parse_params['ver']}-quick_{parse_params['quick']}-{parse_params['savenote']}"]], return_without_fname=True)
        # if assume_only_one_parseset and len(pathlist)>1:
        #     assert False, "found >1 set, need to mnaually prune"
        # pathdir = pathlist[0]

        def _findrow(x):
            # first, try to find beh-aligned perms
            P = self.load_trial_data_wrapper(pathdir, x['trialcode'], 
                x['unique_task_name'],name_ver=name_ver, suffix="behalignedperms", return_none_if_nofile=True)
            if P is None:
                # if cant find, then load origianl parses
                P = self.load_trial_data_wrapper(pathdir, x['trialcode'], 
                    x['unique_task_name'],name_ver=name_ver)
            if finalize:
                P.finalize()

            if not hasattr(P, "ParsesBase"):
                # recent coded
                P.ParsesBase = []

            return P
            # paththis = f"{pathdir}/trialcode_{x['trialcode']}.pkl"
            # with open(paththis, "rb") as f:
            #     tmp = pickle.load(f)
            # return tmp

        colname = f"parser_{suffix}"
        print(f"*Loaded parses into {colname}")
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, colname)

        return pathdir


    def parser_list_of_parses(self, indtrial, kind="summary", 
        parser_names = ("parser_graphmod", "parser_nographmod")):
        """ return list of parses for this trial, in summary format, 
        i.e., each parse is list of dicts.
        IN:
        - parser_names, col in self.Dat
        OUT:
        - list (nparse) of list (ntraj) of dicts.
        (is concatneated, if there are multiple parser_names)
        """
        list_of_p = []
        for name in parser_names:
            P = self.Dat.iloc[indtrial][name]
            plist = P.extract_all_parses_as_list(kind=kind)
            # print(plist[0])
            # assert False
            if kind=="summary":
                for i, p in enumerate(plist):
                    for pp in p:
                        pp["parser_name"] = name
                        pp["ind_in_parser"] = i
            list_of_p.extend(plist)
        return list_of_p


    def parser_list_of_parsers(self, indtrial, parser_names = ("parser_graphmod", "parser_nographmod")):
        """ returns list of parsers
        e.g., [P1, P2]
        """
        list_of_parsers = [self.Dat.iloc[indtrial][pname] for pname in parser_names]
        return list_of_parsers


    def parser_load_precomputed_posteriors(self, model_id):
        """ Load list of posetrtior scores (scalars, one for each row of self.Dat), precomputed
        using the model (model_id)
        IN:
        - model_id, [expt, mod], where expt and mod are strings, e.g., ["pilot", "bent"]
        OUT:
        - assigns scores to a new column, called "parser_post_{expt}_{mod}"
        """
        SDIR = f"{base_dir}/main/model_comp/planner"
        sdir = f"{SDIR}/{model_id[0]}/dset_{self.identifier_string()}-vs-mod_{model_id[1]}"
        path = f"{sdir}/posterior_scores.pkl"
        with open(path, "rb") as f:
            scores = pickle.load(f)
        assert len(scores)==len(self.Dat)
        colname = f"parser_postscores_{model_id[0]}_{model_id[1]}"
        self.Dat[colname] = scores
        print("added scores to columns in self.Dat:", colname)


    def parser_flatten(self, parser_names = ("parser_graphmod", "parser_nographmod")):
        """ take parsers (each a single object) from multiple columns (each a parser_name)
        and combines all and flattens into a new column called "parser_all_flat"
        - each parse represented as a list of p, where p is a dict holding all representations.
        - also holds reference back to origianl Parser, in case want to call it.
        NOTE:
        - also save location of origin in the first p for each list of p (i.e., each parse)
        """
        
        for pname in parser_names:
            assert pname in self.Dat.columns, "this parser no exist"

        list_list_parses = []
        for indtrial in range(len(self.Dat)):
            list_list_parses.append(self.parser_list_of_parses(indtrial, parser_names=parser_names))

            # Put origin in the first p for each parse.
            origin = self.Dat.iloc[indtrial]["origin"]
            for list_p in list_list_parses[-1]: # this trial
                list_p[0]["origin"] = origin

        self.Dat["parses_behmod"] = list_list_parses
        print("flattend parsers and moved to new col: ", "parses_behmod")
        print("Parsers:", parser_names)

    def parserflat_extract_strokes(self, ind):
        """ extract list of parses, where each parse is a strokes (i.e. a list of Nx2 array)
        """
        
        list_list_p = self.Dat.iloc[ind]["parses_behmod"]
        return [[p["traj"] for p in list_p] for list_p in list_list_p]

    def parser_names(self):
        """ gets names of all columns that start with 'parser_'
        OUT:
        - list of str.
        """
        return [col for col in self.Dat.columns if "parser_"==col[:7]]

    ############### GENERAL PURPOSE, FOR LOADING
    def load_trial_data_wrapper(self, pathdir, trialcode, unique_task_name, 
        return_none_if_nopkl=False, name_ver="trialcode", suffix="",
        return_none_if_nofile=False):
        """ looks for file. if not find, then looks in dict to find other trial that has 
        same taskname
        INPUT:
        - name_ver, str, which version to use, 
        --- {"trialcode", "unique_task_name"}
        - return_none_if_nofile, if True, then returns None if this file doesnt eixst. useful for 
        searching for files.
        NOTE:
        - Assumes that saved datapoint is shared across trials that have same task
        - Assumes that if find file iwth this trialcode/unique_task_name, then is correct
        """
        import os

        if name_ver=="trialcode":
            paththis = f"{pathdir}/trialcode_{trialcode}"
        elif name_ver=="unique_task_name":
            paththis = f"{pathdir}/{unique_task_name}"
        else:
            print(name_ver)
            assert False

        if len(suffix)>0:
            paththis += f"-{suffix}"

        paththis+=".pkl"

        if os.path.isfile(paththis):
            # then load it
            try:
                with open(paththis, "rb") as f:
                    tmp = pickle.load(f)
            except Exception as err:
                print(paththis)
                raise err
            return tmp
        else:
            # Failed - return none?
            if return_none_if_nofile:
                return None

            # Not allowed to fail..
            print("Cant find " , paththis)
            if return_none_if_nopkl:
                # if cant find file, then just return None
                return None
            else:
                # try to find it looking for same task, different trial. if fail, then raise error.
                # Read dict
                pathdict = f"{pathdir}/dict_trialcode_taskname.pkl"
                with open(pathdict, "rb") as f:
                    dict_trialcode_taskname = pickle.load(f)
                if dict_trialcode_taskname[trialcode]==unique_task_name:
                    list_tc_same_task = [k for k,v in dict_trialcode_taskname.items() if v==unique_task_name]
                    # try to load each of these trialcodes
                    for tc in list_tc_same_task:
                        tmp = self.load_trial_data_wrapper(pathdir, tc, unique_task_name, return_none_if_nopkl=True)
                        if tmp is not None:
                            return tmp
                    print(trialcode, unique_task_name)
                    assert False, "did not find task, both pkl, or thru other tasks with same taskname."
                else:
                    assert False, "task name changed since you saved this data?"
        assert False, "shouldnt have gotten here.."



    ################ PLANNER MODEL, like a simpler version of BPL
    def _get_planner_sdir(self):
        assert len(self.Metadats)==1, "only run this if one dataset"
        return f"{self.Metadats[0]['path']}/planner_model"

    def planner_extract_save_all_parses(self, permver = "all_orders_directions",
        num_max=1000):
        """ extract and save all permtuations for each task, using the input ver. 
        will save separately depending on ver (and run)
        """
        assert len(self.Metadats)==1, "only run this if one dataset"

        from pythonlib.tools.stroketools import getStrokePermutationsWrapper
        from pythonlib.tools.expttools import makeTimeStamp
        sdir = f"{self._get_planner_sdir()}/parses/{permver}-{makeTimeStamp()}"
        os.makedirs(sdir, exist_ok=True)
        print("Saving parses to : ", sdir) 

        for row in self.Dat.iterrows():
            trialcode = row[1]["trialcode"]
            strokes_task = row[1]["strokes_task"]

            strokes_task_perms = getStrokePermutationsWrapper(strokes_task, ver=permver, 
                num_max=num_max)

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(strokes_task_perms, f)

    def planner_load_presaved_parses(self, permver = "all_orders_directions",
        assume_only_one_parseset=True):
        """ Extract for each trial, parses, permtuations of task, pre-extracted using 
        planner_extract_save_all_parses
        INPUTS:
        - assume_only_one_parseset, then fails if finds >1 presaved ste.
        NOTES:
        - will fail unless can find one and only one for each trial in self
        """
        pathbase = f"{self._get_planner_sdir()}/parses"
        pathlist = findPath(pathbase, [[permver]], return_without_fname=True)
        if assume_only_one_parseset and len(pathlist)>1:
            assert False, "found >1 set, need to mnaually prune"
        pathdir = pathlist[0]

        def _findrow(x):
            paththis = f"{pathdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp

        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner")

        return pathdir

    def planner_score_beh_against_all_parses(self, permver = "all_orders_directions",
        distfunc = "COMBO-euclidian-euclidian_diffs",
        confidence_ver = None):
        """ for each trial the beh is scored against all parses
        parses must have been presaved (indexed by permver)
        Here, saves these scores.
        INPUTS:
        - distfunc, passes into scoreAgainstBatch, could be string or func
        - confidence_ver, passes into scoreAgainstBatch (leave conf ver as none since easy to compute, and currently requires sorting.)

        """
        from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
        from pythonlib.drawmodel.strokedists import scoreAgainstBatch

        pathdir = self.planner_load_presaved_parses(permver=permver)

        sdir = f"{pathdir}/beh_parse_distances"
        os.makedirs(sdir, exist_ok=True)
        print("Saving distances to : ", sdir) 

        # save params
        params = {
        "permver":permver,
        "distfunc":distfunc,
        "confidence_ver":confidence_ver,
        }
        writeDictToYaml(params, f"{sdir}/params.yaml")

        for i, row in enumerate(self.Dat.iterrows()):
            trialcode = row[1]["trialcode"]
            strokes_beh = row[1]["strokes_beh"]
            # strokes_task = row[1]["strokes_task"]
            strokes_task_perms = row[1]["parses_planner"]

            print(len(strokes_beh))
            print(len(strokes_task_perms))
            print(len(strokes_task_perms[0]))

            distances_unsorted = scoreAgainstBatch(strokes_beh, strokes_task_perms, 
                    distfunc = distfunc, confidence_ver=confidence_ver, sort=False)[0]
            # leave conf ver as none since easy to compute, and currently requires sorting.
            # leave sort False, so that distances are always corresponding to presaved parses.

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(distances_unsorted, f)

            if i%200==0:
                print(i)

    def planner_score_parses(self, permver = "all_orders_directions",
        scorever="dist_traveled"):
        """ scores parses , eg by efficiency
        INPUTS
        - permver, is to index, to extract the correct parses.
        """
        from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
        from pythonlib.drawmodel.strokedists import scoreAgainstBatch

        pathdir = self.planner_load_presaved_parses(permver=permver)

        sdir = f"{pathdir}/parse_scores"
        os.makedirs(sdir, exist_ok=True)
        print("Saving parse scores to : ", sdir) 

        # save params
        params = {
        "scorever":scorever,
        }
        writeDictToYaml(params, f"{sdir}/params.yaml")

        # get function, given scorever
        if isinstance(scorever, str):
            if scorever=="dist_traveled":
                from pythonlib.drawmodel.features import computeDistTraveled
                scorefun = lambda strokes: computeDistTraveled(strokes, include_origin_to_first_stroke=False, 
                                include_transition_to_done=False)
            else:
                print(scorever)
                assert False, "not coded"
        else:
            # asusme is a function
            scorefun = scorever

        # compute
        for i, row in enumerate(self.Dat.iterrows()):
            trialcode = row[1]["trialcode"]
            strokes_task_perms = row[1]["parses_planner"]
            distances_unsorted = [scorefun(strokes) for strokes in strokes_task_perms]

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(distances_unsorted, f)

            if i%200==0:
                print(i)

    def planner_load_everything(self, permver = "all_orders_directions"):
        """ Loads (1) parses; (2) beh-parse distances; (3) parses scores
        """

        # Load parses
        pathdir = self.planner_load_presaved_parses(permver=permver)
        print("Loaded parses to column: parses_planner")

        # Load beh task scores
        sdir = f"{pathdir}/beh_parse_distances"
        def _findrow(x):
            paththis = f"{sdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner_behtaskdist")
        print("Loaded beh-task dsitrances to column: parses_planner_behtaskdist")

        # Load task scores
        sdir = f"{pathdir}/parse_scores"
        def _findrow(x):
            paththis = f"{sdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner_taskscore")
        print("Loaded parse scores to col: parses_planner_taskscore")


    def planner_assign_sequence_id(self):
        """ for a given task, number sequences as 0, 1, 2, ..., where sequence number is
        based on order in saved planner pareses. 
        e.g., all trials with 0 have the same discrete parse, nbased on closest match to planner parses.
        RETURNS:
        - new col in self.Dat, "sequence_id"
        NOTE:
        - if ties, then will asign to sequence with lower index.
        """

        self.planner_load_everything()
        def F(x):
            parses_planner_behtaskdist = x["parses_planner_behtaskdist"]
            # find index if minimum dist
            return np.argmin(parses_planner_behtaskdist)
        self.Dat = applyFunctionToAllRows(self.Dat, F, "sequence_id")


    def planner_plot_summary(self, ind_list, n_task_perms_plot="all"):
        """ plots all task perms, along with their distance to beh, and their efficneciy scores
        INPUTS:
        - ind, index into self.Dat
        """
        from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences_quick

        ranklist = []
        conflist =[]
        sumscorelist =[]
        for ind in ind_list:
            strokes_beh = self.Dat["strokes_beh"].values[ind]
            strokes_task_perms = self.Dat["parses_planner"].values[ind]
            beh_task_distances = self.Dat["parses_planner_behtaskdist"].values[ind]
            task_inefficiency = self.Dat["parses_planner_taskscore"].values[ind]

            rank, conf, sumscore, strokest = rank_beh_out_of_all_possible_sequences_quick(
                strokes_beh, strokes_task_perms, beh_task_distances, 
                task_inefficiency, plot_strokes=True, 
                plot_rank_distribution=True, plot_n_strokes=n_task_perms_plot)

            ranklist.append(rank)
            conflist.append(conf)
            sumscorelist.append(sumscore)

        fig, axes = plt.subplots(1,3, figsize=(9,3))
        x = range(len(ind_list))

        ax = axes.flatten()[0]
        ax.plot(x, ranklist, "-ok", label="rank")
        ax.axhline(0)
        ax.set_title("rank")

        ax = axes.flatten()[1]
        ax.plot(x, conflist, "-ok")
        ax.axhline(0)
        ax.set_title("confidence")

        ax = axes.flatten()[2]
        ax.plot(x, sumscorelist, "-ok")
        ax.axhline(1)
        ax.set_title("summary score")


    


    ################ PARSES
    # Working with parses, which are pre-extracted and saved int he same directocry as datasets.
    # See ..drawmodel.parsing

    def _loadParses(self, parses_good=False):
        """ load all parases in bulk.
        Assumes that they are arleady preprocessed.
        - Loads one parse set for each dataset, and then combines them.
        - parses_good, then finds the /parses_good dolfer.
        """
        PARSES = {}
        for ind in self.Metadats.keys():
            
            path = self.Metadats[ind]["path"]
            if parses_good:
                path = path + "/" + "parses_good"
            path_parses = f"{path}/parses.pkl"
            path_parses_params =  f"{path}/parses_params.pkl"

            with open(path_parses, "rb") as f:
                parses = pickle.load(f)
            with open(path_parses_params, "rb") as f:
                parses_params = pickle.load(f)
                
            PARSES[ind] = parses
        return PARSES

    def parsesLoadAndExtract(self, fail_if_empty=True, print_summary=True, parses_good = False):
        """ does heavy lifting of extracting parses, assigning back into each dataset row.
        does this by matching by unique task name.
        - if any parse is not found, then it will be empty.
        - fail_if_empty, then asserts that each row has found parses.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        _checkPandasIndices(self.Dat)

        def _extractParses(PARSES, ind_metadat, taskname):
            """ helper, to get a single list of parses for this task"""
            df = PARSES[ind_metadat]
            dfthis = df[df["unique_task_name"]==taskname]
            if len(dfthis)==0:
                print("did not find parses, for the following, returning []!!")
                print(taskname)
                print(ind_metadat)
                return []
            elif len(dfthis)>1:
                print(f"found >1 {len(dfthis)} parse! returning the first one, for the follpowing")
                print(taskname)
                print(ind_metadat)

            return dfthis["parses"].values[0]

        PARSES = self._loadParses(parses_good=parses_good)

        def F(x):
            """ pull out list of parses"""
            ind = x["which_metadat_idx"]
            task = x["unique_task_name"]
            return _extractParses(PARSES, ind, task)

        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="parses")

        if print_summary:
            # -- sanity check, did we miss any parses?
            print("num parses -- num cases")
            print(self.Dat["parses"].apply(lambda x:len(x)).value_counts())

        if fail_if_empty:
            counts = set(self.Dat["parses"].apply(lambda x:len(x)).values)
            assert 0 not in counts, "at least one row failed to extract parses..."

        print("-- parses gotten!")

    def strokesToSplines(self, strokes_ver="strokes_beh", make_new_col=True,
        add_fake_timesteps=True):
        print("converting to splines, might take a minute...")
        from ..drawmodel.splines import strokes2splines
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import fakeTimesteps      

        def F(x):
            strokes = x[strokes_ver]
            return strokes2splines(strokes)

        if make_new_col:
            newcolname = strokes_ver + "_splines"
        else:
            newcolname = strokes_ver

        self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")
        self.Dat[newcolname] = self.Dat["tmp"]
        del self.Dat["tmp"]

        if add_fake_timesteps:
            print("adding fake timesteps")
            for row in self.Dat.iterrows():
                fakeTimesteps(row[1][newcolname])

    def strokes_smooth_preprocess(self, window_time=None, PLOT_EXAMPLE=False):
        """ Smooth all the strokes in self.Dat.  REturns strokes without
        modifgyning self.Dat. To replace strokes in self,
        just take output and pass into self.Dat["strokes_beh"].
        PARAMS:
        - window_time, in sec, the hanning winodw.
        NOTES:
            -- 0.15 is good, for Diego. could even go higher.
            -- 0.3 probably good, for Diego, for getting substrokes.
            -- 0.4 can even work, for smoothing, for Diego, since his strokes can
            be long and take >1 sec.
        NOTE: tried also strokesFilter, but that didnt work well for such low freq filtereing
        needed here, and often would clip data to smaller.
        NOTE: this automatically sanity checks that outptu endpoints are close to inputs'.
        """

        if window_time is None:
            # Basic smoothing that should always do
            if self.animals(True)[0]=="Diego":
                window_time = 0.2
            elif self.animals(True)[0]=="Pancho":
                window_time = 0.1
            elif self.animals(True)[0]=="Luca":
                window_time = 0.12
            else:
                assert False

        # 1) General preprocessing, smoothing
        from pythonlib.tools.stroketools import strokesFilter, smoothStrokes

        list_strokes = self.Dat["strokes_beh"]

        # do smoothing
        # window_type = "flat"
        fs = self.get_sample_rate_alltrials()
        window_type = "hanning"
        adapt_win_len= "adapt"
        list_strokes_filt = []
        for strokes in list_strokes:
            strokes_filt = smoothStrokes(strokes, fs,
                window_time=window_time, window_type=window_type,
                     adapt_win_len=adapt_win_len)
            list_strokes_filt.append(strokes_filt)

            if PLOT_EXAMPLE:
                from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper
                from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse

                fig, axes = plt.subplots(2,2)

                ax = axes.flatten()[0]
                plotDatStrokesTimecourse(strokes, ax=ax)

                ax = axes.flatten()[1]
                plotDatStrokesTimecourse(strokes_filt, ax=ax)
                ax.set_title("strokesFilter() --> Filtered")

                ax = axes.flatten()[2]
                plotDatStrokesWrapper(strokes, ax)

                ax = axes.flatten()[3]
                plotDatStrokesWrapper(strokes_filt, ax)

                # Find velocity
                fig.savefig("/tmp/tmp.png")

                assert False, "or el;se its too many plots."

        return list_strokes_filt

    def parsesChooseSingle(self, convert_to_splines=True, add_fake_timesteps=True, 
        replace=False):
        """ 
        pick out a random single parse and assign to a new column
        - e..g, if want to do shuffle analyses, can run this each time to pick out a random
        parse
        - replace, then overwrites, otherwise fails.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import fakeTimesteps      
        import random

        assert "parses" in self.Dat.columns, "need to first extract parses"
        _checkPandasIndices(self.Dat)

        def F(x):
            """ pick out a isngle random parse"""
            strokes = random.choice(x["parses"])
            return strokes
    
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_parse", replace=replace)

        if convert_to_splines:
            print("converting to splines, might take a minute...")
            self.strokesToSplines("strokes_parse", make_new_col=False)

            # from ..drawmodel.splines import strokes2splines
            # def F(x):
            #     strokes = x["strokes_parse"]
            #     return strokes2splines(strokes)

            # self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")
            # self.Dat["strokes_parse"] = self.Dat["tmp"]
            # del self.Dat["tmp"]

        if add_fake_timesteps:
            print("adding fake timesteps")
            for row in self.Dat.iterrows():
                fakeTimesteps(row[1]["strokes_parse"])

        print("done choosing a single parse, it is in self.Dat['strokes_parse']!")


    def parses_load_motor_programs(self):
        """ load pre-saved programs
        loads best parse (MP) presaved, BPL.
        (see strokesToProgram) for extraction.
        RETURNS:
        - self.Dat adds a new column "parses_motor_programs", which is a list, same len as parses,
        holding programs. 
        - self.Parses, which stores things related to params.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # check that havent dione before
        # assert not hasattr(self, "Parses"), "self.Parses exists. I woint overwrite."
        assert "parses_motor_programs" not in self.Dat.columns

        MPdict_by_metadat = self._loadMotorPrograms(ver="parses")

        # save things
        self.Parses = {}
        self.Parses["params_MP_extraction"] = {k:MP["params"] for k, MP in MPdict_by_metadat.items()}

        count=0
        def F(x):
            nparses = len(x["parses"])
            idx_mdat = x["which_metadat_idx"]
            trialcode = x["trialcode"]
            MPlist = []
            if count%100==0:
                print(trialcode)    
            count+=1
            for i in range(nparses):
                MP = self._extractMP(MPdict_by_metadat, idx_mdat, trialcode, parsenum=i)
                assert MP is not None, "did not save this MP..."
                MPlist.append(MP)
            return MPlist

        self.Dat = applyFunctionToAllRows(self.Dat, F, 'parses_motor_programs')        


    ############## BEHAVIOR ANALYSIS
    def score_visual_distance(self, DVER = "position_hd_soft", return_vals=False):
        """
        extract scalar score for each trial, based on position-wise distance
        - return_vals, then returns list of vals, without modifiying self.Dat
        """
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        # Distance function
        def F(x):
            strokes_beh = x["strokes_beh"]
            strokes_task = x["strokes_task"]
            return distscalarStrokes(strokes_beh, strokes_task, DVER, 
                                     do_spatial_interpolate=True, do_spatial_interpolate_interval = 10)
        if return_vals:
            df = applyFunctionToAllRows(self.Dat, F, "hdoffline")
            return df["hdoffline"].to_list()
        else:
            self.Dat = applyFunctionToAllRows(self.Dat, F, "hdoffline")


    def extract_beh_features(self, 
        feature_list = ("angle_overall", "num_strokes_beh", "num_strokes_task", "circ", "dist")):
        """ extract features, one val per row, 
        INPUT:
        - feature_list, list of strings. instead of string, if pass in function, then will use that.
        func: strokes --> scalar. 
        RETURNS:
        - (modifies self.Dat in place)
        - feature_list_names, which are the col names (FEAT_...)
        NOTE:
        - for features which operate at single traj level, takes mean over all trajs for a single strokes.
        """
        from pythonlib.drawmodel.features import strokesAngleOverall, strokeCircularity, strokeDistances
        feature_list_names = []
        import warnings

        # Otheriwse shows unecesary waribng: PerformanceWarning: DataFrame is highly fragmented.
        # This is usually the result of calling `frame.insert` many times, which has poor performance.
        # Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
        # https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            # from warnings import simplefilter
            # simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

            # get overall angle for each task
            for f in feature_list:
                if not isinstance(f, str):
                    # Then should be function handle
                    x = [f(strokes) for strokes in self.Dat["strokes_beh"].values]
                else:
                    if f=="angle_overall":
                        x = [strokesAngleOverall(strokes) for strokes in self.Dat["strokes_beh"].values]
                    elif f=="num_strokes_beh":
                        x = [len(strokes) for strokes in self.Dat["strokes_beh"].values]
                    elif f=="num_strokes_task":
                        # number of strokes in ground truth task
                        x = [len(strokes) for strokes in self.Dat["strokes_task"].values]
                    elif f=="circ":
                        x= [np.mean(strokeCircularity(strokes)) for strokes in self.Dat["strokes_beh"].values]
                    elif f=="dist":
                        x = [np.mean(strokeDistances(strokes)) for strokes in self.Dat["strokes_beh"].values]
                    elif f=="hdoffline":
                        x = self.score_visual_distance(return_vals=True)
                    else:
                        print(f)
                        assert False

                print(f"Num nan/total, for {f}")
                print(sum(np.isnan(x)), "/", len(x))
                # self.Dat[f"FEAT_{f}"] = x
                self.Dat[f"FEAT_{f}"] = x
                feature_list_names.append(f"FEAT_{f}")

        print("Added these features:")
        print(feature_list_names)
        
        return feature_list_names




    ############# EXTRACT PREPROCESSED STROKES - for subsequent analyses
    def extractStrokeLists(self, filtdict=None, recenter=False, rescale=False):
        """ extract preprocessed strokes (task and beh) useful for 
        computing visual distances
        INPUT: 
        - D, Dataset
        - filtdict
        --- e.g., F = {"epoch":[1], "taskgroup":["G3"]}
        OUT:
        - makes copy of D (note, will not have the interpolated strokes in it)
        """
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        # -- preprocess strokes
        if filtdict:
            Dthis = self.filterPandas(filtdict, "dataset")
        else:
            Dthis = self.copy()

        # ---- recenter
        params =[]
        if recenter:
            params.append("recenter")
        if rescale:
            params.append("rescale_to_1")
        if len(params)>0:
            Dthis.preprocessGood(ver=None, params=params)

        # Pull out dataframe
        # -- get strokes
        strokes_beh_list = Dthis.Dat["strokes_beh"].tolist()
        strokes_task_list = Dthis.Dat["strokes_task"].tolist()

        # ---- interpolate once
        def dfunc(strokes_beh, strokes_task):
            return distscalarStrokes(strokes_beh, strokes_task, "position_hd_soft", 
                                     do_spatial_interpolate=True, do_spatial_interpolate_interval = 10,
                                        return_strokes_only=True)
        S = [dfunc(sb, st) for sb, st in zip(strokes_beh_list, strokes_task_list)]
        
        # break out
        strokes_beh_list = [SS[0] for SS in S]
        strokes_task_list = [SS[1] for SS in S]
        
        return strokes_beh_list, strokes_task_list, Dthis    


    ############# DAT dataframe manipualtions
    def grouping_conjunctions_print_variables_save(self, var, list_vars_others, path,
        n_min=0, ignore_values_called_ignore=True, plot_counts_heatmap_savedir=None,
        DF = None):
        """
        Help print existing conjucntions of variables in self.Dat.
        PARAMS:
        - var, str, a single variable whose levels' variation you care about as an
        independnet variable, mainpuation.
        - list_vars_others, list of str, variables you awnt to condition on. will
        find levels of var conditions on each level fo this.
        - path, string, path (with extention) to save at.
        RETURNS:
        - prints text file at path (give)
        """

        if DF is None:
            DF = self.Dat
            
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        # n_min = 0 # leave here, to plot all levels. 
        PRINT = False
        DEBUG = False
        lenient_allow_data_if_has_n_levels = 2 # 2 allows plotting all cases
        dfout, dict_dfs = extract_with_levels_of_conjunction_vars(DF, var, list_vars_others,
                                                                  n_min_across_all_levs_var=n_min, PRINT=PRINT,
                                                                  lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels,
                                                                  DEBUG=DEBUG, PRINT_AND_SAVE_TO=path,
                                                                  ignore_values_called_ignore=ignore_values_called_ignore,
                                                                  plot_counts_heatmap_savepath=plot_counts_heatmap_savedir)
        return dfout, dict_dfs

    def grouping_append_col(self, grp_by, new_col_name, use_strings=True, strings_compact=True):
        """ append column with index after applying grp_by, 
        as in df.groupby, where the new val is  string, from
        str(list), where list is the grp_by levels for that
        row.
        PARAMS:
        - strings_compact, if True, then tries to make shorter strings (without losing info)
        RETURNS:
        - modified self.Dat
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        self.Dat = append_col_with_grp_index(self.Dat, grp_by, 
            new_col_name, use_strings=use_strings, strings_compact=strings_compact)
        print("appended col to self.Dat:")
        print(new_col_name)


    def dat_append_col_by_grp(self, grp_by, new_col_name):
        assert False, "moved to grouping_append_col"

    def grouping_print_conjunctions_summary_good(self, vars, PRINT=False, n_min = None):
        """
        Good printing of conjuctions of vars, including sample size and indices in self.Dat
        :param vars:
        :return: grpdict, grplev:indices
        """

        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        # Get trials under each epoch
        out = grouping_append_and_return_inner_items(self.Dat, vars)

        if PRINT:
            print("======================= Marginals")
            for v in vars:
                print(v, " --- ", sort_mixed_type(self.Dat[v].unique()))

            print("======================= Conjucntions keys (n trials)")
            print("-- Vars:", vars)
            for k, v in out.items():
                print(k, " -- n=", len(v))

            print("======================= Conjucntions and indices")
            for k, v in out.items():
                print(k, v)

        if n_min is not None:
            out = {k:v for k,v in out.items() if len(v)>=n_min}

        return out

    def grouping_print_n_samples(self, list_groupouter_grouping_vars, Nmin=0, savepath=None,
        save_as="txt"):
        """ Print n trials for each of conjucntive levels, multiple grouping vars.
        """
        from pythonlib.tools.pandastools import grouping_print_n_samples
        return grouping_print_n_samples(self.Dat, list_groupouter_grouping_vars, Nmin, savepath, save_as=save_as)


    def grouping_get_inner_items(self, groupouter="task_stagecategory", 
            groupinner="index", sort_keys=False,
            n_min_each_conj_outer_inner=1,
            take_top_n_inner=None):
        """ Return dict of unique items (levels of groupinner), grouped
        by groupouter levels. 
        PARAMS:
        - groupouter, string, the first grouping.
        - groupinner, string, the second grouping. either a column or "index"
        RETURNS:
        - groupdict, where each key is a level of groupouter, and
        items are the unique values of groupinner that exist for that
        groupouter level.
        EXAMPLE:
        - if groupouter = date and groupinner = character, then returns
        {date1:<list of strings of unique characters for date1>, 
        date2:<list of strings ....}
        """
        from pythonlib.tools.pandastools import grouping_get_inner_items
        return grouping_get_inner_items(self.Dat, groupouter, groupinner, sort_keys=sort_keys,
            n_min_each_conj_outer_inner=n_min_each_conj_outer_inner,
            take_top_n_inner=take_top_n_inner)

    ################# EXTRACT DATA AS OTHER CLASSES
    def behclass_preprocess_wrapper(self, reset_tokens=True, skip_if_exists=True,
                                    reclassify_shape_using_stroke_version="default",
                                    tokens_gridloc_snap_to_grid=False):
        """ Wrapper of general preprocess steps for entire datset. SKIPS if it detects
        that BehClass already extracted.
        Also skips if the self.TokensVersion is "regenerated_from_raw", which means all tokens, shapes
        etc will never use BehClass...
        PARAMS;
        - reset_tokens, then resets the tokens in TaskClass, if they exist

        NOTE: This is the ONLY place that tokens are extracted initially.
        """

        if not hasattr(self, "TokensVersion"):
            # Just the default ...
            self.TokensVersion = "taskclass"

        if self.TokensVersion == "regenerated_from_raw":
            # Then skip
            # Otherwise it WILL fail, beucase all TaskClass instances are
            # locked. (i.e., self._TokensLocked==True)
            return
        else:
            exists = self.behclass_check_if_tokens_extracted()
            if skip_if_exists and exists:
                # dont rerun
                pass
            else:
                # generate
                self._behclass_generate_alltrials(reset_tokens=reset_tokens,
                                                  reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
                                                  tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid)

                # Prune cases where beh did not match any task strokes.
                # NOTE: added here 2/21/24, used to be in dataset_strokes.py (_prepare_dataset)
                self.behclass_clean()

    def behclass_generate(self, indtrial, expt=None, reset_tokens=False):
        """ Generate the BehaviorClass object for this trial
        PARAMS:
        - expt, will try to extract if can't then you must enter
        RETURNS:
        - Beh, class instance
        - NOTE: doesn't modify self

        """
        from pythonlib.behavior.behaviorclass import BehaviorClass

        # TODO: shape_dist_thresholds, for only keeping good matches.
        # params = {
        # "D":self,
        # "ind":ind,
        # "shape_dist_thresholds":shape_dist_thresholds
        # }

        if expt is None:
            list_expt = self.expts()
            if len(list_expt)!=1:
                assert False, "you must enter expt"
            else:
                expt = list_expt[0]

        params = {
            "D":self,
            "ind":indtrial,
            "expt":expt
        }
        Beh = BehaviorClass(params, "dataset", reset_tokens=reset_tokens)
        return Beh

    def _behclass_generate_alltrials(self, reset_tokens=False,
                                     reclassify_shape_using_stroke_version="default",
                                     tokens_gridloc_snap_to_grid=False):
        """ Generate list of behClass objects, one for each trial,
        and stores as part of self.
        RETURNS:
        - self.Dat["BehClass"], list of beh class iunstance.

        NOTE: This is the ONLY place that tokens are extracted initially.

        """
        print("Extracting Behclass for each trial, may take a while...")
        ListBeh = [self.behclass_generate(i, reset_tokens=reset_tokens) for i in range(len(self.Dat))]
        self.Dat["BehClass"] = ListBeh 
        try:
            self._behclass_alignsim_compute()
            self._behclass_tokens_extract_datsegs(use_global_grid=True,
                                                  reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
                                                  tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid)         
        except Exception as err:
            # dont exit with partial behclass
            if "BehClass" in self.Dat.columns:
                del self.Dat["BehClass"]
            raise err

        print("stored in self.Dat[BehClass]")



    def _behclass_tokens_extract_datsegs(self, use_global_grid=True,
                                         ind=None,
                                         reclassify_shape_using_stroke_version="default",
                                         tokens_gridloc_snap_to_grid=False):
        """ Extract, single time, all task datsegs toekns.
        PARAMS:
        - use_global_grid, bool, if True, then gets gridx and gridx across
        all tasks in this dataset. otherwise uses the grid specific to each
        task.
        - ind, either None (gets all) or a single trial.

        NOTE: This is the ONLY place that tokens are extracted when dataset is constructed.
        """ 

        assert ind is None, "dont dot his."

        if use_global_grid:
            input_grid_xy = self.taskclass_get_grid_xy_over_all_tasks()
        else:
            input_grid_xy = None

        # if extra tforms exist, then shapes need to use the actuals trokes to be redefined.
        extra_tforms_exist = self.taskclass_check_prims_extra_params_tforms_exist()
        if (not reclassify_shape_using_stroke_version=="default") or extra_tforms_exist:
            # Then you either signalled you want to reclassify (ie not default) or you must because extra tforms exist.
            reclassify_shape_using_stroke = True
        else:
            reclassify_shape_using_stroke = False

        # Autoatmically determine shape names based on clustering sgtropkes? Useful for novel prims.        
        if reclassify_shape_using_stroke_version=="cluster_by_sim":
            print("Clustering shapes to label them (strings)")
            _, list_list_clustidx = self.shapesemantic_cluster_taskclass_strokes_to_rename_shapes()
            assert len(list_list_clustidx)==len(self.Dat)

        # if ind is None:
        print("Running D._behclass_tokens_extract_datsegs")
        # if expt==""
        for i in range(len(self.Dat)):
            Beh = self.Dat.iloc[i]["BehClass"]

            if reclassify_shape_using_stroke_version=="cluster_by_sim":
                list_cluster_by_sim_idx = list_list_clustidx[i]
            else:
                list_cluster_by_sim_idx = None

            Beh.alignsim_extract_datsegs(input_grid_xy=input_grid_xy, recompute=True,
                                            reclassify_shape_using_stroke=reclassify_shape_using_stroke,
                                            reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
                                            tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid, 
                                            list_cluster_by_sim_idx=list_cluster_by_sim_idx)
            if i%200==0:
                print(i, "_behclass_tokens_extract_datsegs")
        # else:
        #     Beh = self.Dat.iloc[ind]["BehClass"]

        #     Beh.alignsim_extract_datsegs(input_grid_xy=input_grid_xy, recompute=True,
        #                                  reclassify_shape_using_stroke=reclassify_shape_using_stroke,
        #                                  reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
        #                                  tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid,
        #                                  shape_name_overwrite=shape_name_overwrite)

    def _behclass_alignsim_compute(self, remove_bad_taskstrokes=True,
            taskstrokes_thresh=0.4):
        """ Compute beh-task alignment. This is first step before extracting
        datsegs, or discretized behavior.
        PARAMS:
        - remove_bad_taskstrokes, bool, removes taskstrokes that don't have any beh
        match - these are usually skipped storke.s
        - taskstrokes_thresh, scalar threshold below which strokes will be removed. this is
        similarity score. 0.4 is chosen as good for gridlinecircle Pancho.
        RETURNS:
        - modifies "BehClass" column in self.Dat
        """

        for i in range(len(self.Dat)):
            Beh = self.Dat.iloc[i]["BehClass"]
            Beh.alignsim_compute(remove_bad_taskstrokes=remove_bad_taskstrokes, 
                taskstrokes_thresh=taskstrokes_thresh)
            if i%200==0:
                print(i, "_behclass_alignsim_compute")

    def behclass_extract(self, inds_trials = None):
        """ Get list of behclass for these trials
        - Gets precomputed
        """
        if inds_trials is None:
            inds_trials = range(len(self.Dat))
        return [self.Dat.iloc[i]["BehClass"] for i in inds_trials]

    def behclass_find_behtask_token(self, indtrial, indstroke, version="beh"):
        """ Find the token (combines info on beh and task) that matches this trial
        and indstroke
        PARAMS;
        - indtrial, int index into self.Dat.
        - indstroke, int index with strokes, either beh or task (version)
        - version, string, eitehr beh or task, which indstroke to search for.
        RETURNS:
        - token, tuple, (<list of beh stroke inds>, <list of beh strokeclass>, task datseg (single))
        """
        out = self.behclass_extract_beh_and_task(indtrial)[3] 
        if version=="beh":
            for o in out:
                if indstroke in o[0]:
                    return o
            assert False, "didnt find this beh stroke..."
        elif version=="task":
            return out[indstroke]
        else:
            assert False

    def behclass_extract_taskstroke_inds_in_beh_order(self, indtrial, 
        version="first_touch_using_max_align"):
        """ Return indices into taskstrokes (the default indices) in order
        best aligned to beh strokes. By defualt excludes taskstrokes that were not "gotten"
        at all. This is in order of "first touch"
        RETURNS:
        - list of ints, e.g,, [0 4 2] means got taskstrokes 0, 4, and 2 in that order, and
        missed strokes 3 and 1.
        """
        Beh = self.Dat.iloc[indtrial]["BehClass"]
        if version=="first_touch":
            # In order that sorts the taskstrokes the best to match beh. this is like first tocuh.
            # THis means that more taskstrokes inds can be gotten than exist beh strokes, 
            # if a behs troke gets multipel taskstrokes. 
            inds = Beh.Alignsim_taskstrokeinds_sorted
        elif version=="each_beh_max_align":
            # independelty check for each beh what its best matching task stroke is.
            # this allow a beh to match to mulitple task storkes.
            inds = Beh.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices
        elif version=="first_touch_using_max_align":
            # The unique indices in 
            # Beh.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices and return
            # then in order
            inds = []
            for i in Beh.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices:
                if i not in inds:
                    inds.append(i)
        else:
            print(version)
            assert False

        # TO LOOK FOR DIFFERENCS ACORSS THE ABOVE>
        # for ind in range(len(D.Dat)):
        #     tmp1 = D.behclass_extract_taskstroke_inds_in_beh_order(ind, "first_touch")
        #     tmp2 = D.behclass_extract_taskstroke_inds_in_beh_order(ind, "each_beh_max_align")
        #     tmp3 = D.behclass_extract_taskstroke_inds_in_beh_order(ind, "first_touch_using_max_align")
        #     if not tmp1==tmp3:
        #         print(ind, tmp1, tmp2, tmp3)

        return inds


    def behclass_extract_beh_and_task(self, indtrial):
        """ Extract both beh (oist of PrimitiveClass) and task
        (List of datsegs) for this trial, they will be length of 
        num beh strokes.
        This requires doing alignsim stuff fifirst.
        RETURNS:
        - primlist, list of strokeClass instances, each a beh stroek 
        [in order and num of BEH]
        - datsegs_behlength, tokens, same length as primlist, for the best match for
        each beh stroke [in order and num of BEH]
        - datsegs_tasklength_firsttouch, tokens, same length as task seg that weret touched, 
        in order of first time each task stroke gotten by beh. [in order of BEH, but num of TASK TOUCHED]
        - out_combined. Combined representaion, list (length num taskstrokes) of tuples, and in
        order that they are gotten (identical to datsegs_tasklength_firsttouch). Each tuple: 
        (inds_beh, strokesbeh, dseg_task), where inds_beh are indices into Beh.Strokes indiicating
        for which beh strkes this task stroke was the first task stroke the bhe stroke touched,
        strokesbeh are those sliced strokes, and dseg_task is the single dseg for thsi taskstroke,
        """

        self.behclass_preprocess_wrapper()

        Beh = self.Dat.iloc[indtrial]["BehClass"]

        # task datsegs (get in both (i) length of task and (ii) length of beh.
        out_combined, datsegs_behlength, datsegs_tasklength_firsttouch = Beh.alignsim_extract_datsegs_both_beh_task()

        # Assocaite each beh prim with a datseg.
        if False:
            # These can differ if beh stroke doesnt match task
            assert len(Beh.Strokes) == len(datsegs_behlength)

        # assert [s() for s in Beh.Strokes] == self.Dat.iloc[indtrial]["strokes_beh"]

        return Beh.Strokes, datsegs_behlength, datsegs_tasklength_firsttouch, out_combined

    def behclass_clean(self):
        """ Clean by removing any trials where all beh strokes fail to m atch even a single
        taks stroke"""

        # Figure out the bad trials
        trials_bad = []
        trials_good = []
        for ind in range(len(self.Dat)):
            Beh = self.Dat.iloc[ind]["BehClass"]
            n1 = len(Beh.alignsim_extract_datsegs())
            # n1 = len(Beh.Alignsim_taskstrokeinds_foreachbeh_sorted)
            # n2 = len(Beh.Strokes)
            if n1==0:
                trials_bad.append(ind)
            else:
                trials_good.append(ind)

        # Only keep good trials.
        print("Removing these trials: ")
        print(trials_bad)
        self.subsetDataframe(trials_good)

    def behclass_check_if_tokens_extracted(self):
        """
        Returns True if all BehClass for each trial already has
        datsegs (tokens) extracted.
        """

        if self._BehClassExtracted:
            return True
        else:
            if "BehClass" not in self.Dat.columns:
                return False
            for ind in range(len(self.Dat)):
                Beh = self.Dat.iloc[ind]["BehClass"]
                if Beh.Alignsim_Datsegs is None:
                    return False
            self._BehClassExtracted = True
            return True

    ################# Cue - related features
    def cue_extract_cuestim_order_flipped(self, SANITY=False):
        """ determine whether cue-stim order is flipped
        PARAMS:
        - SANITY, bool, if true, checks that behcodes are ordered as expected for
        each trial based on whether flipped. this takes time since must load 
        raw ml2 beh data.
        NOTE:
        Also do sanity checks that this flipping is correctly reflected in the 
        ordering of the behcodes
        RETURNS:
        - Append a new column in self.Dat, self.Dat["CUE_csflipped"], bool.
        NOTE: looks in supervision params already extracted, unless you run sanity check.
        """

        if SANITY:
            # Extract raw ml2 data, will need below.
            self.ml2_extract_raw()

            def _behcode_get_occurances_ordered(behcodes_num,behcodes_time, codes_print=None, PRINT=False):
                # Return list of occurances of behcodes, keeping only those in codes_print, returning
                # in chronological order.   

                if codes_print is None:
                    # Hard codede times of behcodes that are relevant for determining that order has 
                    # correctly been flipped.
                    codes_print = [
                        11, # fix cue on
                        16, # touch fix,
                        132, # rulecue 2
                    #     91, 92, 21, # guide (default)
                        91, # guide (default)
                    #     93, 94, 22, # guide (flipped)
                        22, # guide (flipped)
                        71, # go
                    ]

                assert np.all(np.diff(behcodes_time)>0)

                list_nums = []
                for num, time in zip(behcodes_num, behcodes_time):
                    if num in codes_print:
                        if PRINT:
                            print(num, " -- ", time)
                        list_nums.append(num)

                return list_nums

            # list_nums = _behcode_get_occurances_ordered(codes_print, behcodes_num, behcodes_time)
            # print(list_nums)


        # COLLECT WHETHER IS FLIPPOED
        list_flipped = []
        for ind in range(len(self.Dat)):
            if SANITY:
                # Just use whatever has already been extracted
                flipped1 = self.supervision_extract_params(ind)["CUESTIM_FLIP"]
                flipped = self.blockparams_extract_single_taskparams(ind)["fix_tp"]["flip_cue_image_order"]==1
                assert flipped == flipped1, "must be a coding bug."

                
                if SANITY:
                    behcodes_num, behcodes_time = self.ml2_utils_getTrialsBehCodes(ind)
                    list_nums = _behcode_get_occurances_ordered(behcodes_num, behcodes_time)
                    
                    if False:
                        print(flipped, "--", list_nums)
                    
                    # NOTE: allow 11 and 16 to be both orders, since sometimes if touch fixation 
                    # too early,then touch code occurs first.
                    if flipped:
                        assert list_nums==[11, 16, 22, 91, 71] or list_nums== [16, 11, 22, 91, 71]
                    else:
                        assert list_nums==[11, 16, 132, 91, 71] or list_nums== [16, 11, 132, 91, 71]
            else:
                # Just use whatever has already been extracted
                flipped = self.supervision_extract_params(ind)["CUESTIM_FLIP"]
            
            # Save a new column
            list_flipped.append(flipped)

        if SANITY:
            print("PAssed sanity check!!")

        print("Appendded column: CUE_csflipped")
        self.Dat["CUE_csflipped"] = list_flipped

    ################# Supervision params
    # ie params that online guide supervision
    def supervision_extract_params(self, ind):
        """ Extract dict of supervision params for this tryial, uses the BASE params data.
        """
        from .dataset_preprocess.supervision import extract_supervision_params

        if not hasattr(self, 'SupervisionParamsDict'):
            self.SupervisionParamsDict = {}

        trialtuple = self.trial_tuple(ind, concise=True)

        # if already extracted...
        if trialtuple in self.SupervisionParamsDict.keys():
            return self.SupervisionParamsDict[trialtuple]
        
        # else extract it, save it, and return it.
        self.SupervisionParamsDict[trialtuple] = extract_supervision_params(self, ind)
        return self.SupervisionParamsDict[trialtuple]

    def supervision_extract_params_as_columns(self, list_keys):
        """ Extract params as columns in self.Dat
        PARAMS:
        - list_keys, list of string keys into superviison params, each will become  anew col
        """

        # Initialize
        outdict = {key:[] for key in list_keys}

        # Iterate over all trials, and extract
        for ind in range(len(self.Dat)):
            prms = self.supervision_extract_params(ind)
            for key in list_keys:
                outdict[key].append(prms[key])

        # Add as columns
        for key in list_keys:
            self.Dat[f"superv_{key}"] = outdict[key]
            print(f"Appended self.Dat[superv_{key}]")


    def _supervision_check_is_online_instruction(self, ind, color_is_considered_instruction=False):
        """ Returns True if this trial used any form of online supervision
        PARAMS;
        - color_is_considered_instruction, bool(False), whether to consider color cues as supervision.
        By default is no, because it is very weak supervision that must be learned, which constrast with 
        strong supervision (sequence)
        """

        prms = self.supervision_extract_params(ind)
        
        sup_seq = prms["SEQUENCE_SUP"]!="off" 
        sup_col = prms["COLOR_ON"]
        sup_guidedyn = prms["GUIDEDYN_ON"]

        if not color_is_considered_instruction:
            # Then ignore color
            sup_col = False

        if sup_seq or sup_col or sup_guidedyn:
            # Then at least one kind of supervision
            return True
        else:
            return False

    def supervision_check_is_instruction_using_color(self, ind, 
        list_methods_instructive = ("rank",)):
        """ Check whethre this trial was using color as instruction
        (e.g., color encoding rank of each stroke)
        PARAMS:
        - list_methods_instructive, if COLOR_METHOD is in this list, then this trial
        is True
        """       

        # 1) Get supervision params 
        prms = self.supervision_extract_params(ind)

        # 2) Check if is color instructions
        if prms["COLOR_ON"] and prms['COLOR_METHOD'] in list_methods_instructive:
            return True
        else:
            return False

    def supervision_check_is_instruction_using_color_assign(self, assign_to_column="INSTRUCTION_COLOR"):
        """ indicate for each trial whether it is using color instruction
        methods that would be considered instructive (and therefore warrants being called a different rule/epocj).
        This does NOT include color cues for rules.
        assign_to_column, if str, then updates D.Dat[assign_to_column], otherwise does
        not modify D.Dat
        """
        # 1) indicate for each trial whether it is using color instruction
        # - methods that would be considered instructive (and therefore warrants being called a different rule/epocj)
        list_issup = []
        for ind in range(len(self.Dat)):
            list_issup.append(self.supervision_check_is_instruction_using_color(ind))            
        self.Dat[assign_to_column] = list_issup

    def supervision_epochs_extract_epochkind(self):
        """ adds a column to D.Dat, which is epochkind, a category of epoch, such as
        "direction" for any direction rule. These are currently hand coded.
        """
        from pythonlib.dataset.modeling.discrete import MAP_EPOCH_EPOCHKIND
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        try:
            def F(x):
                return MAP_EPOCH_EPOCHKIND[x["epoch"]]
            self.Dat = applyFunctionToAllRows(self.Dat, F, "epochkind")
        except Exception as err:
            print(self.Dat["epoch"].value_counts())
            print("Probably simply need to ass this (all) epochs to MAP_EPOCHKIND_EPOCH in discrete.py")

        print("Updated self.Dat with new column: epochkind")

        
    def supervision_epochs_extract_orig(self):
        """ Extracts original name (e.g, AnBmTR|0 --> AnBmTR) even if you have already appended color
        superv info. This is a hacky solution to the original problem of not saving these names
        RETURNS:
        - places in self.Dat["epoch_orig"]
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            if isinstance(x["epoch"], str):
                ind = x["epoch"].find("|")
                if ind>=0:
                    # then has | --> remove it.
                    return x["epoch"][:ind]
                else:
                    return x["epoch"]
            else:
                return x["epoch"]
        # print(self.Dat["epoch_orig"])
        self.Dat = applyFunctionToAllRows(self.Dat, F, "epoch_orig")
        print("Extracted into self.Dat[epoch_orig]")

    def supervision_epochs_merge_these(self, list_epochs, new_epoch_name,
            key="epoch", assert_list_epochs_exist=True):
        """ Converts epochs of names in list_epochs into the epoch newname
        PARAMS:
        - assert_list_epochs_exist, check that each epoch actuall exists. useful sanity
        check.
        RETURNS:
        - modifies "epoch" in self.Dat.
        """

        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if assert_list_epochs_exist:
            for epoch in list_epochs:
                if sum(self.Dat[key]==epoch)==0:
                    print("---")
                    print("The epochs that exist:")
                    print(self.Dat["epoch"].unique())
                    print("The epochs you requested:")
                    print(list_epochs)
                    assert False, "you made mistake in list_epochs?"

        assert isinstance(list_epochs, list)
        # assert isinstance(new_epoch_name, str)
        
        print(f"Mergin these {key}'s .. ")
        print(list_epochs)
        print(f"Into this new {key}:", new_epoch_name)

        def F(x):
            if x[key] in list_epochs:
                return new_epoch_name  
            else:
                return x[key]

        self.Dat = applyFunctionToAllRows(self.Dat, F, key)

    def supervision_epochs_remove_baseline_trials(self):
        """ Modifies self.Dat to exlcude rows whos epochs that are basleine, i.e.
        with "base" or "baseline" in epioch
        """
        indskeep = self.Dat[~(self.Dat["epoch"].isin(["base", "baseline"]))].index.tolist()
        self.subsetDataframe(indskeep)
    
    def supervision_reassign_epoch_byvars(self, vars, new_col_name = "epoch"):
        """ Update epoch, using vars as grouping vars with epoch
        PARAMS:
        - vars, list of str, into self.Dat
        """
        print(" ------------- ")
        print("Old epoch values:")
        print(self.Dat["epoch"].value_counts())

        print(" ------------- ")
        self.grouping_append_col(vars, new_col_name, use_strings=True, strings_compact=True)

        print(" ------------- ")
        print("New epoch values:")
        print(self.Dat["epoch"].value_counts())

    def supervision_reassign_epoch_rule_by_color_instruction(self):
        """
        Replace epoch in self.DAt with <epoch>|1 if is color instruction
        :return:
        """
        # 1) conjunction of color and epoch
        self._supervision_reassign_epoch_rule_by_color_instruction("epoch", "epoch_color", False)

        # 2) which epochs have multipel color values? 
        list_epochs_to_update = []
        list_epochs = self.Dat["epoch"].unique().tolist()
        for ep in list_epochs:
            n = len(self.Dat[self.Dat["epoch"]==ep]["epoch_color"].unique())
            if n>1:
                # Then multiple color instruction values. use epoch_color
                list_epochs_to_update.append(ep)
        
        print("For these epochs, replacing epoch with epoch_color:")
        print(list_epochs_to_update)

        # 3) replace epoch with epoch_color, if necessary
        def F(x):
            if x["epoch"] in list_epochs_to_update:
                return x["epoch_color"]
            else:
                return x["epoch"]
        self.Dat = applyFunctionToAllRows(self.Dat, F, "epoch")

    def supervision_reassign_epoch_rule_by_sequence_mask_supervision(self):
        """
        Replace self.Dat["epoch"] by appending onto each rows epoch "|S" if
        this trial is a sequence mask supervision that is using solid_sequence_mask
        (i.e,, this i a test on purpose, not a training seuqence mask).
        :return: modifies self.Dat["epoch"] if is supervision.
        """

        # Collect epocjh for each row
        list_epoch = []
        list_is_seqsup = []
        for i, row in self.Dat.iterrows():

            # Criteria for calling this sequence mask
            a = row["INSTRUCTION_COLOR"]==False
            b = row["superv_COLOR_METHOD"]=="solid_sequence_mask"
            c = row["supervision_online"]==True
            d = row["epoch"][-2:]!="|S" # dont add again, if already exists

            if a & b & c & d:
                # is superivison
                list_epoch.append(f"{row['epoch']}|S")
                list_is_seqsup.append(True)
            else:
                # Not superv, use default epoch name
                list_epoch.append(row['epoch'])
                list_is_seqsup.append(False)
        self.Dat["epoch"] = list_epoch
        self.Dat["superv_is_seq_sup"] = list_is_seqsup
        print("Modified self.Dat[epoch]")

    def _supervision_reassign_epoch_rule_by_color_instruction(self, old_col_name = "epoch", 
            new_col_name="epoch_color", 
            overwrite=False):
        """ Replaces epoch column with conjunction of (i) current value in epoch and
        (ii) whether trial is color-based supervision. Idea is that if using color,
        then this is an entirely different kind of "rule"
        RETURNS:
        - modifies self.Dat["epoch"] (moves old version to self.Dat["epoch_old"]). 
        NOTE: fails to run if detects that epoch_old exists.
        NOTE: also adds a column called INSTRUCTION_COLOR
        """

        if not overwrite:
            if new_col_name in self.Dat.columns:
                if f"{new_col_name}_old" in self.Dat.columns:
                    assert False, "avoid running this multpel times."
                else:
                    self.Dat[f"{new_col_name}_old"] = self.Dat[f"{new_col_name}"]

        # 1) indicate for each trial whether it is using color instruction
        # - methods that would be considered instructive (and therefore warrants being called a different rule/epocj)
        if "INSTRUCTION_COLOR" not in self.Dat.columns:
            self.supervision_check_is_instruction_using_color_assign()
        assert "INSTRUCTION_COLOR" in self.Dat.columns

        # list_issup = []
        # for ind in range(len(self.Dat)):
        #     list_issup.append(self.supervision_check_is_instruction_using_color(ind))            
        # self.Dat["INSTRUCTION_COLOR"] = list_issup

        # 2) get conjuction of epoch and instruction
        grouping_vars = [old_col_name, "INSTRUCTION_COLOR"]
        self.grouping_append_col(grouping_vars, new_col_name, use_strings=True, strings_compact=True)

        print("Reassigned rules taking conjucntion of old rules x color instruction")
        print("New epochs")
        print(self.Dat[new_col_name].value_counts())

    def supervision_semantic_string_append_RAW(self, colname):
        """ Append a new column with string indicating superivison for this trial,
        on RAW supervision data.
        """
        tmp = []
        for ind in range(len(self.Dat)):
            tmp.append(self.supervision_semantic_string_extract_RAW(ind))
        self.Dat[colname] = tmp
        print("Append column to self.Dat: ", colname)

    def supervision_semantic_string_extract_RAW(self, ind):
        """ Human-readable string indicating features for supervision for this trial, based
        on RAW supervision data.
        Appends features to string if they are not "off".
        RETURNS:
        - string, |-separated. if no superivison at all, then returns NONE
        """ 

        prms = self.supervision_extract_params(ind)
        supervstr = ""
        ##### ONLINE SUPERVISION
        if prms["SEQUENCE_SUP"]=="mask":
            # Supervision stroke by stroke.
            # even if SEQUENCE_ALPHA==0, this is still visible. 
            supervstr+="|seqsup"
        elif prms["SEQUENCE_SUP"]=="off":
            # no seqsupervision
            pass
        elif prms["SEQUENCE_SUP"]=="char_strokes":
            supervstr+="|seqsup"
        else:
            print(prms)
            assert False

        ##### COLOR (CUES AND INSTRUCTION)
        if prms["COLOR_ON"]==True or prms["COLOR_ON"]==1:
            if prms["COLOR_METHOD"]=="solid":
                # Color cue (fixation cue, strokes, etc)
                supervstr+="|colcue"
            elif prms["COLOR_METHOD"]=="rank":
                supervstr+="|colrank"
            elif prms["COLOR_METHOD"]=="randomize_each_stroke":
                # This can be in charseq, when colors are assigned
                # randomly, e.g, Pancho 230125
                supervstr+="|colrnd"
            elif prms["COLOR_METHOD"]=="solid_sequence_mask":
                # This then is like solid, but not using rule, instead signalling
                # that this is trial using sequence_mask.
                supervstr+="|colsqmsk"
            else:
                print(prms)
                print(self.Dat.iloc[ind]["epoch"])
                assert False                

            if prms["COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR"]=="1111":
                # Strokes are colored during guide and draw
                supervstr+="|strkcolGD"
            elif prms["COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR"]=="1101":
                # Strokes are colored during guide but not during draw
                supervstr+="|strkcolG"
            elif prms["COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR"]=="1001":
                # Only cue is colored. strokes never colored
                supervstr+="|strkcolOFF"
            else:
                print(prms)
                assert False
        elif prms["COLOR_ON"]==False:
            pass
        else:
            assert False

        if len(supervstr)==0:
            # Then no supervision at all
            supervstr = "NONE"

        return supervstr

    def supervision_summarize_into_tuple(self, method="verbose", print_summary=False,
            new_col_name = None):
        """ Summarize those supervision params that potnetially provide online instruction
        , these can define distinct stages of blocks
        PARAMS:
        - method, str in {'concise', 'verbose'}, how detailed to make fields in tuple. concise
        groups all alphas together (for sequence training). 
        RETURNS:
        - new col in self.Dat, "supervision_stage_new"
        """

        # - grouping keys which can potentially influence online behavior (instructive).
        # grouping_keys = ["SEQUENCE_SUP", "SEQUENCE_ALPHA", "COLOR_ON", "COLOR_METHOD", "SOUNDS_STROKES_DONE", "GUIDEDYN_ON"]

        # IND_TO_CHECK = 0 # skip if the first trial is not new tasks. assumes that
        # # your dataset doesnt mix trials qith new and old tasks...
        # if not D.taskclass_is_new_version(IND_TO_CHECK):
        #     # Old task version, ignore this.
        #     return None

        assert new_col_name is not None, "to avoid errors, you must enter it"
        # supervision_stage_new
        # supervision_stage_concise

        if method in ["concise"]:
            # grouping_keys = ["SEQUENCE_SUP", "COLOR_ON", "COLOR_METHOD", "GUIDEDYN_ON"]    
            grouping_keys = ["SEQUENCE_SUP", "COLOR_ON", "COLOR_METHOD", "COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR", "GUIDEDYN_ON"]    
        elif method in ["concise_cuestim"]:
            # grouping_keys = ["SEQUENCE_SUP", "COLOR_ON", "COLOR_METHOD", "GUIDEDYN_ON"]    
            grouping_keys = ["SEQUENCE_SUP", "COLOR_ON", "COLOR_METHOD", "COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR", "GUIDEDYN_ON", "CUESTIM_FLIP"]    
        elif method=="verbose":
            # grouping_keys = ["SEQUENCE_SUP", "SEQUENCE_ALPHA", "COLOR_ON", "COLOR_METHOD", "GUIDEDYN_ON", "VISUALFB_METH"]
            grouping_keys = ["SEQUENCE_SUP", "SEQUENCE_ALPHA", "COLOR_ON", "COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR", "COLOR_METHOD", "GUIDEDYN_ON", "VISUALFB_METH"]
        else:
            assert False
        grouping_keys_prefix = [f"superv_{key}" for key in grouping_keys]

        # 1) Extract each param to a column
        self.supervision_extract_params_as_columns(grouping_keys)

        # 2) get their conjunction.
        self.grouping_append_col(grouping_keys_prefix, new_col_name, use_strings=True, 
            strings_compact=True)

        # 3) prin summary
        if print_summary:
            print("*** SUMMARY of new column, after making supervision tuple (column: supervision_stage_new):")
            print(self.Dat[new_col_name].value_counts())
            from pythonlib.tools.pandastools import grouping_get_inner_items
            tmp = grouping_get_inner_items(self.Dat, new_col_name, "block")
            print(f"**** SUMMARY: Levels of {new_col_name} : blocks")
            for k, v in tmp.items():
                print("--", k, ":", v)

    def supervision_summarize_whether_is_instruction(self, color_is_considered_instruction=False):
        """ [FINAL WORD} on whether trial is considered having hand-holding
        online instruction, across equence and color usp methods.
        (and any others).
        For each trial, determine True/False whether this had online
        supervision, based on key parameters like color, sequence presentation, etc
        --> Appends new column self.Dat["supervision_online"]
        """
        list_issup = []
        for ind in range(len(self.Dat)):
            list_issup.append(self._supervision_check_is_online_instruction(ind,
                color_is_considered_instruction=color_is_considered_instruction))            
        self.Dat["supervision_online"] = list_issup
        print("ADded new column: supervision_online")

        # NOTE: superv_semantic coud be used insetead... Here, just making a note of that.
        if False:
            self.supervision_semantic_string_append_RAW("superv_semantic")
            self.grouping_print_n_samples(["superv_semantic", "supervision_online"])

    ############## EPOCHSET stuff
    def _epochset_extract_common_stroke_index(self, list_stroke_index, 
        return_as = "list"):
        """ [NOT USED MUCH...]Group trials so that all trials which have, across epochs,
        (same char, same stroke at the given index) into one group, and
        all others into leftover group
        PARAMS;
        - list_stroke_index, list of ints, each a stroke index of beh order. tells which 
        indices must be same task stroke, to call it sepochset.
        e..g, list_stroke_index = [0,3] means that this char must have the same taskstroke
        (i.e, shape/loc) for beh stroke 0 and 3, across epochs (if doing it correctly, based
        ont asksequencer).
        e..g, list_stroke_index = [0] finds chars with same first stroke across epochs, but
        ignore whether second stroke and up is same.
        - return_as, 
        --- "list", then is list of len(self.Dat), holding the epochste
        --- "dict", then is dict, {epochset:inds into self.Dat}
        RETURNS:
        NOTE: keeps all epochsets that have at least 1 epoch, and the ones with 1 epoch merges into a
        single epochset called tuple(["LEFTOVER"])
        """
        assert isinstance(list_stroke_index, list)
        Dcopy = self.copy()

        # Assign column "char_seq" indicated the correct seuqence for each peoch.
        Dcopy.sequence_char_taskclass_assign_char_seq(sequence_keep_these_indices=list_stroke_index)

        # keep only chars with same first stroke across epochs
        Dcopy.epochset_extract_common_epoch_sets(merge_sets_with_only_single_epoch=True)

        if return_as=="list":
            return Dcopy.Dat["epochset"].tolist()
        elif return_as=="dict":
            return Dcopy.grouping_get_inner_items("epochset")
        else:
            print(return_as)
            assert False
        # # trialcodes_keep = self.Dat[self.Dat["epochset"]==epochs_all]["trialcode"].tolist()
        
        # return trialcodes_keep

    def epochset_apply_sequence_wrapper(self, versions_ordered=None):
        """ Iteratively label trials with epochsets, starting with the least special
        [GOOD]. 
        PARAMS:
        - versions_ordered, list of str, trial-level variables for use in defining epochset, applied
        in order, so the first one is overwritten by epochsets (not leftovers) detected by subsequent ones.
        Make the last one the most important (unique, dont' want to miss out).
        """
        if versions_ordered is None:
            versions_ordered = ["char", "same_beh_first_stroke", "same_beh"]

        for i, version in enumerate(versions_ordered):
            if i==0:
                # First one, least special, apply to directly to D.
                # And do not merge any sets, e.g, to keep single prims separate from others.
                self.epochset_extract_wrapper(version, mutate=True, merge_sets_with_only_single_epoch=False)
            else:
                # The rest, only update trials that are not LEFTOVER
                # (Update subset of trialcodes that can be assigned a more interesting label for epochset
                # map_epochset_trialcode = D.epochset_extract_wrapper("same_beh_first_stroke", exclude_leftover=True)
                # D.assign_insert_value_using_trialcode_key("epochset", map_epochset_trialcode)
                map_epochset_trialcode = self.epochset_extract_wrapper(version, exclude_leftover=True)
                self.assign_insert_value_using_trialcode_key("epochset", map_epochset_trialcode)

    def epochset_extract_matching_motor_wrapper(self, HACK=True):
        """
        Extract epochsets (Eacha  diff column) in each case only haveing two possible
        sets, either yes or no, indicating matched motor beh across epochs within the set.
        Useful, eg., if want to find AnBmCk rule + AnBmCk with colrank + AnBmCk with seqsup, each
        case having matched motor beh across those 3 epochs, with motor beh defined by the actual
        shape and loc sequence in bahvior.
        :return: Adds two columns to self.Dat:
        -- "epochset_shape" and "epochset_loc"
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index

        # First, add column defining motor beh for each trial
        self.seqcontext_extract_locations_in_beh_order_append_column(colname="behseq_locs")
        self.seqcontext_extract_shapes_in_beh_order_append_column(colname="behseq_shapes")
        self.Dat = append_col_with_grp_index(self.Dat, ["behseq_shapes", "behseq_locs"], "behseq_shapes_locs")

        def _prune_to_single_epoch_orig(list_epoch_orig):
            """ list_epoch_orig --> list of a single peoch_orig,
            by taking the one with most datapts.
            """
            # if multiple, then take the one with more trials.
            idx = int(np.argmax([sum(self.Dat["epoch_orig"]==epoch_orig) for epoch_orig in list_epoch_orig]))
            list_epoch_orig = [list_epoch_orig[idx]]
            return list_epoch_orig

        #### Get epochsets for speicifc comparisongs
        # - Shapes
        list_epoch_orig = self.grammarparses_rules_involving_shapes(True)
        if len(list_epoch_orig)>1:
            if HACK:
                list_epoch_orig = _prune_to_single_epoch_orig(list_epoch_orig)
            else:
                print(list_epoch_orig)
                assert False, "sholud probably scrap this entire meothd.."
        self.epochset_extract_common_epoch_sets("behseq_shapes_locs", "epoch_orig",
                                                epochset_col_name="epochset_shape", epochset_desired=list_epoch_orig)
        print("Added column: self.Dat[epochset_shape]")

        # Direction
        list_epoch_orig = self.grammarparses_rules_involving_direction(True)
        if len(list_epoch_orig)>1:
            if HACK:
                list_epoch_orig = _prune_to_single_epoch_orig(list_epoch_orig)
            else:
                print(list_epoch_orig)
                assert False, "sholud probably scrap this entire meothd.."
        self.epochset_extract_common_epoch_sets("behseq_shapes_locs", "epoch_orig",
                                                epochset_col_name="epochset_dir", epochset_desired=list_epoch_orig)
        print("Added column: self.Dat[epochset_shape]")

    def epochset_extract_wrapper(self, version, params=None, epoch_label="epoch",
        exclude_leftover=False, mutate=False, merge_sets_with_only_single_epoch=True,
        only_keep_epochsets_containing_all_epochs=False):
        """ 
        [GOOD] Various methods to compute epochset information. 
        PARAMS:
        - version, str, how to classify sets
        - params, None, or optional params specific to version
        - exclude_leftover, then any trials in "LEFTOVER" epochsets will not include in
        the output dict.
        - mutate, bool, if True, then modifies self.Dat to assign trials to epochsets. 
        RETURNS:
        - dict, {epochset:list of trialcodes}
        """

        D = self.copy()

        if version=="char":
            # Epochset using characters (ignores what is correct beh sequence).
            APPEND_PREFIX = "char"
            D.epochset_extract_common_epoch_sets(
                trial_label="character",
                merge_sets_with_only_single_epoch=merge_sets_with_only_single_epoch,
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",),
                append_prefix=APPEND_PREFIX,
                only_keep_epochsets_containing_all_epochs=only_keep_epochsets_containing_all_epochs
            )
        elif version=="same_beh":
            # Epochset defined by characters with same beh across epochs.
            D.sequence_char_taskclass_assign_char_seq()
            APPEND_PREFIX = "same"
            D.epochset_extract_common_epoch_sets(
                trial_label="char_seq",
                merge_sets_with_only_single_epoch=merge_sets_with_only_single_epoch,
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",),
                append_prefix=APPEND_PREFIX,
                only_keep_epochsets_containing_all_epochs=only_keep_epochsets_containing_all_epochs
            )

        elif version=="same_beh_first_stroke":
            # Epochset defined by characters with same first stroke across epochs.
            # regardless of second stroke and onwards.
            D.sequence_char_taskclass_assign_char_seq(sequence_keep_these_indices=[0])
            APPEND_PREFIX = "same_stroke_0"
            D.epochset_extract_common_epoch_sets(
                trial_label="char_seq",
                merge_sets_with_only_single_epoch=merge_sets_with_only_single_epoch,
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",),
                append_prefix=APPEND_PREFIX,
                only_keep_epochsets_containing_all_epochs=only_keep_epochsets_containing_all_epochs                
            )
        elif version=="same_beh_first_two_stroke":
            # Epochset defined by characters with same first stroke across epochs.
            # regardless of second stroke and onwards.
            D.sequence_char_taskclass_assign_char_seq(sequence_keep_these_indices=[0, 1])
            APPEND_PREFIX = "same_stroke_0"
            D.epochset_extract_common_epoch_sets(
                trial_label="char_seq",
                merge_sets_with_only_single_epoch=merge_sets_with_only_single_epoch,
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",),
                append_prefix=APPEND_PREFIX,
                only_keep_epochsets_containing_all_epochs=only_keep_epochsets_containing_all_epochs                
            )
        elif version=="char_seq_same_first_n_strokes":
            # If a character occurs across all epochs, and across them it has the same 
            # first stroke (tasksequencer), then place all of these trials into the extracted
            # epochset. The rest of trials place in to LEFTOVER

            assert False, "old code, change it."
            # Get trials that have same first strokes
            trialcodes_first = self._epochset_extract_common_stroke_index([0])

            # Get trials that have same second stroke
            trialcodes_second = self._epochset_extract_common_stroke_index([1])

            trialcodes_both = [t for t in trialcodes_first if t in trialcodes_second]
            trialcodes_first_notsecond = [t for t in trialcodes_first if t not in trialcodes_second]

            chars_both = sorted(D.Dat[D.Dat["trialcode"].isin(trialcodes_both)]["character"].unique().tolist())
            chars_first_notsecond = sorted(D.Dat[D.Dat["trialcode"].isin(trialcodes_first_notsecond)]["character"].unique().tolist())

            print("BOTH")
            for x in chars_both:
                print(x)

            print("FIRST")
            for x in chars_first_notsecond:
                print(x)

            # for each trial assign it a classification
            assert len([t for t in trialcodes_both if t in trialcodes_first_notsecond])==0 # - make sure exlcusive

            names = []
            for ind in range(len(D.Dat)):
                tc = D.Dat.iloc[ind]["trialcode"]
                
                if tc in trialcodes_both:
                    names.append("both")
                elif tc in trialcodes_first_notsecond:
                    names.append("first_not_second")
                else:
                    names.append("neither")
                    
            D.Dat["strokes12_same"] = names

            tmp = D.grouping_get_inner_items("character", "strokes12_same")
            for val in tmp.values():
                assert len(val)==1, "a char can only be one..."         

        if mutate:
            self.Dat["epochset"] = D.Dat["epochset"]

        # Return mapping to trialcode.
        map_epochset_trialcode = D.grouping_get_inner_items("epochset", "trialcode")
        if exclude_leftover:
            map_epochset_trialcode = {k:v for k, v in map_epochset_trialcode.items() if not k==(APPEND_PREFIX, "LEFTOVER")}

        return map_epochset_trialcode       

    def epochset_extract_common_epoch_sets(self, trial_label = "char_seq", epoch_label="epoch",
        n_min_each_conj_outer_inner=1, n_max_epochs=None, epochset_col_name="epochset",
        PRINT = False,
        merge_sets_with_only_single_epoch=False,
        merge_sets_with_only_single_epoch_name = tuple(["LEFTOVER"]),
        only_keep_epochsets_containing_all_epochs = False,
        append_prefix=None,
        epochset_desired = None, epochset_desired_name_for_leftover = tuple(["LEFTOVER"])):
        """
        Find groups of trials that share some feature (e.g., same character) and also occur across multiple
        epochs. Group the trials that occur across the exact same epochs. e..g, this finds tasks that 
        have same beh across epochs
        PARAMS:
        - trial_label, int, how to label each trial, i.e., what variable to use for grouping trials, e..g
        if "character", then groups trials by character,
        - n_max_epochs, int or None. if int, then for each <trial_label> this is the max num epochs it can have.
        if have more, then keeps the top n (n_max_epochs) based on n trials.
        - merge_sets_with_only_single_epoch, bool, if True, then any epochset that has only one level of epoch, 
        collect those and combine into a single epochset. this way you don't throw them out. new epochset
        name is "LEFTOVER"
        -= only_keep_epochsets_containing_all_epochs, if True, then keeps only these epochsets, and all
        the rest calls them <merge_sets_with_only_single_epoch_name>
        - append_prefix, either None (ignores) or str, in which case appends this string to 
        start of each epochset. e.g., (epoch1, epoch2) --> (append_prefix, epoch1, epoch2).
        - epochset_desired, either None (ignore) or list of str (e.g, ['llCV3|0', 'llCV3|0|S']) which is an
        epochset that you are looking for. Then there will only be 2 epochsets: ['llCV3|0', 'llCV3|0|S'] (all cases that
        whose epochset inlcudes this epochset, e.g, ('UL', 'llCV3|0', 'llCV3|0|S') would be included), and
        <epochset_desired_name_for_leftover> for all other trials. IN other words, get epochsets with at laest these
        epochs, and possibly more.
        """

        # Some defaults -- not useful, but allows running this using None inputs, which I use in dataset preprocess
        # for neural data.
        if trial_label is None:
            trial_label = "character"

        assert isinstance(merge_sets_with_only_single_epoch_name, tuple)

        # - For each char_seq, get its list of epochs that it is present in
        groupdict = self.grouping_get_inner_items(trial_label, epoch_label, 
            n_min_each_conj_outer_inner=n_min_each_conj_outer_inner, take_top_n_inner=n_max_epochs)
        # - make the epoch set hashable, and sorted
        groupdict = {charseq:tuple(sorted(epoch_set)) for charseq, epoch_set in groupdict.items()}

        ####### Looking for speicifc epochset?
        if epochset_desired is not None:
            epochset_desired = tuple(sorted(epochset_desired)) # imprtant
            # list_includes_desired_epochs = []
            groupdict_new = {}
            for charseq, epochset in groupdict.items():
                epochset_contains_desired_epochs = all([e in epochset for e in epochset_desired])
                # list_includes_desired_epochs.append(epochset_contains_desired_epochs)

                # print(epochset, " ---- ", epochset_contains_desired_epochs)

                if epochset_contains_desired_epochs:
                    groupdict_new[charseq] = epochset_desired
                else:
                    groupdict_new[charseq] = epochset_desired_name_for_leftover
            groupdict = groupdict_new

        if PRINT:
            list_epochsets_unique = sorted(set([x for x in list(groupdict.values())]))
            print("Unique classes of epochs spanned by individual tasks:")
            print(list_epochsets_unique)

        # - For each trial, map to one of these sets )
        def F(x):
            epoch_set = groupdict[x[trial_label]]
            return epoch_set
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        self.Dat = applyFunctionToAllRows(self.Dat, F, epochset_col_name)
        print(f"Defined new column: {epochset_col_name}")
        
        if PRINT:
            print("... value_counts:")
            print(self.Dat[epochset_col_name].value_counts())

        if merge_sets_with_only_single_epoch:
            print("... merge_sets_with_only_single_epoch... ")
            # Collect all the epochsets that have only one epoch, then merge them.
            # Beucase they would be useless (thrown out) if only one epoch.
            var = epoch_label
            groupdict = self.grouping_get_inner_items(epochset_col_name, var)

            list_epochsets_with_only_one_epoch = []
            for epochset, epochs_within_this_epochset in groupdict.items():
                if len(epochs_within_this_epochset)==1:
                    list_epochsets_with_only_one_epoch.append(epochset)
                    print(epochset, "only has one epoch!: ", epochs_within_this_epochset)

            # Merge these all into one new epochset
            self.supervision_epochs_merge_these(list_epochsets_with_only_one_epoch, 
                merge_sets_with_only_single_epoch_name, epochset_col_name)

        if only_keep_epochsets_containing_all_epochs:
            epochs_all = tuple(sorted(self.Dat["epoch"].unique().tolist())) # tuple of all epochs
            # for es in self.Dat["epochset"].tolist():
            #     print(es, sorted(es), sorted(es)==epochs_all)
            # assert False
            list_epochset = [es if tuple(sorted(es))==epochs_all else merge_sets_with_only_single_epoch_name for es in self.Dat["epochset"].tolist()]
            # print(epochs_all)
            # print(self.Dat["epochset"].tolist())
            # print(list_epochset)
            # assert False
            self.Dat["epochset"] = list_epochset

        if append_prefix is not None:
            list_es = self.Dat["epochset"].tolist()
            list_es_new = []
            for es in list_es:
                assert isinstance(es, tuple)
                list_es_new.append(tuple([append_prefix] + list(es)))
            self.Dat["epochset"] = list_es_new

        if PRINT:
            print("-- Final epochsets:")
            print(self.Dat[epochset_col_name].value_counts())

            print(f"-- FINAL RESULTS ({epochset_col_name}, {trial_label}, {epoch_label}):")
            self.grouping_print_n_samples([epochset_col_name, trial_label, epoch_label])

            print(f"-- FINAL RESULTS ({epochset_col_name}, {epoch_label}, {trial_label}):")
            self.grouping_print_n_samples([epochset_col_name, epoch_label, trial_label])

    ############### PROBES stuff
    def probes_extract_blocks_with_probe_tasks(self):
        """ Return list of blocks which include data, and which
        include probe tasks (looks at data, not at blockparams)
        RETURNS:
        - list of sorted ints (blocks)
        """
        return sorted(self.Dat[self.Dat["probe"]==1]["block"].unique().tolist())

    def grammarparses_rules_epochs_superv_summarize_wrapper(self, PRINT=False, include_epochset=False):
        """
        GOOD - wrapper to extract for each row the key features of the epoch, which are
        in generic terms that apply across epxeirments, refelcting, e.g, whether is
        color_rank supevision, sequence_sup, random(preset) sequence, AnBmnCk, etc.
        Also useufl printiing of summary of conjuicntions of these features
        :return:
        """

        #TODO add whether is direction rule.

        # (1) Extract columns
        if "epoch_orig_rand_seq" not in self.Dat.columns:
            self.grammarparses_rules_random_sequence()

        if "epoch_is_AnBmCk" not in self.Dat.columns:
            self.grammarparses_rules_shape_AnBmCk()

        if "epoch_is_DIR" not in self.Dat.columns:
            self.grammarparses_rules_direction()

        if "superv_is_seq_sup" not in self.Dat.columns:
            self.supervision_reassign_epoch_rule_by_sequence_mask_supervision()

        # Whether color is used to indicate seuqence (does NOT include color cue for rules).
        if "INSTRUCTION_COLOR" not in self.Dat.columns:
            self.supervision_check_is_instruction_using_color_assign()

        # (2) PRint summary
        if PRINT:
            from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
            # Get trials under each epoch
            vars = ["epoch_orig", "epoch", "epoch_rand", "INSTRUCTION_COLOR", "superv_is_seq_sup", "epoch_orig_rand_seq", "epoch_is_AnBmCk", "epoch_is_DIR"]
            if include_epochset:
                vars = ["epochset"] + vars
            grpdict = self.grouping_print_conjunctions_summary_good(vars, PRINT=True)

    def grammarparses_rules_direction(self):
        """
        For each row, appends column "epoch_is_DIR", bool, indicating
        whether is pure direction rule. NOTEL This onlyu looks at epoch_orig, and so
        it would be called True even if it is color_rank or sequence_sup (probe tasks
        designed with same motor).
        """

        # Get list of epochs that use shape rules.
        rules = self.grammarparses_rules_involving_direction(return_as_epoch_orig=True)

        list_DIR = []
        for i, row in self.Dat.iterrows():
            if row["epoch_orig"] in rules:
                # Then this is a dir rule, without supervision by color or sequence mask
                list_DIR.append(True)
            else:
                list_DIR.append(False)

        self.Dat["epoch_is_DIR"] = list_DIR
        print("Appended column: self.Dat[epoch_is_DIR]")

    def grammarparses_rules_shape_AnBmCk(self):
        """
        For each row, appends column "epoch_is_AnBmCk", bool, indicating
        whether is pure shape rule. NOTEL This onlyu looks at epoch_orig, and so
        it would be called True even if it is color_rank or sequence_sup (probe tasks
        designed with same motor).
        """

        # Get list of epochs that use shape rules.
        shape_rules = self.grammarparses_rules_involving_shapes(return_as_epoch_orig=True)

        # Include only those epochs that are also not color instruction, or seqsup.
        if "superv_is_seq_sup" not in self.Dat.columns:
            self.supervision_reassign_epoch_rule_by_sequence_mask_supervision()

        list_AnBmCk = []
        for i, row in self.Dat.iterrows():
            a = row["superv_is_seq_sup"]
            b = row["INSTRUCTION_COLOR"]
            c = row["epoch_orig"] in shape_rules

            # if (c) and (not a) and (not b): # STOPPED also checking that it is not colrank or seqsup
            if (c):
                # Then this is a shape rule, without supervision by color or sequence mask
                list_AnBmCk.append(True)
            else:
                list_AnBmCk.append(False)

        self.Dat["epoch_is_AnBmCk"] = list_AnBmCk
        print("Appended column: self.Dat[epoch_is_AnBmCk]")

    ############### Sequence / GRAMMAR stuff, i.e., realted to sequence training
    def grammarparses_rules_random_sequence(self, PRINT=False):
        """
        For each rule (epoch_orig), determine if it is using random preset sequences.
        Has NOTHING to do with color, i.e., ignres the |1 suffix that indicates is color instruction.
        Useful for grouping all epochs which are actually color_rank with random seuqence.
        :return: map_epochorig_to_israndseq, dict, epoch_orig:bool (True means is random_)
        AND appends a new column "epoch_orig_rand_seq"
        AND appends "epoch_rand", which is iether "presetrand" (if is random instruction,
        or epoch otherwise
        """

        ruledict_for_each_rule = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"]

        map_epochorig_to_israndseq = {}
        for epoch_orig in self.Dat["epoch_orig"].unique().tolist():
            # a = ("rnd" in epoch_orig.lower()) or ("rand" in epoch_orig.lower())
            a = True
            b = ruledict_for_each_rule[epoch_orig]["categ"] == "preset" and ruledict_for_each_rule[epoch_orig]["subcat"] == "null"
            map_epochorig_to_israndseq[epoch_orig] = a & b

        # Add a column
        self.Dat["epoch_orig_rand_seq"] = [map_epochorig_to_israndseq[row["epoch_orig"]] for i, row in self.Dat.iterrows()]

        # Add a column
        list_epoch_rand = []
        for i, row in self.Dat.iterrows():
            if row["epoch_orig_rand_seq"]==True and row["INSTRUCTION_COLOR"]==True:
                epoch_rand = "presetrand"
            elif row["epoch_orig_rand_seq"]==True and row["INSTRUCTION_COLOR"]==False and row["superv_COLOR_METHOD"] == "solid_sequence_mask":
                # Random seuqence, but no color instruction. The only case this is possible (I think)
                # is if using seuqence  mask... in which case superv_COLOR_METHOD=="solid_sequence_mask".
                epoch_rand = "presetrand"
            elif row["epoch_orig_rand_seq"]==False and row["INSTRUCTION_COLOR"]==True:
                # using rule, but coloring (probe)
                epoch_rand = row["epoch"] # .e.g, AnBmCk2|1
            elif row["epoch_orig_rand_seq"]==False and row["INSTRUCTION_COLOR"]==False:
                # using rule, no color
                epoch_rand = row["epoch"] # .e.g, AnBmCk2|0
            else:
                assert False, "how is this possible? random but no cue?"
            list_epoch_rand.append(epoch_rand)
        self.Dat["epoch_rand"] = list_epoch_rand

        if PRINT:
            vars = ["epoch_orig_rand_seq", "INSTRUCTION_COLOR", "epoch_rand", "epoch_orig", "epoch"]
            grpdict = self.grouping_print_conjunctions_summary_good(vars)

        return map_epochorig_to_israndseq

    def grammarparses_rules_involving_direction(self, return_as_epoch_orig=False):
        """ Return list of rulestrings of the rules that exist in dataset which
        use knowledge of location/direction
        RETURNS:
            - list_rules_using_shapes, list of rulestrings. can be empty.
        """
        map_rulestr_ruledict = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"]
        list_rules = []
        for rulestr, ruledict in map_rulestr_ruledict.items():
            if ruledict["categ"]=="dir":
                list_rules.append(rulestr)

        if return_as_epoch_orig:
            # Convert from rulestring to epoch_orig
            map_rulestr_ruledict = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"]
            list_rules = [map_rulestr_ruledict[rs]["params"] for rs in list_rules]

        return list_rules

    def grammarparses_rules_involving_shapes(self, return_as_epoch_orig=False):
        """ Return list of rulestrings of the rules that exist in dataset which
        use knowledge of shapes. i.e. those that are categ "ss"
        RETURNS:
            - list_rules_using_shapes, list of rulestrings. can be empty.
        """
        map_rulestr_ruledict = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"]
        list_rules_using_shapes = []
        for rulestr, ruledict in map_rulestr_ruledict.items():
            if ruledict["categ"]=="ss":
                # then this is shape sequence
                list_rules_using_shapes.append(rulestr)

        if return_as_epoch_orig:
            # Convert from rulestring to epoch_orig
            map_rulestr_ruledict = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"]
            list_rules_using_shapes = [map_rulestr_ruledict[rs]["params"] for rs in list_rules_using_shapes]

        return list_rules_using_shapes

    def grammarparses_rules_extract_map_rule_to_rulekind(self):
        """
        Get mapping (dict) from epoch_orig --> (rulecat, rulesubcat), e.g,
        {'UL': ('dir', 'null'),
         'llCV3FstStk': ('preset', 'null'),
         'llCV3': ('ss', 'rankdir')}
        :return: dict (see above)
        """

        ruledict_for_each_rule = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"]
        # for epoch in self.Dat["epoch_orig"].unique():
        #     ruledict_for_each_rule[epoch]
        map_epochorig_to_rulekind = {epoch_orig:(ruledict["categ"],ruledict["subcat"]) for epoch_orig, ruledict in ruledict_for_each_rule.items()}
        return map_epochorig_to_rulekind

    def grammarparses_rules_extract_info(self):
        """ Return dict holding infor for all rules (epochs)
        epoch: Dict holding:
        - rule_dict
        - for each rule, rules consstent with ti
        """
        from pythonlib.dataset.modeling.discrete import rules_map_rule_to_ruledict_extract_auto, _rules_consistent_rulestrings_extract_auto
        # list_ruledict_all_related_rules = rules_map_rule_to_ruledict_extract_auto(self)
        # list_rule = 
        # dict_rules_consistent_with_each_rule = {}

        list_rules_exist = self.Dat["epoch_orig"].unique().tolist()
        list_rules_exist = [r for r in list_rules_exist if not r in ["base", "baseline"]]

        # For each existing rule, get ruledicts consistent with it
        dict_ruledicts_consistent_with_each_existing_rule = {}
        map_rulestr_ruledict = {}
        for rule in list_rules_exist:
            ld = _rules_consistent_rulestrings_extract_auto([rule], return_as_dict=True)[0]
            dict_ruledicts_consistent_with_each_existing_rule[rule] = ld
            # Collect all ruledicts.
            for ruledict in ld:
                rs = ruledict["rulestring"]
                if rs in map_rulestr_ruledict.keys():
                    assert ruledict==map_rulestr_ruledict[rs]
                else:
                    map_rulestr_ruledict[rs] = ruledict

        # ruledict_for_each_rule = {}
        # for list_ruledicts in dict_ruledicts_consistent_with_each_existing_rule.values():
        #     for ruledict in list_ruledicts:
        #         rule_short = ruledict["params"] # 'llV1R',
        #         rule = ruledict["rulestring"] # 'ss-chain-llV1R'}
        #         if rule_short in list_rules_exist:
        #             if rule not in ruledict_for_each_rule:
        #                 ruledict_for_each_rule[rule] = ruledict
        #             else:
        #                 assert ruledict_for_each_rule[rule] == ruledict

        out = {
            "list_rules_exist":list_rules_exist,
            "dict_ruledicts_consistent_with_each_existing_rule":dict_ruledicts_consistent_with_each_existing_rule,
            "ruledict_for_each_rule":rules_map_rule_to_ruledict_extract_auto(self),
            "map_rulestr_ruledict":map_rulestr_ruledict
        }

        try:
            out["list_rules_exist_as_rulestring"] = [out["ruledict_for_each_rule"][r]["rulestring"] for r in out["list_rules_exist"]]
        except Exception as err:
            for k, v in out.items():
                print(k, " -- ", v)
            print(" ________________ ruledict_for_each_rule")
            print(out["ruledict_for_each_rule"].keys())
            raise err
        return out

    def grammarparses_ruledict_rulestring_extract(self, ind):
        """ Return the ruledict and rulestring for this trial, based on
        its epoch
        """
        epoch = self.Dat.iloc[ind]["epoch_orig"]
        ruledict = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"][epoch]
        return ruledict["rulestring"], ruledict

    def grammarparses_rulestrings_exist_in_dataset(self):
        """ return list of rulestrings that exist in this dtaset, ignoring cases that at "baseline"
        """
        list_epoch = self.Dat["epoch_orig"].unique().tolist()
        list_rulestring = []
        for epoch in list_epoch:
            if epoch not in ["base", "baseline"]:
                ruledict = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"][epoch]
                rs = ruledict["rulestring"]
                list_rulestring.append(rs)
        return list_rulestring

    # def grammar_rules_extract_rul
    def grammarparses_print_plot_summarize(self, ind):
        """
        [Good] print and plot all things to summarize things about this trial.
        Wrapper to help me remmember how to plot summary of
        this truials beh and parsaes
        """
        print("=========================================")
        print("BEH (taskstroke inds): ", self.grammarparses_extract_beh_taskstroke_inds(ind))

        tokens = self.taskclass_tokens_extract_wrapper(ind, which_order="beh_using_task_data")
        if "chunk_rank" in tokens[0].keys():
            print("=========================================")
            print("chunk_rank:", [(t["chunk_rank"]) for t in tokens])
            print("chunk_within_rank:", [(t["chunk_within_rank"]) for t in tokens])
            print("chunk_within_rank_fromlast:", [(t["chunk_within_rank_fromlast"]) for t in tokens])
            print("chunk_n_in_chunk:", [(t["chunk_n_in_chunk"]) for t in tokens])
        
        print("=========================================")
        self.grammarparses_grammardict_return(ind, True)

    # def grammarparses_print_plot_summary(self, ind):
    #     """ Wrapper to help me remmember how to plot summary of
    #     this truials beh and parsaes
    #     """
    #     self.grammarparses_grammardict_return(ind, True)

    def grammarparses_grammardict_return(self, ind, doplot=False):
        """
        """
        tc = self.Dat.iloc[ind]["trialcode"]
        gd = self.GrammarDict[tc]
        if doplot:
            # only plot parses for the active rule on this trial
            rs = self.grammarparses_ruledict_rulestring_extract(ind)[0]
            gd.print_plot_summary(doplot=doplot, only_this_rulestring=rs)
        return gd

    def _grammarparses_parses_generate(self, ind, list_rulestring = None):
        """ Generate GrammarDat object and save, for this trial
        PARAMS:
        - ind, index in D.Dat
        - list_rulestring, list of str, each a rule that is applied to this task to genereate parses
        """
        if not hasattr(self, 'GrammarDict'):
            self.GrammarDict = {}

        tc = self.Dat.iloc[ind]["trialcode"]

        if tc not in self.GrammarDict.keys():
            # Generate a new GD
            from pythonlib.grammar.GrammarDat import GrammarDat

            input_data_dict = {
                "dataset":self,
                "ind_dataset":ind
            }
            GD = GrammarDat(input_data_dict, input_version = "dataset")
            self.GrammarDict[tc] = GD

        GD = self.GrammarDict[tc]
        GD.parses_generate_batch(list_rulestring)

        # Return the grammardict, holding all the parses
        return GD

    def _grammarparses_parses_extract(self, ind, list_rulestrings, fail_if_empty=True):
        """ Extract set of parses for each rule.
        PARAMS;
        - list_rulestrings, list of str <cat>-<subcat>-<rule>
        RETURNS:
        - dict[rule] = parses, each a list of possible orderings
        """
        GD = self._grammarparses_parses_generate(ind, list_rulestrings) 
        outdict = {}
        for rulestring in list_rulestrings:
            parses = GD.parses_extract_generated(rulestring)
            if fail_if_empty and len(parses)==0:
                print(rule)
                print(parses)
                assert False, "epty..."
            outdict[rulestring] = parses
        return outdict

    def grammarparses_parses_extract_trial(self, ind):
        """
        RETURNS:
        - parses, list of tuples, eaxh a seq of ints.
        """
        rs = self.grammarparses_ruledict_rulestring_extract(ind)[0]
        g = self.grammarparses_grammardict_return(ind)
        parses = g.parses_extract_generated(rs)
        return parses

    def grammarparses_extract_beh_taskstroke_inds(self, ind):
        """ Return the beh strokes, in format of taskstroke indices.
        as list of ints, same len as beh, based oint he order in which each
        taskstroke ind was first touchhed! So might be shorter than the num beh strokes...
        """
        taskstroke_inds_beh_order = self.behclass_extract_taskstroke_inds_in_beh_order(ind)
        if False:
            # not true any more, since is using first_touch_using_max_align
            assert taskstroke_inds_beh_order == [tok["ind_taskstroke_orig"] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_firsttouch")]
        return taskstroke_inds_beh_order

    def grammarparses_chunks_plot_example(self, ind):
        """
        Plot example trial and draing, and print chunk data (from Tokens)
        to confirm by eye this is accurate
        :param ind:
        :return:
        """
        # SANIYT CHECKS
        # self.taskclass_tokens_extract_wrapper(ind, "beh_firsttouch", plot=True, return_as_tokensclass=True)
        tok = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=False)

        self.grammarparses_print_plot_summarize(ind)
        print(" ")
        print("CHUNK RANK: ", [t["chunk_rank"] for t in tok])
        print("CHUNK WITHIN RANK:", [t["chunk_within_rank"] for t in tok])
        print("CHUNK WITHIN RANK FROM LAST:", [t["chunk_within_rank_fromlast"] for t in tok])

    def grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(self, ind,
        return_n_in_chunk=False):
        """
        For this trial, assign each beh stroke to a chunk index and to index within chunk,
        based on the matching parse, for the rule of this trial. e.g., lines to circles, 
        lines, then circle.s 
        NOTE: if this is a failure trial, could be wierd. recormmended to run:
        D.preprocessGood(params=["one_to_one_beh_task_strokes", "correct_sequencing_binary_score"])
        NOTE: Fails if:
        - the length of beh(first touch) is diff from len (beh strokes)\
        - incorrect beh, so no parses found (then cannot assign chunk)
        RETURNS:
        - chunks, list of ints, 
        - chunks_within, list of ints, 
        - (also modifes tokens for this trial)
        EG: ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])
        """
        epoch = self.Dat.iloc[ind]["epoch_orig"]
        rs = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"][epoch]["rulestring"]
        
        beh = self.grammarparses_extract_beh_taskstroke_inds(ind)
        tmp = self.taskclass_tokens_extract_wrapper(ind, which_order="beh_using_task_data")
        assert len(beh)==len(tmp), "beh_firsttouch doesnt exist beh. cannot do assignment."

        try:
            GD = self.grammarparses_grammardict_return(ind)
        except AssertionError as err:
            print("Run this?? D.grammarparses_successbinary_score()")
            raise err

        # get the original chunksclass for this index
        idx = GD._score_beh_in_parses_find_index_match(beh, rs)
        if idx is None:
            print("-----------------")
            print(idx)
            print(beh, rs)
            assert False, "then there is no match... (run this? D.preprocessGood(params=[one_to_one_beh_task_strokes, correct_sequencing_binary_score]))"
        else:
            C = GD.ParsesGeneratedYokedChunkObjects[rs][idx]
        
        # based on this hier, assign chunks ids to each stroke
        PLOT = False
        Tk = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", plot=PLOT, return_as_tokensclass=True)
        return Tk.chunks_update_by_chunksobject(C, return_n_in_chunk=return_n_in_chunk)
        # chunks, chunks_within = Tk.chunks_update_by_chunksobject(C, return_n_in_chunk=return_n_in_chunk)
        # print(ind, epoch, beh, C.Hier, Tk.chunks_update_by_chunksobject(C))
        # _chunk_sequence_is_correct(Tk.Tokens, print_chunk_seq=True)        
        # return chunks, chunks_within

        ###### OLD CODE RELATED TO EXTRACTION OF CHUNK BOUNDARIES!! i THINK ALL ARE PULLED INTO ABOIVE.
        # This coped from 230615_analy_motor_timing_reaction
        ##### Preprocess - get chunks boundaries

        # [OLD!! Use the stuff below]
        # from pythonlib.dataset.modeling.discrete import rules_map_rule_to_ruledict_extract_auto
        # map_epoch_rulestring = rules_map_rule_to_ruledict_extract_auto(D)


        # ruleinfo = D.grammarparses_rules_extract_info()
        # def _mapper_shape_to_rank_for_this_epoch(epoch):
        #     # map shapes to their abstract "role"
        #     # epoch = D.Dat.iloc[ind]["epoch_orig"]
            
        #     if epoch in ["base"]:
        #         return None
        #     else:
        # #         ruledict = map_epoch_rulestring[epoch]
        #         ruledict = ruleinfo["ruledict_for_each_rule"][epoch]
        #         if ruledict["categ"]=="ss" and ruledict["subcat"]=="rank":
        #             shapes_in_order_of_roles = ruledict["params_good"]
        #         else:
        #             print(ruledict)
        #             assert False, "code here: how to get list of shapes"
        #         assert isinstance(shapes_in_order_of_roles, list) 
        #         assert isinstance(shapes_in_order_of_roles[0], str)

        #         map_shape_to_rank = {}
        #         for i, shape in enumerate(shapes_in_order_of_roles):
        #             map_shape_to_rank[shape] = i
        #         return map_shape_to_rank

        # dict_epoch_to_mappers = {}
        # for epoch in D.Dat["epoch"].unique():
        #     mapper = _mapper_shape_to_rank_for_this_epoch(epoch)
        #     assert mapper is not None, "thne there is not 'correct' rank order. this is baseline?"
        #     dict_epoch_to_mappers[epoch] = mapper  
                
            

        # dict_epoch_to_mappers

        # # for each trial, determine the chunks, and assign it to each stroke
        # PLOT = False
        # for ind in range(len(D.Dat)):
        #     tokens = D.taskclass_tokens_extract_wrapper(ind, plot=PLOT, return_as_tokensclass=True)
        #     epoch = D.Dat.iloc[ind]["epoch"]
        #     map_shape_to_rank = dict_epoch_to_mappers[epoch]
        #     tokens.chunks_update_by_shaperank(map_shape_to_rank)
            

        #### Find lollipops

        # # confirm that correct trials have correct chunk sequencing
        # def _chunk_sequence_is_correct(tokens, print_chunk_seq=False):
            
        #     if print_chunk_seq:
        #         chunk_seq = [(tok["chunk_rank"], tok["chunk_within_rank"]) for tok in Tk.Tokens]
        #         print(chunk_seq)

        #     cr_prev = 0
        #     cwr_prev = -1
        #     for tok in Tk.Tokens:
        #         if tok["chunk_rank"]==cr_prev:
        #             # in same chunk as prev. 
        #             if not tok["chunk_within_rank"]==cwr_prev+1:
        #                 return False
        #         elif tok["chunk_rank"]==cr_prev+1:
        #             # new chunk. must reset cwr
        #             assert tok["chunk_within_rank"]==0, "I do not understand why"
        #         elif tok["chunk_rank"]>cr_prev:
        #             # new chunk. skipped a chunk, but assume it is becuase it doesnt
        #             # eixts in this task. this is an ok assukptiong, since will fail later if failes.
        #             assert tok["chunk_within_rank"]==0, "I do not understand why"            
        #         else:
        #             # new chunk, and is lower rank. this is wrong
        #             return False

        #         cr_prev = tok["chunk_rank"]
        #         cwr_prev = tok["chunk_within_rank"]
        #     return True

        # _chunk_sequence_is_correct(Tk.Tokens, print_chunk_seq=True)        

    def grammarparses_score_this_trial(self, ind):
        """ Score whether this trials behaviro is consistent with
        each of the rule's parses
        RETURNS:
        - dict[rulestring]=bool, where True means this trial;s behavior matches oen fo the
        parses for this rule.
        """
        taskstroke_inds_beh_order = self.grammarparses_extract_beh_taskstroke_inds(ind)
        GD = self.grammarparses_grammardict_return(ind)       
        list_rulestring = self.grammarparses_rules_extract_info()["list_rules_exist_as_rulestring"]
        res = {}
        for rs in list_rulestring:
            res[rs] = GD._score_beh_in_parses(taskstroke_inds_beh_order, rs)
        return res

    def grammarparses_chunk_transitions_gaps_extract(self, ind):
        """
        REturn info for each gap between strokes (transition), realted to chunk transitions, gap durations etc.
        Useful for analyzing how gap duration relates to chunk transitions.
        :return:
        """

        # Get data
        durations, gaps = self.strokes_durations_gaps(ind)
        Tk = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True)

        assert len(gaps) == len(Tk.Tokens)-1

        # Collect data across gaps.
        res = [] # length num gaps
        for i, (tok1, tok2) in enumerate(zip(Tk.Tokens[:-1], Tk.Tokens[1:])):
            # Iterate over each gap

            dat = {}
            dat["index_trial"] = ind
            dat["index_gap"] = i

            # Transition information: shape and location
            dat["gap_dur"] = gaps[i]

            # Transition information: chunks
            for feat in ["chunk_rank", "chunk_within_rank", "shape"]:
                dat[f"gap_{feat}"] = (tok1[feat], tok2[feat])

            res.append(dat)

        return pd.DataFrame(res)


    def grammarparses_chunk_transitions_gaps_extract_batch(self, bin_gaps=True, PLOT=True,
                                                           plot_savedir=None):
        """
        Extract dataframe holding each gap, with information about that gap related to gap duration and
        chunk trnaitions.
        :return:
        - self.Dat, addes columns like "chunkgap_(0, 1)_durbin" holding the bin/class for gap durations
        that trnaiston bwetween (0,1) chunks, for that trial, usually 0,1 (string) bins for short and long
        gap.
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        import seaborn as sns

        # And extract syntax_concrete column
        if "syntax_concrete" not in self.Dat.columns:
            self.grammarparses_syntax_concrete_append_column()

        # This only can run if this day has shape rules
        if len(self.grammarparses_rules_involving_shapes())>0:

            # 1. Collect gaps dataframe across all trials
            list_df = []
            for ind in range(len(self.Dat)):
                df = self.grammarparses_chunk_transitions_gaps_extract(ind)
                df["syntax_concrete"] = [self.Dat.iloc[ind]["syntax_concrete"] for _ in range(len(df))]
                df["behseq_locs"] = [self.Dat.iloc[ind]["behseq_locs"] for _ in range(len(df))]
                df["behseq_shapes"] = [self.Dat.iloc[ind]["behseq_shapes"] for _ in range(len(df))]
                df["epoch"] = self.Dat.iloc[ind]["epoch"]

                list_df.append(df)
            dfgaps = pd.concat(list_df).reset_index(drop=True)

            # 2. Append new columns
            dfgaps = append_col_with_grp_index(dfgaps, ["epoch", "syntax_concrete", "gap_chunk_rank"], "ep_sy_gcr")
            dfgaps = append_col_with_grp_index(dfgaps, ["epoch", "syntax_concrete"], "epoch_syntax")
            dfgaps = append_col_with_grp_index(dfgaps, ["epoch", "syntax_concrete", "behseq_shapes", "behseq_locs"], "ep_sy_sh_lo")
            dfgaps = append_col_with_grp_index(dfgaps, ["syntax_concrete", "behseq_shapes", "behseq_locs"], "sy_sh_lo")
            dfgaps = append_col_with_grp_index(dfgaps, ["behseq_shapes", "behseq_locs"], "sh_lo")
            dfgaps["gap_chunk_rank_str"] = ["".join([str(xx) for xx in x]) for x in dfgaps["gap_chunk_rank"]]

            # 3. Classify gaps as short or long. (Bin all gaps)
            # NOTE: This may leave many or most rows without a bin (in which case they are bin -1) because it conditions
            # on somethign very specific (so will fail if too much variability).
            if bin_gaps:
                from pythonlib.tools.pandastools import bin_values_conditioned_on_class
                if False:
                    # This can be very different result from conditioning on shapes and locs sequence...
                    dfgaps = bin_values_conditioned_on_class(dfgaps, "gap_dur",  ["epoch", "syntax_concrete", "index_gap"], 2, 1, "test",
                                                             bin_by_rank=True).reset_index(drop=True)
                else:
                    dfgaps = bin_values_conditioned_on_class(dfgaps, "gap_dur",  ["epoch", "syntax_concrete",
                                                                                  "behseq_shapes", "behseq_locs", "index_gap"],
                                                             2, 1, "gap_dur_bin", bin_by_rank=True).reset_index(drop=True)

            if PLOT:
                # Plot, showing gap durations all, coloring by if is high bin
                fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", col="epoch_syntax", col_wrap=3, alpha=0.25,
                            hue="gap_dur_bin")
                savefig(fig, f"{plot_savedir}/gap_dur-sequences-binned-1.pdf")

                fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", col="epoch_syntax", col_wrap=3, hue="gap_dur_bin", kind="point")
                savefig(fig, f"{plot_savedir}/gap_dur-sequences-binned-2.pdf")

                # Plots asking if gap duration during chunk-transition is larger than
                fig = sns.catplot(data=dfgaps, x="gap_chunk_rank_str", y="gap_dur", hue="gap_dur_bin", col="epoch",
                            col_wrap=5, alpha=0.2)
                savefig(fig, f"{plot_savedir}/gap_dur-chunk_gaps-binned-1.pdf")

                fig = sns.catplot(data=dfgaps, x="gap_chunk_rank_str", y="gap_dur", hue="gap_dur_bin", col="epoch",
                            col_wrap=5, kind="point")
                savefig(fig, f"{plot_savedir}/gap_dur-chunk_gaps-binned-2.pdf")

                dfgaps_this, dict_dfthis = extract_with_levels_of_conjunction_vars(dfgaps, "gap_dur_bin", ["epoch", "behseq_locs", "behseq_shapes"],
                                        n_min_across_all_levs_var=2, lenient_allow_data_if_has_n_levels=2,
                                        prune_levels_with_low_n=True, plot_counts_heatmap_savepath=f"{plot_savedir}/epoch-behseq_locs_behseq_shapes-counts.pdf")
                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", col="ep_sy_sh_lo", col_wrap=4, alpha=0.4,
                            hue="gap_dur_bin")
                savefig(fig, f"{plot_savedir}/gap_dur-sequences-binned-specific-1.pdf")


            # Assign back to D.Dat, new columns, each indicating whether that trial's transition for a particular chunk_gap (e..g,
            # (0,1) is high or lower than median duration.

            # Only do transitions between chunks (oitherwise will fail asssertion below since ther's mult (0,0) per trial)
            list_g = [g for g in dfgaps["gap_chunk_rank"].unique() if not g[0]==g[1]] # for (0,0) etc, is not well-defined

            for g in list_g:
                self.Dat.loc[:, f"chunkgap_{g}_durbin"] = -1 # empty

            for g in list_g:
                for binthis in dfgaps["gap_dur_bin"].unique():
                    trials = dfgaps[(dfgaps["gap_chunk_rank"]==g) & (dfgaps["gap_dur_bin"]==binthis)]["index_trial"].tolist()
                    assert np.all(self.Dat.loc[trials, f"chunkgap_{g}_durbin"] == -1), "this isn t possible, unless there >1 case of this per trial."
                    self.Dat.loc[trials, f"chunkgap_{g}_durbin"] = binthis
                print(f"chunkgap_{g}_durbin")

            # Optioally, plot trials from D, colored by gap duration bin, to confirm that bin classes were correctly
            # assigned back to D. Look for the chunk trnaistion gap having clearly separteted blue and red dots.
            if False: # Need to rpelace by hand: taskcat_by_rule and 3,0|2,1|1,0|0,1|-1,0|-2,1|-3,0'
                taskcat_by_rule = ((6, 0, 1, 0), 7)
                bindur = "1"
                inds_short = D.Dat[(D.Dat["taskcat_by_rule"]==taskcat_by_rule) & (D.Dat["behseq_locs"] == '3,0|2,1|1,0|0,1|-1,0|-2,1|-3,0') & (D.Dat["chunkgap_(0, 1)_durbin"]==bindur)].index.tolist()
                bindur = "2"
                inds_long = D.Dat[(D.Dat["taskcat_by_rule"]==taskcat_by_rule) & (D.Dat["behseq_locs"] == '3,0|2,1|1,0|0,1|-1,0|-2,1|-3,0') & (D.Dat["chunkgap_(0, 1)_durbin"]==bindur)].index.tolist()

                gaps_short = [D.strokes_durations_gaps(i)[1] for i in inds_short]
                gaps_long = [D.strokes_durations_gaps(i)[1] for i in inds_long]

                fig, ax = plt.subplots()

                for gaps in gaps_short:
                    ax.plot(range(len(gaps)), gaps, "-ob",)
                for gaps in gaps_long:
                    ax.plot(range(len(gaps)), gaps, "-or",)

            ####### compare epochs, diff gap durations?
            dfgaps_this, dict_dfthis = extract_with_levels_of_conjunction_vars(dfgaps, "epoch", ["behseq_locs", "behseq_shapes"],
                                                    n_min_across_all_levs_var=2, lenient_allow_data_if_has_n_levels=2,
                                                    prune_levels_with_low_n=True, plot_counts_heatmap_savepath=f"{plot_savedir}/epoch_counds.pdf")
            if len(dfgaps_this)>0:
                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="epoch", col="syntax_concrete", col_wrap=5, height=4, alpha=0.25)
                savefig(fig, f"{plot_savedir}/gap_dur_vs_index-epochs-1.pdf")
                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="epoch", col="syntax_concrete", col_wrap=5, height=4,
                            kind="point")
                savefig(fig, f"{plot_savedir}/gap_dur_vs_index-epochs-2.pdf")

                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="epoch", col="sy_sh_lo", col_wrap=5, height=4, alpha=0.25)
                savefig(fig, f"{plot_savedir}/gap_dur_vs_index-epochs-sy_sh_lo-1.pdf")
                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="epoch", col="sy_sh_lo", col_wrap=5, height=4,
                            kind="point")
                savefig(fig, f"{plot_savedir}/gap_dur_vs_index-epochs-sy_sh_lo-2.pdf")

        else:
            dfgaps = None

        return dfgaps

    def grammarparses_syntax_epochset_quick_classify_same_diff_motor(self):
        """
        Quick reclassificying of epochsets to just reflect whether the motor beahvior across
        epochs (in the set) is same or different.

        Quick and dirty, becuase this shold relaly be done in the initial extraction of
        epochsets.

        Does it by taking epochsets which are tuplke and first item is "same" --> same motor.
        All others will be called "diff".
        E.g., for the following, the firrst 2 are "diff" motor and the last is "same".
        "(same_stroke_0, UL, llCV3)"
        "(char, UL, llCV3)"
        "(same, UL, llCV3)"

        :return:
        assert column: self.Dat["epochset_diff_motor"]
        """

        # classify if epochset starts with "char"
        # New column "epochset_diff_beh_across_epochs"
        diff_beh_across_epochs = []
        for i, row in self.Dat.iterrows():
            if isinstance(row["epochset"], tuple) and row["epochset"][0]=="same":
                diff_beh_across_epochs.append(False)
            elif isinstance(row["epochset"], tuple):
                diff_beh_across_epochs.append(True)
            # if isinstance(row["epochset"], tuple) and row["epochset"][0]=="char":
            #     diff_beh_across_epochs.append(True)
            # elif isinstance(row["epochset"], tuple) and row["epochset"][0]=="same_stroke_0":
            #     # Diff after first stroke. keep
            #     diff_beh_across_epochs.append(True)
            elif not isinstance(row["epochset"], tuple):
                diff_beh_across_epochs.append(True)
            else:
                print(row["epochset"])
                assert False, "is this same or diff motor?"
        self.Dat["epochset_diff_motor"] = diff_beh_across_epochs

    def grammarparses_syntax_role_append_to_tokens(self, PRINT=False):
        """
        GOOD Syntax -- assign each stroke a "syntax rule" which is a function of its epoch_orig and its
        (e.g.) chunk and within-chunk values. e.g, for shape repeat rules, it is (chunkidx, index_within).
        Makes sure to use the chunk index associated with shape, even if the first shape is sometimes skipped
        (i.e., is tied to shaep).
        :return: appends to tokens(ver, beh_using_task_data), key called "syntax_role"
        """

        ##### Extract all tokens
        dftok = self.tokens_extract_variables_as_dataframe(["shape", "ind_taskstroke_orig", "chunk_rank",
                                                            "chunk_within_rank"],
                                                "beh_using_task_data", ["epoch_orig", "epoch", "syntax_concrete"])

        ##### Get  map from epoch_orig to rulekind
        # Use epoch_orig, becuase this references the origianl ruledict
        map_epochorig_to_rulekind = self.grammarparses_rules_extract_map_rule_to_rulekind()

        # For all shape rules... for this day and rule, get mapping from shape to chunk
        # epochs_using_shape = self.grammarparses_rules_involving_shapes(return_as_epoch_orig=True)

        # nchunks = dftok["chunk_rank"].max()+1

        ###### Get mapping from shape to chunk index
        # Use "epoch", beucase for Pancho when do AnBm with 2 sahpes, want the syntax role to be independent of the shape set.
        map_shape_to_chunkidx_byepoch = {}
        map_chunkidx_to_shape_byepoch = {}
        for epoch in self.Dat["epoch"].unique():
            map_shape_to_chunkidx_byepoch[epoch] = {}
            map_chunkidx_to_shape_byepoch[epoch] = {}

            ### FOr some reason, this worked and then started failing...
            # get df, sorted in increasing average chunk_rank (ie for each chunk get its average chunk rank then sort them).
            # dfthis = dftok[dftok["epoch"] == epoch][["shape", "chunk_rank"]].groupby(["shape"]).mean().sort_values("chunk_rank").reset_index(drop=True)
            # for i, shape in enumerate(dfthis.index):
            #     map_shape_to_chunkidx_byepoch[epoch][shape] = i
            #     map_chunkidx_to_shape_byepoch[epoch][i] = shape

            # if not isinstance(dfthis.index.tolist()[0], str):
            #     dfthis = dftok[dftok["epoch"] == epoch][["shape", "chunk_rank"]].groupby(["shape"]).mean().sort_values("chunk_rank").reset_index(drop=False)
            dfthis = dftok[dftok["epoch"] == epoch][["shape", "chunk_rank"]].groupby(["shape"]).mean().sort_values("chunk_rank").reset_index(drop=False)
            for i, shape in enumerate(dfthis["shape"]):
                map_shape_to_chunkidx_byepoch[epoch][shape] = i
                map_chunkidx_to_shape_byepoch[epoch][i] = shape

        ########### RUN: for each token, classify its syntax role.
        list_role = []
        for i, tok in dftok.iterrows():
            epoch_orig = tok["epoch_orig"]
            epoch = tok["epoch"]
            if map_epochorig_to_rulekind[epoch_orig][0] == "ss":
                # Shape sequence (ABC) or (AABBBCC) or (AnBmCk)...
                map_shape_to_chunkidx = map_shape_to_chunkidx_byepoch[epoch]
                try:
                    chunkidx = map_shape_to_chunkidx[tok["shape"]]
                except Exception as err:
                    print(map_shape_to_chunkidx)
                    print(tok["shape"])
                    print(dftok)
                    print(dfthis)
                    raise err
                syntax_role = (chunkidx, tok["chunk_within_rank"])
            elif map_epochorig_to_rulekind[epoch_orig][0] == "dir":
                # Direction rule (flat).
                # stroke index is syntax role.
                syntax_role = (tok["stroke_index"],)
            elif map_epochorig_to_rulekind[epoch_orig][0] == "rowcol":
                # rows or cols. this has hierarchical structure
                syntax_role = (tok["chunk_rank"], tok["chunk_within_rank"])
            elif map_epochorig_to_rulekind[epoch_orig][0] == "preset" and map_epochorig_to_rulekind[epoch_orig][1] == "null":
                # Random seuqence (hard coded in matlab task).
                # Stroke index is sytnax role.
                syntax_role = (tok["stroke_index"],)
            elif map_epochorig_to_rulekind[epoch_orig][0] == "ch" and map_epochorig_to_rulekind[epoch_orig][1] == "dir2":
                # (AB)n, with any direction within chunks.
                syntax_role = (tok["chunk_rank"], tok["chunk_within_rank"])
            else:
                print(epoch)
                print(epoch_orig)
                print(map_epochorig_to_rulekind)
                print(tok)
                assert False
            list_role.append(syntax_role)
        dftok["syntax_role"] = list_role

        if PRINT:
            from pythonlib.tools.pandastools import grouping_print_n_samples
            print(["epoch_orig", "shape", "chunk_rank", "chunk_within_rank", "syntax_role"])
            grouping_print_n_samples(dftok, ["epoch_orig", "shape", "chunk_rank", "chunk_within_rank", "syntax_role"])

        # Place back into tokens
        self.tokens_assign_dataframe_back_to_self(dftok, "syntax_role", "beh_using_task_data")

        print("DOne! added key to tokens(beh_using_task_data) : syntax_role")

    def grammarparses_syntax_concrete_append_column(self, PRINT=False):
        """
        Append column hoklding syntax (e.g.,, (1,3,2))
        based on the epoch_orig
        of each trial (using task, ignoring beh).
        For trials that dont have syntax-relatd epoch orig, hacks by using
        the syntax for the first rule in the list of rule that are
        shape-related (this rarely mattres)
        :return: appends column "syntax_concrete" to self.Dat, holding
        either tuple of ints or (if no shape rules today)  or ("IGNORE")
        (i.e, item will alwyas be tuple)
        """

        # Get mapping from epoch_orig to syntax, for each trial
        map_epoch_orig_to_list_syntax = self.grammarparses_classify_tasks_syntax_based_on_rule()

        # if map_epoch_orig_to_list_syntax is None:
        #     self.Dat["syntax_concrete"] = "IGNORE"
        # else:
        # for each trial, determine what is its syntax (based on its epoch_orig)
        list_syntax = []
        for i in range(len(self.Dat)):
            epoch_orig = self.Dat.iloc[i]["epoch_orig"]
            if epoch_orig in map_epoch_orig_to_list_syntax:
                syntax = map_epoch_orig_to_list_syntax[epoch_orig][i]
            else:
                # This is epoch doesnt have a syntax. Just take the first epoch, should be using this anyway
                syntax = list(map_epoch_orig_to_list_syntax.values())[0][i]
            list_syntax.append(syntax)
        self.Dat["syntax_concrete"] = list_syntax

        if PRINT:
            self.grouping_print_conjunctions_summary_good(["syntax_concrete", "taskcat_by_rule"], PRINT=True)

    def grammarparses_classify_tasks_syntax_based_on_rule(self):
        """
        Get list of syntaxes (tuples of ints), one for each trial,
        which is concrete syntax for each rule. Returns
        dict epoch_orig --> syntax
        :return: dict (see above) or None (if no shapes rules exist)
        """
        from pythonlib.dataset.modeling.discrete import tasks_categorize_based_on_rule

        # rs_using_shape = self.grammarparses_rules_involving_shapes()
        # if len(rs_using_shape)==0:
        #     return None
        # else:
        #     map_epoch_orig_to_list_syntax = {}
        #     for rs in rs_using_shape:
        #         # assert len(rs_using_shape)==1
        #         # rs = rs_using_shape[0]
        #         # rulestring --> epoch
        #         epoch_orig = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"][rs]["params"]
        #         list_syntax = tasks_categorize_based_on_rule(self, epoch_orig)
        #         map_epoch_orig_to_list_syntax[epoch_orig] = list_syntax
        #     return map_epoch_orig_to_list_syntax

        def _tupleize(s):
            """ Convert s to tuple, either converting list, or by containig in tuple."""
            if isinstance(s, list):
                return tuple(s)
            elif isinstance(s, tuple):
                return s
            else:
                return tuple([s])

        map_epoch_orig_to_list_syntax = {}
        for epoch_orig in self.Dat["epoch_orig"].unique():
        # for rs in rs_using_shape:
            # assert len(rs_using_shape)==1
            # rs = rs_using_shape[0]
            # rulestring --> epoch
            # epoch_orig = self.grammarparses_rules_extract_info()["map_rulestr_ruledict"][rs]["params"]
            list_syntax = tasks_categorize_based_on_rule(self, epoch_orig)
            list_syntax = [_tupleize(s) if not isinstance(s, tuple) else s for s in list_syntax]
            map_epoch_orig_to_list_syntax[epoch_orig] = list_syntax
        return map_epoch_orig_to_list_syntax

    def grammarparses_classify_tasks_categorize_based_on_rule(self):
        """
        Wrapper for methods to classify each task based on something like a "syntax parse"
        # e.g., ((3, 1, 1, 0),) means for AnBmCk, you have n=3, m=1, k=1.
        # The last 0 means no leftover items.
        :return: Modifes self.Dat adding new column taskcat_by_rule (flexible type, usualyl tuple of ints)
        """
        from pythonlib.dataset.modeling.discrete import tasks_categorize_based_on_rule_mult
        tasks_categorize_based_on_rule_mult(self)

    def grammarparses_classify_sequence_error(self, ind, PRINT_PLOT=False):
        """
        Errors in rule choice

        Uses opposite rule
        Uses easier rule
        Uses default rule (i.e., "jump to closest stroke")

        Errors in sequencing
        Correct shape, incorrect order
        Incorrect shape, correct order within [variation: Terminate current chunk, then does next chunk normally]
        Random errors
        """
        from pythonlib.tools.listtools import sequence_first_error

        if False:
            # using matlab version
            g = D.grammarmatlab_extract_beh_and_task(ind)

            # beh order
            beh = g["taskstroke_inds_beh_order"]
            task = g["taskstroke_inds_correct_order"]

            # map from taskstroke_inds to tokens
            print(beh, task)

        ## Extract behavior and task parses
        beh = self.grammarparses_extract_beh_taskstroke_inds(ind)
        parses = self.grammarparses_parses_extract_trial(ind)
        assert len(parses)==1, "this only coded for cases with single determinstic sequence otherwise is complex"
        task = list(parses[0])

        ## Classify features for the first stroke that is an error
        # Find first error
        i_error, taskind_chosen, taskind_correct = sequence_first_error(beh, task)

        if PRINT_PLOT:
            print("... results from sequence_first_error")
            print(i_error, taskind_chosen, taskind_correct)

        if i_error is None:
            # then no stroke error...
            error_same_shape = None
            error_chose_closer = None
        else:
            # Tokens
            tokens = self.taskclass_tokens_extract_wrapper(ind, "task")
            map_taskind_tok = {}
            for i, t in enumerate(tokens):
                map_taskind_tok[i] = t

            # Individula toks for specific strokes
            t_chosen = map_taskind_tok[taskind_chosen]
            t_correct = map_taskind_tok[taskind_correct]

            # Error: same shape wrong location?
            shape_correct = t_correct["shape"]
            shape_chosen = t_chosen["shape"]
            error_same_shape = shape_correct==shape_chosen

            # Error: jumped to closest stroke?
            if i_error>0:
                t_before_error = map_taskind_tok[beh[i_error-1]] # token of stroke before the last(failed) stroke
                loc_chosen = np.array(t_chosen["loc_concrete"])
                loc_prev = np.array(t_before_error["loc_concrete"])
                loc_corr = np.array(t_correct["loc_concrete"])
                dist_chosen = np.linalg.norm(loc_chosen - loc_prev)
                dist_correct = np.linalg.norm(loc_corr - loc_prev)
                error_chose_closer = dist_chosen<dist_correct
            else:
                error_chose_closer = None

        ## First, is beh consistent with each of the rules ont his day
        # compare beh against each parse
        g = self.grammarparses_grammardict_return(ind, False)
        rulestring, _ = self.grammarparses_ruledict_rulestring_extract(ind)
        rulestrings_check = self.grammarparses_rulestrings_exist_in_dataset() # get rules that are used int his day
        rulestrings_correct = []
        list_initiated_other_rule = []
        for rule in rulestrings_check:
            parses_this_rule = g.parses_extract_generated(rule)
            parses_this_rule = [list(p) for p in parses_this_rule]

            correct = False
            for p in parses_this_rule:
                if beh==p[:len(beh)]:
                    # then beh is correct until beh ends
                    correct=True
                    break  
            if correct:
                # note this down
                rulestrings_correct.append(rule)    

            # is the last beh stroke going for the earliest remaining stroke in parse?
            if i_error is None:
                # Then checkign this doesnt make sense
                list_initiated_other_rule = []
            else:
                for p in parses_this_rule:

                    # i.e got distrated and started the other rule
                    remaining_idx_in_p = [idx for idx in p if idx not in beh[:-1]]
                    initiated_other_rule = beh[-1] == remaining_idx_in_p[0]
                    if initiated_other_rule:
                        list_initiated_other_rule.append(rule)
                        break

        if PRINT_PLOT:
            print("Beh is correct under these rules:")
            print(rulestrings_correct)

        # summary this error
        error_dict = {
            "correct":i_error is None, # correct up to abort.
            "rulestrings_correct":rulestrings_correct,
            "list_initiated_other_rule":list_initiated_other_rule,
            "error_same_shape":error_same_shape,
            "error_chose_closer":error_chose_closer,
            "rule_other_correct":(len(rulestrings_correct)>0 and rulestring not in rulestrings_correct), # Is a different rule correct?
            "rule_other_reinitiated":len(list_initiated_other_rule)>0
        }           

        # Sanity
        if error_dict["correct"]:
            assert error_dict["list_initiated_other_rule"] == []
            assert error_dict["error_same_shape"] is None
            assert error_dict["error_chose_closer"] is None
            assert error_dict["rule_other_correct"]==False

        if PRINT_PLOT:
            self.grammarmatlab_extract_beh_and_task(ind, True)

        return error_dict

    def grammarparses_successbinary_print_summary(self):
        """
        """

        assert False, "see todo"
        # Check that success_binary_quick is from parses analysis.
        # Then run grammar_successbinary_print_summary


    def grammarparses_successbinary_score_wrapper(self, print_summary=False, DEBUG=False):
        """ Good, determine if beh is success based on matching any of the possible
        parses given each trial's rule.
        PARAMS:
        - print_summary, bool, then useful printing for each trial, sanity checks as well.
        RETURNS:
        - bm, object holding results.
        - appends columns to self.Dat, including success_binary_quick
        """
        from  pythonlib.dataset.dataset_analy.grammar import preprocess_dataset_recomputeparses

        if "base" in self.Dat["epoch"] or "baseline" in self.Dat["epoch"]:
            assert False, "You must remove baseline epochs, they dont have defined parses."

        # 1) Score each trial based on parses
        bm = preprocess_dataset_recomputeparses(self, DEBUG=DEBUG)

        # D.sequence_extract_beh_and_task(230, True)
        if print_summary:
            self.grammarparses_successbinary_print_summary()

        if DEBUG:
            # For each trial, print its semantic outcome
            for i in range(len(self.Dat)):
                tc = self.Dat.iloc[i]["trialcode"]
                print(i, tc, self.grammarparsesmatlab_score_wrapper(i))

            # pick a speicfic trial and plot and print it.
            ind = 783
            tc = self.Dat.iloc[ind]["trialcode"]
            bm.DatLong[bm.DatLong["trialcode"] == tc]

            self.grammarmatlab_extract_beh_and_task(ind, True)
            
        return bm

    # def grammarmatlab_extract_beh_and_task(self, ind, ploton=False):
    #     """ Goal is to replace "seuqence" module with grammar
    #     """
    #     return self.grammarmatlab_extract_beh_and_task(ind, ploton)

    def grammarmatlab_tasksequencer_rules_matlab(self, ind):
        """ REturn the tasksequencer rules used in matlab to generate
        the ObjectClass seuqence
        """

        # 1) get the matlab params

        # 2) [optional] convert to a string code for post-processing.

        assert False, 'this is alread in self.Dat["epoch_rule_tasksequencer"] To reextract, see epoch_grouping_reassign_by_tasksequencer'

    
    def grammarmatlab_successbinary_print_summary(self):
        """
        Useful printing for each trial, sanity checks as well, for success_binayr,
        which I think could be eitehr matlab sequence or parses, based on which code was run
        """

        print("ind, isprobe, SUCCESS, nbeh, ntask, one_to_one")
        for ind in range(len(self.Dat)):
            SUCCESS = self.Dat.iloc[ind]["success_binary_quick"]
        # D.sequence_extract_beh_and_task(ind, True) 
            nbeh = len(self.Dat.iloc[ind]["strokes_beh"])
            ntask = len(self.Dat.iloc[ind]["strokes_task"])
            isprobe = self.Dat.iloc[ind]["probe"]
            one_to_one = self.sequence_compute_one_to_one_beh_to_task(ind)
            if SUCCESS and nbeh>ntask:
                print(ind, isprobe, SUCCESS, nbeh, ntask, one_to_one, " *** (success, but too MANY strokes)")
                assert one_to_one==False
            elif SUCCESS and nbeh<ntask:
                print(ind, isprobe, SUCCESS, nbeh, ntask, one_to_one, " ### (success, but too FEW strokes)")
                assert one_to_one==False
            elif SUCCESS and nbeh==ntask:
                print(ind, isprobe, SUCCESS, nbeh, ntask, one_to_one)
                assert one_to_one==True
            elif not SUCCESS and nbeh>=ntask:
                print(ind, isprobe, SUCCESS, nbeh, ntask, one_to_one, " !!! (fail, but got lots of beh strokes ...) ")
                # assert one_to_one==False
            else:
                # Then not success...
                print(ind, isprobe, SUCCESS, nbeh, ntask, one_to_one)

    def grammarmatlab_successbinary_score(self, print_summary=False):
        """ Good, score each trial based on the ground truth matlab sequence, a singel
        determistic sequence.
        PARAMS:
        - print_summary, bool, then useful printing for each trial, sanity checks as well.
        RETURNS:
        - bm, object holding results.
        - appends columns to self.Dat, including success_binary_quick
        """
        from  pythonlib.dataset.dataset_analy.grammar import preprocess_dataset_matlabrule

        # 1) Score each trial based on parses
        bm = preprocess_dataset_matlabrule(self)

        # D.sequence_extract_beh_and_task(230, True)
        if print_summary:
            self.grammarmatlab_successbinary_print_summary()

        return bm

    def grammarmatlab_wrapper_extract(self, return_as_bmh_object=True):
        """ Extract grammar data for each trial
        RETURNS:
        - bm, beh_model_holder, with dataset (rows) matching self.Dat
        """
        from pythonlib.dataset.dataset_analy.grammar import preprocess_dataset
        assert False, "either (i) rename this so is clear this is the matlab rules, or (ii) rewrite this to use generate_scored_beh_model_data_long"
        bm, _, _ = preprocess_dataset(self, return_as_bmh_object=return_as_bmh_object)
        return bm


    #####################################################
    def sequence_tokens_check_taskclass_locked(self):
        """
        Check if all TaskClass (self.Dat["Task"]) are locked from generating new
        tokens, which I usualyl do if I have replaced tokens with those for
        chars (using clust shapes) and want to block any possible use of the
        original grid versiuon tokens.
        :return: bool, True if all rows of self.Dat are locked.
        """
        all_locked = True
        for i, row in self.Dat.iterrows():
            if (not hasattr(row["Task"], "_TokensLocked")) or (row["Task"]._TokensLocked == False):
                all_locked = False
                break
        return all_locked

    def sequence_tokens_clear_behclass_and_taskclass_and_lock(self):
        """ Guaranteed to completely remove any trace of extracted
        datsegs, by deleling behclass entirely, and removing
        tokens from taskclass. Similar to sequence_tokens_clear_behclass_and_taskclass
        but just stronger.
        And lock means it blocks code from generating tokens in future.
        """

        self.sequence_tokens_clear_behclass_and_taskclass()

        for i in range(len(self.Dat)):
            self.Dat.iloc[i]["Task"]._TokensLocked = True
            self.Dat.iloc[i]["BehClass"]._TokensLocked = True

        # Move BehClass to other column
        self.Dat["_BehClass"] = self.Dat["BehClass"]
        del self.Dat["BehClass"]

    def sequence_tokens_clear_behclass_and_taskclass(self):
        """ Remove cached tokens, datsegs by deleting them.
        Useful if you want to reextract them.
        """
        for i in range(len(self.Dat)):
            Task = self.Dat.iloc[i]["Task"]
            Beh = self.Dat.iloc[i]["BehClass"]
            if hasattr(Task, "_DatSegs"):
                Task._tokens_delete()
            if hasattr(Beh, "Alignsim_Datsegs"):
                del Beh.Alignsim_Datsegs        
            if hasattr(Beh, "Alignsim_Datsegs_BehLength"):
                del Beh.Alignsim_Datsegs_BehLength

    def sequence_tasksequencer_shapeseq_assign(self):
        """
        For each trial, extract its correct set of shapes
        based on tasksequencer rule, as a tuple of ints, where the ints
        are codes for shapes, sorted so that order doesnt matter.
        Auto extracts the map based on ruledict.
        If a trial's rule is not shape sequence, then gives it "UNKNOWN"  
        RETURNS:
        - assigns new column to self.Dat["taskconfig_shp_code"], with 
        sorted tuple of ints (codes to shapes).
        """

        # if "taskconfig_shp" not in self.Dat.columns:
        self.taskclass_shapes_loc_configuration_assign_column()

        list_epoch = self.Dat["epoch_orig"].unique().tolist()
        ruledict_by_epoch = self.grammarparses_rules_extract_info()["ruledict_for_each_rule"]

        ## Get map betwen index and shape, separaltey for each epoch
        MAP_CODE_SHAPE_byepoch = {}
        MAP_SHAPE_CODE_byepoch = {}
        for epoch in list_epoch:
            if ruledict_by_epoch[epoch]["categ"]=="ss":
                
                # Then this rule is about shapse. Force it to extract some sheapse.
                if ruledict_by_epoch[epoch]["subcat"]=="rankdir":
                    # Then this is sahpe sequence
                    shapes_ordered = ruledict_by_epoch[epoch]["params_good"][0]
                elif ruledict_by_epoch[epoch]["subcat"]=="rank":
                    # Then this is sahpe sequence
                    shapes_ordered = ruledict_by_epoch[epoch]["params_good"]
                else:
                    print(1, ruledict_by_epoch)
                    print(2, ruledict_by_epoch[epoch])
                    print(3, ruledict_by_epoch[epoch]["categ"])
                    print(4, ruledict_by_epoch[epoch]["subcat"])
                    assert False, "find the list of shapes."
                assert isinstance(shapes_ordered, (list, tuple))
                assert isinstance(shapes_ordered[0], str)            

                # give a code
                map_code_shape = {}
                map_shape_code = {}
                for i, sh in enumerate(shapes_ordered):
                    map_code_shape[i] = sh
                    map_shape_code[sh] = i
                    
                MAP_CODE_SHAPE_byepoch[epoch] = map_code_shape
                MAP_SHAPE_CODE_byepoch[epoch] = map_shape_code

        ## For each trial, get its list of shapes, in codenum
        try:
            list_shcode =[]
            for ind in range(len(self.Dat)):
                shapes = self.Dat.iloc[ind]["taskconfig_shp"]
                epoch_orig = self.Dat.iloc[ind]["epoch_orig"]
                if epoch_orig in MAP_SHAPE_CODE_byepoch.keys():
                    map_shape_code = MAP_SHAPE_CODE_byepoch[epoch_orig]
                    shapes_code = tuple(sorted([map_shape_code[sh] for sh in shapes]))
                else:
                    shapes_code = tuple(["UNKNOWN"])
                list_shcode.append(shapes_code)
        except Exception as err:
            print(1, ind)
            print(2, shapes)
            print(3, epoch_orig)
            print(4, MAP_SHAPE_CODE_byepoch)
            print(5, MAP_SHAPE_CODE_byepoch[epoch_orig])
            print(6, map_shape_code)
            print(7, shapes_code)
            raise err
            
        print("New column in self.Dat[taskconfig_shploc_code]")
        self.Dat["taskconfig_shp_code"] = list_shcode

        return MAP_CODE_SHAPE_byepoch, MAP_SHAPE_CODE_byepoch


    def sequence_char_taskclass_assign_char_seq(self, ver="task_matlab", 
            sequence_keep_these_indices=None):
        """ Assign a new column "char_seq" which is conjunction of character 
        and sequence, either beh sequence or task (groud truth matlab) sequene.
        This useful if want to find common beh across epochs, and there is variability 
        in how a character is done, or when considering "random rank" seuqqence epochs
        PARAMS
        - ver, how to define sequnce. 
        --- beh, the beh sequence on the trial
        --- task_matlab, the matlab objectclass sequence
        - sequence_keep_these_indices, list of indices into seuqence, to slice seuqencwe. 
        e.g., useful if you only care about indices 2 and 3...
        If this is longer than the sequence for any char, then uses Nones for non-existing indicecs
        RETURNS:
        - new column (char_seq) for each trial
        """

        # for any trial, get its (task, sequence) tuple
        def _get_char_sequence(ind):
            char = self.Dat.iloc[ind]["character"]
            if ver=="task_matlab":
                sequence = tuple(self.grammarmatlab_extract_beh_and_task(ind)["taskstroke_inds_correct_order"])
            else:
                print(ver)
                assert False, "code it!!"

            if sequence_keep_these_indices:
                sequence = tuple([sequence[i] if i<len(sequence) else None for i in sequence_keep_these_indices ])
            return (char, sequence)

        # - append new column charseq
        list_charseq = []
        for i in range(len(self.Dat)):
            list_charseq.append(_get_char_sequence(i))
            
        self.Dat["char_seq"] = list_charseq      
        print(f".. Appended new column 'char_seq', version: {ver}")

    def sequence_strokes_compute_01_sameness_status(self):
        """
        DEtermine sameness of the first 2 strokes for each trial's char
        across epochs (all) --> either same both strokes, same first diff second,
        or both different. 
        RETURNS:
        - modifies self.Dat, adding column "strokes01_sameness", which has
        string categorical value (one of three possibiltiies.)
        NOTE: only keeps chars that occur across all epochs.
        """

        # If there is only one epoch, then no point running
        if len(self.Dat["epoch"].unique().tolist())==1:
            self.Dat["strokes01_sameness"] = "neither"
            return

        # Get trialcodes that have same first storke and same (first, second) stroke
        map_epochset_trialcode_0 = self.epochset_extract_wrapper("same_beh_first_stroke", 
                                                             only_keep_epochsets_containing_all_epochs=True,
                                                             exclude_leftover=True)

        map_epochset_trialcode_01 = self.epochset_extract_wrapper("same_beh_first_two_stroke",
                                                             only_keep_epochsets_containing_all_epochs=True,
                                                             exclude_leftover=True) 

        if len(map_epochset_trialcode_0)==0 or len(map_epochset_trialcode_01)==0:
            # then didnt have even a single char that have multiple epochs...
            self.Dat["strokes01_sameness"] = "neither"
            return

        try:
            assert len(map_epochset_trialcode_0.keys())==1, "kind of hacky, might need to fix this... PROBAABLY YOU NEED OT REMOVE BASELINE FIRST"
            assert len(map_epochset_trialcode_01.keys())==1
        except Exception as err:
            print(map_epochset_trialcode_0)
            print("---")
            print(map_epochset_trialcode_01)
            map_epochset_trialcode_0 = self.epochset_extract_wrapper("same_beh_first_stroke", 
                                                         only_keep_epochsets_containing_all_epochs=False,
                                                         exclude_leftover=True)

            print("probaly problem is having extra epochs that lead this flag to fail (only_keep_epochsets_containing_all_epochs)")
            print("SOlution is to remove unneeded epochs")
            print(map_epochset_trialcode_0)
            print(self.Dat["epoch"].value_counts())
            raise err
        trialcodes_0 = list(map_epochset_trialcode_0.values())[0] # list of tc
        trialcodes_01 = list(map_epochset_trialcode_01.values())[0]

        # same first, diff 2nd stroke
        trialcodes_0_not1 = [tc for tc in trialcodes_0 if tc not in trialcodes_01]

        # assign back into self.Dat
        names = []
        for ind in range(len(self.Dat)):
            tc = self.Dat.iloc[ind]["trialcode"]
            
            if tc in trialcodes_01:
                names.append("both")
            elif tc in trialcodes_0_not1:
                names.append("first_not_second")
            else:
                names.append("neither")
                
        self.Dat["strokes01_sameness"] = names

        
        print("Appended to self.Dat: strokes01_sameness")

    def sequence_compute_each_task_stroke_only_one_beh(self, ind):
        """ This is like one to one, but allows that the num
        beh strokes is less than num task strokes. just enforces that
        each task stroke is matched by max 1 beh stroke. ie. not allowed
        to use two+ beh strokes on one task stroke.
        RETURNS: bool.
        """

        # Get task strokes in order of gotten, and for each get the
        # beh strokes for which this task stroek was the first gotten
        # e.g,
        # [[0], [1], [], [2], [3]] means 3rd task stroke was not gotten, or it
        # was gotten along with 2nd task stroke by the 2nd beh stroke.
        this = [t[0] for t in self.behclass_extract_beh_and_task(ind)[3]] 
        # n_beh = len(self.Dat.iloc[ind]["strokes_beh"])
        # n_task = len(self.Dat.iloc[ind]["strokes_task"])
        matches = all([len(t)<=1 for t in this]) # eahc task stroke gotten is matched to its own beh stroke
        return matches

    def sequence_compute_one_to_one_beh_to_task(self, ind):
        """ Compute whether one to one mappibng between beh and task strokes.
        i.e, each beh stroke matehd to a singel task stroke, and vice versa.
        alsop means must have gotten all strokes
        """

        # Get task strokes in order of gotten, and for each get the
        # beh strokes for which this task stroek was the first gotten
        # e.g,
        # [[0], [1], [], [2], [3]] means 3rd task stroke was not gotten, or it
        # was gotten along with 2nd task stroke byt the 2nd beh stroke.
        this = [t[0] for t in self.behclass_extract_beh_and_task(ind)[3]] 
        n_beh = len(self.Dat.iloc[ind]["strokes_beh"])
        n_task = len(self.Dat.iloc[ind]["strokes_task"])

        matches = all([len(t)==1 for t in this]) # eahc task stroke gotten is matched to its own beh stroke
        same_n_strokes = n_task == len(this) # got all task strokes.
        
        one_to_one = matches and same_n_strokes

        if one_to_one:
            # NOTE: it is not true the other way round...
            assert n_beh == n_task
        return one_to_one

    ##################################### CHAR CLUSTER RESULTS
    def charclust_shape_labels_extract_presaved_from_DS(self, skip_if_labels_not_found=False):
        """ Good - Load the shape labels already computed and saved using
        DS (see character_cluster_extract_shape_labels.py). And then
        stores in self.Dat['charclust_shape_seq'], each item is
        tuple of strings.
        GUARANTEES:
        - character tasks:
        --- every trial and stroke in DS must be found in self (but ONLY CHECKS this
        for "character" task_kind trials (others are given their shapes in actual) is gotten.
        ACTUALLY, guarantees some fraction (usually 0.9) of char trials gets all stroes.
        The failures will use IGN.
        - non-character tasks:
        --- tries to find. if not, then uses defaults (datsegs).
        PARAMS:
        - skip_if_labels_not_found, bool, whether to fail (False) or return None (True) if
        no pre-saved labels are found.
        - replace_seq_context_cols, bool (True), then replaces columsn in self.Dat related
        to shape sequence, e.g., seqc_0_shape ...
        RETURNS:
        - WIthout pruning self, updates shape labels in self.Dat
        - trialcodes_chars_failed, list of tcs whch did not get all their strokes matched.
        """
        from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes

        nstart = len(self.Dat)

        # First, get all the strokes.
        DS = preprocess_dataset_to_datstrokes(self, "all_no_clean") # get all strokes.
        DS = DS.clustergood_load_saved_cluster_shape_classes(
            skip_if_labels_not_found=skip_if_labels_not_found)
        if DS is None:
            # Then no data found
            print("Skipping cluster labels extractioun! no data found")
        else:

            # plot examples for each shape
            if False:
                DS.plotshape_multshapes_egstrokes_grouped_in_subplots(key_subplots="shape_label", n_examples=5);

            # Assign back to D, to get sequence context using DS.
            DS.dataset_replace_dataset(self)
            # inds_keep = []
            shapes_keep = []
            clust_sim_maxes_keep = []
            n_failures = 0
            n_success = 0
            trialcodes_chars_failed = []
            did_replace = []
            for i, row in self.Dat.iterrows():
                tc = row["trialcode"]
                if row["task_kind"]=="character":
                    # then look for shape labels. IF fail, this bad, relace with ignore, and note failure.
                    tmp = DS.dataset_extract_strokeslength_list(tc, "shape_label", if_fail="return_none")
                    if tmp is not None:
                        shapes = tuple(tmp)
                        clust_sim_maxes = tuple(DS.dataset_extract_strokeslength_list(tc, "clust_sim_max", if_fail="error"))
                        n_success += 1
                        did_replace.append(True)
                    else:
                        # Use the labels based on ground-truth shapes.
                        shapes = ["IGN" for _ in range(len(row["strokes_beh"]))]
                        clust_sim_maxes = tuple([np.nan for _ in range(len(shapes))])
                        n_failures += 1
                        trialcodes_chars_failed.append(tc)
                        did_replace.append(False)
                else:
                    # If tc exists, than pull it out. Else is ok to skip, will use the orignial datsegs.
                    tmp = DS.dataset_extract_strokeslength_list(tc, "shape_label", if_fail="return_none")
                    if tmp is not None:
                        shapes = tuple(tmp)
                        clust_sim_maxes = tuple(DS.dataset_extract_strokeslength_list(tc, "clust_sim_max", if_fail="error"))
                        did_replace.append(True)
                    else:
                        # Use the labels based on ground-truth shapes.
                        shapes = self.seqcontext_extract_shapes_in_beh_order(i)
                        clust_sim_maxes = tuple([np.nan for _ in range(len(shapes))])
                        did_replace.append(False)

                # inds_keep.append(i)
                shapes_keep.append(shapes)
                clust_sim_maxes_keep.append(clust_sim_maxes)

            if n_success/(n_failures + n_success)<0.9:
                print(n_success, n_failures)
                print(len(DS.Dat["trialcode"].unique()))
                print(len(D.Dat["trialcode"].unique()))
                assert False, "why skipped so many? Id expect the only reason to skipt o be very rare losses of strokes (e.g, noise reduction)"

            # Assign them back to
            # self.Dat = self.Dat.iloc[inds_keep].reset_index(drop=True)
            self.Dat["charclust_shape_seq"] = shapes_keep
            self.Dat["charclust_shape_seq_scores"] = clust_sim_maxes_keep
            self.Dat["charclust_shape_seq_did_replace"] = did_replace

            print("Appended column of shape labels: charclust_shape_seq")

            # if replace_seq_context_cols:
            #     # Extract seq context info, given the new labels.
            #     from pythonlib.dataset.dataset_preprocess.seqcontext import preprocess_dataset
            #
            #     # make sure all seqc are replaced
            #     n = 100
            #     for i in range(n):
            #         if f"seqc_{i}_shape" not in self.Dat.columns:
            #             break
            #     nmax = max([i, 9]) # man n strokes to take for seqc...
            #
            #     preprocess_dataset(self, n_strok_max=nmax, version="char_clust_labels")
            #     print("Replaced all seqc_{}_shape columns with char clust labels!")

            assert len(self.Dat)==nstart

            return trialcodes_chars_failed

    ######################################
    def grammarparsesmatlab_score_wrapper_append(self):
        """ For each trial, give string code for its outcome, which takes
        into account whether the sequence was correct, even if it was aborted
        RETURNS;
        - appends column grammar_score_string to D.Dat
        """
        results = []
        for ind in range(len(self.Dat)):
            results.append(self.grammarparsesmatlab_score_wrapper(ind)) 
        self.Dat["grammar_score_string"] = results
        print("Appended D.Dat[grammar_score_string]")

    def grammarparsesmatlab_score_wrapper(self, ind, PRINT=False):
        """ [GOOD] For this trial (ind), return a semantic wrapper of the behavior, based on 
        what has already been computed and saved in D.Dat, for seuqence accuracy either
        relative to parses or matlab.
        THIS uses either matlab or parses, deoending on which version generated success_binary_quick.
        RETURNS:
        - outcome, a string that is interpretable.
        """

        # signature of the trial
        a = self.Dat.iloc[ind]["success_binary_quick"]
        b = self.Dat.iloc[ind]["beh_sequence_wrong"]
        c = self.Dat.iloc[ind]["beh_too_short"]
        d = self.Dat.iloc[ind]["exclude_because_online_abort"]
        e = self.Dat.iloc[ind]["aborted"]

        # assign it one and only one outcome
        beh_tuple = (a,b,c,d,e)

        online_abort_but_sequence_correct_so_far = beh_tuple == (False, False, True, True, True)
        online_abort_but_sequence_correct_complete = beh_tuple == (False, False, False, True, True)
        sequence_correct = (beh_tuple == (True, False, False, False, False)) or (beh_tuple == (True, False, False, False, True)) # online abort is rare, but can happen due to hotkey
        sequence_incorrect_online_abort = (beh_tuple == (False, True, True, False, True)) or (beh_tuple == (False, True, False, False, True))
        sequence_incorrect_but_no_abort = (beh_tuple == (False, True, True, False, False)) or (beh_tuple == (False, True, False, False, False))
        done_early_but_sequence_correct_so_far = beh_tuple == (False, False, True, True, False) # e.g., pressed done button early.

        # make sure one and only one outcome
        if sum([online_abort_but_sequence_correct_so_far, 
             online_abort_but_sequence_correct_complete, 
             sequence_correct,
             sequence_incorrect_online_abort, 
             sequence_incorrect_but_no_abort,
             done_early_but_sequence_correct_so_far])!=1:
            print([online_abort_but_sequence_correct_so_far, 
                     online_abort_but_sequence_correct_complete, 
                     sequence_correct,
                     sequence_incorrect_online_abort, 
                     sequence_incorrect_but_no_abort,
                     done_early_but_sequence_correct_so_far])
            print(a,b,c,d,e)
            print(ind)
            print(self.Dat.iloc[ind]["trialcode"])
            self.grammarmatlab_extract_beh_and_task(ind, True)
            assert False

        if online_abort_but_sequence_correct_so_far:
            return "online_abort_but_sequence_correct_so_far"
        elif online_abort_but_sequence_correct_complete:
            return "online_abort_but_sequence_correct_complete"
        elif sequence_correct:
            return "sequence_correct"
        elif sequence_incorrect_online_abort:
            return "sequence_incorrect_online_abort" 
        elif sequence_incorrect_but_no_abort:
            # usually is probes
            return "sequence_incorrect_but_no_abort"
        elif done_early_but_sequence_correct_so_far:
            # usually is probes
            return "done_early_but_sequence_correct_so_far"
        else:
            assert False

    # def sequence_extract_beh_and_task(self, ind, ploton=False):
    def grammarmatlab_extract_beh_and_task(self, ind, ploton=False):
        """ Wrapper to extract behavior (taskstrokes ordered by beh) and 
        task (e.g., taskstroke inds ordered by chunk, and whether there is color
        supervision)
        NOTE: 
        - if you havent yet run: supervision_summarize_into_tuple, then this runs automatically.
        """

        # 1) Get beh sequence (i.e., sequence of taskstroke inds, based on order gotten by beh)
        taskstroke_inds_beh_order = self.behclass_extract_taskstroke_inds_in_beh_order(ind)
        if ploton:
            Beh = self.Dat.iloc[ind]["BehClass"]
            datsegs = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", plot=False)
            for x in datsegs:
                print(x)
            Beh.alignsim_plot_summary()

            print("*** Behavior order: ", taskstroke_inds_beh_order)
        # 2) Get target sequence
        out = self.objectclass_wrapper_extract_sequence_chunk_color(ind, ploton)
        C = out["active_chunk"]
        if C is not None:
            taskstroke_inds_correct_order = C.extract_strokeinds_as("flat")
        else:
            taskstroke_inds_correct_order = None
        if ploton:
            print("*** Correct order: ", taskstroke_inds_correct_order)

        # 3) What there sequence supervision?
        if "supervision_stage_concise" not in self.Dat.columns: # KEEP THIS - or else runs a lot
            self.supervision_summarize_into_tuple("concise", new_col_name = "supervision_stage_concise")
        supervision_tuple = self.Dat.iloc[ind]["supervision_stage_concise"]

        # COLLECT all
        gramdict = {}
        gramdict["taskstroke_inds_beh_order"] = taskstroke_inds_beh_order
        gramdict["taskstroke_inds_correct_order"] = taskstroke_inds_correct_order
        gramdict["active_chunk"] = C
        gramdict["supervision_tuple"] = supervision_tuple
        gramdict["epoch"] = self.Dat.iloc[ind]["epoch"]

        return gramdict


    #################
    # def seqcontext_replace_gridloc_with_recomputed_loc_chars(self):
    #     """
    #     BEcuase for "character" task_kind the gridloc is not defined, here
    #     replace gridloc in seqc variables with the recomputed gridloc based
    #     on stroke onset, binned within chars only.
    #     Does this only for seqc_{i}_loc variables.
    #     Copies seqc_{i}_locon_bin_in_loc --> seqc_{i}_loc
    #
    #     And finally regenerates shape x loc variables.
    #
    #     :return: modify self.Dat
    #     """
    #     from pythonlib.tools.pandastools import append_col_with_grp_index
    #
    #     assert False, "this is garvbage. should modify Tokens"
    #
    #     # BASE = 10
    #     n_strokes = self.seqcontext_count_n_strokes()
    #     # For all tokens, if is char, replace gridloc with gridloc_within
    #     inds = self.Dat["task_kind"]=="character"
    #     for n in range(n_strokes):
    #         if False: # This kept failing..,.
    #             # add 10, so that doesnt get confused with original gridloc
    #             # (i.e., these are different coordinate systems)
    #             tmp = np.stack(self.Dat.loc[inds, f"seqc_{n}_locon_bin_in_loc"])+BASE
    #             # D.Dat.loc[inds, f"seqc_{n}_loc"] = pd.Series([tuple(x) for x in tmp.tolist()])
    #             self.Dat.loc[inds, f"seqc_{n}_loc"] = [tuple(x) for x in tmp.tolist()]
    #         else:
    #             self.Dat.loc[inds, f"seqc_{n}_loc"] = self.Dat.loc[inds, f"seqc_{n}_locon_bin_in_loc"]
    #
    #     # Regenerate shape-loc variables, now that loc is updated.
    #     for n in range(n_strokes):
    #         self.Dat = append_col_with_grp_index(self.Dat, [f"seqc_{n}_shape", f"seqc_{n}_loc"], f"seqc_{n}_shapeloc")
    #         self.Dat = append_col_with_grp_index(self.Dat, [f"seqc_{n}_shape", f"seqc_{n}_loc"], f"seqc_{n}_loc_shape")

    def seqcontext_count_n_strokes(self):
        """
        Count n strokes exist as columns in self.Dat with names like
        "seqc_2_shape".
        E.g., If "seqc_2_shape" exists but "seqc_3_shape" does not,
        then returns 3.
        Returns 0 if no columns exist at all.
        :return: n_strokes, int.
        """
        n_strokes = 0
        for i in range(100):
            if f"seqc_{i}_shape" in self.Dat.columns:
                n_strokes = i+1
            else:
                return n_strokes
        assert False

    def seqcontext_plot_examples_and_print_context(self, nexamples=10, nstrokes_max=5):
        """ Quick ehlper to plot n trials, and also print their sequence context
        (shapes) for inspection that things are alright.
        Best run in Notebook
        """

        # Plot examples
        fig, axes, inds = self.plotMultTrials2(nexamples)
        display(self.Dat.loc[inds, ["trialcode"] + [f"seqc_{i}_shape" for i in range(nstrokes_max)] + ["charclust_shape_seq_scores"]])
        display(self.Dat.loc[inds, ["trialcode"] + [f"seqc_{i}_loc" for i in range(nstrokes_max)]])

    def seqcontext_behseq_cluster_concrete_variation(self, SAVEDIR, LIST_VAR_BEHORDER=None,
                                                     # groupby="taskcat_by_rule",
                                                     groupby="FEAT_num_strokes_task", # 4/11/24 - this is better since it generalizes across syntax stuff.
                                                     DEBUG=False):
        """
        Classify trials based on their "concrete order", such as the specific sequence of shapes, locations, or
        directions (location diffs).
        First, gets all trials within each level of <groupby>, and then conputes pairwise dist mat across all trials using
        their beh sequences in <var_behorder>, then clusters that distmat to assign each trial a label.
        So, if <var_behorder> is "behseq_locs" and <grouby> is "taskcat_by_rule" then classifies trials with similar
        sequence of gridlocs within each taskcat_by_rule
        :param LIST_VAR_BEHORDER: list of variables (beh seq) to to clustering for, iterating over them
        :param groupby:
        e.g., FEAT_num_strokes_task
        :param plot_savedir:
        :return: new columns for each var_behorder in LIST_VAR_BEHORDER, called var_behorder_clust, using str ints,
        "0", "1", ...

        # e..g, "behseq_shapes_clust", "behseq_locs_clust"
        """
        from pythonlib.tools.distfunctools import distmat_construct_wrapper
        from neuralmonkey.analyses.state_space_good import dimredgood_pca
        from pythonlib.tools.statstools import cluster_kmeans_with_silhouette_score
        from pythonlib.tools.plottools import makeColors

        MIN_TRIALS = 8 # if less than this, then all call same cluster.
        EPSILON = 0.005 # very small value, so if just few uniqe vals, calustering still works.

        def _decide_n_clust(dfthis, var_behorder):
            if False:
                # Old, too often got 2 when should force more
                n_clusters_min_max = [2, 6] # err on side of fewer clusters, to get more trials per.
            else:
                # Force to get more clusters...

                # - get tabulation of counts
                x = dfthis[var_behorder].value_counts().reset_index()

                # - n clust
                max_n_clust = int(np.min([
                    len(x), # cannot have more clusts than the num unique items
                    len(dfthis)/10, # shoot for at least 10 trials per clust.
                    20, # a manually defined hard cap
                ]))
                max_n_clust = np.max([2, max_n_clust]) # or else fails sillhoute.

                min_n_clust = int(np.max([
                    3, # silhoutte score will fail if 1. make 3 to get more.
                    len(dfthis)/20, # if have lots of data, then force partitioning into clusters
                ]))
                min_n_clust = np.min([min_n_clust, max_n_clust])
                if DEBUG:
                    print("Ntot, nunique:", len(dfthis), len(x))
                # print(x)
                n_clusters_min_max = [min_n_clust, max_n_clust]
            return n_clusters_min_max

        if LIST_VAR_BEHORDER is None:
            LIST_VAR_BEHORDER = ["behseq_shapes", "behseq_locs", "behseq_locs_diff"]
                                 # "behseq_locs_diff_x", "behseq_locs_diff_y"]
                                 # "behseq_locs_x", "behseq_locs_y", #  (is too slow).

        assert groupby in self.Dat.columns, "need to extract thi sfirst.."

        def _dist(x1, x2):
            """ Compute scalar distance between pairs of trials's sequences"""
            assert len(x1)==len(x2)
            if isinstance(x1[0], int) and isinstance(x2[0], int):
                # e.g., (-1,1, 10), (1,2, 1)
                # l2 distance
                return np.linalg.norm(np.asarray(x1) - np.asarray(x2))
                # return sum([np.abs(xx2 - xx1) for xx1, xx2 in zip(x1, x2)])
            elif isinstance(x1[0], str) and isinstance(x2[0], str):
                # e.g., ("test", "2"), ("test", "1")
                return sum([xx2==xx1 for xx1, xx2 in zip(x1, x2)])
            elif isinstance(x1[0], tuple) and isinstance(x1[0][0], int):
                # e.g,     ((1,1), (1,2), (-1,-2)) vs ((1,2), (0,-2), (-1,2))
                return sum([_dist(xx1, xx2) for xx1, xx2 in zip(x1, x2)])
            else:
                print(x2)
                print(x2)
                print(type(x1[0]))
                print(type(x2[0]))
                assert False, "code it"

        ### First, extract sequences, keeping them as tuples, for easy distance computation.
        # 1) shape seuqence
        self.seqcontext_extract_shapes_in_beh_order_append_column(abbrev_string_code=False)

        # 2) Location sequence
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only=None, colname="behseq_locs", abbrev_string_code=False)
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only="x", colname="behseq_locs_x", abbrev_string_code=False)
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only="y", colname="behseq_locs_y", abbrev_string_code=False)

        # 3) Direction sequence (location diff)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only=None, colname="behseq_locs_diff", abbrev_string_code=False)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only="x", colname="behseq_locs_diff_x", abbrev_string_code=False)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only="y", colname="behseq_locs_diff_y", abbrev_string_code=False)

        for var_behorder in LIST_VAR_BEHORDER:

            plot_savedir = f"{SAVEDIR}/var_behorder={var_behorder}"
            os.makedirs(plot_savedir, exist_ok=True)

            ##### Intialize labels
            # so that stays string. And can do sanity check after, cant have emptys.
            self.Dat[f"{var_behorder}_clust"] = "empty"

            ##### Group
            for grplev in sorted(self.Dat[groupby].unique().tolist()):

                print("Clustering, for ", grplev)
                # First, collect all trials of a given taskcatbyrule
                dfthis = self.Dat[self.Dat[groupby] == grplev]
                vals = dfthis[var_behorder].tolist()
                idxs = dfthis.index.tolist()

                n_clusters_min_max = _decide_n_clust(dfthis, var_behorder)
                print(var_behorder, ".. Using this min/max n clusters: ", n_clusters_min_max)
                # Get cluster labels.
                if len(set(vals))==1 or len(vals) < MIN_TRIALS:
                    # Then all are same cluster
                    cluster_labels = ["0" for _ in range(len(vals))]
                    # If there are only a few unique cases, then just call those clusters
                else:
                    dmat = distmat_construct_wrapper(vals, vals, _dist, PLOT=False)

                    # add very small value, so that clustering does not fail.
                    dmat = dmat + EPSILON * np.random.rand(dmat.shape[0], dmat.shape[1])


                    # do pca on distmat
                    plot_pca_explained_var_path = f"{plot_savedir}/{groupby}-lev={grplev}-pca_var.pdf"
                    plot_loadings_path = f"{plot_savedir}/{groupby}-lev={grplev}-pca_loadings.pdf"
                    Xpcakeep, Xpca, pca = dimredgood_pca(dmat, n_components=2, plot_pca_explained_var_path=plot_pca_explained_var_path,
                                                         plot_loadings_path=plot_loadings_path, npcs_keep_force=2)

                    # Cluster distance matrix
                    cluster_labels, fig, fig_final = cluster_kmeans_with_silhouette_score(Xpcakeep, n_clusters_min_max=n_clusters_min_max,
                                                                          PLOT_SIL=False, PLOT_FINAL=True,
                                                                          return_figs=True)
                    savefig(fig_final, f"{plot_savedir}/{groupby}-lev={grplev}-clust_final.pdf")

                # Sanity check that sequence matches eye test
                if False:
                    print(vals[0])
                    print(idxs[0])
                    self.grammarparses_print_plot_summarize(idxs[0])

                #### PLOT RESULTS
                pcols = makeColors(len(set(cluster_labels)))
                ncols = 3
                nrows = int(np.ceil(len(set(cluster_labels))/ncols))
                SIZE = 3
                fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*(1.25*SIZE), nrows*SIZE), sharex=True, sharey=True)
                for labget, ax in zip(sorted(set(cluster_labels)), axes.flatten()):
                    valsthis = [vals[i] for i, lab in enumerate(cluster_labels) if lab == labget]
                    for v in valsthis:
                        if isinstance(v[0], int):
                            v_jitter = np.array(v) + 0.25*np.random.rand(len(v))
                            ax.plot(range(len(v)), v_jitter, "-o", label=labget, color=pcols[int(labget)], alpha=0.1)
                        elif isinstance(v[0], tuple) and isinstance(v[0][0], int):
                            # 2d plot
                            v_jitter = np.array(v) # (npts, 2)
                            v_jitter = v_jitter + 0.25*np.random.rand(v_jitter.shape[0], v_jitter.shape[1])
                            ax.plot(v_jitter[:, 0], v_jitter[:,1], "-", label=labget, color="k", alpha=0.1)
                            ax.scatter(v_jitter[:, 0], v_jitter[:,1], c=range(v_jitter.shape[0]), label=labget, alpha=0.4, cmap="plasma")
                            for i, pt in enumerate(np.array(v)):
                                ax.text(pt[0], pt[1], i, alpha=0.2, fontsize=15)
                        elif isinstance(v[0], str):
                            ax.plot(range(len(v)), v, "-o", label=labget, color=pcols[int(labget)], alpha=0.1)
                        else:
                            assert False
                    # ax.plot(valsthis, "-o", label=labget, color=pcols[int(labget)])
                    ax.set_title(f"cluster lab: {labget}")
                    ax.set_xlabel("index in trial")
                    ax.set_ylabel("value")
                savefig(fig, f"{plot_savedir}/{groupby}-lev={grplev}-final_cluster_sequences.pdf")

                ##### Save mapping between val and clustlab
                map_clust_vals = {}
                for v, c in zip(vals, cluster_labels):
                    if c in map_clust_vals and v not in map_clust_vals[c]:
                        map_clust_vals[c].append(v)
                    else:
                        map_clust_vals[c] = [v]
                from pythonlib.tools.expttools import writeDictToTxtFlattened
                path = f"{plot_savedir}/{groupby}-lev={grplev}-map_clust_vals.txt"
                writeDictToTxtFlattened(map_clust_vals, path, header="clust | vals", sorted_by_keys=True)

                ##### PLOT some exmaple beh trials
                import random
                if len(idxs)>=20:
                    indspick = sorted(random.sample(range(len(idxs)), 20))
                else:
                    indspick = range(len(idxs))
                idxs_plot = [idxs[i] for i in indspick]
                # vals_plot = [vals[i] for i in indspick]
                clusts_plot = [cluster_labels[i] for i in indspick]
                fig, axes, _ = self.plotMultTrials2(idxs_plot, titles=clusts_plot)
                savefig(fig, f"{plot_savedir}/{groupby}-lev={grplev}-example_drawings.pdf")


                ############## SAVE LABELS
                # print(idxs)
                # print(cluster_labels)
                assert np.all(self.Dat.loc[idxs, f"{var_behorder}_clust"]=="empty"), "you are overwriting... not sure why"
                self.Dat.loc[idxs, f"{var_behorder}_clust"] = cluster_labels

                plt.close("all")

        ######## FINAL THINGS
        # Sanity -- check that no empty labels.
        assert sum(self.Dat[f"{var_behorder}_clust"] == "empty")==0, "not sure why"

        # Reextract beh labels as strings (default)
        # 1) shape seuqence
        self.seqcontext_extract_shapes_in_beh_order_append_column(abbrev_string_code=True)

        # 2) Location sequence
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only=None, colname="behseq_locs", abbrev_string_code=True)
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only="x", colname="behseq_locs_x", abbrev_string_code=True)
        self.seqcontext_extract_locations_in_beh_order_append_column(x_or_y_only="y", colname="behseq_locs_y", abbrev_string_code=True)

        # 3) Direction sequence (location diff)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only=None, colname="behseq_locs_diff", abbrev_string_code=True)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only="x", colname="behseq_locs_diff_x", abbrev_string_code=True)
        self.seqcontext_extract_location_diffs_in_beh_order_append_column(x_or_y_only="y", colname="behseq_locs_diff_y", abbrev_string_code=True)


    def seqcontext_extract_shapes_in_beh_order_append_column(self, colname = "behseq_shapes", abbrev_string_code=True):
        """
        Append column that holds the shape sequence on each trial, i.e. the actual stroke sequence,
        each represnted as a string, like "l840|l830|l840|l830|l840|l830".
        :param colname:
        :return:
        """
        self.Dat[colname] = [self.seqcontext_extract_shapes_in_beh_order(i, abbrev_string_code) for i in range(len(self.Dat))]
        print("Appended column do D.Dat:", colname)

    def seqcontext_extract_shapes_in_beh_order(self, ind, abbrev_string_code=False):
        """ Return list of strings (shapes) drawn on this trial,
        using tokens, based on touch (not just the frist touch, but
        inlcuding all beh strokes)
        PARAMS:
        - abbreg_string_code, returns string like 'l840|l830|l840|l830|l840|l830'
        RETURNS:
            - shapes, tuple of strings.
        """

        def _shape_string_abbreviate(sh):
            """
            - sh, like "line-8-4-0"
            RETURNS:
            - like l840, string
            """
            from pythonlib.tools.stringtools import decompose_string
            tmp = decompose_string(sh)
            if not len(tmp)==4:
                print(sh)
                print(tmp)
                assert False
            shcode = tmp[0][0] + tmp[1] + tmp[2] + tmp[3]
            return shcode

        shapes = tuple([tok["shape"] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")])
        assert len(shapes) == len(self.Dat.iloc[ind]["strokes_beh"])

        if abbrev_string_code:
            shapes = "|".join([_shape_string_abbreviate(sh) for sh in shapes])

        return shapes

    def seqcontext_extract_locations_in_beh_order_append_column(self, x_or_y_only=None, colname = "behseq_locs", abbrev_string_code=True):
        """
        Append column that holds the shape sequence on each trial, i.e. the actual stroke sequence,
        each represnted as a string, like "l840|l830|l840|l830|l840|l830".
        :param colname:
        :return:
        """
        self.Dat[colname] = [self.seqcontext_extract_gridloc_in_beh_order(i, abbrev_string_code, x_or_y_only=x_or_y_only) for i in range(len(self.Dat))]
        print("Appended column do D.Dat:", colname)

    def seqcontext_extract_location_diffs_in_beh_order_append_column(self, x_or_y_only=None, colname = "behseq_locs_diff", abbrev_string_code=True):
        """
        Append column that holds the shape sequence on each trial, i.e. the actual stroke sequence,
        each represnted as a string, like "l840|l830|l840|l830|l840|l830".
        :param colname:
        :return:
        """
        self.Dat[colname] = [self.seqcontext_extract_gridloc_diff_in_beh_order(i, abbrev_string_code, x_or_y_only=x_or_y_only) for i in range(len(self.Dat))]
        print("Appended column do D.Dat:", colname)

    def seqcontext_extract_gridloc_diff_in_beh_order(self, ind, abbrev_string_code=False,
                                                x_or_y_only=None):
        """
        Return sequence of location differences from previous stroke, in beh stroke order, with methods to simplify and/or take specific dimensions (x or y)
        PARAMS:
        - abbrev_string_code, bool, if True, then returns string, like
        '2,0|1,1|0,0|-1,-1|1,-1|-2,0|-1,1' or '2|1|0|-1|1|-2|-1'
        - x_or_y_only, eithe rNone(get both xa nd y) or x or y, in latter 2 cases get int, in former gets tuple of 2 ints.
        """

        # Get gridloc differences.
        Tk = self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True)
        Tk.sequence_gridloc_direction()

        if x_or_y_only=="x":
            locations = tuple([tok["gridloc_rel_prev"][0] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")][1:])
        elif x_or_y_only=="y":
            locations = tuple([tok["gridloc_rel_prev"][1] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")][1:])
        else:
            assert x_or_y_only is None, "either x, y or None"
            locations = tuple([tok["gridloc_rel_prev"] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")][1:])

        assert len(locations) == len(self.Dat.iloc[ind]["strokes_beh"]) - 1

        if abbrev_string_code:
            if x_or_y_only is None:
                locations = "|".join([",".join([str(l) for l in loc]) for loc in locations])
            else:
                locations = "|".join([str(loc) for loc in locations])

        return locations

    def seqcontext_extract_gridloc_in_beh_order(self, ind, abbrev_string_code=False,
                                                x_or_y_only=None):
        """
        Return sequence of location in beh stroke order, with methods to simplify and/or take specific dimensions (x or y)
        PARAMS:
        - abbrev_string_code, bool, if True, then returns string, like
        '2,0|1,1|0,0|-1,-1|1,-1|-2,0|-1,1' or '2|1|0|-1|1|-2|-1'
        - x_or_y_only, eithe rNone(get both xa nd y) or x or y, in latter 2 cases get int, in former gets tuple of 2 ints.
        """

        if x_or_y_only=="x":
            locations = tuple([tok["gridloc"][0] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")])
        elif x_or_y_only=="y":
            locations = tuple([tok["gridloc"][1] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")])
        else:
            assert x_or_y_only is None, "either x, y or None"
            locations = tuple([tok["gridloc"] for tok in self.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")])

        assert len(locations) == len(self.Dat.iloc[ind]["strokes_beh"])

        if abbrev_string_code:
            if x_or_y_only is None:
                locations = "|".join([",".join([str(l) for l in loc]) for loc in locations])
            else:
                locations = "|".join([str(loc) for loc in locations])

        return locations

    def seqcontext_delete_all_columns(self):
        """ Helper to delete all columns from self.Dat which contain string 'seqc'
        """
        cols_delete = [col for col in self.Dat.columns if "seqc_" in col]
        print("Deleting these columns with seqc in name:", cols_delete)
        self.Dat = self.Dat.drop(cols_delete, axis=1)

    def seqcontext_preprocess(self, plot_examples=False, force_run=False):
        """ Extract new columns into self.Dat, for each trial noting sequence 
        context inforamtion ,such as n strokes, shape of first stroke, etc
        """

        # First, clear old values.
        self.seqcontext_delete_all_columns()

        # if "seqc_0_shape" not in self.Dat.columns or force_run:
        from pythonlib.dataset.dataset_preprocess.seqcontext import preprocess_dataset
        preprocess_dataset(self, plot_examples=plot_examples)

        # also extract gridsize
        list_gridsize = []
        for i in range(len(self.Dat)):
            T = self.Dat.iloc[i]["Task"]
            list_gridsize.append(T.PlanDat["TaskGridClass"]["Gridname"])
        self.Dat["gridsize"] = list_gridsize
        print("Appended columns gridsize!")

    ################ SAVE
    def make_path_savedir_directory_notebook_for_figures(self, analysis_name):
        return self.make_savedir_for_analysis_figures(analysis_name)
        
    def make_savedir_for_analysis_figures_BETTER(self, analysis_name):
        """ BETTER means:
        ii) is animal_date_expt instead of animal_expt_date. This is easier for sorting.
        """
        return self.make_savedir_for_analysis_figures(analysis_name, date_first=True)

    def make_savedir_for_analysis_figures(self, analysis_name, date_first=False):
        from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
        # animal = self.animals()
        # expt = self.expts()
        # rulelist = self.rules()

        idstring = self.save_generate_string_identifier_wrapper(concise=False, 
            date_first=date_first)
        if len(idstring)>35:
            # get concise
            idstring = self.save_generate_string_identifier_wrapper(concise=True, date_first=date_first)

        SDIR_MAIN = f"{PATH_ANALYSIS_OUTCOMES}/main/{analysis_name}/{idstring}"
        print("SAVING at: ", SDIR_MAIN)
        os.makedirs(SDIR_MAIN, exist_ok=True)
        return SDIR_MAIN


    def make_fig_savedir_suffix(self, filterDict):
        """ wrapper to help make suffix for saving, encoding
        animal, expt, and filter params in filterDict (which assumed
        to be used in self.filterPandas to filkter datraset)
        """

        suff = f"{'_'.join(self.animals())}-{'_'.join(self.expts())}"

        for k, v in filterDict.items():
            if isinstance(v[0], bool):
                v = [str(vv) for vv in v]
            if k=="insummarydates":
                suff += f"-insumd_{'_'.join(v)}"
            elif k=="random_task":
                suff += f"-rndmt_{'_'.join(v)}"
            elif k=="monkey_train_or_test":
                suff += f"-mktt_{'_'.join(v)}"
            elif k=="taskgroup":
                suff += f"-tskgrp_{'_'.join(v)}"
            else:
                print("dont know this: ", k)

        return suff

    def save(self, savedir, columns_to_keep = None):
        """ saves dataset (the entire object) in
        savedir/dataset_beh.pkl
        """

        if columns_to_keep is not None:
            D = self.copy()
            D.Dat = D.Dat.loc[:, columns_to_keep]
        else:
            D = self

        import pickle
        path = f"{savedir}/dataset_beh.pkl"
        with open(path, "wb") as f:
            pickle.dump(D, f)

    def save_state(self, SDIR_MAIN, SDIR_SUB, add_tstamp = True):
        """
        RETURNS:
        - SDIR_THIS, dir where this saved.
        """
        import os

        if add_tstamp:
            ts = makeTimeStamp()
            # os.makedirs(SDIR_MAIN, exist_ok=True)
            SDIR_THIS = f"{SDIR_MAIN}/{SDIR_SUB}-{ts}"
        else:
            SDIR_THIS = f"{SDIR_MAIN}/{SDIR_SUB}"
        os.makedirs(SDIR_THIS, exist_ok=True)
        print(SDIR_THIS)

        # Dat
        self.Dat.to_pickle(f"{SDIR_THIS}/Dat.pkl")

        # Metadats
        with open(f"{SDIR_THIS}/Metadats.pkl", "wb") as f:
            pickle.dump(self.Metadats, f)
        
        # BPL
        if hasattr(self, "BPL"):    
            with open(f"{SDIR_THIS}/BPL.pkl", "wb") as f:
                pickle.dump(self.BPL, f)

        return SDIR_THIS


    ############# PLOTS
    def plotMP(self, MP, ax):
        """ helper for plotting motor program
        """
        from ..bpl.strokesToProgram import plotMP
        plotMP(MP, ax=ax)


    def _plot_single_trial_highlighting_overlaid_strokes(self, ind, strokes_overlay, 
            ax=None, underlay_beh_or_task="task",
            underunderlay_task=True, underunderlay_color = "w",
            overlay_single_color="r", underlay_single_color="k"):
        """ Low-level plot of single trial with overlaid strokes (any). Useful if want to highlight
        subset of strokes 
        PARAMS:
        - strokes_overlay, list of np arary.
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper, plotDatStrokes
        if ax is None:
            fig, ax = plt.subplots(1,1)

        if underlay_beh_or_task=="beh" and underunderlay_task:
            # then underlay with a faint task
            self.plot_single_trial(ind, ax, single_color=underunderlay_color, ver="task")

        # plot trial
        if underlay_beh_or_task=="beh":
            self.plot_single_trial(ind, ax, single_color=underlay_single_color)
        elif underlay_beh_or_task=="task":
            self.plot_single_trial(ind, ax, single_color=underlay_single_color, ver="task")
        else:
            assert False

        plotDatStrokesWrapper(strokes_overlay, ax, color=overlay_single_color, 
            add_stroke_number=False)    

    def plot_single_trial_highlighting_overlaid_strokes(self, ind, ind_strokes, 
            ax=None, overlay_beh_or_task="beh", underlay_beh_or_task="task",
            overlay_single_color="r", underlay_single_color="k"):

        if overlay_beh_or_task=="beh":
            strokes = self.Dat.iloc[ind]["strokes_beh"]
        elif overlay_beh_or_task=="task":
            strokes = self.Dat.iloc[ind]["strokes_task"]
        else:
            assert False

        strokes_overlay = [strokes[i] for i in ind_strokes]
        return self.plot_single_trial_highlighting_overlaid_strokes(ind, strokes_overlay,
            ax, underlay_beh_or_task, overlay_single_color=overlay_single_color, 
            underlay_single_color=underlay_single_color)

        # if ax is None:
        #     fig, ax = plt.subplots(1,1)

        # # plot trial
        # if underlay_beh_or_task=="beh":
        #     self.plot_single_trial(ind, ax, single_color=underlay_single_color)
        # elif underlay_beh_or_task=="task":
        #     self.plot_single_trial(ind, ax, single_color=underlay_single_color, ver="task")
        # else:
        #     assert False

        # if overlay_beh_or_task=="beh":
        #     self.plot_single_trial(ind, ax, single_color=overlay_single_color, 
        #         strok_indices=ind_strokes, add_stroke_number_beh=False)
        # elif overlay_beh_or_task=="task":
        #     assert False, "cant do this! ind_strokes is defined wrt to beh strokes..."
        #     # self.plot_single_trial(ind, ax, single_color="r", ver="task", strok_indices=ind_strokes)
        # else:
        #     assert False

    def plot_strokes(self, strokes, ax=None, single_color=None, 
            ver="beh", strok_indices=None, add_stroke_number_beh=True,  alpha=0.55):
        """ Simple low-level helper for plotting strokes
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper, plotDatStrokes
        
        if ax is None:
            fig, ax = plt.subplots(1,1)

        if strok_indices:
            strokes = [strokes[i] for i in strok_indices]

        if ver=="beh":
            plotDatStrokesWrapper(strokes, ax, color=single_color, 
                add_stroke_number=add_stroke_number_beh,  alpha=alpha)    
        elif ver=="task":
            plotDatStrokes(strokes, ax, each_stroke_separate=True, 
                plotver="onecolor", add_stroke_number=False, 
                mark_stroke_onset=False, pcol=single_color, number_from_zero=False, alpha=alpha)
        else:
            assert False

    def plot_mult_trials_overlaid_on_axis(self, inds, ax, 
                                          single_color=None, 
            ver="beh", strok_indices=None, add_stroke_number_beh=True, alpha=0.55,
            nrand=None):
        """
        As title says...
        """
        if nrand is not None and len(inds)>nrand:
            import random
            inds = sorted(random.sample(inds, nrand))

        for idx in inds:
            self.plot_single_trial(idx, ax, single_color, ver,
            strok_indices, add_stroke_number_beh, alpha=alpha)

    def plot_single_trial(self, idx, ax=None, single_color=None, 
            ver="beh", strok_indices=None, add_stroke_number_beh=True,  alpha=0.55):
        """ Low-level code to plot a single trial, beh or task strokes, on an axis.
        PARAMS:
        - single_color, either None (colors by ordinal), or str color code
        - ver, str, {'task', 'beh'}
        """
        if ver=="beh":
            strokes = self.Dat.iloc[idx]["strokes_beh"]
        elif ver=="task":
            strokes = self.Dat.iloc[idx]["strokes_task"]
        else:
            assert False
        return self.plot_strokes(strokes, ax, single_color, ver, strok_indices, 
            add_stroke_number_beh,  alpha=alpha)

    def plot_trials_after_slicing_within_range_values(self, colname, minval, 
        maxval, plot_hist=True, nrand=20):
        """ Plot example trials that are wihitn this range of values for
        a given column, e.g,., frac_touched
        """

        if plot_hist:
            self.Dat[colname].hist()
        # d1 = 0.6
        # d2 = 0.7
        inds = self.Dat[(self.Dat[colname]>minval) & (self.Dat[colname]<maxval)].index.tolist()
        print("This many trials found:", len(inds))
        fig, axes, idxs = self.plotMultTrials(inds, nrand=nrand)
        titles = self.Dat.iloc[idxs][colname].tolist()
        self.plotMultTrials(idxs, "strokes_task", titles=titles)


    def plotSingleTrial(self, idx=None, things_to_plot = ("beh", "task"),
        sharex=True, sharey=True, params=None, task_add_num=False,
        number_from_zero=True):
        """ 
        Plot a single trial
        PARAMS;
        - idx, index into Dat, 0, 1, ... 
        --- if None, then picks a rnadom trial
        - things_to_plot, list of str.
        - params, only matters for some things.
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse, plotDatWaterfall
        
        dat = self.Dat
        if idx is None:
            import random
            idx = random.choice(range(len(dat)))

        assert not isinstance(idx, list)

        # === Plot a single trial
        ncols=4
        nplots = len(things_to_plot)
        nrows = int(np.ceil(nplots/ncols))

        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=(ncols*3, nrows*3))

        for thing, ax in zip(things_to_plot, axes.flatten()):
            if thing=="beh":
                # Plot behavior for this trial
                # - the ball marks stroke onset. stroke orders are color coded, and also indicated by numbers.
                # strokes = dat["strokes_beh"].values[idx]
                # plotDatStrokes(strokes, ax, each_stroke_separate=True, number_from_zero=number_from_zero)
                self.plot_single_trial(idx, ax)
            elif thing=="task":
                # overlay the stimulus
                stim = dat["strokes_task"].values[idx]
                if task_add_num:
                    add_stroke_number=True
                    mark_stroke_onset=True
                else:
                    add_stroke_number=False
                    mark_stroke_onset=False

                plotDatStrokes(stim, ax, each_stroke_separate=True, 
                    plotver="onecolor", add_stroke_number=add_stroke_number, 
                    mark_stroke_onset=mark_stroke_onset, pcol="k", number_from_zero=number_from_zero)
            elif thing=="beh_tc":
                # === Plot motor timecourse for a single trial (smoothed velocity vs. time)
                # fig, axes = plt.subplots(1,1, sharex=True, sharey=True)
                strokes = dat["strokes_beh"].values[idx]
                plotDatStrokesTimecourse(strokes, ax, plotver="vel", label="vel (pix/sec)")
            elif thing=="parse":
                if "parses" in dat.columns:
                    if "strokes_parse" not in dat.columns:
                        assert False, "need to tell me which pares to plot, run self.parsesChooseSingle()"
                    strokes = dat["strokes_parse"].values[idx]
                    plotDatStrokes(strokes, ax, each_stroke_separate=True, number_from_zero=number_from_zero)
            elif thing=="bpl_mp":
                if "motor_program" in dat.columns:
                    # from ..bpl.strokesToProgram import plotMP
                    MP = dat["motor_program"].values[idx]
                    # plotMP(MP, ax=ax)
                    self.plotMP(MP, ax=ax)
            elif thing=="parse_frompool_mp":
                # Then you need to tell me which mp (0, 1, ...)
                ind = params["parse_ind"]
                MP = dat["parses_motor_programs"].values[idx][ind]
                self.plotMP(MP, ax=ax)

            elif thing=="beh_splines":
                assert "strokes_beh_splines" in dat.columns, "need to run convertToSplines first!"
                strokes = dat["strokes_beh_splines"].values[idx]
                plotDatStrokes(strokes, ax, each_stroke_separate=True, number_from_zero=number_from_zero)
            else:
                print(thing)
                assert False

            ax.set_title(thing)
        return fig


    def plotMultStrokesByOrder(self, strokes_list, ncols = 5, titles=None, 
        naked_axes=False, mark_stroke_onset=True, add_stroke_number=True, 
        titles_on_y=False, SIZE=2.5, plotkwargs={}):
        """ Helper to call plotMultStrokesColorMap so that plots by order
        """
        colors = [np.arange(0, len(s)) for s in strokes_list]
        return self.plotMultStrokesColorMap(strokes_list, colors, 
            ncols, titles, naked_axes, mark_stroke_onset,
            add_stroke_number, titles_on_y, SIZE, plotkwargs)



    def plotMultStrokesColorMap(self, strokes_list, strokes_vals_list, 
        ncols = 5, titles=None, naked_axes=False, mark_stroke_onset=False,
        add_stroke_number=True, titles_on_y=False, SIZE=2.5, plotkwargs={}):
        """ helper to plot multiplie trials when already have strokes extracted)
        Assumes want to plot this like behavior.
        """
        from ..drawmodel.strokePlots import plotDatStrokesMapColor
        import random
        from pythonlib.tools.plottools import plotGridWrapper

        plotfunc = lambda strokes_strokesvals, ax: plotDatStrokesMapColor(
            strokes_strokesvals[0], ax, strokes_strokesvals[1],
            naked_axes=naked_axes, mark_stroke_onset=mark_stroke_onset, 
            add_stroke_number=add_stroke_number, **plotkwargs)

        list_data = [(strokes, strokes_vals) for strokes, strokes_vals in 
            zip(strokes_list, strokes_vals_list)]
        
        fig, axes= plotGridWrapper(list_data, plotfunc, ncols=ncols, titles=titles,
            naked_axes=naked_axes, origin="top_left", titles_on_y=titles_on_y, 
            SIZE=SIZE, return_axes=True)

        return fig, axes


    def plotMultStrokes(self, strokes_list, ncols = 5, titles=None, naked_axes=False, 
        add_stroke_number=True, centerize=False, jitter_each_stroke=False, 
        titles_on_y=False, SIZE=2.5, is_task=False, number_from_zero=True,
        color_by = "by_order"):
        """ helper to plot multiplie trials when already have strokes extracted)
        Assumes want to plot this like behavior.
        """
        from ..drawmodel.strokePlots import plotDatStrokes
        import random
        from pythonlib.tools.plottools import plotGridWrapper
    
        if is_task:
            plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, 
                add_stroke_number=add_stroke_number, centerize=centerize, 
                jitter_each_stroke=jitter_each_stroke, number_from_zero=number_from_zero)
        else:
            plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, 
                add_stroke_number=add_stroke_number, centerize=centerize, 
                jitter_each_stroke=jitter_each_stroke, number_from_zero=number_from_zero)
            
        fig, axes= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left",
            titles_on_y=titles_on_y, SIZE=SIZE, return_axes=True)

        return fig, axes



    def _extractStrokeVels(self, list_strokes, remove_time_column=True,
            version = "speed", fs_downsample=None, DEBUG=False,
                           lowpass_freq_force=None, PLOT=False):
        """ extract stroke as instantaneous velocities
        INPUT:
        - list_inds, list of ints, for which strokes to use. If "all", then gets all.
        - version, str, in {'vel', 'speed'}, determines which to reutrn
        - lowpass_freq_force, if not NOne, then freq for lowpass filtering the
        output vels trace.
        RETURN:
        - list of strokes_vels, which are each list of Nx1, so is
        actually speed, not vel. if 'vel', then they are (N,3), if 'speed'
        then (N,2). Remoive one column if remove time.,
        (None, if strok to oshort to get vel.)
        """
        from pythonlib.tools.stroketools import strokesVelocity

        assert fs_downsample is None, "obsolete"

        if lowpass_freq_force is not None:
            clean = False
        else:
            clean = True

        # print("here", fs_downsample)
        # assert False
        assert isinstance(list_strokes[0], list)
        fs = self.get_sample_rate_alltrials()
        list_strokes_vel = []
        for strokes in list_strokes:
            # strokes_vel, strokes_speed = strokesVelocity(strokes, fs,
            #     fs_new = fs_downsample,  DEBUG=DEBUG,
            #     SKIP_POST_FILTERING_LOWPASS=SKIP_POST_FILTERING_LOWPASS,
            #      lowpass_freq=lowpass_freq_force)
            strokes_vel, strokes_speed = strokesVelocity(strokes, fs, ploton=PLOT,
                                                         DEBUG=DEBUG,
                                                         clean=clean,
                                                         lowpass_freq=lowpass_freq_force)

            if version=="speed":
                strokes_out = strokes_speed
                dims = [0]
            else:
                strokes_out = strokes_vel
                dims = [0,1]

            if remove_time_column:
                strokes_out = [s[:,dims] for s in strokes_out] # remove time

            # Any cases that are empyt vels...
            for i in range(len(strokes_out)):
                if any([np.any(np.isnan(sv)) for sv in strokes_out[i]]):
                    assert False, "this shouldnt happen anymore (1/4/24) after cleaning up vels comptuation"
                    list_strokes_vel.append(None)   
                else:
                    list_strokes_vel.append(strokes_out)
        return list_strokes_vel


    def extractStrokeVels(self, list_inds, version="speed"):
        """ extract stroke as instantaneous velocities
        INPUT:
        - list_inds, list of ints, for which strokes to use. If "all", then gets all.

        RETURN:
        - list of strokes_vels, which are each list of Nx1, so is
        actually speed, not vel.
        (None, if strok to oshort to get vel.)
        """
        from pythonlib.tools.stroketools import strokesVelocity
  
        if list_inds == "all":
            list_inds = self.Dat.index.tolist()
        list_strokes = self.Dat.iloc[list_inds]["strokes_beh"].tolist()
        return self._extractStrokeVels(list_strokes, version=version)
        # list_strokes_vel = []
        # for ind in list_inds:
        #     strokes = self.Dat.iloc[ind]["strokes_beh"]
        #     fs = self.get_sample_rate(ind)
        #     _, strokes_vel = strokesVelocity(strokes, fs, clean=True)
        #     strokes_vel = [s[:,0] for s in strokes_vel] # remove time
        #     for i in range(len(strokes_vel)):
        #         if any([np.isnan(sv) for sv in strokes_vel[i]]):
        #             list_strokes_vel.append(None)   
        #             # print(strokes_vel)
        #             # print(list_inds)
        #             # print(strokes[i])
        #             # assert False
        #         else:
        #             list_strokes_vel.append(strokes_vel)
        # return list_strokes_vel


    def _plot_prepare_strokes(self, which_strokes, idxs, nrand=None, titles=None):
        """
        Helper to extract strokes
        """
        import random

        if isinstance(idxs, int):
            N = len(self.Dat)
            k = idxs
            idxs = random.sample(range(N), k=k)

        if len(idxs)==0:
            return

        if nrand is not None:
            if nrand < len(idxs):
                from pythonlib.tools.listtools import random_inds_uniformly_distributed
                tmp = random_inds_uniformly_distributed(idxs, nrand)
                idxs = [idxs[i] for i in tmp]
                # idxs = sorted(random.sample(idxs, nrand))

        if which_strokes=="parses":
            # then pull out a random parse for each case
            assert "parses" in self.Dat.columns, "need to extract parses first..."
            strokes_list = [[a.copy() for a in strokes] for strokes in self.Dat["parses"].values] # copy, or else will mutate
            # for the indices you want to plot, prune to just a single parse
            for i in idxs:
                ithis = random.randrange(len(strokes_list[i]))
                strokes_list[i] = strokes_list[i][ithis]
        else:
            strokes_list = self.Dat[which_strokes].values

        # Keep only data to plot.
        strokes_list = [strokes_list[i] for i in idxs]

        if titles is None:
            # use trialcodes
            titles = self.Dat.iloc[idxs]["trialcode"].tolist()

        return strokes_list, idxs, titles

    def plotMultTrials2(self, idxs, which_strokes="strokes_beh", nrand=None,
            titles = None, add_stroke_number=True, SIZE=2.5, color_by="order",
            number_from_zero = True, 
            plotkwargs=None):
        """ V2, which uses plotMultStrokes
        """

        if plotkwargs is None:
            plotkwargs = {}

        strokes_list, idxs, titles = self._plot_prepare_strokes(which_strokes, idxs, 
            nrand=nrand, titles=titles)
        trialcodes =  self.Dat.iloc[idxs]["trialcode"].tolist()
 
        is_task = which_strokes=="strokes_task"

        fig, axes = self.plotMultStrokes(strokes_list, titles=titles, add_stroke_number=add_stroke_number,
            SIZE=SIZE, is_task=is_task, number_from_zero=number_from_zero, color_by=color_by, 
            **plotkwargs)
        return fig, axes, idxs


    def plotMultTrials(self, idxs, which_strokes="strokes_beh", return_idxs=False, 
        ncols = 5, titles=None, naked_axes=False, add_stroke_number=True, centerize=False, 
        nrand=None, color_by=None):
        """ plot multiple trials in a grid.
        - idxs, if list of indices, then plots those.
        --- if an integer, then plots this many random trials.
        - which_strokes, either "strokes_beh" (monkey) or "strokes_task" (stim)
        - nrand, sample random N
        - color_by, if None, ignores. otherwise different ways to color strokes.
        --- "speed", instantenous speed.
        NOTE:
        - returns None if fails to get any data.
        """
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper

        strokes_list, idxs, titles = self._plot_prepare_strokes(which_strokes, idxs, 
            nrand=nrand, titles=titles)
        if len(idxs)==0:
            if return_idxs:
                return None, None
            else:
                return None

        if which_strokes=="strokes_beh":
            color_by = "order"

        # Extract the final data
        if color_by is None:
            # Which plotting function?
            if which_strokes in ["strokes_beh", "parses"]:
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, 
                    add_stroke_number=add_stroke_number, centerize=centerize)
            elif which_strokes == "strokes_task":
                # plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, 
                #     add_stroke_number=add_stroke_number)
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, centerize=centerize)
            else:
                assert False

            # Plot
            fig= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left")

        elif color_by=="order":
            # Color strokes so that gradient of colors based on rank number.
            fig, axes = self.plotMultStrokesByOrder(strokes_list, ncols, titles, naked_axes, 
                add_stroke_number=add_stroke_number)
            return fig, axes, idxs

        elif color_by=="speed":
            # instantaneuos speed
            
            # -- function
            assert which_strokes=="strokes_beh", "only coded for this currently"
            from pythonlib.drawmodel.strokePlots import plotDatStrokesMapColor
            # 0: strokes... 1: stroke_vel... 2: vmin... 3: vmax
            plotfunc = lambda data, ax: plotDatStrokesMapColor(
                data[0], ax, data[1], data[2], data[3])

            # - get speed
            strokes_vels_list = self.extractStrokeVels(idxs)
            # dont keep if too short.
            strokes_list = [strokes for strokes, sv in zip(strokes_list, strokes_vels_list) if sv is not None]
            titles = [t for t, sv in zip(titles, strokes_vels_list) if sv is not None]
            idxs = [i for i, sv in zip(idxs, strokes_vels_list) if sv is not None]
            strokes_vels_list = [sv for sv in strokes_vels_list if sv is not None]

            # -- data
            data = []
            for strokes, strokes_vel in zip(strokes_list, strokes_vels_list):
                # Get the min and max vel
                # vmin is min of mins across strokes.
                x = [np.percentile(s, [1, 99]) for s in strokes_vel]
                vmin = np.min([xx[0] for xx in x])
                vmax = np.max([xx[1] for xx in x])
                if np.isnan(vmin):
                    print(x)
                    # print([len(s) for s in strokes_vel])
                    print(strokes_vel)
                    assert False
                data.append([strokes, strokes_vel, vmin, vmax])

            # use the same speed across all trials.
            if len(data)>0:
                vmin = np.min([d[2] for d in data])
                vmax = np.max([d[3] for d in data])
                if np.isnan(vmin):
                    print([d[2] for d in data])
                    assert False
                if np.isnan(vmax):
                    print([d[3] for d in data])
                    assert False
                for d in data:
                    d[2] = vmin
                    d[3] = vmax
            else:
                if return_idxs:
                    return None, None
                else:
                    return None

            # -- Plot
            # for d in data:
            #     print([len(dd) for dd in d[0]])
            #     print([dd[0] for dd in d[0]])

            fig= plotGridWrapper(data, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left")
        else:
            print(color_by)
            assert False, "not coded"

        if return_idxs:
            return fig, idxs
        else:
            return fig
        # return idxs

    def strokes_onsets_offsets_location(self, ind):
        """
        Return the coordinates of onset and offset of beh strokes.
        :param ind:
        :return: onlocs, list of np (2,) arrays (x,y) pixel lcoation of each strok onset
        :return: offlocs, list of np (2,) arrays (x,y) pixel lcoation of each strok offset
        """
        strokes = self.Dat.iloc[ind]["strokes_beh"]
        onlocs = [s[0, :2] for s in strokes]
        offlocs = [s[-1, :2] for s in strokes]
        return onlocs, offlocs

    def strokes_onsets_offsets_location_append(self):
        """
        Append "stroke_on_locs" and "stroke_off_locs",
        each row a tuple, len n strokes, holding
        onset or offset location, as (2,) arrays
        :return:
        """
        list_on = []
        list_off = []
        for ind in range(len(self.Dat)):
            onlocs, offlocs = self.strokes_onsets_offsets_location(ind)
            list_on.append(tuple(onlocs))
            list_off.append(tuple(offlocs))
        self.Dat["stroke_on_locs"] = list_on
        self.Dat["stroke_off_locs"] = list_off


    def strokes_durations_gaps(self, ind):
        """ Return duraton of strokes and gaps for this trial
        RETURNS:
            - list of stroke durations, sec
            - gap duratiosn
        """
        ons, offs = self.strokes_onsets_offsets(ind)
        gap_durations = [onthis - offthis for onthis, offthis in zip(ons[1:], offs[:-1])]
        stroke_durations = [offthis - onthis for onthis, offthis in zip(ons, offs)]
        return stroke_durations, gap_durations

    def strokes_onsets_offsets(self, ind):
        """
        Get onsets and offset times of strokes.
        RETURNS:
        - onsets, list of nums, time in sec, onsets of strokes
        - offsets, list ofn ums
        """

        strokes_beh = self.Dat.iloc[ind]["strokes_beh"]
        ons = [s[0, 2] for s in strokes_beh]
        offs = [s[-1, 2] for s in strokes_beh]
        return ons, offs

    def strokes_to_velocity_speed(self, strokes):
        """ Helper to convert strokes to velocities
        PARAMS;
        - ver, either "vel" or "speed"
        RETURNS:
        - strokes-like, with speed or vel
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse, plotDatStrokesVelSpeed
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper
        from pythonlib.tools.stroketools import strokesVelocity

        # get sampling rate
        assert False, "not ready"
        fs = self.get_sample_rate_alltrials()
        return strokesVelocity(strokes, fs)[i]


    def plot_strokes_timecourse_speed_vel(self, strokes, ax, plotver="speed", 
            align_to="first_touch", 
            overlay_stroke_periods=False,
            nolegend=False, alpha=0.8):
        """ Helper to plot strokes on ax, represnting either speed or velocity
        PARAMS;
        - 
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse, plotDatStrokesVelSpeed
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper

        # get sampling rate
        fs = self.get_sample_rate_alltrials()
        if align_to=="first_touch":
            # make the first touch 0
            x = strokes[0][0,:]
            strokes = [s-x for s in strokes]
        else:
            assert align_to is None

        # Which plotting function?
        return plotDatStrokesVelSpeed(strokes, ax, fs, plotver,
            overlay_stroke_periods=overlay_stroke_periods, nolegend=nolegend,
            alpha=alpha)

        
    def plotMultStrokesTimecourse(self, strokes_list, idxs=None, plotver="speed",
        return_idxs=False, ncols = 5, titles=None, naked_axes=False, aspect=0.8,
        align_to=None, overlay_stroke_periods=False):
        """ Helper to plot timecourse
        PARAMS:
        - plotver, str, {'speed', 'raw'}
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse, plotDatStrokesVelSpeed
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper

        # get sampling rate
        if idxs is not None:
            list_fs = [self.get_sample_rate(i) for i in idxs]
            assert len(np.unique(list_fs))==1, "why have diff sample rate? easy fix, make plotfunc take in fs in data tuple"
            fs = list_fs[0]
        else:
            # just pick the first trial
            fs = self.get_sample_rate(0)

        if align_to=="first_touch":
            # make the first touch 0
            strokes_list = [[s.copy() for s in strokes] for strokes in strokes_list] 
            for strokes in strokes_list:
                t0 = strokes[0][0,2]
                for s in strokes:
                    s[:,2] -= t0
        else:
            assert align_to is None
        # whether to modify the time column for strokes
        # by default, is aligned to first touch in strokes.

        # elif align_to=="first_stroke_onset":
        #     # do nothing - this is default.
        #     pass
        # else:
        #     print(align_to)
        #     assert False, "not coded"

        # Which plotting function?
        plotfunc = lambda strokes, ax: plotDatStrokesVelSpeed(strokes, ax, fs, plotver,
            overlay_stroke_periods=overlay_stroke_periods)

        # Plot
        fig= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, 
            titles=titles,naked_axes=naked_axes, origin="top_left", aspect=aspect)

        if return_idxs:
            return fig, idxs
        else:
            return fig


    def plotMultTrialsTimecourse(self, idxs, plotver="speed", which_strokes="strokes_beh", return_idxs=False, 
        ncols = 3, titles=None, naked_axes=False, nrand=None, align_to="go_cue"):
        """ Plot a grid of trials timecourses.
        NOT FUNCTIONAL:
        align_to

        """ 
 
        strokes_list, idxs, titles = self._plot_prepare_strokes(which_strokes, idxs, 
            nrand=nrand, titles=titles)
        if len(idxs)==0:
            return

        if False:
            for strokes in strokes_list:
                try:
                    assert strokes[0][0,2]==0., "I made mistake, not actually aligning by 0 by default"
                    # assert strokes_list[0][0,2]==0., "I made mistake, not actually aligning by 0 by default"
                except Exception as err:
                    print(strokes)
                    raise err

        if False:
            if align_to=="go_cue":
                # copy strokes.
                strokes_list = [[s.copy() for s in strokes] for strokes in strokes_list] 
                for ind, strokes in zip(idxs, strokes_list):

                    # get time from gocue to first touch
                    motor = self.get_motor_stats(ind)
                    x = motor["time_raise2firsttouch"] * 1000 # convert from s to ms.

                    if np.isnan(x):
                        print("Here")
                        print(x)
                        print(ind)
                        print(self.Dat.iloc[ind])
                        assert False
                    # add to all times
                    # print(strokes)
                    for s in strokes:
                        s[:,2] += x
                    # print(strokes)
                    # assert False

                    # ME = Dthis.Dat.iloc[1]["motorevents"]
                    # a = ME["ons"] - ME["raise"]

                    # MT = Dthis.Dat.iloc[1]["motortiming"]
                    # b = MT["time_raise2firsttouch"]

                    # assert a==b, "I forgot what my entries mean"

        self.plotMultStrokesTimecourse(strokes_list, idxs, plotver, 
            return_idxs=return_idxs, ncols=ncols, 
            titles=titles, naked_axes=naked_axes, aspect=2, align_to=align_to)



    def plotwrapper_training_task_examples(self, SDIR, niter = 3, nrand = 10):
        """ Plot grid of example training tasks, separating plots for each 
        epoch

        """
        # For each epoch, plot tasks and beh
        import os
        sdir = f"{SDIR}/training_task_grid_examples"
        os.makedirs(sdir, exist_ok=True)
        print("saving at:", sdir)

        # For each epoch, plot trian tasks.
        list_epoch = self.Dat["epoch"].unique().tolist()
        for i in range(niter):
            for epoch in list_epoch:
                print(epoch)
                inds = self.Dat[(self.Dat["probe"] == False) & (self.Dat["epoch"]==epoch)].index.tolist()
                titles = ["" for _ in range(len(inds))]
                for j, titlesthis in enumerate([titles, None]):
                    if j==0:
                        # randomly sample inds
                        figbeh, _, indsthis = self.plotMultTrials(inds, "strokes_beh", return_idxs=True, nrand=nrand,
                                                                naked_axes=True, add_stroke_number=False, titles=titlesthis)
                    else:
                        # use the current sampled inds
                        figbeh, _, _ = self.plotMultTrials(indsthis, "strokes_beh", return_idxs=False, nrand=nrand,
                                                                naked_axes=True, add_stroke_number=False, titles=titlesthis)
                    figtask = self.plotMultTrials(indsthis, "strokes_task", return_idxs=False, nrand=nrand,
                                                           naked_axes=True, add_stroke_number=False, titles=titlesthis)

                    if titlesthis is None:
                        figbeh.savefig(f"{sdir}/{epoch}-iter_{i}-beh.pdf")
                        figtask.savefig(f"{sdir}/{epoch}-iter_{i}-task.pdf")
                    else:
                        figbeh.savefig(f"{sdir}/{epoch}-iter_{i}-beh-notitles.pdf")
                        figtask.savefig(f"{sdir}/{epoch}-iter_{i}-task-notitles.pdf")
                        
        plt.close("all")   

        
    def plotOverview(self, ignore_large_plots=False):
        """ quick plot of kinds of tasks present across days
        # To visualize the experimental structure:
        # notes:
        # - tvalday is time of trial. e..g, the first digit is the day, the value after decimal is fraction of day (not
        # exactly from midnight to midnight, but within the "experimental session"). so 2.5 means day 1, halfway through the 
        # day (from start of expt to end of expt)
        # - to get the actual time, see "tval" qwhich is fraction of day from 0:00 to 24:00
        # RETURNS: 
        # - list of figs.
        """
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel

        figlist = []

        fig = sns.catplot(data=self.Dat, x="tvalday", y="taskgroup", hue="task_stagecategory", 
            row="expt", col="epoch", alpha=0.5)
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="monkey_train_or_test", 
            row="date", col="epoch", alpha=0.5)
        rotateLabel(fig)
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="block", y="taskgroup", hue="monkey_train_or_test", 
            row="date", col="epoch", aspect=2, alpha=0.5)
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="tvalday", y="taskgroup", hue="monkey_train_or_test", 
            row="task_stagecategory", col="epoch", aspect=2, alpha=0.5)
        figlist.append(fig)

        # fig = sns.catplot(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="online_abort", 
        #     row="date", col="epoch")
        # rotateLabel(fig)

        # fig = sns.scatter(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="online_abort", 
        #     row="date", col="epoch")
        # rotateLabel(fig)

        # One supblot for each date-session, block vs. trial (colored by epoch)
        fig = sns.FacetGrid(self.Dat, col="date_sess", hue="epoch", col_wrap=2, sharey=True, sharex=True, aspect=2, height=4)
        fig.map(sns.scatterplot, "trial", "block", alpha=0.5)
        fig.add_legend()
        figlist.append(fig)
        figlist.append(fig)

        fig = sns.FacetGrid(self.Dat, row = "monkey_train_or_test", col="date_sess", hue="epoch", sharey=True, sharex=True, aspect=2, height=4)
        fig.map(sns.scatterplot, "trial", "supervision_stage_new", alpha=0.5)
        fig.add_legend()
        figlist.append(fig)

        fig = sns.FacetGrid(self.Dat, row = "monkey_train_or_test", col="date_sess", hue="epoch", sharey=True, sharex=True, aspect=2, height=4)
        fig.map(sns.scatterplot, "trial", "supervision_stage_new", alpha=0.5)
        fig.add_legend()
        figlist.append(fig)

        fig = sns.FacetGrid(self.Dat, row = "taskgroup", col="date_sess", hue="supervision_stage_new", sharey=True, sharex=True, aspect=2, height=4)
        fig.map(sns.scatterplot, "trial", "block", alpha=0.5)
        fig.add_legend()
        figlist.append(fig)

        nchar = len(self.Dat["character"].unique())
        ntg = len(self.Dat["taskgroup"].unique())
        if nchar > 20 or ntg>10:
            ignore_large_plots=True

        if ignore_large_plots==False:
            nchar = len(self.Dat["character"].unique())
            fig = sns.FacetGrid(self.Dat, row = "taskgroup", col="monkey_train_or_test", hue="supervision_stage_new", 
                                sharey=True, sharex=True, aspect=1, height=nchar/10)
            fig.map(sns.scatterplot, "trial", "character")
            try:
                fig.add_legend()
            except ValueError as err:
                # this means facet titles are too large?
                # raise ValueError('left cannot be >= right')
                pass
            figlist.append(fig)


            fig = sns.FacetGrid(self.Dat, row = "taskgroup", col="monkey_train_or_test", hue="task_stagecategory", 
                        sharey=True, sharex=True, aspect=1, height=nchar/10)
            fig.map(sns.scatterplot, "trial", "character")
            try:
                fig.add_legend()
            except ValueError as err:
                # this means facet titles are too large?
                # raise ValueError('left cannot be >= right')
                pass
            figlist.append(fig)

        return figlist

    def plotOverviewScoresRewardsFeatures(self, savedir):
        """
        Related to online features, scores, rewards, during experiment.
        """
        import seaborn as sns
        ### SCores, rewards, features
        f = "rew_total"
        if False:
            # This plot too large... not needed
            fig = sns.catplot(data=self.Dat, x="block", y="rew_total", hue="probe", col="date_sess", col_wrap=3, kind="violin")
            fig.savefig(f"{savedir}/feat-{f}_byblock-1.pdf")
        fig = sns.catplot(data=self.Dat, x="block", y="rew_total", row="probe", col="date_sess", aspect=2, height=2.5)
        fig.savefig(f"{savedir}/feat-{f}_byblock-2.pdf")

    ############### related to positions of things in sketchpad
    def sketchpad_fixation_append_as_string(self):
        """ append to each row the fixation location as a string
        e.g., if xy is ['-0, -154'], then appends: '-0|-154' 
        as new column: origin_string
        """
        from pythonlib.tools.nptools import stringify
        origin_string = stringify(np.stack(self.Dat["origin"]))
        self.Dat["origin_string"] = origin_string
        print("Appended a new column: origin_string")

    def sketchpad_fixation_plot_locations(self):
        """ Quick plot of all locations across trials
        """
        xs = np.stack(self.Dat["origin"].values)[:,0]
        ys = np.stack(self.Dat["origin"].values)[:,1]

        fig, axes = plt.subplots(1,2)

        ax = axes.flatten()[0]
        ax.plot(xs, "ob", label="x")
        ax.set_ylabel('x location')

        ax = axes.flatten()[1]
        ax.plot(ys, "or", label="y")
        ax.set_ylabel('y location')



    def sketchpad_fixation_button_position(self, ind):
        """
        Return the location of fix button, in pixels
        RETUNRS:
        - (2,) array, x,y, pixel locations
        """
        return self.Dat.iloc[ind]["origin"]

    def sketchpad_done_button_did_reach_append_col(self,
                                                   max_frac_failures=0.02):
        """ append column "done_did_reach" in self.Dat.
        Also does sanityc check that not too mnahy casees of failed
        extraction (see note in _sketchpad_done_button_did_reach DEBUG).
        """

        list_reach = []
        failures = []
        for ind in range(len(self.Dat)):
            did_reach, done, t_done_touch, _ = self._sketchpad_done_button_did_reach(ind)

            # count how mnay cases have nan but shouldnt
            if done and np.isnan(t_done_touch):
                # This is a but in drawmonkey. its fixed but need to re-extract dset
                failures.append(ind)

            # Save result
            list_reach.append(did_reach)

        self.Dat["doneb_did_reach"] = list_reach

        if len(failures)/len(list_reach) > max_frac_failures:
            print(failures, len(failures)/len(list_reach))
            assert False, "Too many faiglulres (nans). reextract this dataset from drawmonkye"

    def _sketchpad_done_button_did_reach(self, ind,
                                        max_delay=1.,
                                         DEBUG=False):
        """ Return True if he did reach from end of
        last stroke towards done button, in reasonable time.
        and no abort.
        PARAMS:
        - max_delay, time in sec, from end of last stroke to touch of done,
        below which this would qualify as a "reaach" trial. 1 sec picked empriocally
        as easlity capturing all data for Diego, 230630.
        """

        done = self.Dat.iloc[ind]["trial_end_method"]=="pressed_done_button"
        t_done_touch = self.Dat.iloc[ind]["motorevents"]["done_touch"]
        t_last_stroke_off =  self.Dat.iloc[ind]["strokes_beh"][-1][-1, 2]
        delay = t_done_touch-t_last_stroke_off

        if DEBUG:
            if np.isnan(delay):
                # THis means is old dataset.. 1/13/24, This I fixed in
                # extraction of done timing in drawmonkey (see utils.py:
                # TIME OF DONE BUTTON TOUCH
                print("----")
                print(done)
                print(self.Dat.iloc[ind]["motorevents"])
                print(t_done_touch)
                print(t_last_stroke_off)

        did_reach_for_done = done and delay<max_delay

        return did_reach_for_done, done, t_done_touch, t_last_stroke_off

    def sketchpad_done_button_position(self, ind):
        """
        Return the location of done button, in pixels, 
        for this trial
        - (2,) array, x,y, pixel locations
        """

        return self.Dat.iloc[ind]["donepos"]

    ############### PRINT INFO
    def printOverview(self):
        from pythonlib.tools.pandastools import printOverview
        printOverview(self.Dat)

    def print_trial_block_epoch_summary(self, savedir=None):
        """ each line is trial-block-epoch, split by date-session
        Useful overview of all trials (where trial is actual beh trial)
        PARAMS:
        - savedir, if not None, then give path to directory will save file:
        savedir/trial_block_epoch.yaml
        RETURNS:
        - dict
        """

        from pythonlib.tools.expttools import writeDictToYaml

        outdict = {}
        list_sess_dict = sorted(self.Dat["date_sess"].unique())
        for ds in list_sess_dict:
            t= self.Dat[self.Dat["date_sess"]==ds]["trial"]
            b=self.Dat[self.Dat["date_sess"]==ds]["block"]
            e =self.Dat[self.Dat["date_sess"]==ds]["epoch"]
            
            tmp = [f"{tt}-{bb}-{ee}" for tt, bb, ee in zip(t, b, e)]
            outdict[ds] = tmp
        
        if savedir is not None:
            path = f"{savedir}/date_sess-trial_block_epoch.yaml"
            print("Saving to: ", path)
            writeDictToYaml(outdict, path)

        return outdict


    ################ ANALYSIS HELPERS

    def analy_get_all_inds_with_same_task(self):
        """
        returns indices for other trials
        
        """
        assert False, "havent figured out how to remove the ind for each own trial."
        def F(x):
            print(dir(x))
            task = x["unique_task_name"]
            inds = self.Dat[self.Dat["unique_task_name"]==task].index.to_list()
            print(inds)
            assert False

        self.Dat = applyFunctionToAllRows(self.Dat, F, "inds_same_task")

    def analy_subsample_trials_for_example(self, scores, inds, method="uniform",
        score_range=None, n=20):
        """ Extract subset of trials based on some filter for some score.
        e.g. get uniform distribution based on percentiles for score, or
        get top 20, or bottom 100, etc.
        Doesn't use self.Dat, you must enter the data here.
        Useful for generating/plotting examples across range of dataset
        PARAMS:
        - scores, np.array, scalar values used for sorting
        - inds, list or np.array, same length as scores. list of ids, usually ints, but could be tuples.
        - method, str, {'top', 'bottom', 'uniform'}
        - score_range, [min, max], will restrict to within this range. applies before taking
        top or uniform, etc. None means use entire range.
        - n, how many to take? will return in incresaing order of scores.
        RETURNS:
        - scores_sub, inds_sub, same length subsamples.
        """

        if isinstance(scores, pd.core.series.Series):
            scores = scores.tolist()
            # or else the indices will mess up sorting

        if isinstance(scores, list):
            scores = np.asarray(scores)
        assert isinstance(inds, list)

        # sort scores
        indsort = np.argsort(scores)
        scores = scores[indsort]
        inds = [inds[i] for i in indsort]

        # Keep only within range
        if score_range is None:
            score_range = [min(scores), max(scores)] 
        indrange = np.where((scores>=score_range[0]) & (scores<=score_range[1]))[0]
        scores = scores[indrange]
        inds = [inds[i] for i in indrange]

        # you can't request for than available
        if n>len(scores):
            n = len(scores)

        if method=="uniform":
            # take uniform sample 
            indstake = np.floor(np.linspace(0, len(scores)-1, n))
            indstake = [int(i) for i in indstake]
            scores_sub = scores[indstake]
            inds_sub = [inds[i] for i in indstake]
        elif method=="top":
            scores_sub = scores[-n:]
            inds_sub = inds[-n:]
        elif method=="bottom":
            scores_sub = scores[:n]
            inds_sub = inds[:n]
        elif method=="random":
            assert False, "get random, then sort."
        else:
            assert False

        return scores_sub, inds_sub


    def analy_get_tasks_strongesteffect(self, grouping, grouping_levels, feature,
        dfunc = None, df_in=None, return_vals_sorted=False):
        """ Extract tasks which have greatest difference along some feature (column),
        after aggregating over tasks, separating by some grouping variable.
        Currently only works for goruping using 2 levels (takes difference)
        e.g., can ask: which task shows largest difference for feature "score", across
        epochs 1 and 2?
        INPUTS:
        - dfunc, function for scoring difference, func: x,y --> scalar, where y is feature for
        group level 2.
        - df, if None, then uses self.Dat. otherwises uses df.
        RETURNS:
        - tasklist, which is sorted. first item is that with greatest psoitive difference (level 2 minus level 1)
        """

        from pythonlib.tools.pandastools import pivot_table

        if dfunc is None:
            dfunc = lambda x,y: y-x

        if df_in is None:
            print("Using self.Dat")
            dfthis = self.Dat
        else:
            print("Using inputted df_in")
            dfthis = df_in

        # Remove nan

        # Pivot
        df = pivot_table(dfthis, index=["character"], columns=[grouping], 
                         values = [feature])

        df = df.dropna()
        df = df.reset_index(drop=True)

        # get diff
        x1 = df[feature][grouping_levels[0]] 
        x2 = df[feature][grouping_levels[1]]
        tasklist = df["character"].to_list()
        value_diffs = [dfunc(xx1, xx2) for xx1, xx2 in zip(x1, x2)]

        tmp = [(task, v) for task, v in zip(tasklist, value_diffs)]
        tmp_sorted = sorted(tmp, key=lambda x: x[1])[::-1]
        tasklist_sorted = [t[0] for t in tmp_sorted]
        vals_sorted = [t[1] for t in tmp_sorted]

        # # tmp = [(i, v) for i, v in enumerate(value_diffs)]
        # # inds_sorted = sorted(tmp, key= lambda x: x[1])[::-1]
        # # vals_sorted = [i[1] for i in inds_sorted]
        # # inds_sorted = [i[0] for i in inds_sorted]
        # if True:
        #     print(vals_sorted)
        # # nplot = np.min([10, len(inds_sorted)])
        # # inds_to_plot = inds_sorted[:nplot]

        # tasklist_sorted = [tasklist[i] for i in inds_sorted]
        if return_vals_sorted:
            return tasklist_sorted, vals_sorted
        else:
            return tasklist_sorted


    def analy_singletask_df(self, task, row_variable, row_levels=None, return_row_levels=False):
        """ extract datafarme of just one task, with appended columns named
        "row" and "col" where col is trial num (in order seen in self.Dat) and row
        is level for row_variable. indices will be reset within each row level, but not in
        in the final df.
        RETURNS:
        - df
        - return_row_levels (if flagged on)
        """
        df = self.Dat[self.Dat["character"] == task]

        # what are rows
        if row_levels is None:
            row_levels = sort_mixed_type(df[row_variable].unique().tolist())

        # To number from 0, 1, ... for within each row level, iterate over them
        # and each time reset their index, then save the index.
        out = []
        for i, lev in enumerate(row_levels):
            dfthis = df[df[row_variable]==lev]
            dfthis = dfthis.reset_index(drop=True)
            dfthis["col"] = dfthis.index.tolist()
            dfthis["row"] = i
            out.append(dfthis)

        if return_row_levels:
            return pd.concat(out), row_levels
        else:
            return pd.concat(out)

    def analy_assign_trialnums_within_task(self, groupby=None):
        """ in chron order 0, 1, 2, assigned to each row depending 
        onw hich trial for its unique task. groupby allows to have
        0, 1, 2, for a grouping variable too.
        RETURNS:
        - modifies self.Dat to have new column "trialnum_this_task"
        """
        from pythonlib.tools.nptools import rankItems
        def F(x):
            assert np.all(np.diff(x)>0), "shouldnt matter, but should confirm that now rankItems outputs correct order. then remove this assertion?"
            return rankItems(x)

        assert isinstance(groupby, list)
        gb = ["character"] + groupby
        self.Dat["trialnum_this_task"] = self.Dat.groupby(gb)["tval"].transform(F)


    def analy_match_numtrials_per_task(self, groupby):
        """ returns dataset where make sure each task has same num trials across
        all levels for group. prunes by deleting trials not matched.
        RETURNS:
        - new dataset, copied and pruned.
        """
        _checkPandasIndices(self.Dat)
        # D.Dat = D.Dat.reset_index(drop=True)

        # first Make sure assigned trialnums
        self.analy_assign_trialnums_within_task(groupby)

        tasklist = self.Dat["character"].unique().tolist()

        # Collect inds that are extraneous.
        inds_bad_all = []
        for task in tasklist:
            df = self.Dat[self.Dat["character"]==task]
            maxnum = min(df.groupby(groupby)["trialnum_this_task"].max())
            inds_bad = df.index[df["trialnum_this_task"]>maxnum].tolist()
            inds_bad_all.extend(inds_bad)

        Dtrialsmatch = self.copy()
        print(len(Dtrialsmatch.Dat))
        Dtrialsmatch.Dat = Dtrialsmatch.Dat.drop(inds_bad_all)
        Dtrialsmatch.Dat = Dtrialsmatch.Dat.reset_index(drop=True)
        print(len(Dtrialsmatch.Dat))
        return Dtrialsmatch


    def analy_match_sequence_discrete_per_task(self, groupby, grouping_levels = None,
        ver="same", print_summary=False):
        """ For each task, make sure trials use same ("same") or different ("diff")
        sequence across levels for groupby. Sequences are defined by discrete parse of
        task sequence.
        INPUT:
        - groupby, how to group trials
        - grouping_levels, leave None to use all. otherwise give me.
        - ver, {"same", "diff"}
        --- same, then will only keep a trial if every other grouping level has at least
        one trial with this trial's sequence. Note that can end up with a task having multiple
        sequences.
        e..g, --- mixture2_11-savedset-34-91957
                {'short': [21], 'long': [21, 43, 0]} [start, showing seq for each trial for 2 group levels.]
                {'short': [21], 'long': [21]} [end]
        --- diff, then will only keep trials for which no trial in other grouping levels have the
        same sequence. Note that can end up with a task having multiple
        sequences.
        e..g, --- mixture2_11-savedset-34-91957
                {'short': [21], 'long': [21, 43, 0]} [start, showing seq for each trial for 2 group levels.]
                {'short': [], 'long': [43, 0]} [end]


        """

        if grouping_levels is None:
            grouping_levels = self.Dat[groupby].unique().tolist()

        # Make sure sequence ids already asisgned.
        self.planner_assign_sequence_id()
        Dcopy = self.copy()

        # Go thru each task
        inds_to_remove = []
        tasklist = self.Dat["character"].unique().tolist()
        for task in tasklist:
            # print(task)
            dfthis = self.Dat[self.Dat["character"]==task]
            
            # get mapoing between grouping level and sequences that exist
            levels2sequences = {}
            for lev in grouping_levels:
                levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].unique().tolist()
            
            # get mapping between grouping level and sequences to keep
            # flatten to all indices that exist
            indices_all = set([vv for v in levels2sequences.values() for vv in v])
            # print(indices_all)
            # assert False

            if ver=="same":
                # find the indices that exist across all groups
                def _isgood(ind):
                    # Returns True if keep this ind, false otherwise
                    return all([ind in v for v in levels2sequences.values()])
            elif ver=="diff":
                def _isgood(ind):
                    tmp = [ind in v for v in levels2sequences.values()]
                    # if task=="mixture2_55-savedset-34-09627":
                    #     print("---")
                    #     print(levels2sequences)
                    #     print(ind)
                    #     print(sum(tmp))
                    if sum(tmp)==0:
                        assert False, "bug. this ind must be in at least one grouping level"
                    elif sum(tmp)==1:
                        # good, then is only in the one from which we extracted it to enter here
                        return True
                    elif sum(tmp)>1:
                        # then is present in at least one other grouping level.
                        return False
            else:
                print(ver)
                assert False, "noit coded"

            # update levels2sequences to only good sequences
            levels2sequences_good = {}
            for k, v in levels2sequences.items():
                levels2sequences_good[k] = [vv for vv in v if _isgood(vv)]

            # For each level, only keep trials if its sequence is in the good sequences.
            for row in dfthis.iterrows():
                if row[1]["sequence_id"] in levels2sequences_good[row[1][groupby]]:
                    # then good, keep this.
                    pass
                else:
                    inds_to_remove.append(row[0])
        inds_to_remove = sorted(list(set(inds_to_remove)))
        print("Remoiving these inds")
        print(inds_to_remove)
        self.Dat = self.Dat.drop(inds_to_remove, axis=0).reset_index(drop=True)


        if print_summary:
            def _check_task_sequences(D, task, unique=True):
                # print(task)
                dfthis = D.Dat[D.Dat["character"]==task]

                # get mapoing between grouping level and sequences that exist
                levels2sequences = {}
                for lev in grouping_levels:
                    if unique:
                        levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].unique().tolist()
                    else:
                        levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].tolist()
                print(levels2sequences)

            tasklist = self.Dat["character"].unique().tolist()
            print("SUMMARY - task, before, after, [sequences per trial]")
            for task in tasklist:
                print("---", task)
                _check_task_sequences(Dcopy, task, unique=False)
                _check_task_sequences(self, task, unique=False)

            
    def analy_reassign_monkeytraintest(self, key, list_train, list_test, list_other=None):
        """ redefine monkey train test based on value in column "key".
        if a row[key] is in list_train, then is train, etc.
        - enforces that list_train and list_test no overlap.
        - enforces that each row will have an assignment.
        """

        if list_other is None:
            list_other = []

        # Make sure no common items
        assert len([x for x in list_train if x in list_test])==0
        assert len([x for x in list_train if x in list_other])==0
        
        def func(x):
            if x[key] in list_train:
                return "train"
            elif x[key] in list_test:
                return "test"
            elif x[key] in list_other:
                return "other"
            else:
                print(x)
                assert False

        self.Dat = applyFunctionToAllRows(self.Dat, func, "monkey_train_or_test")
        print("Reassigned train/test, using key:", key)
        print("and values:")
        print("Train = ", list_train)
        print("Test = ", list_test)
        print(" ")
        print("New distribution of train/test:")
        print(self.Dat["monkey_train_or_test"].value_counts())


    ################## PRUNING DATASET
    def prune_min_ntrials_across_higher_levels(self, col_high, col_low, n_min=1):
        """ 
        e.g, only keep characters that have at least 5 trials in each epoch:
        self.prune_min_ntrials_across_higher_levels("epoch", "character", 5)
        PARAMS:
        - col_high, the higher-level column, string, e.g., "epoch"
        - col_low, the lower-level column, e.g., "character".
        RETURNS:
        - df, pruned. (Does not modify self.Dat)
        """
        from pythonlib.tools.pandastools import prune_min_ntrials_across_higher_levels
        df = prune_min_ntrials_across_higher_levels(self.Dat, col_high, col_low, n_min=n_min)
        return df
