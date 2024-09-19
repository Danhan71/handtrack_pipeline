import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows
from .analy_dlist import extract_strokes_monkey_vs_self


def sequence_get_rank_vs_task_permutations_quick(D, permver = "all_orders_directions"):
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences
    """ For each behavhora trial, align its sequence vs. task, then compute the rank
    of the saeuqnece based on efficiency cost. can only compute rank if have same lenth
    otherwise will return nan.
    INPUTS:
    - permver, think of this as index into a particular planner expt/analysis.
    RETURNS:
    - D.Dat modified to have 5 new columns. see code.
    NOTE:
    - "quick" means that parses, parse-beh dist, and parse-scores are all precomputed and saved,
    using the appropriate Dataset methods.
    """
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences_quick

    # load pre-computed planner stuff
    D.planner_load_everything(permver = "all_orders_directions")

    def F(x):
        strokes_beh = x["strokes_beh"]
        strokes_task_perms = x["parses_planner"]
        beh_task_distances = x["parses_planner_behtaskdist"]
        task_inefficiency = x["parses_planner_taskscore"]

        out = rank_beh_out_of_all_possible_sequences_quick(
            strokes_beh, strokes_task_perms, beh_task_distances, 
            task_inefficiency, efficiency_score_ver="weighted_avg", 
            confidence_ver="diff_relative_all")
        # print("this")
        # print(out)
        # print("th")
        if out is None:
            return np.nan
        out = list(out) # dont inlcude the last element, which is strokes task picked.
        out[3] = len(strokes_task_perms) # num perms.
        return out

    # def F(x):
    #     sb = x["strokes_beh"]
    #     st = x["strokes_task"]
    #     if len(sb)!=len(st):
    #         return np.nan
    #     out = rank_beh_out_of_all_possible_sequences(sb, st, return_chosen_task_strokes=False,
    #                                                 plot_rank_distribution=False, 
    #                                                 return_num_possible_seq=True)
    # #     rank, confidence, summaryscore, numseq = out
    #     return out
    # #     return rank_beh_out_of_all_possible_sequences(x["strokes_beh"], x["strokes_task"])

    # print(len(D.Dat))
    # for row in D.Dat.iterrows():
    #     row[1]["rankeffic"] = F(row[1])
    D.Dat = applyFunctionToAllRows(D.Dat, F, "rankeffic")

    # Expand out efficiency scores
    def F(x, ind):
        if np.all(np.isnan(x["rankeffic"])):
            return np.nan
        return x["rankeffic"][ind]

    # for ind in range(4):
    #     D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,ind), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,0), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,1), "effic_confid")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,2), "effic_summary")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,3), "effic_nperms")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,4), "effic_taskscore")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,5), "effic_behtaskdist")
    del D.Dat["rankeffic"]

    def F(x):
        if np.isnan(x["effic_rank"]):
            return False
        else:
            return True
    D.Dat = applyFunctionToAllRows(D.Dat, F, "has_effic_rank")



def sequence_get_rank_vs_task_permutations(D):
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences
    """ For each behavhora trial, align its sequence vs. task, then compute the rank
    of the saeuqnece based on efficiency cost. can only compute rank if have same lenth
    otherwise will return nan.
    RETURNS:
    - D.Dat modified to have 5 new columns. see code.
    """
    def F(x):
        sb = x["strokes_beh"]
        st = x["strokes_task"]
        if len(sb)!=len(st):
            return np.nan
        out = rank_beh_out_of_all_possible_sequences(sb, st, return_chosen_task_strokes=False,
                                                    plot_rank_distribution=False, 
                                                    return_num_possible_seq=True)
    #     rank, confidence, summaryscore, numseq = out
        return out
    #     return rank_beh_out_of_all_possible_sequences(x["strokes_beh"], x["strokes_task"])

    D.Dat = applyFunctionToAllRows(D.Dat, F, "rankeffic")

    # Expand out efficiency scores
    def F(x, ind):
        if np.all(np.isnan(x["rankeffic"])):
            return np.nan
        return x["rankeffic"][ind]

    # for ind in range(4):
    #     D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,ind), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,0), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,1), "effic_confid")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,2), "effic_summary")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,3), "effic_nperms")
    del D.Dat["rankeffic"]

    def F(x):
        if np.isnan(x["effic_rank"]):
            return False
        else:
            return True
    D.Dat = applyFunctionToAllRows(D.Dat, F, "has_effic_rank")



def get_task_pairwise_metrics(D, grouping, func):
    """ groups data bsed on tasks, then runs thru all pairs of trials (same task) and computs
    func. 
    INPUT:
    - grouping, will save the value of this variable, in the output, for each item in each pair.
    - func(x,y), where x and y are two trials, row of dataframe.
    OUTPUT:
    - list of dict, where each dict holds a unique pair of trials
    """

    # Compute all pairwise between all trials (conditions on pairs being from the same task).
    tasklist = set(D.Dat["character"])
    out = []
    for task in tasklist:
        dfthis = D.Dat[D.Dat["character"]==task]

        # get all pairwise scores between all rows.
        c = 0
        for i, row1 in enumerate(dfthis.iterrows()):
            for ii, row2 in enumerate(dfthis.iterrows()):
                if ii>i:
                    # print(i, ii)
                    # print(row1[0])
                    # print(row2[0])
                    # assert False
                    value = func(row1[1], row2[1])
                    # strokes1 = row1[1]["strokes_beh"]
                    # strokes2 = row2[1]["strokes_beh"]
                    # d = get_dist(strokes1, strokes2)

                    monkey_priors = (row1[1][grouping], row2[1][grouping])
                    out.append({
                        "value":value,
                        "task":task,
                        "groupings":sorted(monkey_priors),
                        "indexes":(row1[0], row2[0])})
                    c+=1

        # print(task, c)
    return out




def score_all_pairwise_within_task(D,  GROUPING, 
    GROUPING_LEVELS, DIST_VER = "dtw_segments", DONEG=False):
    """
    for each unique task, gets all pairwise distances for same level, diff level,
    where levels defined by grouping variable (e..g, monkey_prior, epoch etc).
    INPUTS:
    - DIST_VER, str, which kidn of score to use.
    - GROUPING, dataframe column, whose levels are used to split data.
    - DONEG, if True, then will flip sign of distance, so that more neg is more worse.
    - GROUPING_LEVELS, the levels to care about (i.e., within level vs. across level scoring)
    RETURNS:
    - modifies D.Dat, modified to have new columns, where for each trial give score reltaive to 
    all other trials, separated by how those other trials are grouped. if multipel trials 
    compared to it, then takes mean.
    - colnames, the names of new cols. **Will be in same order as GROUPING_LEVELS input!!
    NOTE:
    - Useful, e.g, for positive control based on across-trial varibility given
    the same task(stimulus)
    - NOTE: grouping is synonymous with monkey prior in this code.
    - 
    """
    
    # Distancd function defined.
    def get_dist(strokes1, strokes2):
        """ return scalar dist between strokes"""
        from pythonlib.drawmodel.strokedists import distscalarStrokes
        return distscalarStrokes(strokes1, strokes2, ver=DIST_VER)

    def func(x, y):
        strokes1 = x["strokes_beh"]
        strokes2 = y["strokes_beh"]
        return get_dist(strokes1, strokes2)


    out = get_task_pairwise_metrics(D, GROUPING, func)
    # # Compute all pairwise between all trials (conditions on pairs being from the same task).
    # tasklist = set(D.Dat["character"])
    # out = []
    # for task in tasklist:
    #     dfthis = D.Dat[D.Dat["character"]==task]

    #     # get all pairwise scores between all rows.
    #     c = 0
    #     for i, row1 in enumerate(dfthis.iterrows()):
    #         for ii, row2 in enumerate(dfthis.iterrows()):
    #             if ii>i:
    #                 strokes1 = row1[1]["strokes_beh"]
    #                 strokes2 = row2[1]["strokes_beh"]
    #                 d = get_dist(strokes1, strokes2)

    #                 monkey_priors = [row1[1][GROUPING], row2[1][GROUPING]]
    #                 out.append({
    #                     "score":d,
    #                     "task":task,
    #                     "groupings":sorted(monkey_priors)})
    #                 c+=1

    #     # print(task, c)


    # Repopulate into main DF
    def _extract(task, monkey_prior_this, monkey_prior_other):
        """ extract list of scores for this task, done while under
        monkey_prior_this, compared to when monkey was under monkey_prior_other
        RETURNS:
        - scores, list of scores.
        """

        mkp_index = sorted([monkey_prior_this, monkey_prior_other])
        outthis = [o for o in out if o["task"]==task and o["groupings"]==mkp_index]
        scores = [o["value"] for o in outthis]
        return scores

    colnames = []
    for group_other in GROUPING_LEVELS:
        def F(x):
            scores = _extract(x["character"], x[GROUPING], group_other)
            if len(scores)==0:
                return np.nan
            if any(np.array(scores)<0.001):
                print(scores)
                assert False, "this means there is task compared to itself..."
            s = np.nanmean(scores) # negative to conver to score (from distance)
            if DONEG:
                s = -s
            return s
         
        D.Dat = applyFunctionToAllRows(D.Dat, F, newcolname=f"vs-selfsametask-{group_other}")    
        colnames.append(f"vs-selfsametask-{group_other}")

    return colnames


def score_alignment(D, monkey_prior_col, monkey_prior_levels, score_name_list, suffix=""):
    """ 2 x 2 alignment, where there are two monkey priors, and two scores (model scores).
    each row is one particular level, but has a score for each score_name_list.
    INPUTS:
    - monkey_prior_levels, list of levels.
    - score_name_list, list of names (coilumns)
    OUT:
    - modifies D in place.
    NOTE:
    ** ORDER matters, monkey_prior and score_name must be aligned.
    - score is assumed to be more positive --> better score.
    """

    assert False, "note that should use instead model_behmodel_handler.analy_compute_alignment_wrapper. Comment this out to continue."
    
    for prior, score in zip(monkey_prior_levels, score_name_list):
        print(f"{prior} - aligned with - {score}")
        assert prior in score, "are they not aligned?"

    # 1) get difference of scores
    if len(suffix)>0:
        suffix = "_" + suffix
    colname_m2_m1 = f"mod2_minus_mod1{suffix}"
    D.Dat[colname_m2_m1] = D.Dat[score_name_list[1]] - D.Dat[score_name_list[0]]

    # 2) for each row, its alignemnet 
    def F(x):
        if x[monkey_prior_col]==monkey_prior_levels[0]:
            return -x[colname_m2_m1]
        elif x[monkey_prior_col]==monkey_prior_levels[1]:
            return x[colname_m2_m1]
        else:
            print("FAILED)")
            print(monkey_prior_col, monkey_prior_levels)
            print(x[monkey_prior_col])
            print(x)
            assert False

    newcol = f"alignment_trials{suffix}"
    print("New alignmment col: ", newcol)
    D.Dat = applyFunctionToAllRows(D.Dat, F, newcol)

    return colname_m2_m1, newcol

def taskmodel_assign_score(D, expt="lines5"):
    """ Quick and dirty, to replicate what did in Probedat, for scoring datsate.
    Uses probedat here just to extract Params.
    Note: only works for lines5 currently - the chunkmodel functin.
    OUTPUT:
    - new columnds in D.Dat, with names as in model_score_names
    - model_score_names, list of strings.
    """
    from pythonlib.drawmodel.taskmodel import Model, makeParseFunction, getParamsWrapper
    from pythonlib.drawmodel.taskmodel import Dataset as DatasetModel

    assert expt=="lines5", "otherwise make this more general purpose params"

    # -- 2) Fit different models.
    model_score_names = []
    for likeliver, priorver, parse_ver, chunkmodel, posterior_ver, name in zip(
        ["segments", "segments"],
        ["uniform","uniform"], 
        ["chunks", "chunks"],
        ["3line", None],
        ["maxlikeli", "maxlikeli"],
        ["3line", "linePlusL"]):

#         PD = ProbedatTaskmodel([])

        # 3) Buidl model
        if name in ["linePlusL", "linePlusL_combine", "onechunk"]:
            chunkmodel = makeParseFunction(name)
            
        assert chunkmodel is not None, "need to replace this with ParseFunction.."
        PARAMS, PARAMS_MODEL = getParamsWrapper(priorver=priorver, parse_ver=parse_ver, 
                                            chunkmodel=chunkmodel, name=name,
                                            posterior_ver=posterior_ver, 
                                            likeliver=likeliver)

        PARAMS_DATA, PARAMS_MODEL = getParamsWrapper(priorver, parse_ver, chunkmodel,
            name, posterior_ver, likeliver)

        # Prepare dataset
        strokes = D.Dat["strokes_beh"].to_list()
        strokes_task = D.Dat["strokes_task"].to_list()
        fix = D.Dat["origin"].to_list()
        tasks = [D.Dat["Task"].iloc[i].Params["input_params"].Task for i in range(len(D.Dat))]
        for stask, f, T in zip(strokes_task, fix, tasks):
            T["strokes"] = stask
            T["fixpos"] = f
    
        # Score using this model.
        mod = Model(PARAMS_MODEL)
        data = DatasetModel(strokes, tasks, PARAMS=PARAMS)
        data.applyModel(mod)

        # Assign scores back into dataset
        assert len(D.Dat)==len(data.trials)
        namethis = f"MOD_{name}"
        D.Dat[namethis] = [t["posterior"] for t in data.trials]
        model_score_names.append(namethis)


    return model_score_names




