""" analysis related to notebook: analysis_line2_090720
- looking at structured gneralization, in line2 experiment for
Panchoa nd red"""

from tools.utils import *

class OldTaskError(Exception):
    pass

def PROBEDATfromFD(FD, cleanup_remove_if_no_touch=True):
    """ Make ProbeDat instance from this set of filedatas
    PARAMS:
    - FD, 
    NOTE:
     MAKE SURE TO call this from  loadProbeDatWrapper"
    """
    PROBEDAT = []
    currentdate = []
    for i, F in enumerate(FD):
        fd = F["fd"]
        
        # to keep track of trial within day
        if F["date"]!=currentdate:
            currentdate = F["date"]
            ct = 1
            
        for t in getIndsTrials(fd):

            # is this a good trial?
            if cleanup_remove_if_no_touch:
                GOOD = getTrialsTouched(fd, t)
            else:
                GOOD = True
                
            if GOOD:
                probe = getTrialsTaskProbeInfo(fd, t)

                if probe is None:
                    if False:
                        raise OldTaskError
                
                task = getTrialsTask(fd, t)

                if getTrialsTaskIsFix(fd, t):
                    randomtask = False
                else:
                    randomtask = True

                kind = getTrialsTaskProbeKind(fd,t)

                if False:
                    # was trying to get fd as lambda. didn't work.
                    import copy
                    animal = copy.copy(F["animal"])
                    expt = copy.copy(F["expt"])
                    date = copy.copy(F["date"])
                    ii = copy.copy(i)

                PROBEDAT.append({
                    "filedata":fd,
                    # "filedata":lambda : FD[i]["fd"],
                    # "ii":ii,
                    # "filedata":lambda : FD[ii]["fd"],
                    # "filedata":lambda: [F["fd"] for F in FD if F["animal"]==animal and F["expt"]==expt and F["date"]==date][0],
                    "animal":F["animal"],
                    "expt":F["expt"],
                    "date":F["date"],
                    "session":F["session"],
                    "trial":t,
                    "trial_day":ct,
                    "kind":kind,
                    "unique_task_name": getTrialsUniqueTasknameGood(fd, t),
                    "task_stagecategory":task["stage"],
                    "block":getTrialsBlock(fd, t),
                    "random_task":randomtask})

                # PROBEDAT[-1]["filedata"] = lambda : FD[PROBEDAT[-1]["ii"]]["fd"],
                
                # --- append everything in probedat
                if probe is not None: # is None for tasks not post 8/30/20 (i.e. not from makeDrawTasks)
                    for k, v in probe.items():
                        PROBEDAT[-1][k] = v
                    
                ct+=1
        
    # === add beh eval scores to RPOBEDAT
    for P in PROBEDAT:
        O = getTrialsBehEvaluationFeatures(P["filedata"], P["trial"], include_others="all")
        for k, v in O.items():
            P[k] = v

    return PROBEDAT

def getStandardizedTime(PROBEDAT, metaDat):
    """ get real-world time for each trial, but using a 
    decimalized version that is not actual time, but is
    good for plotting timecourses"""
    from pythonlib.tools.datetools import standardizeTime
    day1 = metaDat["sdate"]
    day1 = str(day1) + "-000000"
    for P in PROBEDAT:
        dt = getTrialsDateTime(P["filedata"], P["trial"], fmt="str")
        # dtval = standardizeTime(dt, day1, daystart=0.417, dayend=0.792)
        dtval = standardizeTime(dt, day1)
        P["tval"] = dtval
    return PROBEDAT



def probeDatIndexWithinDay(probedat, unique_task_name, reverse_order=False):
    """ figure out, for this unique task, each rendition its
    index within day, in chron order - i.e., first presentation
    today (acorss all sessions) is given a 0, etc.
    - adds a key to probedat: idx_today_uniquetask
    - modifies in place and also returns.
    - countlist gives number of cases found for each day.
    """
    countlist = []
    datelist = set([p["date"] for p in probedat])
    for date in datelist:
        print(f"-- date: {date}")

        # first check that all trials are in order
        trial_day =[P["trial_day"] for P in probedat if P["date"]==date]
        try:
            assert np.all(np.diff(trial_day)>0)==True, "not in order?"
        except Exception:
            print(trial_day)
            print(np.diff(trial_day))
            raise(Exception)

        # go through in order, and count up
        ct = 0
        if reverse_order:
            for P in probedat[::-1]:
                if P["unique_task_name"]==unique_task_name and P["date"]==date:
                    print(f"found this trial(day): {P['trial_day']}, assigned idx: {ct}")
                    P["idx_today_uniquetask"] = ct
                    ct+=1
        else:
            for P in probedat:
                if P["unique_task_name"]==unique_task_name and P["date"]==date:
                    print(f"found this trial(day): {P['trial_day']}, assigned idx: {ct}")
                    P["idx_today_uniquetask"] = ct
                    ct+=1
        countlist.append(ct)
    return probedat, countlist



def flattenByStrok(DAT, keys_to_exclude=("features", "strokes"), keep_strok=False):
    """ DAT is list, each element one trial.
    DAT must have a features key, which is list of dicts,
    each dict one strok. 
    e.g., DAT[0]["features"] = [{}, {}, {} ... (nstrokes)].
    - any other key in DAT[0] will alos bne included int he flattened
    output. 
    - keep_strok, then also extracts the strok data.
    """
    strokfeats = []
    for D in DAT:
        if keep_strok:
            assert len(D["strokes"]) == len(D["features"])

#         print(D["features"]) # list of dicts
        for i, feat in enumerate(D["features"]): # feat is dict
            strokfeats.append({
                 "strokenum":i,
            })

            if keep_strok:
                strokfeats[-1]["strok"] = D["strokes"][i]
            
            # -- add whatever keys are there
            for k, v in D.items():
                if k not in keys_to_exclude:
                    strokfeats[-1][k] = v

            # -- add all features
            for k, v in feat.items():
                strokfeats[-1][k] = v
    return strokfeats
    

def probedat2strokefeats(PROBEDAT, DATELIST, only_shared_tasks=True, keep_strok=False):
    """ gets stroke-features in a liust of dicts, each
    element corresponding to a stroke (by default). will also 
    keep infor for each stroke in the same output list of dicts.
    Is flexible so can add features and info to include if wanted.
    - only_shared_tasks, then only keeps tasks that are shared 
    across all dates (i..e based on unique name) - this will inherentlyh
    pick out "fixed" tasks (not random).
    - wil make sure that any task with same name is actually poitn by
    point the same across all days.

    - DATELIST, list of str, eg ["200902", "200903", "200906", "200907"]
    """
    from pythonlib.drawmodel.features import strokeFeatures
    
    # ============= get tasks
    tasklist_eachdate=[]
    for date in DATELIST:
        tasklist_eachdate.append(sorted(set([P["unique_task_name"] for P in PROBEDAT if P["date"]==date])))
    tasklist_all = set([task for tasklist in tasklist_eachdate for task in tasklist])

    # --- only inlcude tasks that occur on both dates
    if only_shared_tasks:
        tasklist_good = []
        for task in tasklist_all:
            if all([task in tasklist for tasklist in tasklist_eachdate]):
                tasklist_good.append(task)
        TASKLIST = sorted(tasklist_good)
    else:
        TASKLIST = sorted(tasklist_all)

    # CHECK: only include tasks that are identical on both dates
    tasklist_good = []
    for task in TASKLIST:

        # get one exmaple task from each day. compare all.
        tasks_to_compare = []
        for date in DATELIST:
            tasksthisday  = [getTrialsTask(P["filedata"], P["trial"])
                                     for P in PROBEDAT if P["date"]==date and P["unique_task_name"]==task]
            if len(tasksthisday)>0:
                tasks_to_compare.append(tasksthisday[0])

        # - do all pairwise comaprisons of task
        tmp = []
        for i in range(len(tasks_to_compare)):
            for j in range(i+1, len(tasks_to_compare)):
                tmp.append(compareTasks(tasks_to_compare[i], tasks_to_compare[j]))
        if all(tmp):
            tasklist_good.append(task)
        else:
            print("removing this task, since different across days")
            print(task)
    TASKLIST = tasklist_good


    # =========== PROCESS
    # --- 1) Get stroke features
    DAT = []
    for P in PROBEDAT:
        if P["date"] in DATELIST and P["unique_task_name"] in TASKLIST:
            DAT.append(
                {
                    "session":P["session"],
                    "trial":P["trial"],
                    "date":str(P["date"]),
                    "strokes":getTrialsStrokesByPeanuts(P["filedata"], P["trial"]),
                    "task":P["unique_task_name"],
                    "datetime":getTrialsDateTime(P["filedata"], P["trial"]),
                    "task_kind":P["kind"],
                    "task_category":P["task_stagecategory"],
                    "random_task":P["random_task"]
                })

    strokesfeatures = strokeFeatures([D["strokes"] for D in DAT])
    for sf, D in zip(strokesfeatures, DAT):
        D["features"] = sf

    # --- 2) Flatten to strok
    strokfeats = flattenByStrok(DAT, keep_strok=keep_strok)
    # len(strokfeats)
    # print(strokfeats[0])
    return strokfeats, TASKLIST


def strokfeats2Dataframe(strokfeats, exptMetaDat, only_first_last_trials=False,
                         matchedstrokes=None, 
                         traintest="bothtraintest", modResGetter=None,
                        modResKeyname="Lstrokeindex", only_if_has_model_res=True,
                        only_if_task_in_all_epochs=True):
    """ convert to dataframe and then process in 
    ways related to expt and model-basked (strok model)
    analysis
    - exptMetaDat, 
    - matchedstrokes, list of stroknums (0, 1.,,, ), that will;
    be considered "good" strokes. useful if want to only analyze 
    partocular strokes
    - traintest, "train" or "test" or "bothtraintest", will
    only keep that.
    - modResGetter, function that gets modelscore for paritcular trial
    modResGetter(date, session, trial, strok_num). leave None to ignore.
    - modResKeyname, to name the score extracted.
    - only_if_has_model_res, throws out data that has no model result.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    import pandas as pd
    #### move all below into plotting function
    # === ADD MODEL RESULTS
    k = modResKeyname
    if modResGetter is not None:
        if k=="Lstrokeindex":
            kactual = "0/(0+1)" # to fix bullshit naming problem I had.
        for s in strokfeats:
            mod_res = modResGetter(date=s["date"], session=s["session"], trial=s["trial"], 
                                        strok_num=s["strokenum"])
            if mod_res is not None:
                s[k] = mod_res[kactual].values[0]
            else:
                s[k] = np.nan                

    ## === 1) CONVERT TO DATAFRAME FOR PLOTTING
    SF = pd.DataFrame(strokfeats)

    ## === 2) Process:refelcting experimental structure

    # --- aggregate over strokes of a desired index
    if matchedstrokes is None:
        # then all strokes are good
        SF["keepstroke"] = True
    else:
        F = lambda x:x["strokenum"] in matchedstrokes
        SF = applyFunctionToAllRows(SF, F, newcolname="keepstroke")

    # --- expt epoch.
    F = lambda x:exptMetaDat["datecategories"][x["date"]]
    SF = applyFunctionToAllRows(SF, F, newcolname="epoch")

    # --- call each task either test or train
    F = lambda x:exptMetaDat["task_train_test"][x["task_kind"]]
    SF = applyFunctionToAllRows(SF, F, newcolname="traintest")

    if traintest in ["train", "test"]:
        SF = SF[SF["traintest"]==traintest]
    elif traintest=="bothtraintest":
        pass

    # == only keep if includes model results
    if only_if_has_model_res:
        if modResGetter is not None:
            SF = SF[~np.isnan(SF[k])]
        else:
            # === add fake k (so can use same plotting code as for model)
            SF[k] = 0


    # - only keep tasks that have at least one datapoint in epoch1 and 2
    if only_if_task_in_all_epochs:
        epochs_to_check = list(set([d for d in exptMetaDat["datecategories"].values()]))
        def F(x, epochs_to_check = epochs_to_check):
            """ True if has data for all epochs"""
            checks  = []
            for ep in epochs_to_check:
                checks.append(ep in x["epoch"].values)
        #     if all(checks):
        #         print(checks)
        #         print(x)
        #         assert False
            return all(checks)
        SF = SF.groupby(["task"]).filter(F)


    #  ==== ONLY INCLUDE LAST (OF FIRST EPOCH) AND FIRST (OF LAST EPOCH) TRIALS
    if only_first_last_trials:
        # -- get last trial for first epoch
        tmp = SF[SF["epoch"]==1]
        tmp = tmp[tmp.groupby(["task", "epoch"])["datetime"].transform(max) == tmp["datetime"]]


        # -- get first trial of last epoch
        tmp2 = SF[SF["epoch"]==2]
        tmp2= tmp2[tmp2.groupby(["task", "epoch"])["datetime"].transform(min) == tmp2["datetime"]]

        # -- combine in a new dataframe
        SF = pd.concat([tmp, tmp2])

    return SF