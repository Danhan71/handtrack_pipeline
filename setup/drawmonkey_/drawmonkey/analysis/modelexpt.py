""" analysis of expts training and testing leraning of different drawing models.
"""
from tools.utils import * 
from tools.plots import *
from tools.analy import *
from tools.calc import *
from tools.analyplot import *
from tools.preprocess import *
from tools.dayanalysis import *
from analysis.line2 import *

from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *
from pythonlib.globals import PATH_DRAWMONKEY_DIR
import os

# metadat_dir = os.path.expanduser("~/data1/code/python/drawmonkey/expt_metadat")
# metadat_dir = f"{PATH_DRAWMONKEY_DIR}/expt_metadat"

def loadMultDataForExpt(expt, animal, whichdates="all", metadatonly=False, rule=None,
        dataset_dir_kind="main"):
    """ wrapper that gets all data (FD) for a given aniaml
    and expt.
    - whichdates, determines which dates to extract.
    -- "all", then all
    -- "summary", then dates I encoded as summary
    - metadatonly, whetrher to jsut get dict telling me for each date
    what are viable sessions.
    RETURNS:
    - (FD, exptMetaDat)
    - sessdict, if metadatonly ==True (and will be much faster.)
    v2: (if span multiopel expts)
    - expt = None, then give me dates:
    - whicdates = ["201001", "201002"]
    - rule, a string identifying the model being learned. This indexes into the metadata yaml
    file that has this rule in its name. New, 6/5/21 onwards.
    - dataset_dir_kind, string in {'daily', 'main'}, determines where to save the yaml file. 
    (daily is fore ach day, mian is for combine across days)    
    """
    from pythonlib.tools.datetools import getDateList
    
    if expt is None:
        metadatonly = False

    # 1) Load metadat
    if expt is None:
        MD = {}
        datelist = sorted(whichdates)
        MD["sdate"] = datelist[0]
        MD["edate"] = datelist[-1]
    else:
        MD = loadMetadat(expt, subject=animal, rule=rule, dataset_dir_kind=dataset_dir_kind)
        print("This is metadat:")
        print(MD)

    # 2 which dates
    if isinstance(whichdates, (tuple, list)):
        if isinstance(whichdates[0], str):
            datelist = whichdates
        elif whichdates[0]>1000:
            # then this must be a date
            datelist = whichdates
        else:
            # This is integer, i.e,, indices to pick out datse
            # e..g, [0, 2] means 1st and 3rd dates
            datelist = getDateList(MD["sdate"], MD["edate"])
            datelist = [datelist[i] for i in whichdates]

    elif whichdates=="all":
        datelist = getDateList(MD["sdate"], MD["edate"])
    elif whichdates=="summary":
        datelist = MD["dates_for_summary"]
    else:
        print(whichdates)
        assert False, "bnot coded"
        
    print("\nGetting these dates:")
    print(datelist)
    
    if metadatonly:
        sessdict = getSessionsList(animal, expt, datelist, return_also_exptname=False)
        return sessdict
    else:
        # 3) Extract FD
        dattoget = []
        for d in datelist:
            # try all the possible real expt names
            for e in MD["exptnames"]:
                dattoget.append([e, animal, d])

        FD = loadMultData(dattoget) 
        exptMetaDat = MD

        # Only keep desired sessions
        print("FD sessions: ", [(x["date"], x["session"]) for x in FD])
        if exptMetaDat["ssess"] is not None:
            # Then the first day starts on session after sess 1
            # Remove fd from first day, session before this
            first_date = exptMetaDat["sdate"]
            first_sess = exptMetaDat["ssess"]
            indices_remove = []
            for i, x in enumerate(FD):
                if int(x["date"])==int(first_date) and x["session"]<first_sess:
                    # Then remove
                    indices_remove.append(i)
            # print(first_date, first_sess, indices_remove)
            FD = [x for i, x in enumerate(FD) if i not in indices_remove]
        if exptMetaDat["esess"] is not None:
            # Then the last day ends before the lsat session fothe day
            last_date = exptMetaDat["edate"]
            last_sess = exptMetaDat["esess"]
            indices_remove = []
            for i, x in enumerate(FD):
                if int(x["date"])==int(last_date) and x["session"]>last_sess:
                    # Then remove
                    indices_remove.append(i)
            FD = [x for i, x in enumerate(FD) if i not in indices_remove]
        print("FD sessions, after removing sessions: ", [(x["date"], x["session"]) for x in FD])
        return FD, exptMetaDat


def loadProbeDatWrapper(FD, MD=None, getnumstrokes=False, 
        cleanup_remove_if_no_touch=True):
    """ loads provedat, which is flattened trials cross all
    in FD, which could even be acorss aniammsl. usually across days
    for a given anmiaml (one expt).
    - various postprocessing 
    - FD and MD can get from loadMultDataForExpt
    - MD should be providde, if None, then output is bare minimum, e.g.
    if want to analyze a day but dont carea bout its experimental params.
    """

    # ==== Flatten all trials across days x animals
    # for each trial collect relevant information
    PROBEDAT = PROBEDATfromFD(FD, cleanup_remove_if_no_touch=cleanup_remove_if_no_touch) 
    
    # --- assign standardized time.
    if MD is not None:
        PROBEDAT = getStandardizedTime(PROBEDAT, MD)

        # -- epoch
        for P in PROBEDAT:
            if "datecategories" in MD.keys():
                try:
                    P["epoch"] = MD["datecategories"][P["date"]]
                except:
                    P["epoch"] = MD["datecategories"][str(P["date"])]
            else:
                P["epoch"] = 1
            if "expt" in MD.keys():            
                P["taskgroup"] = assignTaskGroup(P, metadat = MD)
            else:
                P["taskgroup"] = "unassigned"
            if "dates_for_summary" in MD.keys():           
                P["insummarydates"]=P["date"] in MD["dates_for_summary"]
            else:
                P["insummarydates"] = False

    # === only include data that has beahvior
    if cleanup_remove_if_no_touch:
        PROBEDAT = [P for P in PROBEDAT if getTrialsTouched(P["filedata"], P["trial"])]

    # === extract actual num strokes
    if getnumstrokes:
        for i, P in enumerate(PROBEDAT):
            S = getTrialsStrokesByPeanuts(P["filedata"], P["trial"])
            P["nstrokesactual"] = len(S)

    # === spurious tasks, so just remove (specific lto lines5, 
    # I turned on the probes too qucily
    # at onset of first day fo switch
    PROBEDAT= [P for P in PROBEDAT if not (P["date"]=="200925" and 
                                          P["task_stagecategory"]=="linePlusLv2" and P["taskgroup"]=="G2")]

    return PROBEDAT




def loadMetadat(expt, subject=None, rule=None, dataset_dir_kind="main"):
    """ save here metadat for model-based training expts.
    In general, enter None if skip.
    - if pass in rule, then will look for metadat file called 
    {expt}-{rule}-{subject}.
    """

    if rule is not None:
        assert subject is not None, "yaml metadats are named  {expt}-{rule}-{subject}"


    if dataset_dir_kind=="main":
        metadat_dir = f"{PATH_DRAWMONKEY_DIR}/expt_metadat"
    elif dataset_dir_kind=="daily":
        metadat_dir = f"{PATH_DRAWMONKEY_DIR}/expt_metadat_daily"
    else:
        print(dataset_dir_kind)
        assert False

    if True:
        # Try loading yaml file
        import yaml
        if rule is not None:
            tmp  = f"{expt}-{rule}-{subject}"
        else:
            # old version, one file for all rules and animals.
            tmp = expt
        with open(f"{metadat_dir}/{tmp}.yaml") as file:
            exptMetaDat = yaml.load(file, Loader=yaml.FullLoader)
            if exptMetaDat is None:
                print(file)
                print(expt, subject, rule)
                assert False
        # print(exptMetaDat)
        # assert False


    else:
        exptnames = None # if none, then assumes that expt is the only exptname
        # otherwise enter a list of strings, each a name. will look for any session with 
        # any of these names, 
        if expt=="mem123":
            sdate = 201213
            edate = 201220

            # ==== STROKES MOTOR MODEL PARAMS
            strokmodel_kind=None
            strokmodel_tstamp = None
            
            # ==== expt schedule
            datecategories = None
            dates_for_summary = None # for epoch 1 vs. epoch 2. pick peak of learning.
            # assert False, "need to allow all strokenums, change matched strokes." 
            matchedstrokes = None # these strokes are aggregated on each day. expect these to be comaprable across days.

            # -- diff expt anmes were used in real life
            exptnames = ["mem1", "mem2", "mem3"]


        elif expt=="lines2":
            sdate = 200902
            edate = 200907

            # ==== STROKES MOTOR MODEL PARAMS
            strokmodel_kind='spatial'
            strokmodel_tstamp = '200922_093340_lines2'
            
            # ==== expt schedule
            datecategories = {
                    "200902":1, 
                    "200903":1,
                    "200904":2,
                    "200905":2,
                    "200906":2,
                    "200907":2}
            dates_for_summary = ["200902", "200903", "200906", "200907"] # for epoch 1 vs. epoch 2. pick peak of learning.
            matchedstrokes = [0,1] # these strokes are aggregated on each day. expect these to be comaprable across days.

        # elif expt=="lines5":
        #     sdate = 200921
        #     edate = 201001    
            
        #     # ==== STROKES MOTOR MODEL PARAMS
        #     strokmodel_kind="spatial"
        #     strokmodel_tstamp = "201006_032853_lines5"
            
        #     # ==== expt schedule
        #     datecategories = {
        #             "200921":1, 
        #             "200922":1, 
        #             "200923":1, 
        #             "200924":1, 
        #             "200925":2,
        #             "200926":2,
        #             "200927":2,
        #             "200928":2,
        #             "200929":2,
        #             "200930":2,        
        #             "201001":2}
        #     dates_for_summary = ["200923", "200924", "200929", "200930", "201001"] # for epoch 1 vs. epoch 2. pick peak of learning.
        #     # assert False, "need to allow all strokenums, change matched strokes." 
        #     matchedstrokes = None # these strokes are aggregated on each day. expect these to be comaprable across days.
        
        elif expt=="arc2":
            sdate = 201029
            edate = 201110
            
            # ==== STROKES MOTOR MODEL PARAMS
            strokmodel_kind=None
            strokmodel_tstamp = None
            
            # ==== expt schedule
            datecategories = None
            dates_for_summary = None # for epoch 1 vs. epoch 2. pick peak of learning.
            # assert False, "need to allow all strokenums, change matched strokes." 
            matchedstrokes = None # these strokes are aggregated on each day. expect these to be comaprable across days.

        # if exptnames is None:
        #     exptnames = [expt]

        exptMetaDat = {
        "sdate":sdate,
        "edate":edate,
        "strokmodel_kind":strokmodel_kind,
        "strokmodel_tstamp":strokmodel_tstamp,
        "datecategories":datecategories,
        "dates_for_summary":dates_for_summary,
        "matchedstrokes":matchedstrokes,
        "exptnames":exptnames
        }

    exptMetaDat["expt"] = expt
    exptMetaDat["animal"] = subject
    exptMetaDat = cleanUpMetadat(exptMetaDat)

    return exptMetaDat


def cleanUpMetadat(MD):
    """any entry that is none, replaces with defaiult."""
    from pythonlib.tools.datetools import getDateList

    # If dates were entered as lists of [date, session], break out into date and session
    if isinstance(MD["sdate"], list):
        date, sess = MD["sdate"]
        MD["sdate"] = date
        MD["ssess"] = sess
    else:
        MD["ssess"] = None # all sessions

    if isinstance(MD["edate"], list):
        date, sess = MD["edate"]
        MD["edate"] = date
        MD["esess"] = sess
    else:
        MD["esess"] = None # all sessions


    if MD["datecategories"] is None:
        # assign caregory of 1 (epoch) for each date
        d1 = MD["sdate"]
        d2 = MD["edate"]
        datelist = getDateList(d1, d2)
        dc = {}
        for d in datelist:
            dc[d] = 1
        MD["datecategories"] = dc

    if MD["dates_for_summary"] is None:
        MD["dates_for_summary"] = []
    else:
        # convert to strings
        MD["dates_for_summary"] = [str(d) for d in MD["dates_for_summary"]]

    # If expt is empty, then get all expt names in this date range
    skip=False
    if MD["exptnames"] is not None:
        if len(MD["exptnames"])>0:
            # then you entered a list, ignroe theis
            skip=True
    if not skip:
        print(MD)
        d1 = MD["sdate"]
        d2 = MD["edate"]
        datelist = getDateList(d1, d2)
        sessdict = getSessionsList(MD["animal"], datelist=datelist)
        print("---")
        for k, v in sessdict.items():
            # example v: [(1, 'charpsychorel1'), (2, 'charparts1b')]
            for sessdat in v:
                exptthis = sessdat[1]
                if exptthis not in MD["exptnames"]:
                    MD["exptnames"].append(exptthis)        

        # also append the input expt name
        if MD["expt"] not in MD["exptnames"]:
            MD["exptnames"].append(MD["expt"])

    if len(MD["exptnames"]) == 0:
        MD["exptnames"] = [MD["expt"]]


    MD["task_train_test"] = {
            'probe1_liketrain':"train",
            'probe1_nostrokeconstraint':"train",
            'probe2_liketrain':"train",
            'probe2_nostrokeconstraint':'train',
            'probe3_hdpos':"test",
            'probe1':"train",
            'probe2':"train",
            'probe3':"test",
            'probe4':'test',
            'train':"train"}


    return MD
        

def assignTaskGroup(PROBEDAT_single, metadat, allow_undefined=True):
    """ assigns task based on trainnig./generalization
    catregorye - ge..g, train_fixed, or G3, ...
    - have to ahnd enter for each expt
    INPUTS:
    - allow_undefined, then if using ver2, and this taskset not defined, then calls it "undefined"
    """

    if "tasknames_G1" in metadat.keys():
        assert "G1" not in metadat.keys()
        ver = 1
    elif "G1" in metadat.keys():
        assert "tasknames_G1" not in metadat.keys()
        ver = 2
    else:
        print(metadat)
        assert False

    P = PROBEDAT_single
    expt = metadat["expt"]

    if ver==2:
        # new version, bsaed on setnum
        def _assignTaskGroup(metadat, taskname, setnum=None, 
                            kindlist = ("T1", "T2", "G1", "G2", "G3", "G4")):
            """ extracts what kind of task this is
            INPUT:
            - taskname, str, e.g., mixture2
            - setnum, 
            --- int,
            --- [], means any set for this task.
            NOTE;
            - does sanity check to make sure you entered correctly in yaml, or else will fail.
            """
            
            def _check(kind):
                # go thru each task-set that is in this kind.
                if kind not in metadat.keys():
                    return False
                for x in metadat[kind]:
                    if x[0]==taskname:
                        if setnum is None:
                            # then found it
                            return True
                        elif len(x[1])==0:
                            # then this applies for any set
                            return True
                        elif setnum in x[1]:
                            # then found this task and set
                            return True
                # Then didnt find this task, so is not this kind.
                return False
            
            correctkind = [k for i, k in enumerate(kindlist) if _check(k)]
            
            if len(correctkind)>1:
                print(metadat)
                print(correctkind)
                print(taskname, setnum)
                assert False, "you made a mistake in yaml metadat, same task across diff kinds"
            elif len(correctkind)==0:
                if allow_undefined:
                    return "undefined"
                else:
                    print(metadat)
                    print(taskname, setnum)
                    assert False, "you failed to enter this task in yaml metadat"
            else:
                return correctkind[0]

        if P["kind"]=="train" and P["random_task"]==True:
            taskgroup = "train_random"
        else:
            # NOTE: I used to assuem that cant be protytpe if is saveset, but that is
            # not true, since sometimes the tasks that were saved were first made as proprtypes.

            if P["resynthesized"]!=0:
                for k, v in P.items():
                    if k!="filedata":
                        print(k, v)
                assert False,  "currently only works for saved sets. solution: modify yaml to assign a kind for prototipes."
            
            if not isinstance(P["saved_setnum"], int):
                print(P)
                print(P["saved_setnum"])
                [print(k,v) for k,v in P.items()]
                assert False, "expect that is in all non-saved tasks (i..e, random) should have gotten above (i.e,, None or [])."
            taskgroup = _assignTaskGroup(metadat, P["task_stagecategory"], P["saved_setnum"])
                

    if ver==1:
        if "tasknames_G1" in metadat.keys():
            # later expts (yaml configs) did this
            tasknames_G1 = metadat["tasknames_G1"]
            tasknames_G2 = metadat["tasknames_G2"]
            tasknames_G3 = metadat["tasknames_G3"]
        else:
            # earlier expts, hard coded here.
            if expt=="lines5":
                tasknames_G1 = []
                tasknames_G2 = ["C", "triangle", "S", "h", "tristar", "F", "linePlusLv2"]
                tasknames_G3 = ["2linePlusL", "LplusL", "linePlusLv2", "3linePlusL"]
            elif expt=="arc2":
                tasknames_G1 = []
                tasknames_G2 = []
                tasknames_G3 = ["triangle", "linePlusLv2", "LplusL", "2linePlusL", "3linePlusL",
                "lineLCircle", "lineLCircle2", "triangle_circle"]
            elif expt in ["mem1", "mem2", "mem3", "mem123"]:
                # no generalization, just training.
                tasknames_G1 = []
                tasknames_G2 = []
                tasknames_G3 = []
            else:
                print(expt)
                assert False, "not coded for any other expt"

        taskgroup = ""
        if P["kind"]=="train":
            if P["random_task"]==True:
                taskgroup = "train_random"
            elif P["random_task"]==False:
                taskgroup = "train_fixed"
        elif "probe1" in P["kind"]:
            assert P["random_task"]==False
            taskgroup = "train_fixed"
        elif "probe2" in P["kind"]:
            assert P["random_task"]==True
            taskgroup = "train_random"
        elif "probe3" in P["kind"]:
            assert P["random_task"]==False
            # fixed, and no model-based feedback
            if P["task_stagecategory"] in tasknames_G1:
                taskgroup = "G1"
            elif P["task_stagecategory"] in tasknames_G2:
                taskgroup = "G2"
            elif P["task_stagecategory"] in tasknames_G3:
                taskgroup = "G3"
            else:
                print(P.keys())
                for k, v in P.items():
                    if k!="filedata":
                        print(k, v)
                print(P["task_stagecategory"])
                print(tasknames_G3)
                assert False
                taskgroup = "test_fixed"
        elif "probe4" in P["kind"]:
            assert P["random_task"]==True
            taskgroup = "test_random"

        # specific for lines5
        if expt=="lines5":
            if P["task_stagecategory"]== "linePlusLv2" and P["kind"]=="probe3_hdpos":
                taskgroup="train_fixed"

        if taskgroup == "":
            print(P["kind"])
            print(P["random_task"])
            print(P["unique_task_name"])
            print(P["task_stagecategory"])
            print(tasknames_G3)
            print(P["task_stagecategory"] in tasknames_G3)
            assert False, "did not categorize this task"
    return taskgroup


def extractgrid(FD, datelist, taskunique):
    """ returns number of dates and max num sessions
    useful if want to see scope of data beofre run analsys,
    """
    ndates = 0
    nsess_list = []
    for date in datelist:   
        FDthis = [F for F in FD if F["date"]==date]
        nsess = 0
        for F in FDthis:
            fd = F["fd"]
            sess = F["session"]
            # - check how many trials there are
            trials = [t for t in getIndsTrials(fd) if getTrialsUniqueTasknameGood(fd, t) == taskunique]
            if len(trials)>0:
                nsess+=1
        if nsess>0:
            ndates+=1
        nsess_list.append(nsess)
    return ndates, nsess_list

def plotWaterfallAcrossDateSess(FD, datelist, taskunique, 
                                chunkmodel, chunkmodel_idx=0, flipxy=True):
    """ returns None if even a single plot did not have enough parses to 
    match desired chunkmodel_idx"""


    # === RUN
    waterfallkwargs = {
        "align_by_firsttouch_time":True, 
        "normalize_strok_lengths":False,
        "xaxis":"dist",
        "align_all_strok":False,
        "trialorder":"nogaps"
    }

    ndates, nsess_list = extractgrid(FD, datelist, taskunique)
    nrows = ndates
    ncols = max(nsess_list)

    print(nrows)
    print(ncols)
    if flipxy:
        w, h = (7, 4)
        sharex, sharey = (False, True)
    else:
        w, h = (4,7)
        sharex, sharey = (True, False)
    if waterfallkwargs["trialorder"]=="nogaps":
        sharex, sharey = (True, True)

    figwf, axes = plt.subplots(nrows, ncols, figsize=(ncols*w, nrows*h), sharex=sharex, sharey=sharey, squeeze=False)
    row=0
    for date in datelist:

        # 1)
        FDthis = [F for F in FD if F["date"]==date]

        col=0
        for F in FDthis:
            fd = F["fd"]
            sess = F["session"]
            taskplotted = False
            # - get trials of interest
            trials = [t for t in getIndsTrials(fd) if getTrialsUniqueTasknameGood(fd, t) == taskunique]

            if len(trials)>0:
                tmp = plotMultTrialsWaterfall(fd, trials, ax=axes[row][col], 
                                        colorver="taskstrokenum_fixed", 
                                        chunkmodel=chunkmodel, chunkmodel_idx=chunkmodel_idx,
                                        flipxy=flipxy, 
                                        waterfallkwargs=waterfallkwargs)
                axes[row][col].set_title(f"{date}-s{sess}")
                col+=1
                if isinstance(tmp, str) and tmp=="failed":
                    return None, None
        if col>0:
            # then plotted. go to next row
            row+=1

    #             # === make the last column example of task.
    #             if taskplotted==False:
    #                 ax = axes[row][ncols]
    #                 plotTaskWrapper(fd, trials[0], ax=ax, chunkmodel=chunkmodel)
    #                 taskplotted = True

            if False:
                plotMultTrialsSimple(fd, trials, zoom=True, strokes_ver="peanuts")


    # == separate plot of tasks.
    figtask, axes = plt.subplots(nrows, ncols, figsize=(ncols*w, nrows*h), sharex=sharex, sharey=sharey, squeeze=False)
    row=0
    for date in datelist:

        # 1)
        FDthis = [F for F in FD if F["date"]==date]
        col=0
        for F in FDthis:
            fd = F["fd"]
            sess = F["session"]
            taskplotted = False
            # - get trials of interest
            trials = [t for t in getIndsTrials(fd) if getTrialsUniqueTasknameGood(fd, t) == taskunique]
            if len(trials)>0:
                ax = axes[row][col]
                plotTaskWrapper(fd, trials[0], ax=ax, chunkmodel=chunkmodel)
                taskplotted = True
                axes[row][col].set_title(f"{date}-s{sess}")
                col+=1
        if col>0:
            # then plotted. go to next row
            row+=1

    return figwf, figtask


def modelResultsGetter(PROBEDAT, model="spatial", fit_tstamp = "200922_093340_lines2"):
    """ given PROBEDAT, return a function that you can use to pull out model 
    results for a given trial/stroke
    - model and fit_tstamp index a particular run of model fitting.
    RETURNS:
    - resultGetter, a function that takes in specific details of expt (down to level
    of strokenum) and extracts, from processed dataframe, model fit results.
    NOTE: can fail silently, in that if dosent find something, will return a None.
    should modify so that is known what should or should not exist.
    """
    
    # 1) Load previously saved strok model fits.
#     model = "spatial"
#     fit_tstamp = "200922_093340_lines2"
    from analysis.line2_strokmodelfits import postProcess 

    datlist = set([(P["animal"], P["session"], P["date"],  P["expt"]) for P in PROBEDAT])

    # for each combo, load and put in a dict, so that can align trials later.
    MODELDAT = []
    for dat in datlist:
        try:
            strokdat, DF, DF2, fd = \
                postProcess(dat[0], dat[1], dat[2], dat[3], fit_tstamp=fit_tstamp, model=model,ploton=False)

            print(f"** GOOD! Found for this dat: {dat}")
            MODELDAT.append({
                "animal":dat[0],
                "date":dat[2],
                "expt":dat[3],
                "session":dat[1],
                "DF":DF,
                "DF2":DF2
            })
        except FileNotFoundError as err:
            print(err)
            print('SKIPPING')
            continue

    
    def resultGetter(animal, date, expt, session, trial, strok_num):
        """function to extract model stats for a given trial/stroke, 
        given the MODELDAT listy of dicts
        - returns None if can't find for whatever reason
        By default, extracts the model1/model2 score for the trial, which is 
        one df row. could also choose to extract one row for each model."""

        for M in MODELDAT:
            if M["animal"]==animal and M["date"]==date and M["expt"]==expt and M["session"]==session:
                DF = M["DF"]
                DF2 = M["DF2"]

                if False:
                    DFthis = DF[(DF["trial"]==trial) & (DF["strok_num_0"]==strok_num)]
                else:
                    DFthis = DF2[(DF2["trial"]==trial) & (DF2["strok_num_0"]==strok_num)]

                if len(DFthis)==0:
                    return None
                else:
                    return DFthis
        if False:
            print("DID NOT FIND THIS DATA!")
        return None
    
#     extractDatForStroke(MODELDAT, "Pancho", "200907", "lines2", 1, 10, 1)
    return resultGetter


### INTEGRATE PROBEDAT WITH TASKMODEL



if __name__ == "__main__":
    # Overview plots of beahvior (e..g, example tasks, grid plots, overlaid trials,.
    # Combines plots over all days of the experiment.
    expt = "lines2"

    from pythonlib.tools.datetools import getDateList
    from analysis.line2 import PROBEDATfromFD
    import os
    import seaborn as sns
    # import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Load data across days
    if expt=="lines2":
        sdate = 200902
        edate = 200907
        max_strokenum = 2 # for single stroke plots
    elif expt=="lines5":
        sdate = 200921
        edate = 201001    
        max_strokenum = 5 # for single stroke plots

    datelist = getDateList(sdate, edate)

    # expt = "lines2"
    # for animal in ["Red", "Pancho"]:
    for animal in ["Red", "Pancho"]:
        
        dattoget = []
        for d in datelist:
            dattoget.append([expt, animal, d])
            
        FD = loadMultData(dattoget)

        # saving dir
        SAVEDIR = f"{FD[0]['fd']['params']['figuredir_notebook']}/analysis_modelexpt_multsession/{expt}/multday_{animal}_{sdate}_to_{edate}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(f"saving at {SAVEDIR}")

        # ==== Flatten all trials across days x animals
        # for each trial collect relevant information
        PROBEDAT = PROBEDATfromFD(FD)
        
        # === Focus on a single task. Plot its trajectory and compare difefrent days
        # - 1) visualize candidate tasks
        # a) first, only consider fixed tasks
        print("--- list of fixed tasks:")
        T = sorted(set([P["unique_task_name"] for P in PROBEDAT if P["random_task"]==False]))
        [print(t) for t in T];

        # get all fixed tasks of a particular kind
        kindlist = set([P["kind"] for P in PROBEDAT if P["random_task"]==False])

        task_per_kind = {}
        for kind in kindlist:
            tasklist = set([P["unique_task_name"] for P in PROBEDAT if P["kind"]==kind])
            task_per_kind[kind]=sorted(tasklist)

        print("tasks per kind found")
        for k, v in task_per_kind.items():
            print("----")
            print(f"=={k}")
            [print(vv) for vv in v]


        ## PLOT - all trials, 2d grid sorted by date and task category

        # == get all task stages (catregories) that have data across dates

        # only keep data for fixed tasks.
        PROBEDATthis = [P for P in PROBEDAT if P["random_task"]==False]
        datelist = sorted(set([P["date"] for P in PROBEDATthis]))

        # all categories that have fixed tasks
        stagelist = set([P["task_stagecategory"] for P in PROBEDATthis if P["random_task"]==False])

        # # for each category, check that has data across days
        # if False:
        #     # not done - currently taking all stages
        #     for stage in stagelist:
        #         datelist = set([P["date"] for P in PROBEDATthis if P["task_stagecategory"] == stage])

        # for each stage, make a 2d grid plot (date x task)
        for stage in stagelist:
            tasklist = sorted(set([P["unique_task_name"] for P in PROBEDATthis if P["task_stagecategory"]==stage]))
            if len(tasklist)>100:
                assert False, "why so many tasks for this stage?"

            # -- Plot 2d grid, separated by days
            CAT1 = ["date", datelist]
            CAT2 = ["unique_task_name", tasklist]
            fdlist = None

            # == 1) All behavior trials, all strokes overlaid
            plotargs = {"zoom":True, "plotver":"order", "markersize":4, "alpha":0.25}
            # fdlist = [P["filedata"][0]() for P in PROBEDATthis]
            # fdlist = [FD[P["ii"]]["fd"] for P in PROBEDATthis]

            fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="beh", plotargs=plotargs);
            fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_beh_{stage}.pdf")
            fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="task", plotargs=plotargs);
            fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_task_{stage}.pdf")
            plt.close("all")


            # == 2) Split by stroke number (one plot for each stroke)
            if False:
                plotargs = {"zoom":True, "plotver":"order", "markersize":3, "alpha":0.2}
                strokenums_to_plot_alone=list(range(max_strokenum+1))
                overlay_stroke_mean=False

                fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="behtask", 
                                      plotargs=plotargs, strokenums_to_plot_alone=strokenums_to_plot_alone, 
                                     overlay_stroke_mean = overlay_stroke_mean);
                fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_eachstroke_beh_{stage}.pdf")


            # == 3) All strokes (faint) and overlay average
            if False:
                plotargs = {"zoom":True, "plotver":"order", "markersize":2, "alpha":0.15}

                overlay_stroke_mean=True
                fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="beh", 
                                      plotargs=plotargs, overlay_stroke_mean=overlay_stroke_mean);
                fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_strokemeans_{stage}.pdf")
                       
            plt.close("all")

        # ==== PLOT ALL TRIALS
        tasklist = set([P["unique_task_name"] for P in PROBEDATthis if P["random_task"]==False])
        NMAX = 20 # trials to plot, starting from 1st trial int he day
        for task in tasklist:

            PD = [P for P in PROBEDATthis if P["random_task"]==False and P["unique_task_name"]==task]

            for reverse in [False, True]:
                # -- task presentation num as column
                for P in PD:
                    P["idx_today_uniquetask"] = None
                PD, countlist = probeDatIndexWithinDay(PD, task, reverse_order=reverse);

                # -- how many examples to plot?
                ntoplot = min((max(countlist), NMAX))

                # -- Plot 2d grid, separated by days
                CAT1 = ["date", datelist]
                CAT2 = ["idx_today_uniquetask", range(ntoplot)]
                fdlist = None

                # == 1) All behavior trials, all strokes overlaid
                plotargs = {"zoom":True, "plotver":"order", "markersize":8, "alpha":0.7}
                plot_task_last_col = True
                ver = "behtask"

                fig = plotTrial2dGrid(PD, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver=ver, 
                                      plotargs=plotargs, plot_task_last_col=plot_task_last_col);
                if reverse:
                    fig.savefig(f"{SAVEDIR}/egtrials_datebyexample_revchronorder_{task}.pdf")
                else:
                    fig.savefig(f"{SAVEDIR}/egtrials_datebyexample_chronorder_{task}.pdf")
                plt.close("all")