## stuff to convert Probedat to a standardized dataset for analyses, modeling.
from tools.utils import *
import pandas as pd
import os

from pythonlib.globals import PATH_DATASET_BEH
# base_dir = os.path.expanduser("~/data2/analyses")

def generate_metadat(exptname, date_first, date_last, animal, rule="null",
        dataset_dir_kind="daily", overwrite=False):
    """ Helper to generate yaml file, 
    Generate the yaml file, {exptname}-{rule}-{animal}.yaml"
    PARAMS:
    - exptname, string, just for naming. it would still take all data, regardless exptname, between these days.
    - date_first, date_last, ints, YYMMDD
    - dataset_dir_kind, string in {'daily', 'main'}, determines where to save the yaml file. 
    (daily is fore ach day, mian is for combine across days)
    - overwrite, if GTrue, then overwrite if yaml file already exists.
    """
    
    from pythonlib.tools.expttools import writeDictToYaml
    from pythonlib.globals import PATH_DRAWMONKEY_DIR
    
    # if is daily, then make rule the date, for ease of disticntion from main
    if dataset_dir_kind=="daily":
        assert rule == "null", 'fyi rule will be overwritten by date'
        assert date_first==date_last, "otherwise why is this callled daily?"
        rule = date_first

    metadat = {
        "sdate":date_first,
        "edate":date_last,
        "strokmodel_kind": None,
        "strokmodel_tstamp": None,
        "datecategories": None, 
        "dates_for_summary": [], # for epoch 1 vs. epoch 2. pick peak of learning.
        "matchedstrokes": None, # these strokes are aggregated on each day. expect these to be comaprable across days.
        "exptnames": [], # leave empty to search for all names and get all.
        "T1": [], 
        "G1": [], 
        "G2": [], 
        "G3": [], 
        "G4": [],
        "description": "",
        "finalized": False, 
        "good_expt": True
    }
    
    if dataset_dir_kind=="daily":
        path = f"{PATH_DRAWMONKEY_DIR}/expt_metadat_daily/{exptname}-{rule}-{animal}.yaml"
    elif dataset_dir_kind=="main":
        path = f"{PATH_DRAWMONKEY_DIR}/expt_metadat/{exptname}-{rule}-{animal}.yaml"
    else:
        print("dataset_dir_kind", dataset_dir_kind)
        assert False

    # Confirm that this file doesnt exist'
    if overwrite==False:
        if os.path.exists(path):
            print(path)
            assert False, "This path already exists, are you sure you want to overwite?"

    print("SAVED FILE: ", path)
    writeDictToYaml(metadat, path)
    
    return metadat, path


def Probedat2Dat(P, extraction_params, save=True, keep_all_in_probedat=True):
    """ takes P, probedat class, and simplifies
    into dat, 
    - extraction_params, see example below.
    - save, saves based on dir given in extraction_params
    - keep_all_in_probedat, false, then prunes to just key columns.
    RETURNS:
    - DAT, pd.Dataframe
    - METADAT, dict, 
    """

#     extraction_params = {
#     "expt":expt,
#     "animal":animal,
#     "probedat_filter_params":{
#         "hausdorff_filter":True,
#         "hausdorff_filter_prctile":2.5,
#         },
#     "pix_add_to_sketchpad_edges":20,
#     "savedir":f"{base_dir}/database",
#     "savenote":"formodeling"
# }
    from pythonlib.tools.pandastools import mergeOnIndex

    Pp = P.pandas().reset_index(drop=True)

    # things to keep from P.pandas()
    if keep_all_in_probedat:
        keylist =  ["unique_task_name"] # keep for sanity check, making sure aligned.
        # since will add all in a bit
    else:
        keylist = ["animal", "date", "tval", "tvalday", "epoch", "taskgroup", "unique_task_name"]
    
    # === Convert Probedat to DAT
    DAT = []
    for i in P.getIndsTrials():
        if i%200==0:
            print(i)
        idx = P.fd_trial(i)
        
        try:
            strokes_beh = getTrialsStrokesByPeanuts(*idx)
            strokes_task = getTrialsTaskAsStrokes(*idx)
            out = getTrialsOutcomesWrapper(*idx)
            fixpos = getTrialsFix(*idx)["fixpos_pixels"]
            # origin = getTrialsFix(fd, t)["fixpos_pixels"]
            donepos = getTrialsDoneButtonPos(*idx)
            timestats = getTrialsMotorTimingStats(*idx)
            motorevents = getTrialsTimesOfMotorEvents(*idx)
            pnut_ext = getTrialsPeanutSampCollisExt(*idx)
        except Exception as err:
            print(Pp.iloc[i])
            raise err

        htime = getTrialsHoldTimeActual(*idx)
        if not np.isnan(htime):
            htime = htime[0]

        DAT.append({
            "strokes_beh":strokes_beh,
            "strokes_task":strokes_task,
            "trial_end_method":out["trial_end_method"],
            "online_abort":out["online_abort"]["failure_mode"],
            "task_stagecategory":P.Probedat[i]["task_stagecategory"],
            "origin":fixpos,
            "donepos": donepos,
            "motortiming":timestats,
            "motorevents":motorevents,
            "holdtime":htime,
            "delaytime":getTrialsStimDelayTime(*idx),
            "pnut_ext":pnut_ext.item()
            })
        
        for k in keylist:
            DAT[-1][k] = Pp[k].values[i]
            
    DAT = pd.DataFrame(DAT)

    #### Add things
    # datetime
    def F(fd, t):
        return getTrialsDateTime(fd, t, fmt="str")
    dt = P.getTrialsHelper(F, "all")
    DAT["datetime"] = dt

    # Supervision params
    tmp = P.getTrialsHelper("getTrialsSupervisionParams", "all")
    DAT["supervision_params"] = tmp

    # Online abort params
    tmp = P.getTrialsHelper("getTrialsAbortParams", "all")
    DAT["abort_params"] = tmp

    #################################
    # keep all from pandas?
    if keep_all_in_probedat:
        assert len(DAT)==len(Pp)
        assert np.all(DAT.index == Pp.index)
        assert (DAT["unique_task_name"] == Pp["unique_task_name"]).all()
        DAT = mergeOnIndex(DAT, Pp)

    ## Find "ground truth" sketchpad edges, as in the params.
    spadlist = P.getTrialsHelper("getTrialsSketchpad", "all")
    # Make sure each trials sketpad is identical
    for s1, s2 in zip(spadlist[:-1], spadlist[1:]):
        if s1 is None:
            assert s2 is None
        elif s2 is None:
            assert s1 is None
        else:
            assert np.all(np.isclose(s1, s2))
    sketchpad_edges_pixcentered = spadlist[0]

    # -- optionally add 20 pix to end, since some beh goes a bit over.
    add = extraction_params["pix_add_to_sketchpad_edges"]
    if sketchpad_edges_pixcentered is not None:
        sketchpad_edges_pixcentered[0,:] = sketchpad_edges_pixcentered[0,:] - add
        sketchpad_edges_pixcentered[1,:] = sketchpad_edges_pixcentered[1,:] + add

    METADAT = {
        "sketchpad_edges":sketchpad_edges_pixcentered,
        "metadat_probedat":P.Metadat
    }

    # Save filedata params
    # just use the first filedata
    fd_list = [P.fd(i) for i in P.getIndsTrials()]
    keys_to_skip = ["n_trials", "session", "fname", "max_trials_because_use_resaved_TrialRecord", "fname_dict", 
    "figuredir", "figuredir_general", 'figuredir_notebook', 'figuredir_main', 'n_trialoutcomes', 'max_trial_with_beheval',
    'date', 'expt']
    keys_good = [k for k in fd_list[0]["params"].keys() if k not in keys_to_skip]

    for key in keys_good:
        # check that this key is identical across all fd. if so, then just keep the first.
        assert([fd_list[0]["params"][key] == fd["params"][key] for fd in fd_list])
    fdgood = {k:fd_list[0]["params"][k] for k in keys_good}

    # fd_first = P.fd(0)
    # fd_params = fd_first["params"]
    METADAT["filedata_params"] = fdgood


    if save:
        from pythonlib.tools.expttools import makeTimeStamp
        import os
        import pickle

        ## SAVE AS PKL
        SDIR = extraction_params["savedir"]

        note = extraction_params["savenote"]
        a = extraction_params["animal"]
        e = extraction_params["expt"]
        r = extraction_params["rule"]
        suffix = f"{a}-{e}-{r}-{makeTimeStamp(note, False)}"
        SDIRTHIS = f"{SDIR}/{suffix}"

        os.makedirs(SDIRTHIS, exist_ok=True)

        DAT.to_pickle(f"{SDIRTHIS}/dat.pkl")

        with open(f"{SDIRTHIS}/metadat.pkl","wb") as f:
            pickle.dump(METADAT, f)
        with open(f"{SDIRTHIS}/extraction_params.pkl","wb") as f:
            pickle.dump(extraction_params, f)


    return DAT, METADAT, SDIRTHIS



def generate_dataset_file_from_raw(animal, expt, dataset_dir_kind,
        rulelist=None,
        DO_CLEANUP_BEFORE_SAVE=True, DO_PARSES = False, 
        DO_EXTRACT_TASKS=True, DO_EXTRACT_DATASET = True,
        SKIP_IF_EXISTS = False):
    """ Convert from raw daily files to a single dataset file, based on saved metadat telling 
    how to combine (e.g., whcih days)
    PARAMS:
    - dataset_dir_kind, string in {'daily', 'main'}, determines where to save the yaml file. 
    (daily is fore ach day, mian is for combine across days)
    - rulelist, either None (autoamticlalyt gets and iterates opver all rules) or list of str, each
    a rule with separate metadat file.
    - DO_CLEANUP_BEFORE_SAVE, bool, means: (i) remove trials where not fixation success,
    (ii) remove trials with no touch data recorded. (NOTE: does keep trials with abort)
    - DO_PARSES, bool, Extract permutations of the ground truth task (just of the originally defined strokes)
    (takes a while)
    - DO_EXTRACT_TASKS = True # Extract tasks.
    - SKIP_IF_EXISTS, skips if find already done (previous timestamp)
    """

    from analysis.modelexpt import loadProbeDatWrapper, loadMultDataForExpt
    from analysis.probedatTaskmodel import ProbedatTaskmodel
    from analysis.dataset import Probedat2Dat
    import os, pickle
    from pythonlib.drawmodel.taskgeneral import TaskClass as TaskClassGen
    from pythonlib.dataset.dataset_preprocess.general import get_rulelist

    pix_add_to_sketchpad_edges = 20
    SAVENOTE = ""
    SDIR = f"{PATH_DATASET_BEH}/BEH"

    #############################
    if DO_PARSES:
        assert False, "not yet coded. look into the dataset extraction notebook"
    assert DO_CLEANUP_BEFORE_SAVE==True, "if turn off, then I think is the flag cleanup_remove_if_no_touch in loadProbeDatWrapper, but you need to confirm"

    if rulelist is None:
        rulelist = get_rulelist(animal, expt)#, dataset_dir_kind=dataset_dir_kind)
    else:
        assert isinstance(rulelist, list)
        for rule in rulelist:
            assert isinstance(rule, str)

    for rule in rulelist:
        
        # Check if already exists
        if SKIP_IF_EXISTS:
            import glob
            # path_partial_string = f"{animal}-{expt}-{rule}-{makeTimeStamp(note, False)}"
            path_partial_string = f"{SDIR}/{animal}-{expt}-{rule}-"
            files = glob.glob(f"{path_partial_string}*")
            if len(files)>0:
                # Then found it
                print("[generate_dataset_file_from_raw] SKIPPING, since found previous: ", files, " that matches current: ", path_partial_string)
                continue

        FD, exptMetaDat = loadMultDataForExpt(expt, animal, metadatonly=False, rule=rule, 
            dataset_dir_kind=dataset_dir_kind)
        PD = loadProbeDatWrapper(FD, exptMetaDat) 
        P = ProbedatTaskmodel(PD, exptMetaDat)

        extraction_params = {
            "expt":expt,
            "animal":animal,
#                 "probedat_filter_params":{
#                     "hausdorff_filter":True,
#                     "hausdorff_filter_prctile":2.5,
#                     },
            "rule":rule,
            "probedat_filter_params":{},
            "pix_add_to_sketchpad_edges":pix_add_to_sketchpad_edges,
            "savedir":SDIR,
            "savenote":SAVENOTE,
            "dataset_dir_kind":dataset_dir_kind
        }

        # ==== filter trials based on behavioral criteria, to throw out noise.
        assert len(extraction_params["probedat_filter_params"]) == 0, "Don't do this - do any filtering you want AFTER constructing the dataset"
        ProbedatFiltered = P.filterByBehPerformance(extraction_params["probedat_filter_params"])

        # Reconstruct P
        P = ProbedatTaskmodel(ProbedatFiltered, P.Metadat)

        # === Convert Probedat to DAT
        if DO_EXTRACT_DATASET:
            DAT, METADAT, SDIRTHIS = Probedat2Dat(P, extraction_params, save=True, 
                keep_all_in_probedat=True) 
        
        # Extract and save blockparams
        BlockParamsByDateSessBlock = P.extractBlockParams()
        with open(f"{SDIRTHIS}/BlockParamsByDateSessBlock.pkl", "wb") as f:
            pickle.dump(BlockParamsByDateSessBlock, f)

        # === EXTRACT TASKS
        if DO_EXTRACT_TASKS:    
            print("-- EXTRACTING TASKS")
            # extract all tasks
            Tasks = P.extractTasksAsClass("all")
            trialcodes = P.pandas()["trialcode"]
            unique_task_names = P.pandas()["unique_task_name"]
            task_stagecategorys = P.pandas()["task_stagecategory"]
            taskgroups = P.pandas()["taskgroup"]
            assert len(trialcodes)==len(Tasks)

            # convert all to general class
            Tasks2 = []
            for i, T in enumerate(Tasks):
                Tasks2.append(TaskClassGen())
                Tasks2[-1].initialize(ver="ml2", params=T)
            Tasks = Tasks2

            # save, lsit of dixcts
            TaskDat = []
            for T, tc, u, st, gr in zip(Tasks, trialcodes, unique_task_names,
                                        task_stagecategorys, taskgroups):
                TaskDat.append({
                    "Task":T,
                    "trialcode":tc,
                    "unique_task_name":u,
                    "task_stagecategory": st,
                    "taskgroup":gr
                })

            sdir = f"{PATH_DATASET_BEH}/TASKS_GENERAL/{animal}-{expt}-{rule}-all"
            os.makedirs(sdir, exist_ok=True)
            with open(f"{sdir}/Tasks.pkl", "wb") as f:
                pickle.dump(TaskDat, f)



# DATASET EXTRACTION
if __name__=="__main__":
    # exptlist = ["gridlinecircle"]
    # animallist = ["Pancho"]
    # main_or_daily = "main" # 'main' or 'daily' depends on which metadat folder
    # rulelist = ["baseline", "circletoline", "linetocircle", "lolli"]
    
    exptlist = ["priminvar3"]
    animallist = ["Pancho"]
    main_or_daily = "main" # 'main' or 'daily' depends on which metadat folder
    rulelist = ["null"]

    # exptlist = ["primsingridrand3"]
    # animallist = ["Pancho"]
    # main_or_daily = "daily" # 'main' or 'daily' depends on which metadat folder
    # rulelist = [""]
    # rulelist = ["null"]
    # exptlist = ["primsingrid1"]
    # animallist = ["Pancho"]
    # rulelist = ["null"]

    # SKIPS = []
    for animal in animallist:
        for expt in exptlist:
            generate_dataset_file_from_raw(animal, expt, main_or_daily)

    #     except Exception as err:
    #         if DOSKIP:
    #             SKIPS.append([expt, rule, animal])
    #         else:
    #             raise err
    # print("*** SKIPPED THESE DUE TO ERRORS:")
    # print(SKIPS)