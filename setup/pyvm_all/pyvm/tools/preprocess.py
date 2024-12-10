from matplotlib import pyplot as plt
from .utils import *
import os
from pythonlib.globals import PATH_DATA_BEHAVIOR_RAW, PATH_DATA_BEHAVIOR_RAW_SERVER, PATH_MATLAB, PATH_DATASET_BEH, PATH_DATA_BEHAVIOR_RAW_SERVER2
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
# PATH_MATLAB

# base_dir = os.path.expanduser("~/data2/animals")
BASE_DIR = PATH_DATA_BEHAVIOR_RAW
BASE_DIR_SERVER = PATH_DATA_BEHAVIOR_RAW_SERVER
BASE_DIR_SERVER_2 = PATH_DATA_BEHAVIOR_RAW_SERVER2

assert PATH_DATA_BEHAVIOR_RAW is not None


def filename_decide_storage_server_where_load(animal, date):
    """
    Decide where to load beh data from, with priorities encoded,
    including server and local locations.
    Priority is server > local.
    RETURNS:
    - str name of dir, str path to dir
    (e.g., "base_dir", BASE_DIR)
    """
    extension = "bhv2"
    outdict1 = _findFilenamesGood(animal, date, extension)

    # Sometimes I have the pkl but not the original bhv2, which is saved elsewhere. pkl takes precedence
    extension = "pkl"
    outdict2 = _findFilenamesGood(animal, date, extension)

    outdict = {}
    for k in outdict1.keys():
        if len(outdict2[k])>0:
            # directory holding pkl takes precedence.
            outdict[k] = outdict2[k]
        else:
            outdict[k] = outdict1[k]
    print("OUTDICT", outdict)
    if len(outdict["base_dir_server_2"])>0:
        return "base_dir_server_2", BASE_DIR_SERVER_2
    elif len(outdict["base_dir_server"])>0:
        return "base_dir_server", BASE_DIR_SERVER
    elif len(outdict["base_dir"])>0:
        return "base_dir", BASE_DIR
    else:
        print(animal, date, outdict)
        print("... printing DEBUG notes:")
        _findFilenamesGood(animal, date, extension, DEBUG=True)
        assert False, "Cannot find beh data anywhere..."

def remove_unneeded_h5_files(animal, date):
    """ Remove all h5 expt files, checking whether bhv2 and pkl files
    exist, which means this file is unneeded. see comments within here
    """
    fnames_h5 = findFilename(animal, date, return_all_files=True, extension="h5")
    fnames_bhv2 = findFilename(animal, date, return_all_files=True, extension="bhv2")
    fnames_pkl = findFilename(animal, date, return_all_files=True, extension="pkl")

    for f_h5 in fnames_h5:
        f_bhv2 = f"{f_h5[:-3]}.bhv2"
        f_pkl = f"{f_h5[:-3]}.pkl"

        if os.path.isfile(f_bhv2) and os.path.isfile(f_pkl):
            # then is safe to delete, 
            # - has bhv2 means that h5 is not raw (bhv2 is)
            # - has pkl means h5 has served its purpose (intermeidate step towards pkl)
            print("!!! Deleting this h5 file: ", f_h5)
            os.remove(f_h5)

def cleanupDir(animal, date):
    """ only keeps the latest and largest file for each session. asserts that 
    those are the same file. autoatmically goes thru all sessions. for files not kept, 
    appends "IGNORE-" to path onset.
    - Also removes if name has "test" instead of a session num.
    """
    from pythonlib.tools.expttools import findPath, modPathFname
    sesslist = getSessionsList(animal, "*", [date], extension="bhv2", return_also_exptname=False)[date]
    print(sesslist)

    dir_behavior = filename_decide_storage_server_where_load(animal, date)[1]

    for s in sesslist:
        index = [dir_behavior, [[animal], [date], [date, f"{animal}_{s}"]], None, "bhv2"]
        pathlist1 = findPath(*index, sort_by="date", path_hierarchy_wildcard_on_ends=False)
        pathlist2 = findPath(*index, sort_by="size", path_hierarchy_wildcard_on_ends=False)

        if len(pathlist1)<2:
            continue

        # sort by size and date. 
        if pathlist1[-1] != pathlist2[-1]:
            print("PATHLIST: ", pathlist1)
            assert False, "latest file must also be the largest"

        for path in pathlist1[:-1]:
            print("*** IGNORING THIS FILE", path)
            # assasdfasfd
            modPathFname(path, "IGNORE")
            # assert False

        ## Remove all sessions called "[exptname]_[animal]_test"
        index = [dir_behavior, [[animal], [date], [date, f"{animal}_test"]], None, "bhv2"]
        pathlist = findPath(*index, sort_by="date", path_hierarchy_wildcard_on_ends=False)
        for path in pathlist:
            print("*** IGNORING THIS FILE, since is test session", path)
            modPathFname(path, "IGNORE")


def filename2params(fname, return_ext=False):
    """ convert a .h5 or .pkl file to 
    (datetime, expt, animal, session)
    - Filename must look something like this:
    201218_152930_mem2_Red_2.h5
    - if filename format is incorrect, i.e, sss
    number is not a number (i.e., "test" ibnstad of "2")
    then returns None.
    """
    from os import path
    ff = path.splitext(path.split(fname)[1])[0] # get only the final name
    ext = path.splitext(path.split(fname)[1])[1] 
    
    import re
    uscores = [m.start() for m in re.finditer("_", ff)]
    assert len(uscores)==4, "file format incorrec"

    date = int(ff[:uscores[0]])
    time = int(ff[uscores[0]+1:uscores[1]])
    expt = ff[uscores[1]+1:uscores[2]]
    animal = ff[uscores[2]+1:uscores[3]]
    # print(ff[uscores[3]+1:])

    if ff[uscores[3]+1:]=="test":
        print(f"expect this to be sess integer, but is actually not: {ff[uscores[3]+1:]} [returning None in filename2params]")
        return None
    elif len(ff[uscores[3]+1:])>2:
        print(f"expect this to be sess integer, but is actually too long to be a real session..: {ff[uscores[3]+1:]} [returning None in filename2params]")
        return None
    else:
        sess = int(ff[uscores[3]+1:])
    # except ValueError as err:
    #     print(f"expect this to be sess integer, but is actually not: {int(ff[uscores[3]+1:])} [returning None in filename2params]")
    #     return None

    if return_ext:
        return (date, time, expt, animal, sess, ext)
    else:
        return (date, time, expt, animal, sess)

def _findFilenamesGood(animal, date, extension, DEBUG=False):
    """
    Returns all filenames, across diff storage locations, e.g, server, of the
    paths of behavioral raw files.
    Useful for deciding where to load the data from.
    PARAMS:
    - 
    RETURNS:
    - outdict, e.g.,:
        {'base_dir': [],
        'base_dir_server': ['/home/lucas/mnt/Freiwald_kgupta/kgupta/macaque_data/Diego/240510/240510_113842_primdiego1d_Diego_1.bhv2',
        '/home/lucas/mnt/Freiwald_kgupta/kgupta/macaque_data/Diego/240510/240510_120039_primdiego1d_Diego_2.bhv2',
        '/home/lucas/mnt/Freiwald_kgupta/kgupta/macaque_data/Diego/240510/240510_122029_primdiego1d_Diego_3.bhv2',
        '/home/lucas/mnt/Freiwald_kgupta/kgupta/macaque_data/Diego/240510/240510_123846_primdiego1d_Diego_4.bhv2',
        '/home/lucas/mnt/Freiwald_kgupta/kgupta/macaque_data/Diego/240510/240510_124130_primdiego1d_Diego_4.bhv2'],
        'base_dir_server_2': []}
    """
    import glob

    expt = "*"
    session = "*"

    outdict = {}

    fdir = f"{BASE_DIR}/{animal}/{date}"
    fname_search = f"{fdir}/{date}_*_{expt}_{animal}_{session}.{extension}"
    if DEBUG:
        print("Searching BASE_DIR:", fname_search)
    outdict["base_dir"] = glob.glob(fname_search)

    fdir_server = f"{BASE_DIR_SERVER}/{animal}/{date}"
    fname_search_server = f"{fdir_server}/{date}_*_{expt}_{animal}_{session}.{extension}"
    if DEBUG:
        print("Searching BASE_DIR_SERVER:", fname_search_server)
    outdict["base_dir_server"] = glob.glob(fname_search_server)

    # 3/19/24 - Adding this after moving to lemur, to access the old beh data on gorilla (Pancho), here checks
    # the gorilla backup directory on server.
    fdir_server2 = f"{BASE_DIR_SERVER_2}/{animal}/{date}"
    fname_search_server2 = f"{fdir_server2}/{date}_*_{expt}_{animal}_{session}.{extension}"
    if DEBUG:
        print("Searching BASE_DIR_SERVER_2:", fname_search_server2)
    outdict["base_dir_server_2"] = glob.glob(fname_search_server2)

    if DEBUG:
        print("... Found these directories holding raw beh:")
        print(outdict)
    return outdict

def findFilename(animal, date, expt="*", session="*", extension="h5", 
    fail_if_noexist=True, take_larger_file = True, doprint=False, return_all_files=False,
    return_files_sorted=True, return_file_sizes=False):
    """get list of filenames for this exact session
    PARAMS:
    - date = yymmdd (int)
    - fail_if_noexist if False, then will allow to output empty list if no find
    returns list of filenames.
    - take_larger_file, if True then if multiple files, takes the largest one.
    wil also check assertion that largest one is most recently created.
    - expt and session default is wildcard, gets all.
    - return_all_files, overwrites "fail_if_noexist" and "take_larger_file".
    - return_files_sorted, bool, if True, then sorts alphabetically before returning.
    """
    import glob

    date = date

    if False:
        # Old method
        fdir = f"{base_dir}/{animal}/{date}"
        fname_search = f"{fdir}/{date}_*_{expt}_{animal}_{session}.{extension}"
        fnames = glob.glob(fname_search)
        
        fdir_server = f"{base_dir_server}/{animal}/{date}"
        fname_search_server = f"{fdir_server}/{date}_*_{expt}_{animal}_{session}.{extension}"
        fnames_server = glob.glob(fname_search_server)

        # 3/19/24 - Adding this after moving to lemur, to access the old beh data on gorilla (Pancho), here checks
        # the gorilla backup directory on server.
        fdir_server2 = f"{base_dir_server_2}/{animal}/{date}"
        fname_search_server2 = f"{fdir_server2}/{date}_*_{expt}_{animal}_{session}.{extension}"
        fnames_server2 = glob.glob(fname_search_server2)

        if not fdir_server==fdir:
            # Then these are different directoreis, try to find the unique file across these directoreis.
            if len(fnames)==0 and len(fnames_server)>0:
                fnames = fnames_server
            elif len(fnames)>0 and len(fnames_server)==0:
                fnames = fnames
            elif len(fnames)>0 and len(fnames_server)>0 and (len(fnames)==len(fnames_server)):
                # then assume they are same, just take fnames
                fnames = fnames
            elif len(fnames)>0 and len(fnames_server)>0:    
                # Then use the one that's local
                pass
                # print(fnames)
                # print(fnames_server)
                # assert False, "found pkl on both local and server. delete one of them"
            else:
                # found nothing --> Try server 2
                fnames = fnames_server2
    else:
        # New, first decide which directory, then do load.
        dir_behavior = filename_decide_storage_server_where_load(animal, date)[1]

        fdir = f"{dir_behavior}/{animal}/{date}"
        fname_search = f"{fdir}/{date}_*_{expt}_{animal}_{session}.{extension}"
        fnames = glob.glob(fname_search)


    if return_files_sorted:
        fnames = sorted(fnames)

    if return_all_files:
        if doprint:
            print(f"returning all matching files")
        return fnames

    if len(fnames)==0:
        if doprint:
            print(f"did not find any glob match to {fname_search}!!")
    elif len(fnames)>1:
        if doprint:
            print(f"found {len(fnames)} matches to {fname_search}!")
            print(f"matches are: {fnames}")
        if take_larger_file:
            if doprint:
                print("-- taking one filename with max size (and asserting that is most recent mod)")
            #import os
            fsizes = [os.path.getsize(f) for f in fnames]
            mtimes = [os.path.getmtime(f) for f in fnames] # mod times.
            indthis = fsizes.index(max(fsizes))
            if False:
                assert indthis == mtimes.index(max(mtimes)), "max size and most recent mod are not the same file? which to use."
            fnames = [fnames[indthis]]
    else:
        if doprint:
            print(f"found one matching filename: {fnames}")

    if fail_if_noexist:
        if len(fnames)!=1:
            print(fnames)
            assert False, "failed, since either no file or too many..."
    return fnames

def loadMultData(dattoget):
    """ 
    loads multipel filedata structures into one FD
    - will automaticalyll also collect all sessions for a given date
    - pass in expt as None, to automatically get expt name (could be multple)
    if dattoget is tuple/list:
        dattoget[0] = (expt, animal, date), 
        - e.g.,
            dattoget = []
            datelist = getDateList(sdate, edate)
            animal = "Pancho"
            expt = "lines2"
            for d in datelist:
                dattoget.append([expt, animal, d])
    if dattoget is dict, then:
        dattoget["animal"] = string (single)
        dattoget["expt"] = string (single)
        dattoget["dates"] = list of dates
    """

    if isinstance(dattoget, dict):
        tmp = []
        expt = dattoget["expt"]
        animal = dattoget["animal"]

        for d in dattoget["dates"]:
            tmp.append([expt, animal, d])
        dattoget = tmp

    
    FD = []

    # ==== if expt is wildcard, then here first find all expts
    from pythonlib.tools.expttools import extractStrFromFname
    dattogetnew = []
    for d in dattoget:
        if d[0] is None:
            # expand this, find all expts.
            fnames = findFilename(d[1], d[2], "*", doprint=False, fail_if_noexist=False)
            for f in fnames:
                # get expt
                expt = extractStrFromFname(f, "_", 2)
                dattogetnew.append([expt, d[1], d[2]])
        else:
            dattogetnew.append(d)
    dattoget = dattogetnew

    for (expt, animal, date) in dattoget:

        # simple, just try many sessions...
        fdsessions = []
        N = 12
        for session in range(10):
            print(animal, date, expt, session)
            fd = loadSingleData(animal, date, expt, session, resave_as_dict=False, load_resaved_data=True, 
                              resave_overwrite=False)
            if fd is not None:
                print(f"appending fd for sess {session}")
                FD.append({
                    "animal":animal,
                    "expt":expt,
                    "date":date,
                    "session":session,
                    "fd":fd
                })
                if session==N-1:
                    assert False, "got the end and still found.. you need to search for even more sessions..."


    print("===== SUMMARY")
    for f in FD:
        print("--")
        print(f"{f['animal']}-{f['date']}-{f['session']}: ntrials: {f['fd']['params']['n_trials']}")
        print(f"")
    return FD
    
def loadSingleDataH5(animal, date, expt, session, resave_as_dict):
    import h5py
    import numpy as np
    from tools.utilsh5 import group2dict
    fname = findFilename(animal, date, expt, session, doprint=False)[0]
    # print(fname)
    # with h5py.File(ff, "r") as f:
    #     print(f)
    ################################## LOAD FILE
    print(fname)
    with h5py.File(fname, "r") as dat:

        # try:
        # f = h5py.File(fname, "r")
        # except OSError as err:
        #     print("OS error: {0}".format(err))
        #     print(f"Skipping loading of {fname}, NOT PREPROCESSING!!")
        #     return None

        # data = f["ML"]
        print(dat["ML"].keys())

        ### recovering data when TrialRecord fails to save (modify below params for specific session)
        ##resaveTrialRecordH5##
        if animal=="Diego" and int(date) in [231213] and session in [2]: ##CHANGE for resaveTrialRecordH5
        # if animal=="Luca" and ((int(date)==230106 and session in [1,2,3]) or (int(date)==230113 and session==1) or (int(date)==230115 and session==1)): ##CHANGE for resaveTrialRecordH5
            #### RUN FIRST: dragmonkey/MonkeyLogicCode/tools/resaveTrialRecordH5.m
            #### -- converts TrialRecord.mat to .h5 containing only TR data
            #### -- will then have two .h5 files: one for TR, one for expt data
            #### -- below code merges the two into one .pkl file
            print("Loading TrialRecord.h5 separately (generated with resaveTrialRecordH5.m)")
            dir_behavior = filename_decide_storage_server_where_load(animal, date)
            fnameTrialRecord = f"{dir_behavior}/{animal}/{date}/TrialRecord_s{session}.h5" ##CHANGE for resaveTrialRecordH5 (must use this filename format)
            with h5py.File(fnameTrialRecord, "r") as datTR:
                print("datTR[ML][TrialRecord]", datTR["ML"]["TrialRecord"])
                filedata_TR = {
                    "TrialRecord":datTR["ML"]["TrialRecord"]
                }
                print("filedata_TR[TrialRecord]", filedata_TR["TrialRecord"])
                filedata_TR = group2dict(filedata_TR) # create dict with just TrialRecord data
            print("filedata_TR.keys()", filedata_TR.keys())
            TrialRecord = None
            # MAXTRIALS = len(filedata_TR["TrialRecord"]["User"]["TrialData"])
            MAXTRIALS = len(filedata_TR["TrialRecord"]["User"]["TrialOutcomes"])
        else:
            MAXTRIALS = None
            filedata_TR = None
            try:
                TrialRecord = dat["ML"]["TrialRecord"]
            except KeyError:
                print("Possibly corrupted file, sometimes will fail to save TrialRecord")
                print("To fix, use dragmonkey/MLC/tools/resaveTrialRecordH5 to make TrialRecord.h5")
                raise


        MLConfig = dat["ML"]["MLConfig"]

        ############################## EXTRACT PARAMS FOR THIS FILE
        params = {}

        params["pix_per_deg"] = MLConfig["PixelsPerDegree"][()]

        resolution = MLConfig["Resolution"][()].decode()
        h = int(resolution[0:resolution.find(' x ')])
        tmp = resolution.find(' 59 Hz')
        if tmp==-1:
            tmp = resolution.find(' 75 Hz')
        if tmp==-1:
            tmp = resolution.find(' 60 Hz')
        if tmp==-1:
            disp(resolution)
            assert False, "what is frame rate?"
        v = int(resolution[resolution.find(' x ')+3:tmp])
        assert (h==1024 and v==768), "diff resolution?"
        params["resolution"] = (h, v)
        
        # num trials
        trials = [int(key[5:]) for key in dat["ML"].keys() if (key[:5]=="Trial" and key!="TrialRecord")]
        if len(trials)==0:
            params["n_trials"]=0
        elif MAXTRIALS is not None: # then using resaved TrialRecord.h5
            params["n_trials"] = MAXTRIALS # only have full data up to last saved TR; scrap rest of 0..49 trials
        else:
            params["n_trials"] = max(trials) # then have complete data for all trials; so use all
        # params["n_trials"] = len(TrialRecord["User"]["CurrentTask"])
        params["animal"]=animal
        params["date"] = date
        params["expt"] = expt
        params["session"] = session
        params["fname"] = fname
        params["max_trials_because_use_resaved_TrialRecord"] = MAXTRIALS

        # ################## extract all trials into a dict (e.g, dict[1] = trial data)
        if False:
            # converts to dict - slow
            trials = {}
            for t in range(1, params["n_trials"]+1):
                trials[t] = group2dict(dat["ML"][f"Trial{t}"])
        else:
            # saves as hdf5 group, not dict - fast
            trials = {}
            for t in range(1, params["n_trials"]+1):
                trials[t] = dat["ML"][f"Trial{t}"]
        ########################## OUTPUT A DICT
        if TrialRecord is None:
            filedata = {
                "MLConfig":MLConfig,
                "params":params,
                "trials":trials,
                }
        else:
            filedata = {
                # "data":data,
                "TrialRecord":TrialRecord,
                "MLConfig":MLConfig,
                "params":params,
                "trials":trials,
                }

        ###### decide if convert to dict and save
        if resave_as_dict:
            # 1) conver to dict
            print('Converting to dict - may take a while, like a minute or two')
            # del filedata["data"]    # delete "data", is reduntant
        
            # if True:
            print("-- group2dict")
            filedata = group2dict(filedata)
            # else:
            #   fd = {}
            #   for key in filedata.keys():
            #       fd[key] = group2dict(filedata[key])
            

        if filedata_TR is not None:
            print("Inserting TrialRecord into experiment .bhv2, from TrialRecord.h5 (resaveTrialRecordH5)")
            print(filedata_TR)
            print("here")
            filedata["TrialRecord"] = filedata_TR["TrialRecord"]


        if resave_as_dict:
            # 2) save pickle
            import pickle #import os, pickle
            fname_dict = os.path.splitext(fname)[0] + ".pkl"
            filedata["params"]["fname_dict"] = fname_dict
            print(f"Saving pickle file {fname_dict}")
            with open(fname_dict, "wb") as f2:
                print("-- saving")
                pickle.dump(filedata, f2)

        return filedata     



def loadSingleData(animal, date, expt, session, resave_as_dict=False, 
    load_resaved_data=False, resave_overwrite=True, MINTRIALS=5):
    """f use resave_as_dict, then will
    convert to dict, so will be in memory. will also resave in the same
    directory a pickle file with same name.
    ...
    resave=False, load_resaved=False: [default], loads h5 and outputs h5, by default loads and outputs hdf5 groups, which is fast, but is slower if
    do analyses, since reading from disk.
    reseve=True, load_resaved=False: load h5, then converts to dict and also resaves as pickle
    resave=False, load_resaved=True: will load previously saved .pkl
    [NOTE]: foolproof, will not allow any other arpams
    resave_overwrite = False, then will not resave pkl if it already exists - will just load it..
    - if fewer trials than MINTRIALS, then will return None."""

    # First, check if file exists.
    if load_resaved_data:
        # Then look for pkl
        ext = "pkl"
    else:
        ext = "h5"
    if len(findFilename(animal, date, expt, session, extension=ext, fail_if_noexist=False, doprint=False))==0:
        print(f"- No {ext} file for {animal}, {date}, {expt}, {session} - returning None!")
        return None

    if resave_as_dict==True and resave_overwrite==False:
        # then do not overwrite. look for presaved. if exist, then just load
        fname = findFilename(animal, date, expt, session, extension="pkl", 
            fail_if_noexist=False, doprint=False)
        if len(fname)==1:
            print(f"Found existing presaved pkl file {fname} - Loading this (instead of resaving)")
            load_resaved_data=True
            resave_as_dict=False

    if load_resaved_data:
        import pickle
        assert(resave_as_dict==False), "if you want to load resaved (pkl) then you cannot also load h5 and resave. choose one..."
        fname = findFilename(animal, date, expt, session, extension="pkl", doprint=False)[0]
        with open(fname, "rb") as f3:
            filedata = pickle.load(f3)
        print(f"-- loaded presaved data: {fname}")
    else:
        # load from scratch (i.e., h5 data)
        filedata = loadSingleDataH5(animal, date, expt, session, resave_as_dict)

    # -- Failure modes
    if filedata is None:
        return None
    if len(filedata["trials"].keys()) < MINTRIALS:
        return None

    # print(filedata["TrialRecord"]["User"]["behEvaluation"])
    # print(filedata["trials"].keys())
    # print([getTrialsFixationSuccess(filedata, t) for t in getIndsTrials(filedata)])

    # =============== THINGS TO ADD TO PARAMS
    updateFiledataParams(filedata)

    # === SKIP if no trials fixation succes
    if not any([getTrialsFixationSuccess(filedata, t) for t in getIndsTrials(filedata)]):
        return None

    # === to reduce size in memory, clean up
    cleanupFiledata(filedata)

    # === SANITY CHECKS
    list_end_methods = [getTrialsOutcomesWrapper(filedata, i)["trial_end_method"] for i in getIndsTrials(filedata)]
    nbad = len([x for x in list_end_methods if x=="unknown_probably_hotkey_x"])
    ntot = len(list_end_methods)
    if ntot>50:
        if nbad/ntot>0.05:
            print(nbad, ntot)
            print(list_end_methods)
            assert False, "I guessed these are all unknown_probably_hotkey_x... this too many"
            # NOTE: before 8/24 I hadnt added eventcpde. so only by propcess of elimination did I
            # make this decision. but sholdnt expect >0.05 of trials to be this...
    

    return filedata


def updateFiledataParams(filedata):
    # 1) number of blocks (ideally, even if did not do all of them)
    if False:
        # not sure what this code is for, I don't use...
        # note that blockorder can be like 1,2,3,3, in which case this lead sto error.
        n_blocks = len(filedata["TrialRecord"]["BlockOrder"])
        print(n_blocks)
        print(filedata["TrialRecord"]["BlockOrder"])
        assert(n_blocks == max(filedata["TrialRecord"]["BlockOrder"]))

    animal = filedata['params']['animal']
    date = filedata['params']['date']
    dir_behavior = filename_decide_storage_server_where_load(animal, date)[1]

    filedata["params"]["basedir"] = dir_behavior
    filedata["params"]["figuredir"] = f"{dir_behavior}/{animal}/{date}/figures/{filedata['params']['session']}"
    filedata["params"]["figuredir_general"] = f"{dir_behavior}/{animal}/{date}/figures"
    filedata["params"]["figuredir_notebook"] = os.path.expanduser("~/data2/analyses/notebook")
    filedata["params"]["figuredir_main"] = os.path.expanduser("~/data2/analyses/main")

    params = filedata["params"]
    # @KGG 220712 — commenting out; handled in loadSingleDataH5 (elif MAXTRIALS is not None:)
    # 
    # if params["animal"]=="Pancho" and int(params["date"])== 210306 and params["session"] == 1:
    #     # this day used resaved trialrecord. it is a few trials short of total trials, since some trials did not perofrm.
    #     # new version (3/6/21), this is definitely the length of TrialRecord (num trials completed)
    #     filedata["params"]["n_trialoutcomes"] = len(filedata["TrialRecord"]["User"]["TrialData"])
    #     # Force to remove trials that go beyond this TrialData
    #     MAXTRIALS = filedata["params"]["n_trialoutcomes"]
    #     # print(MAXTRIALS)
    #     for t in range(MAXTRIALS+1, 691+1):
    #         # print(filedata["trials"].keys())
    #         # # print(filedata["trials"]
    #         # print(filedata["trials"][t])
    #         del filedata["trials"][t]
    #     filedata["params"]["n_trials"] = MAXTRIALS
    #     # assert False
    try:
        filedata["params"]["n_trialoutcomes"] = len(filedata["TrialRecord"]["User"]["TrialOutcomes"])
    except KeyError as error:
        filedata["params"]["n_trialoutcomes"] = ()

    # if n_trials and n_trialoutcomes off by 1, reassign to smaller
    if abs(filedata["params"]["n_trialoutcomes"] - filedata["params"]["n_trials"]) == 1:
        min_n = min(filedata["params"]["n_trialoutcomes"], filedata["params"]["n_trials"])
        filedata["params"]["n_trials"] = min_n
        filedata["params"]["n_trialoutcomes"] = min_n

    #import os
    # os.makedirs(filedata["params"]["figuredir"], exist_ok=True)
    # filedata["params"]["sample_rate"] = filedata["trials"][trial]["AnalogData"]["SampleInterval"][()] # period for sampling, in ms
    filedata["params"]["sample_rate"] = filedata["MLConfig"]["AISampleRate"][0]

    # populate task sets in BlockParams
    updateBlockParams(filedata)
    filedata["params"]["max_trial_with_beheval"] = None
    if "behEvaluation" in filedata["TrialRecord"]["User"].keys():
        # print(filedata["TrialRecord"]["User"])
        # print(filedata["TrialRecord"]["User"]["behEvaluation"].keys())
        if (len(filedata["TrialRecord"]["User"]["behEvaluation"]))>0:
            tmp = max([int(a) for a in filedata["TrialRecord"]["User"]["behEvaluation"].keys()])
            filedata["params"]["max_trial_with_beheval"] = int(filedata["TrialRecord"]["User"]["behEvaluation"][f"{tmp}"]["trialnum"])

    filedata["params"]["beh_codes"] = {
            9:"start",
            10:"fix cue",
            11:"fix cue visible",
            13:"frame skip",
            14:"manual rew",
            15:"guide",
            16:'FixationOnsetWTH',
            17:'FixationDoneSuccessWTH',
            18:"end",
            19:'FixationRaiseFailWTH',
            20:"go (draw)",
            21:"guide_on_GA",
            22:"guide_on_GA_delay",
            30:'DelayWhatIsThis', 
            40:'GoWhatIsThis',
            41:"samp1 on",
            42:"samp1 off",
            45:"done",
            46:"post",
            50:"reward",
            51:"free reward",
            52:'WaitThenHold_LT_reward',
            61:'DoneButtonVisible',
            62:'DoneButtonTouched',
            63:'DragAroundSuccess',
            64:'DragAroundAbort',
            65:'DragAroundFirstAbortNow',
            70:'hotkey_x',
            71:'DAstimevent_firstpres',
            72:'DAstimoff_finibeforepause',
            73:'DAstimoff_fini',
            74:'DAsamp1_visible_change',
            75:'DAnewpnutthisframe',
            76:'DAsound_samp1touched',
            78:'DAsound_gotallink',
            80:'ttl_trialon',
            81:'ttl_trialoff',
            91:'GAstimevent_firstpres', 
            92:'GAstimoff_fini', 
            93:'GAstimeventDelay_firstpres',
            94:'GAstimoffDelay_fini',
            101:'fix_square_on',
            102:'fix_square_off',
            103:'fix_square_on_pd',
            111:'photodiode_force_off',
            120:'DAsound_chunk',
            121:'DAsound_strokedone',
            122:'DAsound_chunkupdate',
            123:'DAsound_chunkdone',
            124:'DAsound_firstraise',
            131:'fix_cue_colored_on',
            132:'fix_cue_colored_on_v2',
            133:'fix_cue_colored_off',
            134:'fix_cue_colored_off_v2',
            135:'new_color_cue_off',
            141:'estim_ttl_1_on',
            142:'estim_ttl_2_on',
            151:'estim_ttl_stroke_on',
            152:'estim_ttl_stroke_off',
            200:'skipped_movie_frame'}

    # === screen hz
    idx = filedata["MLConfig"]["Resolution"].find("Hz")
    screen_hz = int(filedata["MLConfig"]["Resolution"][idx-3:idx-1])
    screen_period = 1/screen_hz
    if screen_hz not in [59, 60, 75]:
        print(screen_hz)
        assert False, "what is it?"
    filedata["params"]["screen_hz"] = screen_hz
    filedata["params"]["screen_period"] = screen_period
        


def _cleanup_blockparams_singleindex(blockparams):
    # blockparams could be TaskParams, BlockParams, etc.

    if "TaskSet" in blockparams.keys() or "TaskParams" in blockparams.keys():
        # then this should be repopulated to ["1"]["TaskSet"]
        return {"1":blockparams}
    else:
        return blockparams

def updateBlockParams(filedata):
    """
    Part 1: see below

    Part 2: 
    problem is that some entries in BlockParams are indexed by the task set, while others are not and are supposed to automatically populate:
    filedata["TrialRecord"]["User"]["BlockParams"] returns
    {'1': {'TaskSet': {'tasklist': {'1': 'line'},
   'numtasks': array([[20.]]),
   'MINLINELENGTH': array([[0.05]]),
   'MAXLINELENGTH': array([[0.1]])},
   ...

   but we want:
   {'1': {'TaskSet': {'tasklist': {'1': 'line'},
   'numtasks': {'1': array([[20.]])},
   'MINLINELENGTH': {'1': array([[0.05]])},
   'MAXLINELENGTH': {'1': array([[0.1]])}},
   ...

   This code does just that.
    """

    def _cleanup_blockparams_singleindex(blockparams):
        # blockparams could be TaskParams, BlockParams, etc.
        if len(blockparams)>0:
            if len(blockparams)>0 and "TaskSet" in blockparams.keys() or "TaskParams" in blockparams.keys():
                # then this should be repopulated to ["1"]["TaskSet"]
                return {"1":blockparams}
            else:
                return blockparams
        else:
            return blockparams



    BlockParams = filedata["TrialRecord"]["User"]["BlockParams"]

    ################ Part 1: convert from BlockParams["TaskSet"] to BlockParams["1"]["TaskSet"]
    # i..e, if blockparmas is length 1.
    BlockParams = _cleanup_blockparams_singleindex(BlockParams)
    # do same thing for all TaskParams
    for blocknum in BlockParams.keys():
        if len(BlockParams[blocknum])>0:
            if "TaskParams" in BlockParams[blocknum].keys():
                BlockParams[blocknum]["TaskParams"] = _cleanup_blockparams_singleindex(BlockParams[blocknum]["TaskParams"])


    # if "TaskSet" in BlockParams.keys() or "TaskParams" in BlockParams.keys():
    #     # then this should be repopulated to ["1"]["TaskSet"]
    #     # import copy
    #     # # print(BlockParams)
    #     # tmp = {
    #     #     "1":copy.deepcopy(BlockParams)
    #     # }
    #     # filedata["TrialRecord"]["User"]["BlockParams"] = tmp
    #     filedata["TrialRecord"]["User"]["BlockParams"] = {
    #         "1":BlockParams
    #     }

    ################ Part 2: convert from BlockParams["1"]["TaskSet"]["numtasks"]=... to BlockParams["1"]["TaskSet"]["numtasks"]["1"]= ...
    # BlockParams = filedata["TrialRecord"]["User"]["BlockParams"]

    def _cleanup_TaskSet(TaskSet):
        # first find what all the keys are (ie.., how many task sets, like ["1", "2"])
        list_of_task_sets = []
        for key in TaskSet:
            B = TaskSet[key]
            if isinstance(B, dict):
                # then this guy I explicitly entered infor for all task sets (1,2,3)..
                # otherwise I just put in a number and it automatily populated for all task sets
                list_of_task_sets = [keys for keys in B.keys() if isinstance(B, dict)]
                break

        if not list_of_task_sets:
            print(TaskSet)
            print(list_of_task_sets)
            print(TaskSet["tasklist"])
            print(type(TaskSet["tasklist"]))
            print(len(TaskSet["tasklist"].keys()))
            assert False, "is this because there is actually only one task set?"

        
        # populated by filling in for each task set
        for key in TaskSet:
            B = TaskSet[key]
            if not isinstance(B, dict):
                TaskSet[key] = {task_set: B for task_set in list_of_task_sets}

        return TaskSet



    for block in BlockParams:
        if len(BlockParams[block])>0:
            if "TaskSet" in BlockParams[block].keys():
                # OLD VERSION
                BlockParams[block]["TaskSet"] = _cleanup_TaskSet(BlockParams[block]["TaskSet"])

                # if len(TaskSet["tasklist"].keys())>0:

                #     # first find what all the keys are (ie.., how many task sets, like ["1", "2"])
                #     list_of_task_sets = []
                #     for key in TaskSet:
                #         B = TaskSet[key]
                #         if isinstance(B, dict):
                #             # then this guy I explicitly entered infor for all task sets (1,2,3)..
                #             # otherwise I just put in a number and it automatily populated for all task sets
                #             list_of_task_sets = [keys for keys in B.keys() if isinstance(B, dict)]
                #             break

                #     if not list_of_task_sets:
                #         print(TaskSet)
                #         print(list_of_task_sets)
                #         print(TaskSet["tasklist"])
                #         print(type(TaskSet["tasklist"]))
                #         print(len(TaskSet["tasklist"].keys()))
                #         assert False, "is this because there is actually only one task set?"

                    
                #     # populated by filling in for each task set
                #     for key in TaskSet:
                #         B = TaskSet[key]
                #         if not isinstance(B, dict):
                #             TaskSet[key] = {task_set: B for task_set in list_of_task_sets}

                # else:
                #     print("NO TASKSETS FOUND!! (is ok as long as this is desired)")
            elif "TaskParams" in BlockParams[block].keys():
                # 9/19/22 onwards, each Block can have multiple TaskParams
                for tpnum in BlockParams[block]["TaskParams"]:
                    BlockParams[block]["TaskParams"][tpnum]["TaskSet"] = _cleanup_TaskSet(BlockParams[block]["TaskParams"][tpnum]["TaskSet"])
            else:
                print(BlockParams[block])
                print(BlockParams[block].keys())
                print(BlockParams[block]["TaskParams"].keys())
                assert False, "one of those two two elif conditions above must be true ..."


            # progression levels
            if "progression" in BlockParams[block]:
                if len(BlockParams[block]["progression"]["level"])==0:
                    # then this is eithered "ignored" or an initial block to which 
                    # progression was not applied. newer matlab code labels level as
                    # 0. previously I left it empty. 
                    import numpy as np
                    BlockParams[block]["progression"]["level"] =np.array([0.])

    filedata["TrialRecord"]["User"]["BlockParams"] = BlockParams

    # for bk in BlockParams:
    #     print(BlockParams[bk]["TaskParams"].keys())
    #     for tp in BlockParams[bk]["TaskParams"]:
    #         print(BlockParams[bk]["TaskParams"][tp].keys())
    # # == replay stats, if no replays, then shoudl have empty dict "round"
    # for T in file

        

# def getDateList(sdate=None):
# #     sdate = date(2020, 1, 1)   # start date
#     from datetime import date, timedelta, datetime
#     if sdate is None:
#         sdate=date(2020, 1, 1)
#     edate =  date.today()
#     delta = edate - sdate       # as timedelta
#     date_list = [sdate + timedelta(days=i) for i in range(delta.days+1)]
#     # for i in range(delta.days + 1):
#     #     day = sdate + timedelta(days=i)
#     #     print(day)
#     date_list = [d.strftime("%y%m%d") for d in date_list]
#     return date_list

def loadSingleDataQuick(animal, date, expt, sess):
    """ reloads saved dict. never overwrites.
    a awrapper."""
    filedata = loadSingleData(animal, date, expt, sess, resave_as_dict=False, 
                          load_resaved_data=True, resave_overwrite=False)
    return filedata

def getSessionsList(animal, expt="*", datelist=None, doprint=False, extension="pkl", 
        return_also_exptname=True):
    """ given list of dates and animal, tell me what sessions 
    exist for each date.
    - only looks for pickle files (altreadry preprocessed)
    PARAMS:
    - return_also_exptname, bool, see RETURNS.
    RETURNS:
    - outdict, keys are dates, vals are lists of sessions, each session a tupl eof (sessnum, exptname):
        [if return_also_exptname] e.g., {220702:[(1, 'charpsychorel1'), (2, 'charparts1b')]}
        otherwise is just the essnum, {220702:[1, 2]}
    """
        
    if datelist is None:
        datelist = []
    outdict = {}
    for date in datelist:
        outdict[date]=[]
        
        def check(session):
            f = findFilename(animal, date, expt, session, extension=extension, 
                    fail_if_noexist=False, doprint=doprint)
            for filethis in f:
                # get this files params
                _, _, exptthis, _, sessthis = filename2params(filethis)
                assert sessthis==session, "weird...?"
                if return_also_exptname:
                    outdict[date].append((session, exptthis))
                else:
                    outdict[date].append(session)
            return f
            
        # -- try many sessions
        for session in range(1, 15):
            f = check(session)
            
        # if found  file for the last session, then keep trying.
        if len(f)>0:
            for session in range(15, 30):
                f = check(session)

    return outdict


def cleanupFiledata(fd, print_size=False):
    """ to remove unneeded objects in filedata.
    reduce memory.
    - modifies filedata in place
    - print_size, prints size of fd in memory, pre and post.
    """
    if print_size:
        from pythonlib.tools.pytools import get_size
        print("before cleaning, size filedata (kb)")
        print(get_size(fd)/1000)

    
    # == 1) remove things from fd["trials"]
    keys_taken = []
    keys_to_keep = ["BehavioralCodes", "TrialError", 
    "AnalogData", "TrialDateTime", "BlockCount", "UserVars"]
    for k in fd["trials"].keys():
        # print("---")
        # print(k)
        # print(fd["trials"][k])
        for kk in list(fd["trials"][k].keys()):
            if kk not in keys_to_keep:
                del fd["trials"][k][kk] 
            else:
                keys_taken.append(kk)

    # == 1b) remove from fd["trials"][t]["AnalogData"]
    keys_to_keep = ["Touch", "SampleInterval",  "Button", "Eye"]
    for k in fd["trials"].keys():
        for kk in list(fd["trials"][k]["AnalogData"].keys()):
            if kk not in keys_to_keep:
                del fd["trials"][k]["AnalogData"][kk]

    # == 2) Remove things from fd["TrialRecord"]
    keys_to_remove = ["LastTrialAnalogData", "TaskInfo"]
    for k in list(fd["TrialRecord"].keys()):
        if k in keys_to_remove:
            del fd["TrialRecord"][k]

    # === 3) Remove things from fd["TrialRecord"]["User"]
    # TODO: waiting to modify BPHU in matlab code.
#     ## things to keep
#     Params, 
#     --- fix
#     --- fade

#     ## things to remove
#     TrialData, have x y coords repeated many times.
#     bb, could remove following, but is ok:
#     -- behEvaluator
#     BlockParams_thisblock
#     BlockParamsHotkeyUpdated

    if print_size:
        print("after cleaning, size filedata (kb)")
        print(get_size(fd)/1000)

def _dataset_get_exptname(animal, date):
    """ Returns either NOne (no data) or string, 
    which is exptname of fierst session 
    """
    fnames_pkl = findFilename(animal, date, return_all_files=True, extension="pkl")
    if len(fnames_pkl)==0:
        return None
    else:
        exptlist = list(set([filename2params(f)[2] for f in fnames_pkl if filename2params(f) is not None]))
        expt = sorted(exptlist)[0] # Just take the first name, regardless, will extract all data for today
        return expt

def preprocess_plots_dataset(animal, date, DATASET_RE_EXTRACT):
    """ Generate (i) metadat, (ii) dataset pkl files, and (iii) simple summary drawing plots for the
    sessions in fnames_pkl, gerated for this animal and date.
    """
    from analysis.dataset import generate_metadat
    from analysis.dataset import generate_dataset_file_from_raw
    from pythonlib.dataset.dataset_analy.summary import plotall_summary
    import glob

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        return
    else:
        fnames_pkl = findFilename(animal, date, return_all_files=True, extension="pkl")
    # if len(fnames_pkl)==0:
    #     return
    # else:
    #     exptlist = list(set([filename2params(f)[2] for f in fnames_pkl if filename2params(f) is not None]))

        ##########################
        # Dataset analy, do automaticaly (9/23/22)

        # Check if dataset already extracted
        datfile_exists_1 = len(glob.glob(f"{PATH_DATASET_BEH}/BEH/{animal}-{expt}-{date}-*"))>0
        datfile_exists_2 = len(glob.glob(f"{PATH_DATASET_BEH}/TASKS_GENERAL/{animal}-{expt}-{date}-*"))>0

        # expt = exptlist[0] # Just take the first name, regardless, will extract all data for today
        fdir = f"{os.path.split(fnames_pkl[0])[0]}/figures/dataset/{expt}"
        exists, hasfiles = checkIfDirExistsAndHasFiles(fdir)

        # if hasfiles and datfile_exists_1 and datfile_exists_2:
        #     # Then have datfile, taskfile, and plots... Skip.
        #     return

        rulelist = [str(date)]
        if datfile_exists_1 and datfile_exists_2 and not DATASET_RE_EXTRACT:
            print(f"== SKIPPING DATASET EXTRACTION for {animal}, {date}, {expt} (alrady done)")
            pass
        else:
            # 1) Make metadat
            generate_metadat(expt, date, date, animal, overwrite=True)

            # 2) Generate the datsaet file from raw
            SKIP_IF_EXISTS = False # 2/22/24 - if get to this part of code, should overwrite...
            generate_dataset_file_from_raw(animal, expt, dataset_dir_kind="daily",
                rulelist=rulelist, SKIP_IF_EXISTS=SKIP_IF_EXISTS)

        if not hasfiles:
            # 3) Simple summary plots.
            plotall_summary(animal, expt, rulelist, "daily")
        else:
            print(f"== SKIPPING DATASET SUYMMARY PLOTS for {animal}, {date}, {expt} (alrady done)")

def preprocess_plots_dataset_character_strokiness(animal, date):
    """ Plot and analysis for grammar (i.e., rule learning) experiments
    """
    from pythonlib.dataset.dataset_analy.characters import pipeline_generate_and_plot_all
    from pythonlib.dataset.dataset import load_dataset_daily_helper

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        # D = Dataset([])
        # rulelist = [str(date)]
        # D.load_dataset_helper(animal, expt, ver="mult", rule=rulelist)
        pipeline_generate_and_plot_all(D)            

def preprocess_plots_dataset_primsingrid(animal, date):
    from pythonlib.dataset.dataset_analy.prims_in_grid import preprocess_dataset

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        preprocess_dataset(D, doplots=True)

def preprocess_plots_dataset_singleprims(animal, date):    
    from pythonlib.dataset.dataset_analy.singleprims import preprocess_dataset

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        preprocess_dataset(D, PLOT=True)

def preprocess_plots_dataset_microstim(animal, date):    
    from pythonlib.dataset.dataset_analy.microstim import plot_all_wrapper

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        plot_all_wrapper(D)

def preprocess_plots_psychometric_singleprims(animal, date, var_psycho):    
    # Psychometric in general, but for now is hard coded to only work for 
    # variations in angle rotation.. ACTUALLY works with others too.
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import preprocess_and_plot

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        preprocess_and_plot(D, var_psycho=var_psycho, PLOT=True)

def preprocess_plots_novel_singleprims(animal, date):    
    """
    Novel single prims, generic, doesnt try to figure out psychometric function ,etc.
    GFoal is test motor stats diff between novel and laerned.
    """
    from pythonlib.dataset.dataset_analy.novel_singleprims import preprocess_and_plot

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        DS, SAVEDIR, dfres, grouping = preprocess_and_plot(D, PLOT=True)

def preprocess_plots_psycho_general(animal, date):    
    """
    (See notes below)
    """
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper

    expt = _dataset_get_exptname(animal, date)
    if expt is None:
        print("DIDNT FIND This expt: ", animal, date)
        return
    else:
        D = load_dataset_daily_helper(animal, date)
        psychogood_preprocess_wrapper(D)

def preprocess_plots_dataset_grammar(animal, date):
    """ Plot and analysis for grammar (i.e., rule learning) experiments
    """
    from pythonlib.dataset.dataset_analy.grammar import pipeline_generate_and_plot_all
    from pythonlib.dataset.dataset import Dataset

    expt = _dataset_get_exptname(animal, date) 
    if expt is None:
        print("This experiment not found!", animal, date)
        return
    else:
        D = Dataset([])
        rulelist = [str(date)]
        D.load_dataset_helper(animal, expt, ver="mult", rule=rulelist)
        pipeline_generate_and_plot_all(D)
           

if __name__ == "__main__":
    # Example calls

    # Initial run (and doing PIG and SP plots): python -m tools.preprocess Pancho 220125 220128 Y ps
    # Reextract dataset: python -m tools.preprocess Pancho 220125 220128 E
    # 


    from tools.utils import *
    from pythonlib.tools.datetools import getDateList
    from tools.dayanalysis import *
    from pythonlib.drawmodel.analysis import *
    from pythonlib.tools.stroketools import *
    from tools.analy import extractSessionDf
    import sys
    from datetime import date, timedelta, datetime
    import glob
    # dates = getDateList(sdate=date(2020, 7, 28))
    # dates = getDateList(sdate=date(2020, 9, 10))

    # dates = getDateList(sdate=210114)
    # dates = getDateList(sdate=201219)
    # dates = getDateList(sdate=210109)

    OVERWRITE = False # Raw data conversion and plots
    DATASET_RE_EXTRACT = False

    # Plots default on
    PLOT_DAYANALYSIS = True
    PLOT_PROBEANALYSIS = True
    PLOT_DATASET = True
    
    # Plots default off
    PLOT_DATASET_GRAMMAR = False
    PLOT_DATASET_CHARSTROKI = False
    PLOT_DATASET_PIG = False
    PLOT_DATASET_SINGLEPRIMS = False
    PLOT_DATASET_MICROSTIM = False
    PLOT_PSYCHO_SINGLE_PRIMS_ANGLE = False
    PLOT_PSYCHO_SINGLE_PRIMS_CONTMORPH = False
    PLOT_NOVEL_SINGLE_PRIMS = False
    PLOT_PSYCHO_GENERAL = False

    # Delete unneeeded h5 files
    DELETE_H5 = True

    START_DATE = 230103
    ANIMALS = ["Diego"]

    # 1) Which animals?
    if len(sys.argv)>1:    
        # eg., python -m tools.preprocess Pancho
        animals = [sys.argv[1]]
    else:
        animals = ANIMALS

    # 2) Which dates?
    if len(sys.argv)<3:
        # Use default dates
        # e..g, python -m tools.preprocess Pancho
        dates = getDateList(sdate=START_DATE)
    elif len(sys.argv)==3:
        # a single date
        # e..g, python -m tools.preprocess Pancho 220125
        dates = [sys.argv[2]]
    else:
        # range of dates
        # e..g, python -m tools.preprocess Pancho 220125 220128
        date1 = int(sys.argv[2])
        date2 = int(sys.argv[3])
        dates = list(range(date1, date2+1))


    # Flag to switch off specific plots
    if len(sys.argv)>4:
        # e..g, python -m tools.preprocess Pancho 220125 220128 PLOTVER
        # e..g, python -m tools.preprocess Pancho 220125 220128 ad means turn off day and dataset plots

        PLOTVER = sys.argv[4] # string
        if PLOTVER=="Y":
            # skip. i.e. make all plots
            pass
        elif PLOTVER=="E":
            # reextract dataset files and replot
            DATASET_RE_EXTRACT = True
        elif PLOTVER=="N":
            # ignore all plots
            PLOT_DAYANALYSIS = False
            PLOT_PROBEANALYSIS = False
            PLOT_DATASET = False
        else:
            for code in PLOTVER:
                if code=="a":
                    PLOT_DAYANALYSIS = False
                    print("Turning off plot: PLOT_DAYANALYSIS")
                elif code =="p":
                    PLOT_PROBEANALYSIS = False
                    print("Turning off plot: PLOT_PROBEANALYSIS")
                elif code=="d":
                    PLOT_DATASET = False
                    print("Turning off plot: PLOT_DATASET")
                else:
                    print(code)
                    assert False

    if DATASET_RE_EXTRACT:
        print("Turning on flag: DATASET_RE_EXTRACT")
    if PLOT_DAYANALYSIS:
        print("Turning on flag: PLOT_DAYANALYSIS")
    if PLOT_PROBEANALYSIS:
        print("Turning on flag: PLOT_PROBEANALYSIS")
    if PLOT_DATASET:
        print("Turning on flag: PLOT_DATASET")
        
    # Flag to switch on plots
    if len(sys.argv)>5:
        # e..g, python -m tools.preprocess Pancho 220125 220128 ad g means 
        # means turn off day and dataset plots and turn on grammar
        PLOTVER = sys.argv[5] # string
        for code in PLOTVER:
            if code=="g":
                PLOT_DATASET_GRAMMAR = True
                PLOT_DATASET_PIG = True
            elif code=="c":
                PLOT_DATASET_CHARSTROKI = True
            elif code=="p":
                PLOT_DATASET_PIG = True
            elif code=="s":
                PLOT_DATASET_SINGLEPRIMS = True
            elif code=="m":
                PLOT_DATASET_MICROSTIM = True
                PLOT_DATASET_PIG=True
            elif code=="y":
                # Psychometric single prims (angle)
                PLOT_PSYCHO_SINGLE_PRIMS_ANGLE = True
            elif code=="r":
                # Psychometric single prims (continous morph)
                PLOT_PSYCHO_SINGLE_PRIMS_CONTMORPH = True
            elif code=="n":
                # Novel, general. i.e, simply compare motor stats between learned vs. novel. Doesnt
                # try to determine what kind of novel or psycho.
                PLOT_NOVEL_SINGLE_PRIMS = True
            elif code=="h":
                # Psycho general, the best, currently assuming each los is a single task.
                # (Written for structured morph. Have not generalized to all)
                PLOT_PSYCHO_GENERAL = True
            else:
                print(code)
                assert False

    if PLOT_DATASET_GRAMMAR:
        print("Turning on plot: PLOT_DATASET_GRAMMAR")
    if PLOT_DATASET_CHARSTROKI:
        print("Turning on plot: PLOT_DATASET_CHARSTROKI")
    if PLOT_DATASET_PIG:
        print("Turning on plot: PLOT_DATASET_PIG")
    if PLOT_DATASET_SINGLEPRIMS:
        print("Turning on plot: PLOT_DATASET_SINGLEPRIMS")
    if PLOT_DATASET_MICROSTIM:
        print("Turning on plot: PLOT_DATASET_MICROSTIM")
    if PLOT_PSYCHO_SINGLE_PRIMS_ANGLE:
        print("Turning on plot: PLOT_PSYCHO_SINGLE_PRIMS_ANGLE")
    if PLOT_PSYCHO_SINGLE_PRIMS_CONTMORPH:
        print("Turning on plot: PLOT_PSYCHO_SINGLE_PRIMS_CONTMORPH")
    if PLOT_NOVEL_SINGLE_PRIMS:
        print("Turning on plot: PLOT_NOVEL_SINGLE_PRIMS")
    if PLOT_PSYCHO_GENERAL:
        print("Turning on plot: PLOT_PSYCHO_GENERAL")

    for a in animals:
        for d in dates:

            animal = a
            date = d

            if a in ["Barbossa", "Taz"] and int(d) < 210820:
                # Skip, since the data are weird (touch data), since this had neutral
                # reward thing, that also messed up camera. Not sure why, but not important
                # since not using this data.
                continue

            # moves duplicates (same a, d, s), to keep only latest and largest.
            cleanupDir(a, d)

            # == 1) get all datasets:
            fnames_h5 = findFilename(a, d, return_all_files=True, extension="h5")
            fnames_bhv2 = findFilename(a, d, return_all_files=True, extension="bhv2")
            fnames_raw = fnames_h5 + fnames_bhv2

            ############## CONVERT RAW TO PKL
            # (first check if already done. if so, then skip)
            for f in fnames_raw:
                try:
                    out = filename2params(f, return_ext=True)
                    if out is None:
                        # then this is not a real filename, is test.
                        continue    
                    else:
                        (_, _, expt, _, sess, ext) = out

                except Exception as err:
                    print(f)
                    print(err)
                    raise 
                
                # === 1) Check if need to convert to pkl
                # -- Check if already done.
                    # # Dont use this version, since this confused if there are nultiple files with same session
                    # f2 = findFilename(a, d, expt, sess, extension="pkl", return_all_files=True,
                    #                  doprint=False)
                f2 = glob.glob(os.path.splitext(f)[0] + ".pkl")

                if len(f2)==1:
                    # print(f"== SKIPPING {a}, {d}, {expt}, {sess} since already done (found .pkl)")
                    print(f"== SKIPPING {f} since already done (found .pkl)")
                    # continue
                elif len(f2)>1:
                    print(f2)
                    assert False, "why multipel pkl files?"
                else:
                    # 1) If this is bhv2, then need to first convert to h5.
                    # i.e no pickle file exists
                    
                    if ext==".bhv2":
                        from pythonlib.globals import PATH_DRAGMONKEY
                        print("MATLAB: Converting bhv2 to h5")
                        print(f"matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('{PATH_DRAGMONKEY}')); convert_format('h5', '{f}'); quit\"")
                        # os.system(f"~/data1/programs/MATLAB/R2021a/bin/matlab -nodisplay -nosplash -nodesktop -r \"convert_format('h5', '{f}'); quit\"")
                        os.system(f"{PATH_MATLAB} -nodisplay -nosplash -nodesktop -r \"addpath(genpath('{PATH_DRAGMONKEY}')); convert_format('h5', '{f}'); quit\"")
                        # import subprocess
                        # subprocess.Popen(["/bin/bash", "-i", "-c", "matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", f"convert_format('h5', '{f}');", "quit"])

                        # os.system(f"/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r \"convert_format('h5', '{f}'); quit\"")
                        #os.system(f"~/data1/programs/MATLAB/R2022a/bin/matlab -nodisplay -nosplash -nodesktop -r \"convert_format('h5', '{f}'); quit\"")
                        print("DONE converting")

                    # 2) Convert h5 to pkl.
                    print(f"== converting to PKL and analyzing {a}, {d}, {expt}, {sess}")
                    filedata = loadSingleData(a, d, expt, sess, resave_as_dict=True, 
                                              load_resaved_data=False, resave_overwrite=OVERWRITE)



            ################# PLOTS
            # 1) Extract processed data
            fnames_pkl = findFilename(a, d, return_all_files=True, extension="pkl")

            ### DAY ANALYS PLOTS
            if PLOT_DAYANALYSIS:
                for f in fnames_pkl:
                    try:
                        out = filename2params(f, return_ext=True)
                        if out is None:
                            # then this is not a real filename, is test.
                            continue    
                        else:
                            (_, _, expt, _, sess, ext) = out

                    except Exception as err:
                        print(f)
                        print(err)
                        raise err

                    # === 2) Check if need to make day plots
                    figure_save_dir = f"{os.path.split(f)[0]}/figures/{sess}/dayanalysis"
                    exists, hasfiles = checkIfDirExistsAndHasFiles(figure_save_dir)
                    if exists and hasfiles:
                        print(f"== SKIP day figures for {a}, {d}, {expt}, {sess}, since already done")
                        continue
                    else:
                        print(f"== PLOTTING day figures for {a}, {d}, {expt}, {sess}")
                    # -- Load filedata
                    fd = loadSingleDataQuick(a, d, expt, sess)
                    if fd is None:
                        print(f"== NEVER MIND, skipping (fd is None) day figures for {a}, {d}, {expt}, {sess}")
                        continue
                    if len(getIndsTrials(fd))<5:
                        print(f"== NEVER MIND, skipping (n trials < 5) day figures for {a}, {d}, {expt}, {sess}, since already done")
                        continue

                    # ====== ADDITIONAL THINGS ADDED ON SEP 2020
                    # setup saving dir for figs
                    SAVEDIRDAY = f"{fd['params']['figuredir']}/dayanalysis"
                    #import os
                    if os.path.exists(SAVEDIRDAY):
                        print(f"SKIPPING {SAVEDIRDAY}, since already done")
                    else:
                        os.makedirs(SAVEDIRDAY, exist_ok=True)

                        # 1) overview
                        df = extractSessionDf(fd)

                        try:
                            df = df[df["modelcomp"]<1.5]
                        except:
                            pass

                        # ==== get features to plot.    
                        featurestoplot = getMultTrialsBehEvalFeatures(fd)
                        featurestoplot.append("score_offline")

                        ############### THINGS INCLUDING ABORT TRIAL
                        fig1, fig2, fig3, fig4 = plotOverview_(df, featurestoplot=featurestoplot)
                        fig1.savefig(f"{SAVEDIRDAY}/overview1.pdf")
                        fig2.savefig(f"{SAVEDIRDAY}/overview2.pdf")
                        fig3.savefig(f"{SAVEDIRDAY}/overview3.pdf")
                        fig4.savefig(f"{SAVEDIRDAY}/overview4.pdf")
                        plt.close()

                        # == TIMECOURSE PLOT, SEPARATE BY TASKSTAGE
                        taskstage_list = set(df["taskstage"])

                        for taskstage in taskstage_list:
                            sdirtmp = f"{SAVEDIRDAY}/separate_by_taskstage/{taskstage}"
                            os.makedirs(sdirtmp, exist_ok=True)
                            dfthis = df[df["taskstage"]==taskstage]
                            fig1, fig2, fig3, fig4 = plotOverview_(dfthis, featurestoplot=featurestoplot)
                            
                            fig1.savefig(f"{sdirtmp}/overview1.pdf")
                            fig2.savefig(f"{sdirtmp}/overview2.pdf")
                            fig3.savefig(f"{sdirtmp}/overview3.pdf")
                            fig4.savefig(f"{sdirtmp}/overview4.pdf")
                            plt.close()


                        # 4) Plot behavior subsampling in chronological order
                        trials = [t for t in getIndsTrials(fd) if getTrialsFixationSuccess(fd, t)]
                        Nrand = 80
                        fig = plotMultTrialsSimple(fd, trials, zoom=True, strokes_ver="peanuts", plot_fix=False, rand_subset=Nrand)
                        fig.savefig(f"{SAVEDIRDAY}/trialsRandomChronOrder.pdf")

                        # 5) ==== plot all trials in chron order
                        trials = [t for t in getIndsTrials(fd) if getTrialsFixationSuccess(fd, t)]
                        NplotPerFig = 80
                        nfigs = int(np.ceil(len(trials)/NplotPerFig))
                        for n in range(nfigs):
                            if n==nfigs-1:
                                idx = range(n*NplotPerFig, len(trials))
                                trialsthis = [trials[i] for i in idx]
                            else:
                                idx = range(n*NplotPerFig, (n+1)*NplotPerFig)
                                trialsthis = [trials[i] for i in idx]

                            fig = plotMultTrialsSimple(fd, trialsthis, zoom=True, strokes_ver="peanuts", plot_fix=False)
                            fig.savefig(f"{SAVEDIRDAY}/trialsAllChronOrder-{n}.pdf")

                        # 5) TASK VISUALIZATIONS, SCHEDULE, REPETITION
                        figs = plotTaskSchedules(df)
                        for i, fig in enumerate(figs):
                            fig.savefig(f"{SAVEDIRDAY}/taskSchedule{i}.pdf")
                        plt.close()

                        ################## THINGS REMOVING ABORT TRIALS
                        # 2) relationship between reward and factors that go into reward
                        if False:
                            # Replaced by similar plot in probeanalysis
                            figs = plotReward(df, featurestoplot=featurestoplot)
                            for i, fig in enumerate(figs):
                                fig.savefig(f"{SAVEDIRDAY}/reward_score_{i}.pdf")

                        # 3) PLOT BEHAVIOR FOR TRIALS SORTED BY SCORE (for all features)
                        # - only keep if is not abort
                        df = df[~(df["trial_end_method"]=="online_abort")]

                        import copy
                        scoretypes = copy.copy(featurestoplot)
                        scoretypes.extend(["behscore", "reward"])
                        for score_type in scoretypes:
                            print("Plotting trials by score", score_type)
                            LIST_VER = ["percentiles", "bottomN", "topN"]
                            for ver in LIST_VER:
                                FIGS = plotBehSortedByScore(df, fd, score_type, 40, [ver])
                                if ver in FIGS.keys():
                                    # Then values were not all nan, so figs were created.
                                    figs = FIGS[ver]
                                    for i, fig in enumerate(figs):
                                        print(f"trialsSortedByScore_{score_type}_{ver}_{i}_.pdf")
                                        fig.savefig(f"{SAVEDIRDAY}/trialsSortedByScore_{score_type}_{ver}_{i}_.pdf")
                                plt.close("all")
                                del FIGS
                            # FIGS = plotBehSortedByScore(df, fd, score_type, 40, ["percentiles", "bottomN", "topN"])
                            # for ver, figs in FIGS.items():
                            #     for i, fig in enumerate(figs):
                            #         print(f"trialsSortedByScore_{score_type}_{ver}_{i}_.pdf")
                            #         fig.savefig(f"{SAVEDIRDAY}/trialsSortedByScore_{score_type}_{ver}_{i}_.pdf")
                        
                        #### Same, but separate sorting for each block
                        for blocknum in df["block"].unique():
                            dfthis = df[df["block"]==blocknum]
                            SDIR = f"{SAVEDIRDAY}/trialsSortedByScore_ByBlock/{blocknum}"
                            os.makedirs(SDIR, exist_ok=True)
                            for score_type in scoretypes:
                                print("Plotting trials by score, split by block.", blocknum, score_type)
                                FIGS = plotBehSortedByScore(dfthis, fd, score_type, 40, ["percentiles"])
                                for ver, figs in FIGS.items():
                                    for i, fig in enumerate(figs):
                                        fig.savefig(f"{SDIR}/{score_type}_{ver}_{i}_.pdf")
                                plt.close("all")
                                del FIGS

            # ==== "PROBES" ANALYSIS - plotting, separated by task types
            # - This is done once per day (combining all sessions)
            if PLOT_PROBEANALYSIS:
                from .dayanalysis import plotProbesSummary
                from analysis.line2 import OldTaskError
                from analysis.probedatTaskmodel import *
                from analysis.modelexpt import loadProbeDatWrapper
                if len(fnames_pkl)>0:
                    exptlist = list(set([filename2params(f)[2] for f in fnames_pkl if filename2params(f) is not None]))
                    for expt in exptlist:
                        fdir = f"{os.path.split(fnames_pkl[0])[0]}/figures/probeanalysis/{expt}"
                        exists, hasfiles = checkIfDirExistsAndHasFiles(fdir)
                        if not exists or not hasfiles:
                            print(f"== RUNNING PROBE ANALYSIS for {a}, {d}, {expt} (not alrady done)")

                            os.makedirs(fdir, exist_ok=True)
                            print(f"== PROBE ANALYSIS for {a}, {d}, {expt}")
                            dattoget = (expt, a, d)
                            FD = loadMultData([dattoget])
                            if len(FD)==0:
                                continue
                            try:
                                print("plotting probes summary")
                                plotProbesSummary(FD, fdir)
                            except OldTaskError as err:
                                print("PROBE ANALYSIS: skipping, since this is old task")
                                continue

                            # = PAIRWISE SCATTER COMAPRING ALL BEH FEATURES
                            # added 3/3/21
                            PROBEDAT  = loadProbeDatWrapper(FD)
                            # PROBEDAT = PROBEDATfromFD(FD)
                            if len(PROBEDAT)>0:
                                P  = ProbedatTaskmodel(PROBEDAT)
                                P = P.clean()
                                if len(P.getIndsTrials())>5:

                                    featurelist = P.extract_feature_list()

                                    ########### THINGS KEEPING ABORT TRIALS
                                    # ===== Print out scores for each trials (all features), save to text
                                    P.print_all_feature_scores(fdir)
                                    # === pnut size for each trial.
                                    P.print_all_pnut_size(fdir)

                                    # ==== Print names of all tasks and ntrials for them
                                    P.print_all_tasknames(fdir)

                                    # ==== Plot pnut size timecoursea nd by block
                                    P.plot_timecourse_pnut_size(fdir)

                                    # ===== Plot all trials, split by trial error code
                                    from analysis.probedatTaskmodel import plotAllTrialsByErrorCode
                                    print("plotting trials by eror code")                            
                                    plotAllTrialsByErrorCode(P, fdir)
                                    plt.close("all")

                                    # ===== PROBEDAT SUMMARIES
                                    # 3/11/21 - 
                                    # block transitions
                                    # tasks, sorted by reward
                                    SDIRTHIS = f"{fdir}/block_transitions"
                                    os.makedirs(SDIRTHIS, exist_ok=True)
                                    print("plotting overview blocks")                            
                                    P.plotOverviewBlokks(SDIRTHIS)
                                    plt.close("all")

                                    SDIRTHIS = f"{fdir}/tasks_sorted"
                                    os.makedirs(SDIRTHIS, exist_ok=True)
                                    P.plotOverviewTaskPerformance(SDIRTHIS)
                                    plt.close("all")

                                    ########### THINGS THAT SHOULD REMOVE ABORT TRIALS
                                    P = P.clean_remove_abort_trials()
                                    if P is None:
                                        continue

                                    # ===== Plot distrivtions (pairwixe and marginal) for each beh feature
                                    fdirthis = f"{fdir}/beh_eval_features"
                                    os.makedirs(fdirthis, exist_ok=True)

                                    # ===== Plot all feature scores, and overlay thresholds (xmin xmax) and 
                                    # hard limits used in adaptive updating of params
                                    P.plot_featuredists_overlying_params(fdir)
                                    plt.close("all")

                                    # - Split by block.
                                    blocks_to_plot = list(set(P.pandas()["block"]))
                                    for bk in blocks_to_plot:
                                        filtdict = {"block":[bk]}
                                        PD = P.filterProbedat(filtdict, modify_in_place = False)
                                        if len(PD)==0:
                                            continue
                                        Pfilt = ProbedatTaskmodel(PD)
                                        if len(Pfilt.getIndsTrials())>3:
                                            if False:
                                                # Old version, both pairwise and marginals. This takes too long, like a minute per block
                                                featurelist_this = [f for f in featurelist if f in Pfilt.pandas().columns]
                                                try:
                                                    fig = sns.pairplot(Pfilt.pandas(), vars=featurelist_this, hue="kind")
                                                    fig.savefig(f"{fdirthis}/allfeatures_pairwise-block{bk}.pdf")
                                                except Exception as err:
                                                    pass
                                                    # print("pandas:", Pfilt.pandas())
                                                    # print("featurelist", featurelist_this)
                                                    # raise err
                                            else:
                                                # New version, just the marginals, and overlay limits and thresholds.
                                                Pfilt.plot_featuredists_overlying_params(fdirthis, f"featuredists_overlying_params_block{bk}")
                                        plt.close("all")

                                    # - Combined across blocks.
                                    featurelist_this = [f for f in featurelist if f in P.pandas().columns]
                                    if len(P.pandas())>200:
                                        dfthis = P.pandas().sample(200)
                                    else:
                                        dfthis = P.pandas()
                                    fig = sns.pairplot(dfthis, vars=featurelist_this, hue="kind")
                                    fig.savefig(f"{fdirthis}/allfeatures_pairwise.pdf")
                                    plt.close("all")

                                    # ===== PROBEDAT summaries
                                    # 8/24/21 - 
                                    # Plot distributiosn of features, over blocks.
                                    SDIRTHIS = f"{fdir}/beh_eval_features_timecourse"
                                    os.makedirs(SDIRTHIS, exist_ok=True)
                                    P.plot_featuredists_byblock(SDIRTHIS)
                                    plt.close("all")

            if PLOT_DATASET:
                preprocess_plots_dataset(animal, date, DATASET_RE_EXTRACT)

            if PLOT_DATASET_GRAMMAR:
                preprocess_plots_dataset_grammar(animal, date)

            if PLOT_DATASET_CHARSTROKI:
                preprocess_plots_dataset_character_strokiness(animal, date)

            if PLOT_DATASET_PIG:
                preprocess_plots_dataset_primsingrid(animal, date)

            if PLOT_DATASET_SINGLEPRIMS:
                preprocess_plots_dataset_singleprims(animal, date)

            if PLOT_DATASET_MICROSTIM:
                preprocess_plots_dataset_microstim(animal, date)
                
            if PLOT_PSYCHO_SINGLE_PRIMS_ANGLE:
                preprocess_plots_psychometric_singleprims(animal, date, var_psycho="angle")

            if PLOT_PSYCHO_SINGLE_PRIMS_CONTMORPH:
                preprocess_plots_psychometric_singleprims(animal, date, var_psycho="cont_morph")

            if PLOT_NOVEL_SINGLE_PRIMS:
                preprocess_plots_novel_singleprims(animal, date)

            if PLOT_PSYCHO_GENERAL:
                preprocess_plots_psycho_general(animal, date)
            
            ########## DELETE INTERMEDIATE H5 FILES
            if DELETE_H5:
                remove_unneeded_h5_files(animal, date)