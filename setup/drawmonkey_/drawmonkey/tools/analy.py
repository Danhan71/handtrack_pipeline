"""analysis code, for more deeper analyses. plotting will still be plot"""

from tools.utils import *
from pythonlib.tools.stroketools import getAllStrokeOrders

def _assertNotAlreadyDone(stroke_dict, keyname):
    """ to check that an analsysis not already done"""
    assert keyname not in stroke_dict.keys(), f"cannot continue, {keyname} is already in stroke_dict"
    if "done_processes" in stroke_dict.keys():
        assert keyname not in stroke_dict["done_processes"], f"cannot continue, {keyname} is already in 'done_processes'"

def _addToDoneProcesses(stroke_dict, keyname):
    if "done_processes" in stroke_dict.keys():
        stroke_dict["done_processes"].append(keyname)
    else:
        stroke_dict["done_processes"] = [keyname]

def extractSessionDf(filedata, items = None, onlytrialspassfix=True):
    """get pandas dataframe, each trial in flat structure.
    this is general purpose, can build on this later.
    """
    import pandas as pd

    trials = getIndsTrials(filedata)
    if onlytrialspassfix:
        trials = [t for t in trials if getTrialsFixationSuccess(filedata, t)]

    fd_dict = {
        # "block":[int(f[0]) for f in filedata["TrialRecord"]["BlockCount"]],
        "block":[getTrialsBlock(filedata, t) for t in trials],
        "bloque":[getTrialsBloque(filedata, t) for t in trials],
        "blokk":[getTrialsBlokk(filedata, t) for t in trials],
        "trial":trials,
        "errorcodes":[int(getTrialsOutcomesAll(filedata, t)["errorcode"][0]) for t in trials],
        "fracinkgotten":[getTrialsOutcomesAll(filedata, t)["fracinkgotten"] for t in trials],
        "errorstring":[getTrialsOutcomesAll(filedata, t)["trialoutcomes"]["failure_mode"] for t in trials],
        "taskstage":[getTrialsTask(filedata, t)["stage"] for t in trials],
    }

    # --- behavioral score
    X =[]
    hausdorff = []
    frac_touched = []
    for t in trials:
        BE = getTrialsBehEvaluation(filedata, t)
        if BE is None:
            X.append(np.nan)
            hausdorff.append(np.nan)
            frac_touched.append(np.nan)
        else:
            X.append(BE["beh_multiplier"][0][0])
            # print(BE)
            if "hausdorff" in BE["output"].keys():
                hausdorff.append(BE["output"]["hausdorff"]["value"][0][0])
            else:
                hausdorff.append(np.nan)
            if "frac_touched" in BE["output"].keys():
                frac_touched.append(BE["output"]["frac_touched"]["value"][0][0])
            else:
                frac_touched.append(np.nan)

    fd_dict["behscore"] = X
    fd_dict["hausdorff"] = hausdorff
    fd_dict["frac_touched"] = frac_touched

    # -- inlude all factors in beahvioral score (unelss already donea bove)
    # featurestoplot = []
    # for key, val in getTrialsBlockParams(filedata, 1)["behEval"]["beh_eval"].items():
    #     if val["weight"][0][0]>0:
    #         featurestoplot.append(val["feature"])
    featurestoplot = getMultTrialsBehEvalFeatures(filedata)
    for f in featurestoplot:
        if f not in fd_dict.keys():
            fd_dict[f] = []
            for t in trials:
                BE = getTrialsBehEvaluation(filedata, t)
                if BE is None:
                    fd_dict[f].append(np.nan)
                else:
                    if f in BE["output"].keys():
                        fd_dict[f].append(BE["output"][f]["value"][0][0])
                    else:
                        fd_dict[f].append(np.nan)   

    # --- trial outcomes
    fd_dict["reward"] = [getTrialsOutcomesWrapper(filedata, t)["beh_evaluation"]["rew_total"][0][0] if getTrialsFixationSuccess(filedata,t) else np.nan for t in trials]
    fd_dict["binaryscore"] = [getTrialsOutcomesWrapper(filedata, t)["beh_evaluation"]["binary_evaluation"][0][0] if getTrialsFixationSuccess(filedata,t) else np.nan for t in trials]
    fd_dict["biasscore"] = [getTrialsOutcomesWrapper(filedata, t)["beh_evaluation"]["bias_multiplier"][0][0] if getTrialsFixationSuccess(filedata,t) else np.nan for t in trials]
    # fd_dict["behscore2"] = [getTrialsOutcomesWrapper(filedata, t)["beh_evaluation"]["beh_multiplier"][0][0] if getTrialsFixationSuccess(filedata,t) else np.nan for t in trials]

    # ========== PROGRESSIONS
    # --- block category
    if getMultTrialsBlockCategories(filedata) is not None:
        fd_dict["blockcategory"] = [getTrialsBlockCategory(filedata, t) for t in trials]
    # --- progression level
        # print(1, getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"])
        # print(11, len(getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"]))
        # # print(2, getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"][0])
        # # print(3, getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"][0][0])

        if len(getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"])>0:
            try:
                fd_dict["progressionlevel"] = [int(getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"]) for t in trials]
            except IndexError:
                print("I think this means there is only one blaock...")
                fd_dict["progressionlevel"] = [int(getTrialsBlockParamsHotkeyUpdated(filedata, t)["progression"]["level"]) for t in trials]
        else:
            fd_dict["progressionlevel"] = None


    # ========= FADE VALUES
    if getTrialsFadeValues(filedata, 1) is not None:
        fd_dict["fade_samp1"] = [getTrialsFadeValues(filedata, t)["samp1"][0] for t in trials]
        fd_dict["fade_guide1task"] = [getTrialsFadeValues(filedata, t)["guide1_task"][0] for t in trials]
        fd_dict["fade_guide1fix"] = [getTrialsFadeValues(filedata, t)["guide1_fix"][0] for t in trials]

    # ======= number of replays:
    # nan if this trial did not pass go.
    nreplays = []
    for t in trials:
        replayStats = getTrialsReplayStats(filedata, t)
        if replayStats is None:
            nreplays.append(np.nan)
        elif "count" not in replayStats.keys():
            nreplays.append(np.nan)
        else:
            nreplays.append(replayStats["count"][0][0])
    nreplays = np.array(nreplays)
    fd_dict["num_replays"] = nreplays

    # ====== trial end method
    fd_dict["trial_end_method"] = [getTrialsOutcomesWrapper(filedata, t)["trial_end_method"] for t in trials]

    # ===== get task for this trial
    # string for task name
    fd_dict["task_string"] = [getTrialsTask(filedata, t)["str"] for t in trials]

    # ===== reward max
    fd_dict["reward_max"] = [getTrialsBlockParamsHotkeyUpdated(filedata, t)["params_task"]["rewardmax_success"].item() for t in trials]

    # == recalc hausdorff
    fd_dict["score_offline"] = [getTrialsScoreRecomputed(filedata, t, normalize=True) for t in trials]

    # == figure out if is probe task
    # print(fd_dict)
    # for k, v in fd_dict.items():
    #     print((k, len(v)))
    # print(fd_dict.keys())


    # --- convert to dataframe
    df = pd.DataFrame(fd_dict)
    
    for col in df.columns:
        if isinstance(df[col].values[0], np.ndarray):
            print(col)
            print(df[col].values)
            print(df[col].values[0])
            assert False, "shold take df[col].values[0].item(). or else sns cannot plot it"
    return df


def getMultTrialsStrokes(filedata, trials_list, clean_strokes=True):
    """ basically what getMultTrialsStrokeDict does but this is more
    compact, just get strokes.
    if clean, then does:
    - processed. only strokes within boundaries of
    go and reward. 
    Segment into strokes based on combination of: 
    - time rel go and reward. throws out if is just fixation
    - speed minima
    NOTE: tested for Pancho, 2/26, using lines.
    throw_out_first_stroke = True # this is during fixation
    - also segments strokes based on velocity minima
    ALSO: throws out first stroek if this beh type involves picking up finger.
    """
    if clean_strokes:
        print("[getTrialsStrokes] removing stroke for calc vel since shorter than window")

        strokes_all = []
        for t in trials_list:
            beh_type = getTrialsBlockParamsHotkeyUpdated(filedata, t)["behtype"]["type"]
            if beh_type=="Trace (instant)":
                throw=True
            elif beh_type=="Trace (pursuit,track)":
                throw=False # keep first stroke, since drawing starts on first stroke
            strokes_all.append(getTrialsStrokesClean(filedata, t, throw_out_first_stroke=throw))
    else:
        assert False, "have not coded. see below for what I would do"
        # strokes_all = [getTrialsStrokes(filedata, t, cleanstrokes=True) for t in trials_list]
    return strokes_all
    

########################## STROKE_DICT stuff
def getMultTrialsStrokeDict(filedata, trials_list, clean_strokes=True):
    """
    get strokes for beahvior and tasks
    does processing too
    has options for processing
    - clean_strokes, only works for Trace (instant) tasks, gets strokes
    post "go" cue and ending at reward.
    - task strokes will by default be wtih timesteps starting from end closer
    to the origin
    - will segment strokes based on velocity minima
    """
    # 1) data strokes
    if clean_strokes:
        print("[getTrialsStrokes] removing stroke for calc vel since shorter than window")

        strokes_all_clean = []
        for t in trials_list:
            # beh_type = getTrialsBlockParams(filedata, t)["behtype"]["type"]
            # if beh_type=="Trace (instant)":
            #     throw=True
            # elif beh_type=="Trace (pursuit,track)":
            #     throw=False
            #      # keep first stroke, since drawing starts on first stroke
            # else:
            #     assert False, "dont know this one"
            # strokes_all_clean.append(getTrialsStrokesClean(filedata, t, throw_out_first_stroke=throw))
            strokes_all_clean.append(getTrialsStrokesClean(filedata, t))
        strokes_all = strokes_all_clean
    else:
        assert False, "have not coded. see below for what I would do"
        # strokes_all = [getTrialsStrokes(filedata, t, cleanstrokes=True) for t in trials_list]

    # 2) task strokes
    # tasks_all = [getTrialsTask(filedata, t) for t in trials_list]
    strokes_all_task = [getTrialsTaskAsStrokes(filedata, t, fake_timesteps="from_orig") for t in trials_list]
    stroke_dict = {
         "strokes_all": [tuple(s) for s in strokes_all],
        "strokes_all_task": strokes_all_task,
        "strokes_all_task_orig": [tuple(s) for s in strokes_all_task], # useful so this stays imutable
        "trials_list":trials_list,
    }
    return stroke_dict


def processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all_task", ver="from_orig", replace_key_to_do=False):
    """modifies the timesteps in strokes.
    useful for 'shuffling' and hypothesis test
    type plotting and analysises.
    Does this for both the data strokes and
    task strokes
    - note: timesteps relative to some point will be respect to the saem point for
    all strokes, currently.
    - NOTE: Can compose this function by doing it multiple times in sequence. e.g., can set
    dir for first stroke, and then run again using "from_end_of_previous_stroke" to set the next
    strokes in order based on proximity to first.
    """
        
    def _A(key_to_do):
        if ver=="from_orig":
            strokes_all_fake = [fakeTimesteps(s, point=getTrialsFix(filedata, t)["fixpos_pixels"], 
                                              ver="from_point") 
                                for s, t in zip(stroke_dict[key_to_do], stroke_dict["trials_list"])]
        elif ver=="from_first_touch":
            # the first position actually touched in behavior.
            strokes_all_fake = [fakeTimesteps(s, point=sreal[0][0,[0,1]], 
                                              ver="from_point") 
                                for s, sreal in zip(stroke_dict[key_to_do], stroke_dict["strokes_all"])]
        # if ver=="from_orig":
        #     strokes_all_fake = [fakeTimesteps([s[0]], point=getTrialsFix(filedata, t)["fixpos_pixels"], 
        #                                       ver="from_point") 
        #                         for s, t in zip(stroke_dict[key_to_do], stroke_dict["trials_list"])]
        # elif ver=="from_first_touch":
        #     # the first position actually touched in behavior.
        #     strokes_all_fake = [fakeTimesteps([s[0]], point=sreal[0][0,[0,1]], 
        #                                       ver="from_point") 
        #                         for s, sreal in zip(stroke_dict[key_to_do], stroke_dict["strokes_all"])]
        elif ver in ["in_order", "from_end_of_previous_stroke"]:
            # in origianl order 
            strokes_all_fake = [fakeTimesteps(s, point=[], ver=ver) for s in stroke_dict[key_to_do]]
        else:
            assert False, "dont knwo this ver"
        return strokes_all_fake
    
    stroke_dict[f"{key_to_do}_faketime"] = _A(key_to_do)  
    stroke_dict["faketime_ver"] = ver
    
    if replace_key_to_do:
        print(f"NOTE: replaced {key_to_do} with values in {key_to_do}_faketime")
        stroke_dict[key_to_do] = stroke_dict[f"{key_to_do}_faketime"]
        
    
    if False:
        """useful for seeing what processFakeTimesteps is doing"""
        # 1) Preprocess data
        stroke_dict = getMultTrialsStrokeDict(filedata, trials_list)
        # stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all", ver="from_first_touch")
        stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all_task", ver="from_end_of_previous_stroke",
                                          replace_key_to_do=True)
        stroke_dict = processAngles(stroke_dict)
        stroke_dict_plot = stroke_dict

        # --- filter by angle between first and second strokes.
        stroke_dict_plot = processFilterDat(stroke_dict, {"relative_angle_task_first_two_strokes":[-pi,0]})
        stroke_dict_plot = processFilterDat(stroke_dict, {"relative_angle_task_first_two_strokes":[0,pi]})

        # == 1) Plot in original space, trials + first touch point
        ax = plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all_task", 
                              title="", strokes_to_plot="all", 
                             plotver="raw")

        # plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all_task", 
        #                       title="", strokes_to_plot="first_touch", 
        #                      plotver="one_color", ax=ax)

        plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all", 
                              title="", strokes_to_plot="first_touch", 
                             plotver="one_color", ax=ax)    


    return stroke_dict


def processAngles(stroke_dict, stroke_to_use="all_strokes", num_angle_bins=8,
    use_just_first_two_points=False, force_use_two_points=False):
    """ finds angle of movement.
    - this works well if first stroke is meaningful
    - num_angle_bins, to bin angles, with bin 1 starting from 0rad.
    - stroke_to_use = use_just_first_two_points. useful if first stroke is curvy
    - stroke_to_use = first, uses endpoints of first stroke (firs to last pos)
    - all_strokes, takes entire stroke for each stroke.
    - in each case will output a list of angles for each strokes object    - 
    - note, if an angle is nan (i.e., no distance for vector), then this
    bins using index that is outside range of bins, so bin[index] would lead to 
    IndexError.
    """
    from math import pi
    def _A(strokes_all):
        # get angles [0, 2
        angles_all = [stroke2angle(s, stroke_to_use=stroke_to_use, force_use_two_points=force_use_two_points) for s in strokes_all]

        # bin the angles
        bins = np.linspace(0, 2*pi, num_angle_bins+1)
        angles_all_binned = [np.digitize(a, bins) for a in angles_all]
        return angles_all, angles_all_binned
    
    # --- behavior
    stroke_dict["angles_all"], stroke_dict["angles_all_binned"] = _A(stroke_dict["strokes_all"])

    # --- do the same for task
    stroke_dict["angles_all_task"], stroke_dict["angles_all_task_binned"] = _A(stroke_dict["strokes_all_task"])

    return stroke_dict

if False:
    stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all", ver="from_first_touch")
    stroke_dict = processFakeTimesteps(stroke_dict, filedata, key_to_do="strokes_all_task", ver="from_first_touch",
                                      replace_key_to_do=True)
    stroke_dict = processAngles(stroke_dict)
    [print(len(stroke_dict[key])) for key in stroke_dict.keys()]
    stroke_dict.keys()


# 1) filter to only include data for a given angle bin
def processFilterDat(stroke_dict, filterdict, filedata):
    """
    general purpuse, to filters across all variables in dict,
    based on parameter of interst. that paraemter must also be
    in an array in stroke_dict, with elements matching across all 
    arrays
    e.g., filterdict = {
    "angle_bin":[4, 8],
    "something_else":[params_to_keep]
    }
    - will do the filters in order and only keep things that finally satisfie 
    each filter. e..g, here angle_bin, then something_else
    - how params_to_keep is interpreted can depend on the key
    -----
    - angle_bin, uses the angles for the actual task (which will be assuming that strokes
    go from origin), and not from the behavior.
    - relative_angle_task_first_two_strokes, e.g., [-pi, 0], or [0,pi]. all values within [-pi, pi]
    - not_empty:[], can leave values as whatever. will keep any that have non empty strokes.
    """

    # -- if checking notempty, then will autmoatically figure out the values to filter with
    for key, item in filterdict.items():
        if key=="not_empty":
            filterdict[key]=[True]

    keys_to_check = ['strokes_all', 'strokes_all_task', 'strokes_all_faketime', 'strokes_all_task_faketime',
                      'angles_all', 'angles_all_binned', 'angles_all_task', 'angles_all_task_binned', 'trials_list',
                      "stroke_centers_task", "stroke_centers", "strokes_all_task_orig"]

    # copy stroke_dict, but only the keys that I know will be filtered.
    stroke_dict_out = {}
    for key in keys_to_check:
        if key in stroke_dict.keys():
            stroke_dict_out[key] = stroke_dict[key]

    # copy things that will not iterate over.
    for key in ["faketime_ver"]:
        if key in stroke_dict.keys():
            stroke_dict_out[key] = stroke_dict[key]

    stroke_dict = stroke_dict_out

    

    def stroke_centers(stroke_dict, snum=0, dim=0, beh_or_task="task"):
        """ get the stroke centers (in units from 0,1) for the center of the
        stroke.
        otuputs one scalar for each trail"""
        from tools.utils import convertPix2Relunits
        
        if beh_or_task=="task":
            centers_all = stroke_dict["stroke_centers_task"]
        elif beh_or_task=="beh":
            centers_all = stroke_dict["stroke_centers"]
        output = []
        for centers in centers_all:
            # convert from pixels to (0,1[])
            # print(centers)
            # print(snum)
            # print(dim)
            output.append(convertPix2Relunits(filedata, centers[snum])[dim])
        return output


    def stroke_angles(stroke_dict, snum=0, beh_or_task="task"):
        """ get the stroke centers (in units from 0,1) for the center of the
        stroke.
        otuputs one scalar for each trail"""
        if beh_or_task=="task":
            angles_all = stroke_dict["angles_all_task"]
        elif beh_or_task=="beh":
            angles_all = stroke_dict["angles_all"]
        output = [a[snum] for a in angles_all]
        return output


    # == keys that require filtering
    for key, values in filterdict.items():

        def _A(stroke_dict, k, test_values):
            return [s for s, a in zip(stroke_dict[k], test_values) if a in values]
        def _A_continuous(stroke_dict, k, test_values):
            # checks whether test_value is in range [v[0], v[1]), so values
            # must have 2 elements
            assert len(values)==2
            return [s for s, a in zip(stroke_dict[k], test_values) if (a>=values[0] and a<values[1])]


        ##############################################################

        if key=="angle_bin_task_first_stroke":
            # filter by the angle bin for the ground truth first stroke
            test_values = [a[0] for a in stroke_dict["angles_all_task_binned"]]
            
            # def _A(k):
            #     return [s for s, a in zip(stroke_dict[k], angle_values) if a in values]
            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A(stroke_dict, k, test_values)

        elif key=="relative_angle_task_first_two_strokes":
            # stroke 2 minus stroke 1. within the boundaries of angle.
            # 1) extract the angle differences
            test_values = []

            for a in stroke_dict["angles_all_task"]:
                if len(a)<2:
                    test_values.append(np.nan)
                elif np.any(np.isnan(a[:2])):
                    test_values.append(np.nan)
                else:
                    test_values.append(a[1]-a[0])
            # - convert to [-pi, pi]
            test_values = (np.array(test_values) + np.pi) % (2 * np.pi) - np.pi

            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A_continuous(stroke_dict, k, test_values)

        elif "center_" in key:
            # then e.g., center_s0_d1 means take stroke 0 and d 1(y) and use the
            # center coordinates for that stroke (TASK COORDINATES)
            # A TRICK: if want to compose, could do:
            # f = {
            #     "center_s0_d0_0":[0, 0.5],
            #     "center_s0_d0_1":[0.3, 0.9],
            # } I.E. APPEND A _0 or _1 to end. but would not need to do this. instead
            # can easily compose multiple operations on diff strokes and dimensiosn.

            ind1 = key.find("_s")
            ind2 = key.find("_d")
            snum = int(key[ind1+2:ind2])
            dim = int(key[ind2+2:])
            test_values = stroke_centers(stroke_dict, snum=snum, dim=dim)
            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A_continuous(stroke_dict, k, test_values)

        elif "angletask_" in key:
            # then "angletask_s1" means use the angle for s1, in rad with 0 being to right.
            ind1 = key.find("_s")
            snum = int(key[ind1+2:])
            test_values = stroke_angles(stroke_dict, snum=snum)
            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A_continuous(stroke_dict, k, test_values)
        elif "angletaskbin_" in key:
            # then "angletask_s1" means use the angle for s1, in rad with 0 being to right.
            ind1 = key.find("_s")
            snum = int(key[ind1+2:])
            test_values = [a[snum] for a in stroke_dict["angles_all_task_binned"]]
            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A(stroke_dict, k, test_values)

        elif "not_empty":
            # only keep if you have nonempty strokes
            test_values = [len(s)>0 for s in stroke_dict["strokes_all"]]
            for k in keys_to_check:
                if k in stroke_dict.keys():
                    stroke_dict[k] = _A(stroke_dict, k, test_values)


        # elif key=="center_0_x":
        #     # first stroke, x coord
        #     test_values = stroke_centers(stroke_dict, snum=0, dim=0)
        #     for k in keys_to_check:
        #         if k in stroke_dict.keys():
        #             stroke_dict[k] = _A_continuous(stroke_dict, k, test_values)
        # elif key=="center_0_y":
        #     test_values = stroke_centers(stroke_dict, snum=0, dim=1)
        #     for k in keys_to_check:
        #         if k in stroke_dict.keys():
        #             stroke_dict[k] = _A_continuous(stroke_dict, k, test_values)
        else:
            assert False, "dont knwo this one"

    return stroke_dict        



def processAssignStrokes(stroke_dict, ver="v2"):
    """takses segmented strokes for behavior and task, and 
    algins them. for each beh stroke assigns the task stroke
    that is closest
    v1: (mod haussdorf distance) independent for each beahvior
    stroke, finds its closest task stroke. 
    therefore a given task stroke could be unassigned.
    -Ideally first segments strokes. this should be automatically
    done by default in getMultTrialsStrokeDict, for both task and
    behavior
    v2: [DEFAULT], does DTW (or variant) to align beh and task, and
    takes optimal alignment. from that alignment matches each beh stroke
    to a target stroke.
    """
    _assertNotAlreadyDone(stroke_dict, "stroke_assignments_all")

    if ver=="v1":
        from pythonlib.tools.vectools import modHausdorffDistance
        distances_all =[]
        stroke_assignments_all = []
        for strokes_beh, strokes_task in zip(stroke_dict["strokes_all"], stroke_dict["strokes_all_task"]):
            # - every real stroke must be "assigned" a task stroke (not vice versa)
            # - assign each stroke the task stroke that is the closest
            stroke_assignments = [] # one for each stroke in behavuiopr
            for s_beh in strokes_beh:
                # get distnaces from this behavioal stroke to task strokes
                distances = []
                for s_task in strokes_task:
                    distances.append(modHausdorffDistance(s_beh, s_task))

                # assign the closest stroke
                stroke_assignments.append(np.argmin(distances))
                # just for debugging
                distances_all.append(sorted(distances))

            stroke_assignments_all.append(stroke_assignments)
        stroke_dict["stroke_assignments_all"] = stroke_assignments_all
    elif ver=="v2":
        assert False, "not codede yet!"
        # take the ordering of task that is most aligned with the behavior.
        # then find the assignment of behavior strokes to task.

    return stroke_dict

if False:
    # use this to visualize the distances between the closest and second-closest.
    # for one example I checked and very different.
    plt.figure()
    plt.plot(np.array(distances_all)[:,0], np.array(distances_all)[:,1], 'ok');
    plt.xlim([0, 600])
    plt.ylim([0, 600])

def processRemoveShortStrokes(stroke_dict, thresh_time=0.12, thresh_dist = 28, long_output=False):
    """removes short strokes from behavior, if shorter than
    thresh. 
    TODO: make the treshold adaptive
    0.12 and 28 was based on looking at some of Pancho data,
    for 2/26. should make more adaptive. future.
    - will remove any stroke that fails either time or distance criterior.
    """

    strokes_all_out = []
    durs_all = []
    dists_all =[]
    for strokes in stroke_dict["strokes_all"]:
        
        if long_output:
            durs_all.extend([s[-1,2]-s[0,2] for s in strokes])
            dists_all.extend([np.linalg.norm(s[-1,[0,1]]-s[0,[0,1]]) for s in strokes])

        if thresh_time:
            strokes = [s for s in strokes if s[-1,2]-s[0,2] >=thresh_time]
        if thresh_dist:
            strokes = [s for s in strokes if np.linalg.norm(s[-1,[0,1]]-s[0,[0,1]]) >=thresh_dist]

        strokes_all_out.append(strokes)
        
    
    stroke_dict["strokes_all"] = strokes_all_out
    
    if False:
        ## DEBUG CODE, goes thru random trials, for each trial plots the strokes on canvas
        # and lists the durs and lengths.

        # === go thru random trials, plot and show the times and distances
        targ = {
            "fracsuccess_min":[0.2]
        }
        trials_list = getIndsTrials(filedata, targ)

        trial = random.sample(range(len(trials_list)),1)[0]

        stroke_dict_plot = getMultTrialsStrokeDict(filedata, [trials_list[trial]])
        _, times, lengths = processRemoveShortStrokes(stroke_dict_plot, long_output=True)
        stroke_dict_plot = processBehTaskDistance(stroke_dict_plot)
        # lengths = processRemoveShortStrokes(stroke_dict_plot, long_output=True)[2]

        print(f"trial {[trials_list[trial]]}")
        print(f"times (sec): {times}")
        print(f"lengths: {lengths}")

        strokes_task = stroke_dict_plot["strokes_all_task"][0]
        orders = stroke_dict_plot["behtask_taskorders_all"][0]
        distances = stroke_dict_plot["behtask_distances_all"][0]

        plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all_task", strokes_to_plot="all", plotver="strokes")
        plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all", plotver="strokes")
        plotTrialSimple(filedata, trials_list[trial])

        for o,d in zip(orders, distances):
            ax = plotTrialSimple(filedata, 1, plotver="empty")[0]
            strokesthis = [strokes_task[i] for i in o]
            strokes_this = fakeTimesteps(strokesthis, strokes_this[0][0,[0,1]], "from_point")
            strokes_this = fakeTimesteps(strokesthis, [], "from_end_of_previous_stroke")
            plotDatStrokes(strokes_this, ax=ax, plotver="raw")
            plt.title(f"dist {d}")


    if long_output:
        return stroke_dict, durs_all, dists_all
    else:
        return stroke_dict


def processBehTaskDistance(stroke_dict, symmetric=True):
    """gets distance between strokes for task and behavior. uses
    all possible permutations of the task strokes. gets a list for each
    trial, where the list contains all the distances, one for each 
    ordering of the task
    - distance currently uses something like DTW between strokes.
    - if symmetric, then for each comparison, does both beh vs. model and model vs. beh
    then takes the avereage. This is important since currently the distance metric is assymetric,
    since it forces 'using up' of all strokes on the left side, but not the right. not perfectly
    symmetric, since weighs the behvs model component by 2/3. if False, then 
    will be beh vs mod. 
    - """
    from pythonlib.tools.stroketools import distanceBetweenStrokes
    distances_all = []
    orders_task_all = []
    
    if False:
        _assertNotAlreadyDone(stroke_dict, "behtask_taskorders_all")
        _assertNotAlreadyDone(stroke_dict, "behtask_distances_all")
    for strokes_beh, strokes_task in zip(stroke_dict["strokes_all"], stroke_dict["strokes_all_task_orig"]):

        # print([len(s) for s in strokes_beh])
        # print([len(s) for s in strokes_task])

        # get all stroke orders for this task
        strokes_allorders, stroke_orders_set = getAllStrokeOrders(strokes_task)

        # get all distances across all possible orderings of the task strokes
        # do both ways, beh-->mod and mod-->beh
        distances = []
        for strokes_mod in strokes_allorders:
            d = []
            # - beh--> mod
            d.append(distanceBetweenStrokes(strokes_beh, strokes_mod))
            # - mod --> beh
            d.append(distanceBetweenStrokes(strokes_mod, strokes_beh))
            # print(d)
            
            distances.append((2/3)*d[0] + (1/3)*d[1])
            # print(distances)
            # assert False
            # distances.append(d[0])
#             print(distanceBetweenStrokes(strokes_beh, strokes_mod))
#             print(distanceBetweenStrokes(strokes_mod, strokes_beh))
#             print(d)
#             assert False
            
#         distances.extend([distanceBetweenStrokes(strokes_beh, strokes_mod) for strokes_mod in strokes_allorders])
#         distances.extend([distanceBetweenStrokes(strokes_mod, strokes_beh) for strokes_mod in strokes_allorders])
        orders = list(stroke_orders_set)
        assert len(distances)==len(orders), "should be one for each permutation of order..."
        # print(orders[0])
        # print(distances[0])
        # assert False
        distances_all.append(distances)
        orders_task_all.append(orders)

    stroke_dict["behtask_distances_all"] = distances_all
    stroke_dict["behtask_taskorders_all"] = orders_task_all

    if False:
        ## DEBUG - to plot for random trials the distances and canvas drawings for model and task, for all model permutations.
        trial = random.sample(range(len(trials_list)),1)[0]
    #     print(trial)
    #     print([trials_list[trial]])
        stroke_dict_plot = getMultTrialsStrokeDict(filedata, [trials_list[trial]])
        stroke_dict_plot = processBehTaskDistance(stroke_dict_plot)
        stroke_dict_plot = processRemoveShortStrokes(stroke_dict_plot)
        
        strokes_task = stroke_dict_plot["strokes_all_task"][0]
        orders = stroke_dict_plot["behtask_taskorders_all"][0]
        distances = stroke_dict_plot["behtask_distances_all"][0]
    #     print(distances)
    #     print(orders)
    # #     assert False
        plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all_task", strokes_to_plot="all", plotver="strokes")
        plotDictCanvasOverlay(stroke_dict_plot, filedata, "strokes_all", plotver="strokes")
        plotTrialSimple(filedata, trials_list[trial])

        for o,d in zip(orders, distances):
            ax = plotTrialSimple(filedata, 1, plotver="empty")[0]
            strokesthis = [strokes_task[i] for i in o]
            strokes_this = fakeTimesteps(strokesthis, strokes_this[0][0,[0,1]], "from_point")
            strokes_this = fakeTimesteps(strokesthis, [], "from_end_of_previous_stroke")
            plotDatStrokes(strokes_this, ax=ax, plotver="raw")
            plt.title(f"dist {d}")
    

    return stroke_dict

# ==== reorder task strokes to maximize similarity 
def processReorderStrokes(stroke_dict, filedata, method="distance", symmetric=True,
                         reassign_timestamps=False):
    """reorder the strokes in task (ground truth) following method.
    - distance: minimizes the distance betweent he behaviora nd ground stroek.
    generalyl requires segmenting the beahviro and task and comparing task to beahvior.
    Output will be replaced using teh best ordering
    """

    if method=="distance":
        # 1) calculate distances
        if "behtask_distances_all" not in stroke_dict.keys():
            stroke_dict = processBehTaskDistance(stroke_dict, symmetric=symmetric)
            
        # 2) for each task choose the min distance ordering.
        strokes_all_task_reordered = []
        for strokes, orders, distances in zip(stroke_dict["strokes_all_task"],
                                              stroke_dict["behtask_taskorders_all"],
                                              stroke_dict["behtask_distances_all"]):
            order_to_use = orders[np.argmin(distances)]        
            strokes_all_task_reordered.append([strokes[i] for i in order_to_use])
        
        stroke_dict["strokes_all_task"] = strokes_all_task_reordered
            
        # Re-assign timestamps to match current order.
        if reassign_timestamps:
            stroke_dict = processFakeTimesteps(stroke_dict, filedata,
                                               "strokes_all_task", ver='in_order', replace_key_to_do=True)
            stroke_dict = processFakeTimesteps(stroke_dict, filedata,
                                               "strokes_all_task", ver='from_end_of_previous_stroke', 
                                               replace_key_to_do=True)
    else:
        # 1) calculate distances
        print("THIS IS SLOW - rewrite to just get the orders, without getting the distances")
        if "behtask_taskorders_all" not in stroke_dict.keys():
            stroke_dict = processBehTaskDistance(stroke_dict, symmetric=symmetric)

        # everything else do this by first processing to get scores, and then taking the 
        # order with the best score
        if 'scores_taskorders_all' not in stroke_dict.keys():
            stroke_dict = processScoreReorderedStrokes(stroke_dict, filedata, method_order=method)
        strokes_out = []
        for strokes, orders, scores in zip(stroke_dict["strokes_all_task"],
                                              stroke_dict["behtask_taskorders_all"],
                                              stroke_dict["scores_taskorders_all"]):
            # reorder strokes using the best order
            order_to_take = orders[np.argmax(scores)]
            strokes_reordered = [strokes[i] for i in order_to_take]
            strokes_out.append(strokes_reordered)
        stroke_dict["strokes_all_task"] = strokes_out

    _assertNotAlreadyDone(stroke_dict, "ReorderStrokes")
    _addToDoneProcesses(stroke_dict, "ReorderStrokes")
    return stroke_dict

def processScoreReorderedStrokes(stroke_dict, filedata, method_order="prox_to_origin",
    method_score=""):
    """for each possible reordering of task strokes, gives a score based on a 'model'
    note: this is NOT scoring behavior vs. model. it is more like a "prior" than a 
    likelihood.
    - method_order, what metric/model to use to score strokes
    - method_score, how to normalize this score """

    def getScore(strokes, orig, method_order=method_order, method_score=method_score):
        """ the higher the beter"""
        if method_order=="prox_to_origin":
            # order strokes based on the proximities of their COM to origin
            centers = getCentersOfMass(strokes)
            distances = [getDistFromOrig(c, orig) for c in centers]
            s = np.sum(np.diff(distances)) # this is most positive when strokes are ordered from close to far
        elif method_order=="uniform":
            # all stroke sequences weighed equally
            s = np.random.rand()
        elif method_order=="total_dist_traveled":
            # order strokes so that minimize travel, including from fix position and
            assert False, "have not coded"
        else:
            assert False, "have not coded"
        return s

    def normalizeScores(scores, method_score=method_score):
        """ given list of scores (scalars), normalizes them. e.g., 
        treats them as logits. or places on ordinal scale, etc."""
        if method_score=="":
            # do nothing
            return scores
        else:
            assert False, "not yet coded"

    def getCentersOfMass(strokes, method="use_median"):
        from pythonlib.tools.stroketools import getStrokesFeatures
        if method=="use_median":
            return getStrokesFeatures(strokes)["centers_median"] # list of (x,y) arrays
        else:
            assert False, "not coded"

    def getDistFromOrig(point, orig):
        return np.linalg.norm(point-orig)

    # ==========================
    scores_all = []
    for strokes, orders, distances, trial in zip(stroke_dict["strokes_all_task"],
                                          stroke_dict["behtask_taskorders_all"],
                                          stroke_dict["behtask_distances_all"],
                                          stroke_dict["trials_list"]):
        scores = [] # one for each possible ordering
        orig = getTrialsFix(filedata, trial)["fixpos_pixels"]

        for o in orders:
            strokes_this = [strokes[i] for i in o]
            scores.append(getScore(strokes_this, orig)) # the higher the better

        scores = normalizeScores(scores, method_score)

        scores_all.append(scores) # list, length of trials
    
    if False:
        _assertNotAlreadyDone(stroke_dict, "scores_taskorders_all")
    stroke_dict["scores_taskorders_all"] = scores_all

    if False:
        ## DEBUG - plot all the orderings and print scores.
        import random
        trial = random.randint(1, len(stroke_dict["behtask_taskorders_all"]))

        strokes = stroke_dict["strokes_all_task"][trial]
        for orders, scores in zip(stroke_dict["behtask_taskorders_all"][trial],
                                  stroke_dict["scores_taskorders_all"][trial]):
            strokes_this = [strokes[i] for i in orders]
            strokes_this = fakeTimesteps(strokes_this, [], ver="in_order")
            ax = plotTrialSimple(filedata, 1, plotver="empty")[0]
            plotDatStrokes(strokes_this, ax=ax, plotver="strokes")
            plt.title(f"score, this order, {scores}")        

    return stroke_dict


def processPosteriorScores(stroke_dict, filedata, method_likeli="DTW_modHausdorff",
                           method_prior="distance", method_post="top1"):
    """gets one score for each trial, where higher is better.
    does this by considering teh one behavior sequence with
    all the possible model stroke sequences. how this
    consideration is done is based on method. 
    Both the distance metric (between two stroke sequences)
    (like a likelihood) and the sequence score (i.e,, the prior,
    for the model), are flexible.
    """
    # 1) get likelihood scores
    if method_likeli in ["DTW_modHausdorff"]:
        stroke_dict = processBehTaskDistance(stroke_dict)
    else:
        assert False, "not coded"
    
    # 2) get prior scores
    if method_prior in ["prox_to_origin", "uniform"]:
        stroke_dict = processScoreReorderedStrokes(stroke_dict, filedata, method_order=method_prior)
    elif method_prior in ["distance"]:
        assert method_likeli=="DTW_modHausdorff", "coudl be others, but I shoud think abou thtis"
        pass
        # since will use likeli as prior
    else:
        assert False, "not coded"

    # 3) get posterior
    assert method_post=="top1", "not coded"
    post_scores =[]
    likelis_all = stroke_dict["behtask_distances_all"]
    if method_prior=="distance":
        # dont need priors...
        priors_all = [[] for _ in range(len(likelis_all))]
    else:
        priors_all = stroke_dict["scores_taskorders_all"]
    
    for likelis, priors in zip(likelis_all, 
                               priors_all):
        if method_prior=="distance":
            # then prior is to maximize similarity to beahvior - positive control
            p = np.array([-l for l in likelis])
        else:
            p = np.array(priors)
        c = np.random.choice(np.flatnonzero(p == p.max())) # this randomly chooses, if there is tiebreaker.
        post_scores.append(-likelis[c])
    
    _assertNotAlreadyDone(stroke_dict, "posterior_scores")
    stroke_dict["posterior_scores"] = post_scores

    if False:
        # print(stroke_dict["behtask_distances_all"][0])
        # print(stroke_dict["scores_taskorders_all"][0])
        stroke_dict = _getStrokeDict()
        stroke_dict = processPosteriorScores(stroke_dict)
        trial = 5
        print(stroke_dict["behtask_distances_all"][trial])
        print(stroke_dict["scores_taskorders_all"][trial])
        print(stroke_dict["posterior_scores"][trial])

    # 4) done
    return stroke_dict
    

def processCenters(stroke_dict):
    """ for each stroke gets the position of the cetner of the stroke
    i.e., currently uses the median position in x and y.
    for each strokes will output list of np arrays (x,y)
    """
    from pythonlib.tools.stroketools import getStrokesFeatures
    stroke_centers = []
    stroke_centers_task = []
    for strokes, strokes_task in zip(stroke_dict["strokes_all"],
                                     stroke_dict["strokes_all_task"]):
        stroke_centers.append(getStrokesFeatures(strokes)["centers_median"])
        stroke_centers_task.append(getStrokesFeatures(strokes_task)["centers_median"])
    stroke_dict["stroke_centers"] = stroke_centers
    stroke_dict["stroke_centers_task"] = stroke_centers_task
    return stroke_dict


def processChopStrokes(stroke_dict):
    """[inprogress] if long strokes (behavior),
    chop them up into shorter...
    - useful if want to compare beh to task, but beh
    cannot be segmented in to strokes based on velocity 
    minima accurately"""
    assert False, "in progress!"

print("NOTE: need to not overwrite strokes_all_task, because then the orders saved will stop being accurate. Modify")
