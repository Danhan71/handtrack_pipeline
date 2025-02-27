from pyvm.tools.utilsh5 import *
from pyvm.tools.calc import *
from pythonlib.tools.stroketools import fakeTimesteps, check_strokes_in_temporal_order
from pyvm.tools.tasks import convertTask2Strokes

import numpy as np
import sys
sys.path.append("/home/lucast4")


BLOCKPARAMS_ALREADY_CONVERTED_TO_LIST = True

####################### SINGLE TRIAL EXTRACTION/PROCESSING CODE
def getTrialsBehCodes(filedata, trial):
    """get this trial's beh code timings
    [Note: code 13 (frameskip) only started collecting post 6/20/20
    RETURNS:
    - trial_behcodes, dict
    --- (all items None, if this trial has no codes at all fopr some reaosn.)
    """
    
    trial_behcodes = {}
    
    if filedata["trials"][trial]["BehavioralCodes"]["CodeNumbers"][()] is None:
        trial_behcodes["num"] = None
        trial_behcodes["time"] = None    
        trial_behcodes["string"] = None
    else:
        trial_behcodes["num"] = filedata["trials"][trial]["BehavioralCodes"]["CodeNumbers"][()][0]
        # print(trial_behcodes["num"])
        # print([d for d in trial_behcodes["num"]])
        trial_behcodes["num"] = [int(d) for d in trial_behcodes["num"]]
        
        trial_behcodes["time"] = filedata["trials"][trial]["BehavioralCodes"]["CodeTimes"][()][0]/1000
        
        for n in trial_behcodes["num"]:
            if n not in filedata["params"]["beh_codes"].keys():
                print(n)
                print(filedata["params"]["beh_codes"])
                print(trial_behcodes["num"])
                assert False, "need to add this code in preprocess.py"

        trial_behcodes["string"] = [filedata["params"]["beh_codes"][n] for n in trial_behcodes["num"]]
    
    return trial_behcodes

def getTrialsTimeOfBehCode(filedata, trial, code):
    """given a trial and code, tells you when that code occured happened (in trial time, sec)
    ; code can be int or string"""
    
    bc = getTrialsBehCodes(filedata, trial)
    
    if isinstance(code, str):
        # first convert the string to a number
        code_num = [a for a,b in filedata["params"]["beh_codes"].items() if b==code]
        if len(code_num)==0:
            print("this code string doesnt exist...")
            code = []
        else:
            assert(len(code_num)==1), "are there kmultipel codes with the same name?"
            code = code_num[0]
        
    return [t for n, t in zip(bc["num"], bc["time"]) if n==code]    

def getTrialsIsAbort(filedata, trial):
    """ True if was abort, False otherwise"""
    out = getTrialsOutcomesWrapper(filedata, trial)
    if out["online_abort"]["failure_mode"] is None:
        return False 
    else:
        return True

def getTrialsOutcomesWrapper(filedata, trial):
    """ 
    This is best.
    updated, now separated into two things:
    (1) online abort reason (and relevant stats)
    (2) beh evaluation
    Note: this throws out somet hings in onld
    getTrialsOutcomes, which were recorded for other versions
    of online abort, but I mostly don't use
    - NOTE: this should pick out useful things from:
    getTrialsOutcomes, getTrialsOutcomesAll, and 
    getTrialsBehEvaluation
    """

    outcome = {}

    oc = getTrialsOutcomes(filedata, trial)    
    be = getTrialsBehEvaluation(filedata, trial, exclude_strings=False)
    bb = getTrialsAdapterParams(filedata, trial)
    BP = getTrialsBlockParamsHotkeyUpdated(filedata, trial)
    params_task = BP["params_task"]
    done_button_criterion = params_task["donebutton_criterion"]
    date = filedata["params"]["date"]

    if "trialEndMethod" not in oc.keys():
        # Then was fixation error
        assert oc["failure_mode"] in ["no fixation made", "fixation lost"], "why else would I not have saved trialEndMethod?"
        outcome["trial_end_method"] = "fixation_error"
        # outcome["DA_reportDoneSuccess"] = False
    else:
        # Then got to task.
        # 1) COLLECT INFO
        if params_task["donebutton_criterion"]!="skip" and 62 in getTrialsBehCodes(filedata, trial)["num"]:
            pressed_done_button = True
        else:
            pressed_done_button = False

        # Is this a successfult trial?
        if oc["trialEndMethod"]=="report_raise":
            # DragAroundv2, successful completion. 
            # NOTE before 10/15/21 this sometimes could be "report_raise" yet still be due to hotkey abort. FIXED.
            assert 63 in getTrialsBehCodes(filedata, trial)["num"], "confused - isnt this the eventmarker?"
            if not (filedata["params"]["animal"] == "Luca" and filedata["params"]["date"] == "220902"):
                assert done_button_criterion in ["success", "skip"], "these only reasons woudl get signals for both DA success and done button success"
            if done_button_criterion == "success":
                # Then assume you must have pressed the done button.
                if pressed_done_button:
                    report_success=True
                elif pressed_done_button==False and date=="211015":
                    # Only allowed on 211015. i.e. you cannot report_raise and not have pressed done button...
                        # hotkey abort, but code did not save properly into  bb.FailureMode="hotkey_abort"
                        # and erroneuosly puts "report_raise" into oc["trialEndMethod"]. I expect that should never
                        # reach here after 211015.
                    report_success=False
                elif pressed_done_button==False:
                    report_success=False # 220902 - unique situation, luca finishes task w/o pressing done button
                    # probably hotkey aborted
                else:
                    assert False, "what is this"

            # for luca 220902 - coding error where doneButtonCriterion==go_cue AND reportDone==numstrokes
            # which means that done button pops up by default as soon as trial starts
            # but also trial ends in 'report_raise' if uses up numstrokes
            # so, can have trial with report_raise BUT done_button_criterion isn't just success/skip
            elif done_button_criterion == "go_cue":
                if pressed_done_button:
                    report_success = True
                else:
                    report_success = False
            else:
                # report raise means you succeded criteria.
                report_success=True
        else:
            report_success = False

        # Was there an online abort? Note, could be both success (above) and online abort because of hotkey abort.
        if oc["trialEndMethod"]=="online_abort":
            # could be either online abort or hotkey
            online_abort = True 
        else:
            online_abort = False


        if oc["failure_mode"] == "hotkey_abort":
            # can have hotkey abort, but not save in  oc["trialEndMethod"]
            online_abort = True

        if oc["trialEndMethod"] not in ["report_raise", "online_abort"]:
            if date=="211015":
                # is because done button only show after DAv2 sucecss, BUT can touch that area even before
                # it shows, and this ends the trial. But does not record anything trialEndMethod (empty).
                # Call this online abort, since this is really a failure.
                online_abort=True
                pressed_done_button = False
                report_success=False
            elif int(date)<211015:
                # hacky, many cases are OK, but code was bad. after this date shouldnt happen I think.
                pass
            elif int(date)==211106:
                # Hack, "Lucas" cagetest2. I cant tell why doesnt work..
                pass
            elif (int(date)==211021) & (trial==59):
                # Hack, not sure why...
                pass
            elif oc["failure_mode"]=="hotkey_abort":
                pass
            elif (int(date)==211028) & (trial==171):
                # not sure why
                pass
            elif int(date)>211015 and report_success==False and len(oc["failure_mode"])==0 and len(oc["trialEndMethod"])==0:
                oc["failure_mode"] = "unknown"
                oc["trialEndMethod"] = "unknown"

                # in some cases, e.g, Diego 211028 trial 202 sess 1, get what looks like timeout, but not saved as such. the folliwing things were printed in the else statment below:
                # but if this is report_success==False, then call this a failure.

                # 0
                # 202
                # False
                # False
                # False
                # trial 202
                # trialEndMethod 
                # {'failure_mode': '', 'num_finger_raises': array([[4.]]), 'dist_peanut_failure': array([], shape=(0, 0), dtype=float64), 'trialEndMethod': '', 'trial_report_done_status': ''}
                # {'trialnum': array([[202.]]), 'output': {'dist_total': {'rescale': array([[0.45312039]]), 'value': array([[-1.44445432]])}, 'frac_touched': {'rescale': array([[0.9454736]]), 'value': array([[0.95041322]])}, 'shortness': {'rescale': array([[1.]]), 'value': array([[1.]])}, 'hausdorff': {'rescale': array([[1.]]), 'value': array([[-0.1743927]])}, 'frac_overlap': {'rescale': array([[0.82434211]]), 'value': array([[0.561]])}, 'frac_strokes': {'rescale': array([[nan]]), 'value': array([[nan]])}, 'posterior': {'rescale': array([[nan]]), 'value': array([[nan]])}, 'ft_decim': {'rescale': array([[0.63942308]]), 'value': array([[0.75]])}, 'numstrokes': {'rescale': array([[0.625]]), 'value': array([[-2.]])}, 'numstrokes_frac': {'rescale': array([[0.54545455]]), 'value': array([[-1.36363636]])}}, 'rew_total': array([[59.83250384]]), 'beh_multiplier': array([[0.13707959]]), 'bias_multiplier': array([[1.]]), 'binary_evaluation': array([[1.]]), 'score_final': array([[0.13707959]]), 'idletime': array([[6441.42542149]]), 'screencolor': array([[0.89033632],
                #        [0.30966368],
                #        [0.02741592]])}
                # bb FailureMode 
                # bb AbortNow [[0]]
            elif pressed_done_button==True and len(oc["trialEndMethod"])==0 and bb["FailureMode"] in ["too_far_from_ink", "failed_rule_objectclass"]:
            # elif int(date) in [220319, 220320, 220321, 220322, 220323] and pressed_done_button==True and len(oc["trialEndMethod"])==0 and bb["FailureMode"] in ["too_far_from_ink", "failed_rule_objectclass"]:
                # on these days it is possible that bb.trialEndMoethod is empty.
                # Is abort, but wasnt triggered (flipped a coin). For some reason didnt save trialEndMethod as "report_raise" but should have.
                oc["trialEndMethod"] = "report_raise" # is acrtually trial where failed rule, made wn sound, should have aborted, but these days trying method where allow to continue.
                report_success = True

            elif pressed_done_button==False and len(oc["trialEndMethod"])==0 and bb["FailureMode"] in ["time_limit"]:
                oc["trialEndMethod"] = "online_abort" # is 
                online_abort = True
            elif oc["failure_mode"]=="failed_rule_objectclass" and report_success==False and pressed_done_button==False and len(oc["trialEndMethod"])==0:
                # THis is online abort not sure why didnt save
                oc["trialEndMethod"] = "online_abort"
                online_abort = True
            else:
                # hotkey abort can be not noted in bb, therefore not noted in trial end method.
                # but will def be noted in failure_mode (10/16/21 onwards)
                # assert(bb["FailureMode"]=="hotkey_abort")
                print(oc["failure_mode"])
                print(len(oc["failure_mode"]))
                print(trial)
                print(online_abort)
                print(pressed_done_button)
                print(report_success)
                print("trial", trial)
                print("trialEndMethod", oc["trialEndMethod"])
                print("trialEndMethod", bb["trialEndMethod"])
                print(oc)
                print(be)
                print("bb FailureMode", bb["FailureMode"])
                print("bb AbortNow", bb["abortNow"])
                try:
                    print("abortNowCounter", bb["abortNowCounter"])
                except err as Exception:
                    pass
                assert oc["failure_mode"]=="hotkey_abort"
                # assert False, "10/16/21, I assume this, but maybe not, if not, then check why and if should add to code."

        if oc["failure_mode"]=="hotkey_abort":
            # NOTE, 10/15/21 code updated so this is ALWAYS correct.
            hotkey_abort = True
        else:
            hotkey_abort = False

        if 70 in getTrialsBehCodes(filedata, trial)["num"]:
            # only 8/24/21 and onwards.
            hotkey_x = True
        else:
            hotkey_x = False


        # 2) RETURN mutually exclusive outcomes
        if hotkey_x:
            if pressed_done_button or report_success:
                # Then successed befofre I presed x
                outcome["trial_end_method"] = "postscene_hotkey_abort"
            elif hotkey_abort==False and online_abort==True:
                # Then normal abortion before I pressed x
                outcome["trial_end_method"] = "online_abort"
            else:
                # Then pressed x during the task
                outcome["trial_end_method"] = "online_abort"
                # print(oc["failure_mode"])
                # print(len(oc["failure_mode"]))
                # print(trial)
                # print(online_abort)
                # print(pressed_done_button)
                # print(report_success)
                # print("trial", trial)
                # print("trialEndMethod", oc["trialEndMethod"])
                # print(oc)
                # print(be)
                # print("bb FailureMode", bb["FailureMode"])
                # print("bb AbortNow", bb["abortNow"])

                assert len(oc["failure_mode"])==0 or oc["failure_mode"] in ["unknown", "hotkey_abort"], f"{oc['failure_mode']}"
                oc["failure_mode"] = "hotkey_x" # since, for hotkey_x, this will be empty.
        elif hotkey_abort==True:
            # Then is hotkey_z.

            # print(oc)
            # print(be)
            # print(trial, pressed_done_button, report_success, params_task)
            # print(getTrialsBehCodes(filedata, trial))
            # assert False
            # if online_abort == False:
            #     print(oc)
            #     print(be)
            #     print(trial, pressed_done_button, report_success, params_task)
            #     print(getTrialsBehCodes(filedata, trial))
            #     assert online_abort==True

            if pressed_done_button or report_success:
                # Then pressed after succeeded in task already. Dont callt this online abort
                outcome["trial_end_method"] = "postscene_hotkey_abort"
            else:
                # Then pressed before completed. failure.
                outcome["trial_end_method"] = "online_abort"
        elif hotkey_abort==False and online_abort==True:
            # online abort due to behavior.
            outcome["trial_end_method"] = "online_abort"
        elif report_success and pressed_done_button:
            outcome["trial_end_method"] = "pressed_done_button"
        elif pressed_done_button:
            # Then done button, This might be marked as success (if done button criterion is success)
            outcome["trial_end_method"] = "pressed_done_button"
        elif report_success:
            # Then not done button, but success in task.
            outcome["trial_end_method"] = "report_raise"
        elif int(filedata["params"]["date"])<210824:
            # Then is likely hotkey_x, since I had not added an event_code yet for this. 
            outcome["trial_end_method"] = "unknown_probably_hotkey_x"
        elif not np.isnan(be["score_final"]):
            # Then sometimes done button eventcode doesnt succesfuly signal, but is still success...
            # print(params_task["donebutton_criterion"])
            # # assert(bb["FailureMode"]=="hotkey_abort")
            # print(oc["failure_mode"])
            # print(len(oc["failure_mode"]))
            # print(trial)
            # print(online_abort)
            # print(pressed_done_button)
            # print(report_success)
            # print("trial", trial)
            # print("trialEndMethod", oc["trialEndMethod"])
            # print(oc)
            # print(be)
            # print("bb FailureMode", bb["FailureMode"])
            # print("bb AbortNow", bb["abortNow"])
            assert params_task["donebutton_criterion"]!="skip"
            outcome["trial_end_method"] = "done_but_method_unclear"
        elif report_success==False and pressed_done_button==False:
            outcome["trial_end_method"] = "finished_task_no_press_done_button"
        else:
            # Not sure..
            print(oc)
            print(be)
            print(params_task["donebutton_criterion"])
            print(trial, pressed_done_button, report_success, online_abort, hotkey_abort)
            print(getTrialsBehCodes(filedata, trial))
            assert False

        # 1) Online abort
        outcome["online_abort"] = {
            "failure_mode":oc["failure_mode"] if outcome["trial_end_method"]=="online_abort" else None
        }

    # assert outcome["trial_end_method"] in ["postscene_hotkey_abort", "report_raise", "pressed_done_button", "online_abort", "fixation_error"], f"outcome: {outcome}"


    # # 1) figure out trial end method
    # if "trialEndMethod" not in oc.keys():
    #     # Then was fixation error
    #     assert oc["failure_mode"] in ["no fixation made", "fixation lost"], "why else would I not have saved trialEndMethod?"
    #     outcome["trial_end_method"] = "fixation_error"
    #     # outcome["DA_reportDoneSuccess"] = False
    # else:
    #     if len(oc["trialEndMethod"])>0:
    #         # Could be anything
    #         outcome["trial_end_method"] = oc["trialEndMethod"]
    #         # outcome["DA_reportDoneSuccess"] = True if bb["reportDoneSuccess"][0]==1 else False

    #     elif len(oc["failure_mode"])>0:
    #         # probably hotkey (online abort). this is the only thing with faillure that doesnt have trialEndMethod
    #         if oc["failure_mode"]!="hotkey_abort":
    #             assert False, "just checking. why else?"
    #         outcome["trial_end_method"] = "online_abort"
    #         # outcome["DA_reportDoneSuccess"] = True if bb["reportDoneSuccess"][0]==1 else False
    #         # assert outcome["DA_reportDoneSuccess"] == False

    #     elif params_task["donebutton_criterion"]!="skip" and 62 in getTrialsBehCodes(filedata, trial)["num"]:
    #         # Then used done button, and succesfully pressed it.
    #         outcome["trial_end_method"] = "pressed_done_button"
    #         # outcome["DA_reportDoneSuccess"] = True

    #     elif be is not None:
    #         if len(be["score_final"]):
    #             # Then confident this is fine, a good completed trial.
    #             outcome["trial_end_method"] = "done_but_method_unclear"
    #         else:
    #             print("trial", trial)
    #             for k, v in be.items():
    #                 print(k, v)
    #             print("-")
    #             for k, v in oc.items():
    #                 print(k, v)

    #             # Note: this is what is looks like when done_but_method_unclear, and you print as above.
    #             # Not clear why, maybe becuase hotkey abort? But this would assign bb.FailureMode="hotkey_abort", and
    #             # because, would not explain why num_finger has 2, the others empty.:
    #             # NO: I confirmed that still reaches here on trials where bb["FailureMode"] is empty. 
    #             # trial 45
    #             # trialnum [[45.]]
    #             # output {'dist_total': {'rescale': array([[1.]]), 'value': array([[-0.68415149]])}, 'shortness': {'rescale': array([[0.40778404]]), 'value': array([[0.68415149]])}, 'hausdorff': {'rescale': array([[0.65165321]]), 'value': array([[-0.46446239]])}, 'numstrokes': {'rescale': array([[1.]]), 'value': array([[2.]])}, 'frac_overlap': {'rescale': array([[0.60955224]]), 'value': array([[0.55597015]])}, 'frac_strokes': {'rescale': array([[0.6]]), 'value': array([[0.75]])}, 'posterior': {'rescale': array([[0.2]]), 'value': array([[0.12728089]])}}
    #             # rew_total [[0.]]
    #             # beh_multiplier [[0.05658253]]
    #             # bias_multiplier [[1.]]
    #             # binary_evaluation [[0.]]
    #             # score_final [[0.]]
    #             # -
    #             # failure_mode 
    #             # num_finger_raises [[2.]]
    #             # dist_peanut_failure []
    #             # trialEndMethod 
    #             # trial_report_done_status 

    #             assert False, "what is this trial?"
    #     else:
    #         print("trial", trial)
    #         for k, v in be.items():
    #             print(k, v)
    #         print("-")
    #         for k, v in oc.items():
    #             print(k, v)
    #         assert False, "not sure how this trial ended"


    # # 1) Online abort
    # outcome["online_abort"] = {
    #     "failure_mode":oc["failure_mode"] if outcome["trial_end_method"]=="online_abort" else None
    # }

    ### Remove any keys in be["outcome"] that are not numerical
    # ie. be["outcome"] should be like:
        # dist_total {'rescale': array([[1.]]), 'value': array([[-0.21624363]])}
        # frac_touched {'rescale': array([[0.]]), 'value': array([[0.37777778]])}
        # shortness {'rescale': array([[0.]]), 'value': array([[0.21624363]])}
        # hausdorff {'rescale': array([[0.]]), 'value': array([[-1.30184591]])}
        # frac_overlap {'rescale': array([[nan]]), 'value': array([[nan]])}
        # frac_strokes {'rescale': array([[nan]]), 'value': array([[nan]])}
    # but seomtimes feature_picked:frac_touched is included (my mistake in matlab)
    # example be:
        # trialnum [[49.]]
        # output {'dist_total': {'rescale': array([[1.]]), 'value': array([[-0.24350323]])}, 'frac_touched': {'rescale': array([[nan]]), 'value': array([[nan]])}, 'shortness': {'rescale': array([[0.]]), 'value': array([[0.24350323]])}, 'hausdorff': {'rescale': array([[0.]]), 'value': array([[-2.91434463]])}, 'frac_overlap': {'rescale': array([[0.]]), 'value': array([[0.26408451]])}, 'frac_strokes': {'rescale': array([[0.]]), 'value': array([[0.]])}, 'ft_decim': {'rescale': array([[0.]]), 'value': array([[0.2]])}, 'ft_minobj': {'rescale': array([[0.]]), 'value': array([[0.]])}, 'numstrokes_frac': {'rescale': array([[1.]]), 'value': array([[-0.61538462]])}, 'posterior': {'rescale': array([[0.]]), 'value': array([[-6.42449189]])}, 'pacman': {'rescale': array([[1.]]), 'value': array([[-0.]])}, 'numstrokesorig': {'rescale': array([[0.]]), 'value': array([[-2.]])}, 'circleness_obj': {'rescale': array([[nan]]), 'value': array([[nan]])}, 'pacman_minobj': {'rescale': array([[1.]]), 'value': array([[-0.]])}, 'feature_picked': 'ft_minobj'}
        # rew_total [[0.]]
        # beh_multiplier [[0.]]
        # bias_multiplier [[1.]]
        # binary_evaluation [[1.]]
        # score_final [[0.]]
        # score_originally_computed_beheval [[0.]]
        # idletime [[6005.05925732]]
        # screencolor [[1. ]
        #  [0.2]
        #  [0. ]]
    if be is not None:
        output_new = {}
        output = be["output"]
        for k, v in output.items():
            if isinstance(v, str) or k=="feature_picked":
                # Put it into be
                assert k not in be.keys()
                be[k] = v
            else:
                output_new[k] = v
        be["output"] = output_new

    # 2) beh evaluation
    outcome["beh_evaluation"] = be
    outcome["error_code"] = getTrialsErrorCode(filedata, trial)

    # # If hotkey abort, but after done buttn pressed, then don't call this onlien abort.
    # if outcome["trial_end_method"]=="online_abort" and 45 in getTrialsBehCodes(filedata, trial)["num"] and outcome["online_abort"]["failure_mode"]=="hotkey_abort":
    #     outcome["trial_end_method"] == "postscene_hotkey_abort"
    #     outcome["online_abort"]["failure_mode"] = None

    # # SANITY CHECK
    # if outcome["trial_end_method"] in ["report_raise", "pressed_done_button"]:
    #     assert 45 in getTrialsBehCodes(filedata, trial)["num"], f"sanity check failed. not sure why, {outcome}"
    # elif outcome["trial_end_method"] in ["online_abort", "fixation_error"]:
    #     assert 45 not in getTrialsBehCodes(filedata, trial)["num"], f"sanity check failed. not sure why, {outcome}"
    
    # assert outcome["trial_end_method"] in ["postscene_hotkey_abort", "report_raise", "pressed_done_button", "online_abort", "fixation_error"], f"outcome: {outcome}"

    return outcome



def getTrialsOutcomes(filedata, trial):
    """for this trial, get its otucomes status. 
    This can be tricky since for some cases only noted down trial outcome
    for trials in which fixation was successful. this code should take 
    care of that"""
    if filedata["params"]['n_trials'] == filedata["params"]["n_trialoutcomes"]:
        # then good, can just use trial as the currect trial number
        trialoutcomes = group2dict(filedata["TrialRecord"]["User"]["TrialOutcomes"][f"{trial}"])

        # Some cleaning up.
        bb = getTrialsAdapterParams(filedata, trial)

        # It is possible for bb["FailureMode"] to be hotkey_abort while oc["failure_mode"] is something else, if
        # hotkey abort was pressed after the line in drag.m where oc["failure_mode"]=bb.FailureMode.
        # In this case, reassign oc to what bb says.
        if isinstance(bb, dict):
            if bb["FailureMode"]=="hotkey_abort":
                trialoutcomes["failure_mode"] = "hotkey_abort"
    else:
        print("n_trials", filedata["params"]['n_trials'])
        print("n_trialoutcomes", filedata["params"]["n_trialoutcomes"])
        # filedata["params"]["max_trials_because_use_resaved_TrialRecord"]
        raise ValueError("need to code", "mismatch")
        # this case should have been handled in:
        # -- preprocess.py: loadSingleDataH5 ("elif MAXTRIALS is not None:")
        # -- preprocess.py: updateFiledataParams ("@KGG 220712 — commenting out")
    return trialoutcomes

# print(getTrialsOutcomes(filedata, 100))

def getTrialsNewSaveTasksver(filedata):
    """ Returns True if this is post 4/10/23, saving
    tasks to data
    """
    UserVarsBase = filedata["trials"][1]["UserVars"]
    return "AdapterParams" in UserVarsBase.keys() or "CurrentTask" in UserVarsBase.keys()


def getTrialsAdapterParams(filedata, trial, adapter="bb"):
    """ get adapter params for desired adapter.
    returns empty list if this trial did not pass fixation.
    returns epty list if trial doesnt exist. this may happen for a legit trial
    if at end of day (since must be flanked by pass-fix trials.
    returns None if adapter params not a key"""
    
    # print("----")
    # print(filedata["trials"][trial])

    # if "UserVars" not in filedata["trials"][trial].keys():
    #     print("HERERER",  filedata["trials"][trial])
    #     print("HERERER",  filedata["trials"][trial].keys())
    #     assert False
    # print("----")
    # print(filedata["trials"].keys())
    # print(trial)
    # print(filedata["trials"][trial]["UserVars"].keys())
    # print(filedata["trials"][trial])

    # for k, v in filedata["trials"].items():
    #     print(k, v.keys())
    # assert False

    # if "UserVars" not in filedata["trials"][1].keys():
    #     for k, v in filedata["trials"].items():
    #         print(k, v.keys())
    #     print(filedata["trials"].keys())
    #     print(filedata["trials"][1].keys())
    #     assert False

    if getTrialsNewSaveTasksver(filedata):
        # post 4/10/23 and beyond.

        if "UserVars" not in filedata["trials"][trial].keys():
            # This is only reasonalbe if this is the last trail.
            assert max(filedata["trials"].keys())==trial
            # print("HERERER",  filedata["trials"][trial])
            # print("HERERER",  filedata["trials"][trial].keys())
            return []  
        else:
            UserVars = filedata["trials"][trial]["UserVars"]
            if "AdapterParams" in UserVars.keys():
                return UserVars["AdapterParams"][adapter]
            else:
                return []
    elif "AdapterParams" not in filedata["TrialRecord"]["User"].keys():
        return None
    elif len(filedata["TrialRecord"]["User"]["AdapterParams"])==0:
        return []
    elif f"{trial}" in filedata["TrialRecord"]["User"]["AdapterParams"].keys():
        return filedata["TrialRecord"]["User"]["AdapterParams"][f"{trial}"][adapter]
    else:
        return []


def getTrialsOutcomesAll(filedata, trial):
    """extract the outcome of this trial, puts all outcomes
    into one dict"""
    TrialRecord = filedata["TrialRecord"]

#     trialoutcomes = group2dict(TrialRecord["User"]["TrialOutcomes"][f"{t}"])
    trialoutcomesall = {}
    trialoutcomesall["trialoutcomes"] = getTrialsOutcomes(filedata, trial)
    A = getTrialsAdapterParams(filedata, trial)
    if A is None:
        # old version
        fracinkgotten_ = filedata["TrialRecord"]["User"]["InkGotten"][f"{trial}"][()][0]
    else:
        if A:
            if len(A["InkGotten"])==0:
                # new version, post DragAroundv2, not saving InkGotten.
                fracinkgotten_ = A["InkTouchedWhileVisible"][0]
            else:
                fracinkgotten_ = A["InkGotten"]
        else:
            fracinkgotten_ = [np.nan]

    trialoutcomesall["fracinkgotten"] = np.sum(fracinkgotten_)/len(fracinkgotten_)
    # trialoutcomesall["errorcode"] = filedata["data"][f"Trial{trial}"]["TrialError"][()][0]
    trialoutcomesall["errorcode"] = filedata["trials"][trial]["TrialError"][()][0]
    
    return trialoutcomesall


def getTrialsSpeedMinima(filedata, trial, plot_debug=False, window=0.4):
    """finds timepoints where speed is at local minimum, 
    useful for segmenting strokes
    returns list of times (seconds) where detected pause
    - NOTE: works well for single lines but not as well for curved stuff...
    """
    from scipy.signal import find_peaks, find_peaks_cwt

    vels = getTrialsStrokesVelocity(filedata, trial, window=window)
    if len(vels)==0:
        if False:
            # to visualize why no vels, which is likely because segments shorter than
            # the smoothing winodw.
            plotTrialSimpleTimecourse(filedata, trial)
        return []
    
    # --- various params useful for finding minimum
    vels_flat = np.array([vv for v in vels for vv in v])[:,0] # flatten to T x 2
    prom = (max(vels_flat)-min(vels_flat))/4 # prominence
    dist = 4 # arbitrary, but seemed reasonable
    minheight = -(min(vels_flat) + (max(vels_flat) - min(vels_flat))/10) # only find minima that are really slow.
    # --- Collect all minima (pauses)
    pauses_time = []
    for V in vels:
        v = V[:,0]
        peaks, properties = find_peaks(-v, prominence=prom, distance=dist, height=minheight)
        
        if plot_debug:
            import matplotlib.pyplot as plt
            print(peaks)
            print(properties)
            plt.figure()
            plt.plot(v, '.k')
            plt.plot(peaks, np.zeros(peaks.shape), 'or')

        # -- real time of minima
        if len(peaks)>0:
            for p in peaks:
                pauses_time.append(V[p,1])
#     print(pauses_time)
    return pauses_time


def getTrialsParams(filedata, trial):
    """ get params for this trial. i.e., the params_task field."""
    return filedata["TrialRecord"]["User"]["Params"][f"{trial}"]

def getTrialsTimesOfRaises(filedata, trial):
    """ get times of raises recorded in DragAround
    online. if empty, returns empty array
    """
    
    A = getTrialsAdapterParams(filedata, trial)
    if A is None:
        # then old version
        tor = filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["TimesOfRaises"]
    else:
        if A:
            tor = A["TimesOfRaises"]
        else:
            tor = []
    # try:
    #     # new, after pruning size of data fuile by removing stuff.
    #     tor = getTrialsAdapterParams(filedata, trial)["TimesOfRaises"]
    # except:
    #     tor = filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["TimesOfRaises"]
        
    if len(tor)>0:
        return tor[0]
    else:
        return np.empty(0)

def getTrialsAnalogData(filedata, trial, ver):
    """ HELPER to extract analog data. Identialy to what I have used for touch
    """

    def _get_timebins(dat):
        """
        PARAMS:
        - dat, array of length of samples. len(dat) must be the num timebins.
        """
        assert len(dat.shape)==1
        dat_interval = filedata["trials"][trial]["AnalogData"]["SampleInterval"][()] # period for sampling, in ms
        dat_t = np.arange(0, len(dat)*dat_interval, dat_interval)/1000 # timestamps, in sec
        # sanity cehck
        assert(np.isclose(dat_interval, 1000/filedata["params"]["sample_rate"]))
        return dat_t


    if ver in ["Btn1"]:
        # Returns tbins x 2 (binary logic, timebins in sec
        dat = filedata["trials"][trial]["AnalogData"]["Button"][ver][()][0]
        
        dat_t = _get_timebins(dat)

        # combine and return
        return np.array([(x,t) for x,t in zip(dat, dat_t)])

    elif ver in ["Touch"]:
        # touch data, (tbins x 3), where columsn are (x, y, tsec)
        dat_touch = filedata["trials"][trial]["AnalogData"]["Touch"][()] 

        # extract x and y. (1) convert from deg to pixlels and (2) rotate so it is in perpsective of animal
        dat_touch_x = dat_touch[1]*filedata["params"]["pix_per_deg"][0] 
        dat_touch_y = dat_touch[0]*filedata["params"]["pix_per_deg"][1]
        dat_t = _get_timebins(dat_touch_x)

        return np.array([(x,y,t) for x,y,t in zip(dat_touch_x, dat_touch_y, dat_t)])
    elif ver in ["Eye"]:
        # Returns (N,3) where n is num bins and cols are (x,y,t)
        dat = filedata["trials"][trial]["AnalogData"]["Eye"][()]
        dat_x = dat[0,:]
        dat_y = dat[1,:]
        dat_t = _get_timebins(dat_x)

        # combine and return
        return np.array([(x,y,t) for x,y,t in zip(dat_x, dat_y, dat_t)])
    else:
        assert False


def getTrialsEyeData(filedata, trial, return_as_pixels=True):
    """ Get entire trial eye position data
    REturns (N,3) data
    PARAMS:
    - return_as_pixels, then returns as pixels.

    NOTE: to recalibrate from raw data:
    https://monkeylogic.nimh.nih.gov/board/read.php?3,1078,1078#msg-1078

        That is the transformation matrix, but just knowing where it is will not help you much. Keep reading if you want to know how to use the matrix.

        First of all, you need to fill out some missing information in MLConfig, since MLConfig does not keep the information of the subject screen dimension.

        sx = 1024;  % subject screen width
        sy = 768;   % subject screen height
        MLConfig.Screen.SubjectScreenRect = [0 0 sx sy];
        MLConfig.Screen.SubjectScreenFullSize = [sx sy];
        MLConfig.Screen.SubjectScreenHalfSize = 0.5 * [sx sy];
        MLConfig.Screen.SubjectScreenAspectRatio = sx / sy;


        Then reconstruct the calibration object.

        EyeCal = mlcalibrate('eye', MLConfig, 1);  % 1 means Eye #1


        With the calibration object, you can calculate the degree positions of raw signals like the following.

        deg_xy = EyeCal.sig2deg(raw_xy, [0 0]);  % raw_xy is an n-by-2 matrix. [0 0] is offset; do not change it.


        -----

        FYI, MLConfig at the time of recording is saved in the data file and you can retrieve it like this.

        [data, MLConfig] = mlread;

        In the new version of NIMH ML, MLConfig will keep the subject screen info so that there is no need to fill it out later.     
    """
    xyt = getTrialsAnalogData(filedata, trial, "Eye")

    if return_as_pixels:
        xyt[:,:2] = convertDeg2PixArray(filedata, xyt[:,:2])

    return xyt


def getTrialsTouchingBinary(filedata, trial):
    """ Return for eah time bin wherther is touching
    RETURNS:
    - times, array of times
    - touching, array of 0,1
    """
    xyt = getTrialsTouchData(filedata, trial)
    x = xyt[:,0]
    # y = xyt[:,1]
    touching = 1-np.isnan(x).astype(int)
    times = xyt[:,2]
    return times, touching

def getTrialsTouchData(filedata, trial, post_go_only=False, window_rel_go_reward=None):
    """ to get x,y,t coordinates for this trial.
    rotates and rescales so that is in coordiantes of pixels
    and is matched to what animal sees. and is same as coords for
    tasks
    post_go_only, then gets for times after go. if no go, then empty
    - Otherwise, doesnt do any preprocessing"""

    if window_rel_go_reward is None:
        window_rel_go_reward = []

    if len( window_rel_go_reward)>0:
        assert(post_go_only==False), "can only one of these two two options"
        assert(len(window_rel_go_reward)==2), "need to give seconds preceding go and following reward [-0.1, 0.1]"
    # print('***')
    # print(window_rel_go_reward)


    def _getTrialsTouchData(filedata, trial):
        # gets entire trial, ignoreing go.
        # print(trial)
        return getTrialsAnalogData(filedata, trial, "Touch")
        # dat_touch = filedata["trials"][trial]["AnalogData"]["Touch"][()] 

        # # extract x and y. (1) convert from deg to pixlels and (2) rotate so it is in perpsective of animal
        # dat_touch_x = dat_touch[1]*filedata["params"]["pix_per_deg"][0] 
        # dat_touch_y = dat_touch[0]*filedata["params"]["pix_per_deg"][1]
        # dat_interval = filedata["trials"][trial]["AnalogData"]["SampleInterval"][()] # period for sampling, in ms
        # dat_t = np.arange(0, len(dat_touch_x)*dat_interval,dat_interval)/1000 # timestamps, in sec
        
        # assert(np.isclose(dat_interval, 1000/filedata["params"]["sample_rate"]))
        # return np.array([(x,y,t) for x,y,t in zip(dat_touch_x, dat_touch_y, dat_t)])

    if post_go_only:
        time_go = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")
        if len(time_go) == 0:
            # then did not pass fixation

            # print("Failed to pass fixation  - printing outomces:")
            # print(getTrialsOutcomesAll(filedata, trial))
            return []
        elif len(time_go)>1:
            dat = _getTrialsTouchData(filedata, trial)
            inds = dat[:,2] >= time_go[0]
            return dat[inds, :]            
        else:
            dat = _getTrialsTouchData(filedata, trial)
            inds = dat[:,2] >= time_go
            return dat[inds, :]
    elif window_rel_go_reward:
        time_go = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")
        time_rew = getTrialsTimeOfBehCode(filedata, trial, "reward")
        if len(time_go)==0:
            return []
        else:
            dat = _getTrialsTouchData(filedata, trial)
            if len(time_rew)>0:
                inds = (dat[:,2] >= time_go[0]+window_rel_go_reward[0]) & (dat[:,2] <= time_rew[0]+window_rel_go_reward[1])
            else:
                # then don't require there to be reward - i.e., if did not attain reward
                # (e.g., since failed post-go) then still allow extarction of all data 
                # post go.
                inds = (dat[:,2] >= time_go[0]+window_rel_go_reward[0])

            # print("====")
            # print(time_go)
            # print(dat[0])
            # print(dat[-1])
            # print(sum(inds))
            # print(inds)
            # print(dat[inds][0])
            # print("===")
            return dat[inds, :]
    else:
        return _getTrialsTouchData(filedata, trial) 


def getTrialsSketchpad(fd, t, add=0.):
    """ return actual sketchpad, in format [[-x, -y], [+x, +y]], where
    (0,0) is center of page, i..e, monkey coords. in pixel units, so same as 
    strokes represntations.
    - add, number to pad the edges with, in pix.
    RETURNS:
    - eitehr [0 0 1 1] (if empty) or (2,2) array
    """

    TaskParams = getTrialsTaskParams(fd, t)
    if TaskParams is not None:
        sketchpad_edges_mk = TaskParams["sketchpad"]["edges_monkey"]
    else:
        sketchpad_edges_mk = getTrialsBlockParamsHotkeyUpdated(fd, t)["sketchpad"]["edges_monkey"]
    
    # THis can be empty, if in matlab code dont want to have sketchpad outline, etc.
    if len(sketchpad_edges_mk)==0:
        # sketchpad_edges_mk = np.array([[0., 0.], [1., 1.]])
        sketchpad_edges_mk = np.array([[0., 1.], [0., 1.]])
    elif sketchpad_edges_mk is None:
        sketchpad_edges_mk = np.array([[0., 1.], [0., 1.]])

    pos = convertCoords(fd, sketchpad_edges_mk.T, "monkeynorm2centeredmonkeypix")
    sketchpad_edges_pixcentered = pos

    # -- optionally add 20 pix to end, since some beh goes a bit over.
    sketchpad_edges_pixcentered[0,:] = sketchpad_edges_pixcentered[0,:] - add
    sketchpad_edges_pixcentered[1,:] = sketchpad_edges_pixcentered[1,:] + add

    return sketchpad_edges_pixcentered


def getTrialsTask(filedata, trial):
    """ Get taskstruct for this tak, as a dict (I think?)
    """
    if getTrialsNewSaveTasksver(filedata):
        # post 4/10/23 and beyond.
        if "UserVars" not in filedata["trials"][trial].keys():
            # This is only reasonalbe if this is the last trail.
            assert max(filedata["trials"].keys())==trial
            task = None 
        else:
            UserVars = filedata["trials"][trial]["UserVars"]
            if "AdapterParams" in UserVars.keys():
                task = UserVars["AdapterParams"]["bb"]["taskstruct"]
            else:
                task = UserVars["CurrentTask"]

    elif "CurrentTask" in filedata["TrialRecord"]["User"].keys():
        # old version, stopped saving this on ~ 10/27, since saving in adapter paranms
        # then I restarted saving this on 10/28, since bb is not saved if dont pass fixation...need to save tasks.
        task = group2dict(filedata["TrialRecord"]["User"]["CurrentTask"][f"{trial}"])
    elif "CurrentTask" in filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"].keys() and isinstance(filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["CurrentTask"], dict):
        # current version, for cases where fixation fail, do this.
        task = filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["CurrentTask"]
    elif f"{trial}" in filedata["TrialRecord"]["User"]["AdapterParams"].keys() and \
        "bb" in filedata["TrialRecord"]["User"]["AdapterParams"][f"{trial}"].keys() and \
        "taskstruct" in filedata["TrialRecord"]["User"]["AdapterParams"][f"{trial}"]["bb"].keys():
    # elif getTrialsFixationSuccess(filedata, trial): # Avoid this, goes into infinite recursion.
        task = filedata["TrialRecord"]["User"]["AdapterParams"][f"{trial}"]["bb"]["taskstruct"]
    else:
        # since bb is not saved/
        # this should only happen for a few days around 10/25/2020, since I neglected tos ave CurrentTask 
        # if fail fixation.
        task = None


    # task_x = -(task["y"] - 0.5) * filedata["params"]["resolution"][1]
    # task_y = -(task["x"] - 0.5) * filedata["params"]["resolution"][0]
    if task is not None:
        # print(task.keys())
        task["x_rescaled"] = -(task["y"] - 0.5) * filedata["params"]["resolution"][1]
        task["y_rescaled"] = -(task["x"] - 0.5) * filedata["params"]["resolution"][0]

    # parse the one continuous stroke into single strokes
    # NDOTS = 15 # this is assumed true, since is not a saved param. if I change in matlab code
    # then will save the param


    return task
    # return {
    #     "task":task,
    #     "task_x":task_x,
    #     "task_y":task_y
    # }


# def getTrialsFixTimeOnset(filedata, trial):
#     """ Get time of onset of holding fixation, based on 

def getTrialsFix(filedata, trial):
    """fixation params, 
    shoudl be aligned as animal sees it, and comapitble with task and 
    touchd data"""
    dat = group2dict(filedata["TrialRecord"]["User"]["Params"][f"{trial}"]["fix"])
    # convert fixpos to pixels
    if "fixpos" in dat.keys():
        dat["fixpos_pixels"] = convertDeg2Pix(filedata, dat["fixpos"])
    else:
        if getTrialsBehtype(filedata, trial) == "Trace (instant)":
            # then assume that fixpos is (0,0)
            dat["fixpos_pixels"] = convertDeg2Pix(filedata, [0,0])
            # print(dat)
        else:
            raise KeyError("fixpos doesnt exist for this dataset", "nokey") 
    return dat

def getTrialsFixationSuccess(filedata, trial):
    """ simply true if passed fixation, false otherwise"""

    if getTrialsOutcomesWrapper(filedata, trial)["trial_end_method"]=="fixation_error":
        return False
    else:
        return True

def getTrialsTouched(filedata,trial):
    """ True if temporal overlap between peanuts ands trokes exist, false otehrwise.
    """
    # if len(getTrialsPeanutPos(filedata, trial))>0:
    if len(getTrialsStrokesByPeanuts(filedata, trial))>0:
        return True
    else:
        return False


# def getTrialsBlockParamsSpecific(filedata, trial, item):
#     """ picks out specific item from blockparams, specific
#     to trial (accounting for any hotkeys pressed). 
#     item is variable length list, i.e., keys that will apply
#     in sequence to BP. 
#     - e.g.,: item = ('task_staging', 'task_scheduler')
#     """
#     BP = getTrialsBlockParamsHotkeyUpdated(filedata, trial)
# #     print(BP.keys())
# #     for a in BP["task_staging"].items():
# #         print(a)

#     assert isinstance(item, tuple) or isinstance(item, list)
#     A = BP
#     for i in item:
#         A = A[i]
#     return A

# === Plot number of replays per trial
def getTrialsReplayStats(filedata, trial):
    """ extract replayStats for this trial.
    - returns None if cannot find.
    - count = 0, 1, ...
    - round is dict with keys 1, 2, 3... length same as count
    - note: the last round for replays will NOT be the same
    as the final peanutpos saved for the trial. therefore can have
    entirely empty replay (i..e, when solved on first try)
    """

    # check whether trial ended due to fixation error
    if getTrialsFixationSuccess(filedata, trial):
        if "replayStats" in filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"].keys():
            replayStats = filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["replayStats"]
            # print("--------")
            # print(filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"])
            # print(replayStats)
            # print(trial)

            if len(replayStats)==0:
                # then not using replayStats
                return None
            if "count" not in replayStats.keys():
                # print(replayStats)
                # assert False
                return None
            if "round" not in replayStats.keys():
                # then this did not have replays, so did not save rounds.
                replayStats["round"]={}
            if "allPeanutPos" in replayStats["round"].keys():
                # then this was only one round, so round number wasnt saved
                tmp = {
                "1":replayStats["round"]
                }
                replayStats["round"] = tmp

            assert replayStats["count"][0][0]==len(replayStats["round"].keys())

            return replayStats
        else:
            print(filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"])
            return None
    else:
        return None


def getTrialsPeanutSampCollisExt(filedata, trial):
    """ Return the actual peanut peanut extension, this plus size is total size.
    RETURNS:
    - number, the pnut samp extension.
    """
    bp = getTrialsBlockParamsHotkeyUpdated(filedata, trial)
    return bp["params_task"]["PnutSampCollisExt"]
    # return bp["params_task"]["PnutSampCollisExt"][0][0]



def getTrialsPeanutPos(filedata, trial, replaynum=None):
    """the actual positions of peanuts put down 
    by finger during the task,
    - replaynum, if a number then will check whether there werre 
    replays, if replays, then will take data from the first round.
    (1 means the first behavior) [1, 2, 3...]
    - if replaynum==1, then will return the regular peanut pos if no 
    replays occured. for higher repalynum, will return empty if did not
    get to that replay num.
    """
    

    if False:
        # OLD VERSION - before saved each trial data.
        # this has flaw int hat if p3eanut poist empty, thenm matlab appending did not
        # work well. so those are empty. Now hard to figure oput which trials those are

        assert False, "not ready - reason is that sometime allPeanutPos is not aligned with data trials. see code"    
        if filedata["params"]["n_trials"] != len(filedata["TrialRecord"]["User"]["allPeanutPos"]):
            print('ERROR - allPeanutPos is not aligned with data truials')
            print("Not returning peanut positions - should fix this problem")
            print('fix by noting that skips whenever peanutpos is empty')
            return []
        else:
            return filedata["TrialRecord"]["User"]["allPeanutPos"][f"{trial}"][()].T

    # NEW VERSION
    if getTrialsFixationSuccess(filedata, trial) is False:
        return np.array([])

    def pnut():
        A = getTrialsAdapterParams(filedata, trial)
        if A is None:
            # then use old version
            xyt = filedata["TrialRecord"]["User"]["TrialData"][f"{trial}"]["allPeanutPos"].T
        else:
            if A:
                xyt = A["allPeanutPos"].T
            else:
                xyt = []
        return xyt

    if replaynum is None:
        # get final peanuts
        xyt = pnut()
    # elif getTrialsBlockParamsHotkeyUpdated(filedata, trial)["replay"]["activate"][0][0]==0:
    elif getTrialsBlockParamsHotkeyUpdated(filedata, trial)["replay"]["activate"]==0:
        # no replays activated, get final peanuts
        xyt = pnut()
    else:
        # replay stats can exist. get them
        replayStats = getTrialsReplayStats(filedata, trial)
        assert replayStats is not None, "not sure why? cant be beucause not fix, since I arleady checked that above (could be becuase is trial 1?)"

        if f"{replaynum}" in replayStats["round"].keys():
            # then take this replay
            xyt = replayStats["round"][f"{replaynum}"]["allPeanutPos"].T
        else:
            if replaynum==1:
                # if did any replay, then the first replay would have been labeled "1" and that would have
                # been pulled out above. since was not, that means this trial had no replays. confirm that
                # with assertion, then pull out the final peanut pos as output.
                assert replayStats["count"]==0, "expect that no replays happened (since not even '1' is ink rounds keys) ..."
                # then return the peanut pos, even if there were no replays
                xyt = pnut()
            else:
                # did not do this many replays. also, since you gave me replay >1, that means you 
                # actually want that round, so will not return antyhing in substitute.
                xyt = []
                print("no data for this replay round")

    out = np.array(xyt)
    if len(out)>0:
        out[:,:2] = convertCoords(filedata, out[:,:2], "pix2monkey")
        out[:,2] /= 1000 # convert to sec 
    return out 


def getTrialsRunParamsBlock(filedata, trial):
    """ which runparams block was used on this trial for defining run params?
    RETURNS:
    - int, the block number
    """
    RunParams = filedata["TrialRecord"]["User"]["RunParams"]
    if "TrialData" in RunParams.keys():
        # Then is latest code where was saving
        # for trial in range(1, 500):
        #     print(trial, RunParams["TrialData"][f"{trial}"]["run_params_block"])
        # assert False

        # print(trial)
        # # print(getTrialsBehCodes(filedata, trial))
        # print(RunParams["TrialData"][f"{trial}"])
        # print(RunParams["TrialData"][f"{trial}"]["run_params_block"])
        # print(RunParams["TrialData"][f"{trial}"]["run_params_block"][0])
        if f"{trial}" not in RunParams["TrialData"].keys(): # hack - sometimes a final stretch doesn't save, b/c of fixation error
            return getTrialsRunParamsBlock(filedata, trial-1)
        if len(RunParams["TrialData"][f"{trial}"]["run_params_block"])==0:
            # Then failed to save - this occured in days after first split into runparams, since
            # didnt save if restarted after fixation error. just call this ths same as this trials
            # BlockParams block
            indblock = getTrialsBlock(filedata, trial)
        else:
            indblock = int(RunParams["TrialData"][f"{trial}"]["run_params_block"][0][0])
    else:
        assert int(filedata["params"]["date"]) < 220224
        # Then this was when was using the taskblock
        indblock = getTrialsBlock(filedata, trial)
    return indblock

def getTrialsBlock(filedata, trial):
    """starting from 1, gets block for this trial
    - this is the block ID (i.e., neednt be chronological"""
    assert trial>0
    return int(filedata["TrialRecord"]["BlocksPlayed"][trial-1][0])
    # return int(filedata["TrialRecord"]["BlockCount"][trial-1][0])

def getTrialsBloque(filedata, trial):
    """ I define bloque as current position in sequenc eof blocks. 
    i./e., always 1, 2, 3, 4.... regardless of what the actual 
    block (id) is. 
    - this is only possible post-COVID. before, I did not mark block
    transitions, so transition that repeats block would not be stored
    """    
    
    if False:
        # this is post-covd code, but then I realized that did not need this.
        # see alternative method bleow.
        # I confirmed that these are indentical.
        bloque_onsets = fd["TrialRecord"]["User"]["TrialOfBlockUpdate"]

    if False:
        # to plot to compare blocks vs. bloques.
        bloque = fd["TrialRecord"]["BlockCount"]
        block = fd["TrialRecord"]["BlocksPlayed"]
        blocks_by_bloque = fd["TrialRecord"]["BlockOrder"]

        plt.figure(figsize=(10,5))
        plt.plot(bloque, block, '-ok')
        plt.figure(figsize=(10,5))
        x = list(range(1, len(bloque)+1))
        plt.plot(x, bloque, '-b')
        plt.plot(x, block, '-k')

        for b in bloque_onsets:
            plt.plot([b, b], [0, 60], '-r')

    assert trial>0
    # method 1
    if False:
        # DOES NOT WORK if a block is repeated (will erroneuosly count as on
        # bloque)
        bloque_num = int(filedata["TrialRecord"]["BlockCount"][trial-1][0])
        return bloque_num    
    
    # method 2
    else:
        bloque_onsets = filedata["TrialRecord"]["User"]["TrialOfBlockUpdate"]
        b = np.sum(trial>= bloque_onsets)
        return b
    # try:
    #     assert bloque_num == b, "Probably a repeat block, which did not register as two bloques in 'BlockCount'?"
    # except:
    #     print(bloque_onsets)
    #     print(bloque_num)
    #     print(trial)
        # raise

def getTrialsBlokk(filedata, t):
    """ Blokk, defined as contigous bloques that are the 
    same block. e..g, if 6 bloques, with the following 
    blocks: [1 2 2 3 3 2], then blokks are [1 2 2 3 3 4]
    """
    # print(filedata["trials"][t])
    return int(filedata["trials"][t]["BlockCount"][0][0])
    
    if False:
        # Run the follwioing to see distinction between block,
        # blokk, and bloque.
        BlockCount_list = []
        Block_list = []
        Bk =[]
        Bq = []
        for t in getIndsTrials(fd):
            BlockCount_list.append(getTrialsBlokk(fd, t))
            Block_list.append(fd["trials"][t]["Block"][0][0])
            Bk.append(getTrialsBlock(fd, t))
            Bq.append(getTrialsBloque(fd, t))
        plt.figure()
        plt.plot(BlockCount_list, "--b")
        plt.plot(Block_list, "-k")
        plt.plot(Bk, "--r")
        plt.plot(Bq, "--g")


def getTrialsStrokesVelocityXY(filedata, trial, window=0.05,
    hi_smoothing=False):
    """
    get x,y, velocities - returns strokes_vels, which is list of np arays (N x 3).
    See getTrialsStrokesVelocity for more details.
    - does appropriate smoothings etc.
    """
    from pythonlib.tools.stroketools import strokesVelocity
    assert hi_smoothing==False, "this leads to weird things for short strokes."
    strokes = getTrialsStrokes(filedata, trial, adapt_win_len="adapt", only_if_part_of_stroke_postgo=False,
    dofilter=True)
    fs = filedata["params"]["sample_rate"]
    if hi_smoothing:
        # _, strokes_vels = strokesVelocity(strokes, fs=fs, 
        #     fs_new=20, do_pre_filter=True, lowpass_freq = 5)
        strokes_vels, _ = strokesVelocity(strokes, fs=fs, 
        lowpass_freq = 5)
    else:
        # strokes_vels, _ = strokesVelocity(strokes, fs=fs,
        # lowpass_freq = 15)
        strokes_vels, _ = strokesVelocity(strokes, fs=fs,
        lowpass_freq = None)

    return strokes_vels

def getTrialsStrokesSpeed(filedata, trial, window=0.05,
    hi_smoothing=False):

    return getTrialsStrokesVelocity(filedata, trial, window=window,
        hi_smoothing=hi_smoothing)

def getTrialsStrokesVelocity(filedata, trial, window=0.05,
    hi_smoothing=False):
    """
    note: will remove any stroke that is too short for the window
    UPDATED on 9/16/20 - to use new version of velocity, 5pt differentiation
    # NOTE: this actually outputs speeds, not vels
    - hi_smoothing, controls how much to smooth the final timecourse speed.
    eveyrthing else the same. empricially this is the manjor factor for waht
    final speed looks like (i.e, fs_new is less important). 5hz is good for 
    smooth-primitives (e.,g hump), while ~30 hz is good for seeing faster stuff.
    """
    if False:
        strokes = getTrialsStrokes(filedata, trial, smooth_window=window, adapt_win_len="remove")
        strokes_vels = [dat2velocity(s) for s in strokes] # only for storkes that are longer then smoothing window..
    else:
        from pythonlib.tools.stroketools import strokesVelocity
        strokes = getTrialsStrokes(filedata, trial, adapt_win_len="adapt", only_if_part_of_stroke_postgo=False,
        dofilter=True)

        assert hi_smoothing==False, "this leads to weird things for short strokes."

        fs = filedata["params"]["sample_rate"]
        if hi_smoothing:
            # _, strokes_vels = strokesVelocity(strokes, fs=fs, 
            #     fs_new=20, do_pre_filter=True, lowpass_freq = 5)
            _, strokes_vels = strokesVelocity(strokes, fs=fs, 
            lowpass_freq = 5)
        else:
            _, strokes_vels = strokesVelocity(strokes, fs=fs,
            lowpass_freq = None)
            # _, strokes_vels = strokesVelocity(strokes, fs=fs,
            # lowpass_freq = 15)

    return strokes_vels

def getTrialsBehEvaluationParams(filedata, trial):
    """ Get the beh eval params, i.e., for the scoring function, that was active on this
    trial.
    RETURNS:
    - feature_dict, where keys are featurenames and values are dicts holding the feature's
    params
    """
    feature_dict = {}
    for val in getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behEval"]["beh_eval"]:
        feature_dict[val["feature"]] = {k:v for k, v in val.items() if k!="feature" and isinstance(v, np.ndarray)}
    return feature_dict

def getTrialsBehEvalAdaptiveParams(filedata, trial):
    """ Get adaptive beh evaluator params. which controls
    the automatic updating of scorer
    RETURNS:
    - outdict, holding info for this trial
    """

    # Get the adaptive beh eval limits
    x = getTrialsBlockParamsHotkeyUpdated(filedata, trial)["adaptive_beheval"]
    
    # Get the hard limits for each feature
    tmp = x["hardlimits"]
    # n = len(x["hardlimits"])
    # tmp =[]
    # for i in range(n):
    #     tmp.append(x["hardlimits"][f"{i+1}"])
    features = tmp[0::2]
    limits = tmp[1::2]
    limits_low = []
    limits_high = []
    for l in limits:
        limits_low.append(l[0].squeeze())
        limits_high.append(l[1].squeeze())
    out = []
    for f, low, high in zip(features, limits_low, limits_high):
        out.append({
            "feature": f,
            "limits_low":low,
            "limits_high":high})

    # Collect all info for output
    outdict = {}
    _keys = ["params", "scope"]
    for k in _keys:
        outdict[k] = x[k]
    outdict["hardlimits"] = out
    
    outdict["on"] = x["on"]==1.
    outdict["N"] = x["N"]
    outdict["schedule"] = x["schedule"]

    return outdict

def getTrialsBehEvaluationFeatures(filedata, trial, include_others = None):
    """ gets features, only those that have a weight >0,
    i.e,. those that actually matter for score.
    outputs a dict, where D[feature] = value.
    - By default only includes the raw "faetures" but can include reward and score with flags:
    --- include_others = ['rew_total', 'beh_multiplier', 'bias_multiplier', 'binary_evaluation', 'score_final']
    --- include_others = "all" to include ['rew_total', 'beh_multiplier', 'bias_multiplier', 'binary_evaluation', 'score_final']
    """
    
    if include_others is None:
        include_others = []

    # -- inlude all factors in beahvioral score (unelss already donea bove)
    featurestoplot = []
    if BLOCKPARAMS_ALREADY_CONVERTED_TO_LIST:
        for val in getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behEval"]["beh_eval"]:
            if val["weight"]>0:
                featurestoplot.append(val["feature"])
    else:
        for key, val in getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behEval"]["beh_eval"].items():
            if val["weight"][0][0]>0:
                featurestoplot.append(val["feature"])

    fd_dict = {}
    BE = getTrialsBehEvaluation(filedata, trial)
#     print(BE["output"].keys())
    for f in featurestoplot:
        # print(f, BE["output"][f], len(BE["output"][f]["value"]), BE["output"][f]["value"].shape)
        fd_dict[f] = BE["output"][f]["value"][0][0]

    if include_others=="all":
        include_others = ['rew_total', 'beh_multiplier', 'bias_multiplier', 'binary_evaluation', 'score_final']

    for f in include_others:
        fd_dict[f] = BE[f][0][0]

    return fd_dict



def getTrialsStrokes(filedata, trial, window_rel_go_reward=None, 
    smooth_window=[], concat_into_xyt=False, cleanstrokes=False, 
    adapt_win_len="adapt", only_if_part_of_stroke_postgo=False,
    dofilter=True, filt_hi=15):
    """segments and outputs strokes
    can add things to be smart about segmenting. 
    most basic is to use finger raises
    
    should tell it how much data to take pre-go and post-reward

    smooth_window leave empty to not smooth. enter a number in seconds to do 
    boxcar window smoothing. 
    smooth_window = 0.05 is good.

    Note: only outputs strokes that are longer thatnt he filtering window.

    do either cleanstrokes (overwrites window_rel_go_reward), or
    leave window_rel_go_reward empty (will get all data) or
    enter value for window_rel_go_reward (will check cleanstrokes is False)
    - only_if_part_of_stroke_postgo, if True, then only keeps strokes that have some
    data post-go. (uses first go cue)

    - 9/9/20, modified default to do smoothing. was previously smooth_window=[] and
    adapt_win_len="error"

    - 9/16/20 - by default do filter instead of smoothing. see random_spectralcontent_091620
    notebook for comparison of these two moethod. could go even down to 10/15 hz, see 
    PSD in that notebook, not much power above 5hz.

    PARAMS:
    - window_rel_go_reward, [time_pre_go time_post_rew] in seconds. e.,g [-0.1, 0.1]
    """

    if window_rel_go_reward is None:
        window_rel_go_reward = []
    from pyvm.tools.calc import segmentTouchDat

    if cleanstrokes:
        assert len(window_rel_go_reward)==0, "if want clean strokes, then implie want to use preset good window."
        window_rel_go_reward=[-0.1, 0.1]

    # print(2)
    # print(window_rel_go_reward)
    xyt = getTrialsTouchData(filedata, trial, window_rel_go_reward=window_rel_go_reward) 

    # print(xyt[0])   
    # print("2-end)")
    strokes = segmentTouchDat(xyt) 

    # If adjacent strokes off and on close in time, then assume this is one stroke
    # and is touhscreeen error.
    from pythonlib.tools.stroketools import concatStrokesTimeGap
    MINTIME = 0.065 # 4 frames
    DEBUG = False
    if DEBUG:
        n_in = len(strokes)
        import copy
        strokes_in = copy.deepcopy(strokes)
    strokes = concatStrokesTimeGap(strokes, MINTIME, DEBUG=DEBUG)
    if DEBUG:
        # print and save figures.
        n_out = len(strokes)
        if not n_in==n_out:
            print("===============")
            print("trial", trial)
            print(n_in, n_out)
            print("--- strokes in")
            for iiiii, s in enumerate(strokes_in):
                print(iiiii, s[0, 2], s[-1,2])
            print("--- strokes out")
            for iiiii, s in enumerate(strokes):
                print(iiiii, s[0, 2], s[-1,2])

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for s in strokes_in:
                ax.plot(s[:,0], s[:,1], 'x')
            fig.savefig(f"/tmp/fig_{trial}.png")

            fig, ax = plt.subplots()
            for s in strokes:
                ax.plot(s[:,0], s[:,1], 'x')
            fig.savefig(f"/tmp/fig_{trial}_after.png")

            # assert False
    
    if isinstance(smooth_window, list) and len(smooth_window)==0:
        # default
        pass
    else:
        # do smoothing
        window_type = "flat"
        window_type = "hanning"
        strokes = smoothStrokes(strokes, filedata["params"]["sample_rate"], 
            window_time=smooth_window, window_type=window_type,
                 adapt_win_len=adapt_win_len)

    if dofilter:
        from pythonlib.tools.stroketools import strokesFilter
        strokes = strokesFilter(strokes, [None, filt_hi], filedata["params"]["sample_rate"],
         N=9)

    if only_if_part_of_stroke_postgo:
        time_go = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")

        if len(time_go)==1:
            # for s in strokes:
            #     try:
            #         np.any(s[:,2]>time_go)
            #     except:
            #         print("here")
            #         print(time_go)
            #         print(s)
            try:
                strokes = [s for s in strokes if np.any(s[:,2]>time_go)]
            except Exception:
                print(strokes)
                raise(Exception)
        elif len(time_go)>1:
            # then likely due to "replay"
            # wil get strokes post fuirst go.
            try:
                strokes = [s for s in strokes if np.any(s[:,2]>time_go[0])]
            except Exception:
                print(strokes)
                raise(Exception)
        else:
            strokes = []

    if concat_into_xyt:
        # instead of heirarchical, just output one N x 3 np array, built from strokes.
        # print("---")
        # print(strokes[0])
        strokes = np.array([ss for s in strokes for ss in s])
        # strokes = np.array()
        # print(strokes[0])
        # print("---")

    return strokes


def getTrialsStrokesClean(filedata, t, throw_out_first_stroke=None):
    """ wrapper for
    _getTrialsStrokesClean. if dont pass in throw_out_first_stroke, wioll;
    decide based on the beahvior type.
    NOTE: don't use this!!! (this is old version)
    """
    assert False, "use getTrialsStrokesByPeanuts"

    if throw_out_first_stroke is None:
        beh_type = getTrialsBlockParamsHotkeyUpdated(filedata, t)["behtype"]["type"]
        if beh_type=="Trace (instant)":
            throw_out_first_stroke=True
        elif beh_type=="Trace (pursuit,track)":
            throw_out_first_stroke=False
             # keep first stroke, since drawing starts on first stroke
        else:
            assert False, "dont know this one"
    return _getTrialsStrokesClean(filedata, t, throw_out_first_stroke=throw_out_first_stroke)


def _getTrialsStrokesClean(filedata, trial, throw_out_first_stroke):
    """ processed. only strokes within boundaries of
    go and reward. 
    Segment into strokes based on combination of: 
    - time rel go and reward. throws out if is just fixation
    - speed minima
    NOTE: tested for Pancho, 2/26, using lines.
    throw_out_first_stroke = True # this is during fixation
    - also segments strokes based on velocity minima
    """
    assert False, "use getTrialsStrokesByPeanuts"

    if throw_out_first_stroke:
        print(getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behtype"]["type"])
        assert False, "make sure type is correct"
        assert getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behtype"]["type"]=="Trace (instant)", "I am only sure it makes sense to remove first stroke if is this beh type"

    # -- get strokes
    strokes = getTrialsStrokes(filedata, trial, window_rel_go_reward=[-0.1, 0.1])

    if len(strokes)==0:
        return strokes

    # -- remove first stroke, since this is just fixation touchign
    if throw_out_first_stroke:
        del strokes[0]
    if len(strokes)==0:
        return strokes
    # -- further segment strokes based on speed minima
    pausetimes = getTrialsSpeedMinima(filedata, trial, )
    strokes_new = []
    for s in strokes:
        onsets_new = [] # collects for this stroke
        for p in pausetimes:
            if p>s[0, 2] and p<s[-1, 2]:
                # then this pause is within this stroke
                # find the index to split at
                idx_min = np.argmin(abs(s[:,2] - p))
                if p>=s[idx_min,2]:
                    on = idx_min+1
                else:
                    on = idx_min
                onsets_new.append(on)

        # ---- pull out the stroke(s)
        if not onsets_new:
            # -- then just add the current stroke
            strokes_new.append(s)
        else:
            onsets = [0]
            onsets.extend(onsets_new)
            onsets.extend([len(s)+1])
            # -- then chop up the current stroke
            for on1, on2 in zip(onsets[:-1], onsets[1:]):
                strokes_new.append(s[on1:on2])
    return strokes_new
# ---- debug plotting for getTrialsStrokesClean
if False:
    ax = plotTrialSimple(filedata, trial, plotver="empty")[0]
    plotDatStrokes(strokes_new, ax, plotver="strokes")

    ax = plotTrialSimpleTimecourse(filedata, trial, plotver="empty")
    plotDatStrokesTimecourse(strokes_new, ax, plotver="raw")

    ax = plotTrialSimpleTimecourse(filedata, trial, plotver="empty")
    plotDatStrokesTimecourse([strokes_new[0]], ax, plotver="raw")


def _get_strokes_overlapping_pnuttimes(filedata, strokes_orig, 
    pnut_times, append_sec_stroke_onset=None, append_sec_stroke_offset=None):
    """ returns list of strokes which start before and end after a "peanut strokes" 
    i..e,, if a strok (np array) contains any of the pnut times, then it will be inlcuded in return.
    PARAMS:
    - pnut_times, array of times.
    - append_sec_stroke_onset, to append tofirst t of strokes_orig, to be more leneint. 
    Negative value means look ealeir in time.
    - append_sec_stroke_offset, to append to last t ef strokes_orig, to be more leneint. 
    Negative value means look ealeir in time.
    RETURNS:
    - strokes, subset of strokes_orig.
    NOTE: appens screen period to strokes on and off, to have some leeway.
    """
    per = filedata["params"]["screen_period"] # sec
    if append_sec_stroke_onset is None:
        # append_sec_stroke_onset = -per/2
        append_sec_stroke_onset = -per # 1/13/24 - since sometimes missing done button..
    if append_sec_stroke_offset is None:
        # append_sec_stroke_offset = per/2
        append_sec_stroke_offset = per # 1/13/24 - since sometimes missing done button..
    # strokes = [s for s in strokes_orig if np.any((pnut_times >= s[0,2]-per/2) & (pnut_times <= s[-1,2]+per/2))]
    strokes = [s for s in strokes_orig if np.any((pnut_times >= s[0,2] + append_sec_stroke_onset) & (pnut_times <= s[-1,2] + append_sec_stroke_offset))]
    return strokes

def getTrialsStrokesByPeanuts(filedata, trial, 
                             only_if_part_of_stroke_postgo = True,
                             replaynum=None):
    """ Get touch data, filtered only so that it overlaps in 
    time with putting of peanuts. i.e., first extracts all
    strokes. then asks, for each stroke, whether at least one
    peanut was placed during stroke. if so, then keeps.
    Why not just take peanut? because is sparsely plootted.
    Why not just take bounds within onset and offset of peanut? 
    becasue (at some point before fixed code) not register offset if
    finger not moving, and sometimes doesnt register onset well.
    - replaynum, if a number then will check whether there werre 
    replays, if replays, then will take data from the first round. 
    - include_overlapping_go_cue, then includes the stroke when touching go cue
    - include_overlapping_done_button, then includes stroke for touch done button.
    [USE THIS!!]
    NOTE: 
    - will make sure that include_overlapping_go_cue, include_overlapping_done_button, then
    will make sure that will inclued that stroke. if failes to extract, then will raise error. 
    i.e.,, if include_overlapping_go_cue, then sure that strokes[0] will be the one touching go.
    - If no peanut, then will return [].
    - Is possible that there is peanute, but doesnt overlap with strokes. e..g, if touch of fix
    overlaps go, then this is a peanut, but not counted as a stroke. The stroke later, let's say 
    it hits the wrong thing to start, and then aborts. then that stroke will not have peanut. Therefore
    the only peanut (early) will not overlap the only strokes (later).
    """
    

    if trial>filedata["params"]["max_trial_with_beheval"]:
        # then this is at end of dataset (shoudl be), and no data.
        return []
    strokes_orig = getTrialsStrokes(filedata, trial, 
                              only_if_part_of_stroke_postgo=only_if_part_of_stroke_postgo) # standard

    # expected_n_strokes = len(strokes)-2 # exclude touch for go and touch for done
    # if include_overlapping_go_cue:
    #     expected_n_strokes+=1
    # if include_overlapping_done_button:
    #     expected_n_strokes+=1

    pnut = getTrialsPeanutPos(filedata, trial, replaynum=replaynum)
    if len(pnut)==0:
        return []
    # assert len(pnut)>0, "not sure why empty"
    pnut_times = pnut[:,2]

    # --- take a strok if it encompasses a pnut.
    # append a single screen period (usually 16ms) to onset and offset of stroke, in case pnut is 
    # in the same frame.
    # print(filedata["params"])
    strokes = _get_strokes_overlapping_pnuttimes(filedata, strokes_orig, pnut_times)
    # print(strokes_orig)
    # print(strokes)
    # print(pnut_times)
    # assert False
    # strokes = [s for s in strokes if np.any((pnut_times >= s[0,2]-per/2) & (pnut_times <= s[-1,2]+per/2))]
    
    # if include_overlapping_go_cue:
    #     # include 
    #     x = getTrialsTimeOfBehCode(filedata, trial, "go (draw)")
    #     s = _get_strokes_overlapping_pnuttimes(np.array(x))
    #     assert len(s)==1

    #     strokes = s + strokes

    # if include_overlapping_done_button:

    #     x = getTrialsTimeOfBehCode(filedata, trial, "done")
    #     s = _get_strokes_overlapping_pnuttimes(np.array(x))
    #     print("____________")
    #     print(x)
    #     print(strokes_orig)
    #     assert len(s)==1

    #     strokes = strokes + s

    # check that strokes are in order
    if len(strokes)>0:
        check_strokes_in_temporal_order(strokes)
    
    return strokes    

def getTrialsStrokesOverlappingEventcode(filedata, trial, eventcode="go (draw)",
    append_sec_stroke_onset=None, append_sec_stroke_offset=None):
    """ get stroke overlapping go cue. 
    - event_code, {"go (draw)", "done", ...}
    PARAMS:
    - append_sec_stroke_offset and append_sec_stroke_offset, see inner.
    RETURNS:
    - [np array]
    or returns None if nothing fond.
    """

    # Changed to False (for post go) on 9/7/21 because sometimes raise before go, but is not
    # called a failure... need to geto raise time for those.
    strokes_orig = getTrialsStrokes(filedata, trial, 
                              only_if_part_of_stroke_postgo=False) # standard

    x = getTrialsTimeOfBehCode(filedata, trial, eventcode)
    s = _get_strokes_overlapping_pnuttimes(filedata, strokes_orig, 
        np.array(x), append_sec_stroke_onset=append_sec_stroke_onset, 
        append_sec_stroke_offset=append_sec_stroke_offset)

    if len(s)==0:
        # This is sometimes bad...
        if eventcode=="go (draw)":
            # for time of raise, need to have something for when raised finger on the stroke
            # that overlaps go. Soetmies can raise finger early but still no abort (like 150-200ms)
            # e.g., Pancho 210804-2-271. 
            per = filedata["params"]["screen_period"]
            x_new = np.array(x)
            ct = 0

            time_first_touch = strokes_orig[0][0,2]
            per = filedata["params"]["screen_period"] # sec
            
            if time_first_touch - np.max(x_new) > per/2:
                # Then the first touch is way later than the latest beh code time 
                # (by more than 1 screen periods).
                # There is no overlap found.
                return None

            while len(s)==0 and ct<15:
                # Monkey likely lifted right beofre end of hold so doesnt count.
                # Should call the lift a raise.
                x_new += -per # artifiaclly move the time of interest earliser.
                # print(per)
                # print(strokes_orig)
                # print(x_new)
                # assert False
                s = _get_strokes_overlapping_pnuttimes(filedata, strokes_orig, x_new)
                ct+=1
            if len(s)==0:
                # Then failed, return None. Used to throw error, but I think that considering
                # this failing is more likely.
                return None
                # # Failed
                # print(x)
                # print(x_new)
                # print(s)
                # print(strokes_orig)
                # assert False, "why?"
            return s
        else:
            # For any other evenctcode, just return None
            return None 
    else:
        # good, return it.
        return s

def getTrialsBlockParamsHotkeyUpdated_(filedata, trial, ver, default_blockparams=False):
    """ get this trials params specific to taskparams(old blockaprams)
    or runparams, taking into acocunt any hotkey
    updates that occur on or befoer this trial. REPLACES 
    getTrialsBlockParams, which always takes the default params.
    (but only works for dates after I started saving hotkeys, aorund early 
    Aug 2020.
    PARAMS:
    - ver, string eitehr "RunParams" or "TaskParams.
    """
    from .preprocess import _cleanup_blockparams_singleindex
    # assert False, "11/28/22, need to update"

    # Get the hotkey updated list
    if ver=="RunParams":
        BB = filedata["TrialRecord"]["User"]["RunParams"]["BlockParamsHotkeyUpdated"]
    elif ver=="TaskParams":
        BB = filedata["TrialRecord"]["User"]["BlockParamsHotkeyUpdated"]
    else:
        assert False

    if len(BB)==0 or default_blockparams:
        # no hotkeys used at all in entire session. return the default blockparams.
        return getTrialsBlockParamsDefault_(filedata, trial, ver)
    else:
        # RETARDED - if only one entry, then is structured different from if multiple.
        if "trial_of_update" in BB.keys():
            BB = {"1":BB}

        # Which trials were updates performed?
        # These are first trial where new params were applied
        trials_with_update = [int(B["trial_of_update"]) for B in BB.values()]
        
        # Where does this current trial fall within sequence of updtes?
        if trial < np.min(trials_with_update):
            # then return default params, no updates done yet
            return getTrialsBlockParamsDefault_(filedata, trial, ver) 
        else:
            # find the most recent trial with an update. i..e the most recent params
            indthis = np.max(np.nonzero(np.array(trials_with_update)<=trial)[0])
            trial_to_take = trials_with_update[indthis]

            # get block params
            # - pick out the specific blockparams
            list_BlockParams = [B["BlockParams"] for B in BB.values() if B["trial_of_update"]==trial_to_take]
            assert len(list_BlockParams)==1
            BlockParams = list_BlockParams[0] # dict, keys 1,2,3, ... for each block.

            # -- stupid matlab shit. if only one block, then leavse out the level of "trial"
            BlockParams = _cleanup_blockparams_singleindex(BlockParams)  
            for bk in BlockParams:
                if "TaskParams" in BlockParams[bk]:
                    BlockParams[bk]["TaskParams"] = _cleanup_blockparams_singleindex(BlockParams[bk]["TaskParams"])
            # if "GENERAL" in BlockParams.keys():
            #     tmp = {"1":BlockParams}
            #     BlockParams = tmp 
            
            # pick out this trial's params
            return getSpecificBlockParam(BlockParams, filedata, trial, ver) 

    
def getTrialsBlockParamsHotkeyUpdated(filedata, trial, default_blockparams=False):
    """ [GOOD! only use this for extracting blockaprams. becuase it correctly converts BP to 
    list format, not original dict]
    get this trials params, taking into acocunt any hotkey
    updates that occur on or befoer this trial. REPLACES 
    getTrialsBlockParams, which always takes the default params.
    (but only works for dates after I started saving hotkeys, aorund early 
    Aug 2020.
    - 2/24/22 LT - fixing so this is now default. and also merges RunParams and
    BlockParams(taskparams) - new functioanlity.
    - NOTE: maybe this might not work correctly for the first terial on 
    a block update or block switch (this only applies for taskparams).
    - 12/2/22 - Here, converting from dict to list structures.
    - 12/22/23 - Now that params_task, which used to be in RunParams, is in BlockParams[bk].TaskParams[idx].params_task...
    Here successfully references each trial's taskparams correctly.
    RETURNS:
    - BP, dict, blockparams for this trial.
    """
    
    if "BlockParamsHotkeyUpdatedv2" not in filedata["TrialRecord"]["User"].keys() or default_blockparams==True:
        # MEthod 1, before 11/28/22, where saving entire blockparams every time do hotkey update
        # or if you want defualt, just use this and ignore hotkeys.
        if "RunParams" in filedata["TrialRecord"]["User"].keys():
            list_paramkinds = ["RunParams", "TaskParams"]
        else:
            list_paramkinds = ["TaskParams"] 
        list_bp = [getTrialsBlockParamsHotkeyUpdated_(filedata, trial, ver, default_blockparams=default_blockparams) for ver in list_paramkinds]

        # # Convert from dicts to lists. 12/2/22 - to make compatbile with new vesrion below.
        from pythonlib.tools.monkeylogictools import dict2list2
        # print(type(list_bp[1]))
        # print(list_bp[1].keys())
        # dict2list2(list_bp[1])
        # # dict2list2(list_bp[1])
        # # assert False
        list_bp = [dict2list2(bp, fix_problem_with_single_items=False) for bp in list_bp]

        # print(type(list_bp[0]))
        # # print(len(list_bp[0]))
        # assert False
    else: 
        # Method 2
        list_paramkinds = ["RunParams", "TaskParams"] # 11/28/22 onweards always has both.
        list_bp = [getTrialsBlockParamsAllHotkeyUpdatedv2_(filedata, trial, ver) for ver in list_paramkinds]

        assert any(["params_task" in bp.keys() for bp in list_bp]), "12/20/23 - code changed so that params_task is in TaskParams. Checking that extraction here worked"
        # for bp in list_bp:
        #     print(bp.keys())
        # assert False, "confirm that params_task is in one of them"
        # And include TaskParams 
        if False:
            # Dont do this, since it won't be hotkey updated
            assert False, "instead of this, merge above into TaskParams"
            tp = getTrialsTaskParams(filedata, trial)
            list_bp.append(tp)

    # Merge the RunParams and TaskParams
    BP = merge_list_bp(list_bp)
    return BP


def getSpecificBlockParam(BlockParams, filedata, trial, ver):
    """ 
    Helper to get the specific params for this block. 
    BlockParams, all blocks
    Returns specific one.
    - ver, string for which kind of params.
    - BlockParams, dict with keys 1,2,3, ... for each block. Can extract these,
    see getTrialsBlockParamsDefault
    """

    if ver=="TaskParams":
        # Then is defined by block num
        BP = BlockParams[f"{getTrialsBlock(filedata, trial)}"]
    elif ver=="RunParams":
        # Defined by saved index.
        BP = BlockParams[f"{getTrialsRunParamsBlock(filedata, trial)}"]
    else:
        assert False
    return BP


def getTrialsBlockParamsDefault_(filedata, trial, ver):
    """Default params (not hotkey updated). returns dict (keys 1,2,3 for blocks)
    NOTE: returns it RAW!, without converting to list yet. Use this as base for other codes.
    """
    if ver=="TaskParams":
        BlockParams = filedata["TrialRecord"]["User"]["BlockParams"]

        # # Merge the TaskParams into BlockParams
        # tp =getTrialsTaskParams(filedata, trial)
        # print(tp.keys())
        # print(BlockParams.keys())
        # assert False
    elif ver=="RunParams":
        RunParams = filedata["TrialRecord"]["User"]["RunParams"]
        BlockParams = RunParams["BlockParams"]
    else:
        assert False

    if trial is None:
        return BlockParams
    else:
        return getSpecificBlockParam(BlockParams, filedata, trial, ver)


def _extract_updates_this_trial(updates, fd, DEBUG=False):
    """ For this specific trial, extract information about
    all hotkey updates that were performed
    RETUNRS:
    - trial, int, update were done immediatley _BEFORE_ this trial
    - blocks_updated_for_runparams, list of ints, blocks taht this was appleid to.
    12/22/23 post --> this is only used for (f1, f2) that update RunParams
    - updates_dict, dict of the updtes. with keys = (field1, field2)
    and vals = list of values (list length num blpocks).
    - taskrundict = dict[(f1, f2)] --> str, either run or task,
    - tpinds, list of ints, tp indices matched to the blocks. If multiple updates, then takes the ones
    for "task" updates. 
    
    GOOD: Asserts that blocks and tpinds are identical across all updates - thus can return a list isntead of dict of lists.

    EXAMPLE (5/3/24):
    print(trial, blocks, updates_dict, taskrundict, tpinds)
    32 
    [ 2  3  4  5  6  7  8 10 11 12 13 14 15 16 34] 
    {('params_task', 'rewardmax_success'): [array(113.75), array(113.75), array(113.75), array(113.75), array(113.75), array(125.), array(100.), array(113.75), array(113.75), array(113.75), array(113.75), array(113.75), array(125.), array(100.), array(100.)]} 
    {('params_task', 'rewardmax_success'): 'task'} 
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 2]

    5/19/24 Update - instead of returning blocks and tpinds, returns blocksdict and tpindsdict, which are dicts[(f1, f2)] --> blocks or tpinds,
    beucase there are times when the blocks are different across (f1, f2) values...
    """

    # 1) Convert from dict to list
    # updates_list = dict2list2(updates)

    def _array_to_list(arr):
        """
        arr --> list of int.
        """
        if arr.shape==():
            # then is 0-d array
            li = [int(arr)]
        else:
            li = [int(x) for x in arr]
        return li

    # 2) Which blocks and trial?
    if "blocks_tps_updated" in updates.keys() and int(fd["params"]["date"]) > 231221:
        # New version, 12/22/23 and on. This updates specific taskparams.

        if DEBUG:
            print(updates.keys())
            print(updates['blocks_updated'])
            print(updates['blocks_tps_updated'])
            print(updates['trial_of_update'])
            print(updates['block_field_val'])
        # assert "blocks_updated" not in updates.keys()
        # print(updates.keys())
        # print(updates["blocks_tps_updated"])
        # blocks = [x[0] for x in updates["blocks_tps_updated"]]
        # tpinds = [x[1] for x in updates["blocks_tps_updated"]]
        # assert False, "extract from this function all fields below"
        # tmp = struct;
        # tmp.blocks_tps_updated = blocks_tps;
        # tmp.trial_of_update = TrialRecord.CurrentTrialNumber+1;
        # tmp.block_field_val= allvals;
        # % disp('hotkeys added on this trial...')

        # Updates holds for this one trial, and same blocks across all updates, but trhere
        # can be mulitple updates.
        # i.,e, 
            # blocks_updated - list of ints
            # blocks_tps_updated - array (blcoksa nd tpinds are rows)
            # trial_of_update, int
            # block_field_val, list, length n updtes, each a list of values for that update.

        # e.g., a case with multiple updated fields:
            # blocks_updated
            # [2. 3. 4. 5. 6. 7. 8.]
            # ---
            # blocks_tps_updated
            # [[2. 3. 4. 5. 6. 7. 8.]
            #  [1. 1. 1. 1. 1. 1. 1.]]
            # ---
            # trial_of_update
            # 31.0
            # ---
            # block_field_val
            # [['params_task', 'PnutSampCollisExt', [array(28.), array(26.), array(24.), array(22.), array(20.), array(18.), array(18.)], 'task'], 
            #   ['sizes', 'TIMEOUT', [array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.)], 'run']]


        trial = int(updates["trial_of_update"]) # int

        # 3) Extract each update
        updates_dict = {}
        taskrundict = {}
        blocksdict = {}
        tpindsdict = {}
        
        block_field_val = updates["block_field_val"]
        nupdates = len(block_field_val)

        # Collect a single list of blocks and tpinds. Assume they are same across all field updates. This is ok, since it
        # does sanity check.

        # if nupdates>1:
        #     for k, v in updates.items():
        #         print("---")
        #         print(k)
        #         print(v)
        #     assert False

        blocks_general = None
        tpinds_general = None

        for i in range(nupdates):
            
            assert len(block_field_val[i])==4, "sanitych cekc"

            # print("***", block_field_val)
            f1 = block_field_val[i][0]
            f2 = block_field_val[i][1]
            vals = block_field_val[i][2]
            which_bp = block_field_val[i][3] # task, run

            # Whether to use "blocks" or "blcoks_taskpsraminds" depends on "which_bp"
            # run --> then is RunParams
            # task --> then is TaskParams
            if which_bp=="task":
                assert updates["blocks_tps_updated"].shape[0]==2
                
                if len(updates["blocks_tps_updated"].shape)>1:
                    blocks = updates["blocks_tps_updated"][0,:].astype(int)
                    tpinds = updates["blocks_tps_updated"][1,:].astype(int)
                else:
                    blocks = updates["blocks_tps_updated"][0].astype(int)
                    tpinds = updates["blocks_tps_updated"][1].astype(int)

                blocks = _array_to_list(blocks)
                tpinds = _array_to_list(tpinds)
            elif which_bp=="run":
                blocks = updates["blocks_updated"].astype(int)
                blocks = _array_to_list(blocks)

                # - assign dummy variable for tpinds
                tpinds = [None for _ in range(len(blocks))]
            else:
                print(which_bp)
                assert False, "dont know this... check dragmonkey for waht this means."
            assert len(vals)==len(blocks), "need one val per updated block"

            if False: # OLD, where this assumed (but checked) that every (f1, f2) has same blocks and tpinds...
                # Update values, and sanity check that this field has the saem blocks as previous fields (udptes)
                if blocks_general is None:
                    # The firs tupdate. keep it
                    blocks_general = blocks
                else:
                    # check is same, then keep it.
                    print(blocks_general)
                    print(blocks)
                    assert blocks_general == blocks, "you need to change this code to return blocksdict[(f1, f2)]-->list of ints, instead of blocks as lsit of ints"
                    blocks_general = blocks

                # For tpinds,  
                if tpinds_general is None:
                    # The first update. Keep it
                    tpinds_general = tpinds
                elif all([_i is not None for _i in tpinds]):
                    # The new tpinds is ints (legit)
                    assert tpinds_general == tpinds, "see note above"
                    tpinds_general = tpinds
                elif isinstance(tpinds, list) and tpinds[0] is None:
                    # Then geneal is legit (lsit of ints) so ignore this update. keep the ints
                    pass
                else:
                    print(tpinds)
                    print(tpinds_general)
                    assert False, "what case is this???"
            else: # BEtter, more general, whjere each (f1, f2) is assigned its own blocks, tpinds
                # Other sanity cheecks
                blocksdict[(f1, f2)] = blocks
                tpindsdict[(f1, f2)] = tpinds

            # Update the output
            updates_dict[(f1, f2)] = vals
            taskrundict[(f1, f2)] = which_bp

        if DEBUG:
            print(trial, blocks, updates_dict, taskrundict, tpinds)
            assert False
            
            if nupdates>1:
                for k, v in updates.items():
                    print("---")
                    print(k)
                    print(v)
                print(trial)
                print(blocks_general)
                print(updates_dict)
                print(taskrundict)
                print(tpinds_general)
                assert False

        # e.g., output:
        # INPUT (updates):
            # blocks_updated
            # [2. 3. 4. 5. 6. 7. 8.]
            # ---
            # blocks_tps_updated
            # [[2. 3. 4. 5. 6. 7. 8.]
            # [1. 1. 1. 1. 1. 1. 1.]]
            # ---
            # trial_of_update
            # 31.0
            # ---
            # block_field_val
            # [['params_task', 'PnutSampCollisExt', [array(28.), array(26.), array(24.), array(22.), array(20.), array(18.), array(18.)], 'task'], ['sizes', 'TIMEOUT', [array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.)], 'run']]

        # OUTPUT:
            # [['params_task', 'PnutSampCollisExt', [array(28.), array(26.), array(24.), array(22.), array(20.), array(18.), array(18.)], 'task'], ['sizes', 'TIMEOUT', [array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.)], 'run']]
            # 31
            # [2, 3, 4, 5, 6, 7, 8]
            # {('params_task', 'PnutSampCollisExt'): [array(28.), array(26.), array(24.), array(22.), array(20.), array(18.), array(18.)], ('sizes', 'TIMEOUT'): [array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.), array(6000.)]}
            # {('params_task', 'PnutSampCollisExt'): 'task', ('sizes', 'TIMEOUT'): 'run'}
            # [1, 1, 1, 1, 1, 1, 1]

    else:
        # Old, just updating the flat runparams, and updating all taskparams within a block
        blocks = updates["blocks_updated"]
        blocks = _array_to_list(blocks)
        tpinds = [None for _ in range(len(blocks))]
        trial = int(updates["trial_of_update"])
        
        # 3) Extract each update
        updates_dict = {}
        taskrundict = {}
        blocksdict = {}
        tpindsdict = {}

        block_field_val = updates["block_field_val"]
        nupdates = len(block_field_val)
        for i in range(nupdates):
            f1 = block_field_val[i][0]
            f2 = block_field_val[i][1]
            vals = block_field_val[i][2]

            # Get "which_bp"
            if len(block_field_val[i])>3:
                # run --> then is RunParams
                # task --> then is TaskParams
                which_bp = block_field_val[i][3] # task, run
                assert isinstance(which_bp, str)
                if which_bp=="task":
                    assert "blocks_tps_updated" in updates.keys(), "sanity check"
                elif which_bp=="run":
                    assert "blocks_updated" in updates.keys(), "sanity check"
                else:
                    print(which_bp)
                    assert False, "dont know this... check dragmonkey for waht this means."
            else:
                # Previuos to using this, always was in runparams.
                which_bp = "run"

            # Sanity checks
            assert len(vals)==len(blocks), "need one val per updated block"
            assert len(block_field_val[i])==3, "did this include >2 fields?"
            
            # Store for this (f1,f2)
            updates_dict[(f1, f2)] = vals
            taskrundict[(f1, f2)] = which_bp

            # New, storing separately for each (f1, f2)
            blocksdict[(f1, f2)] = blocks
            tpindsdict[(f1, f2)] = tpinds

        if False:
            # Change the name, to make output compatible with new code
            blocks_general = blocks
            tpinds_general = tpinds

    # trial --> int
    # blocks --> list of ints
    # updates_dict --> dict[(f1, f2)] --> array of new values
    # taskrundict --> dict[(f1, f2)] --> "task" or "run"
    # tpinds --> list of ints (of list of None, if all updates are "run" version)
    # return trial, blocks_general, updates_dict, taskrundict, tpinds_general
    
    # New...
    return trial, blocksdict, updates_dict, taskrundict, tpindsdict


# def _extract_updates_this_trial(updates):
#     """ For this specific trial, extract information about
#     all hotkey updates that were performed
#     RETUNRS:
#     - trial, int, update were done immediatley _BEFORE_ this trial
#     - blocks, list of ints, blocks taht this was appleid to
#     - updates_dict, dict of the updtes. with keys = (field1, field2)
#     and vals = list of values (list length num blpocks).
#     """

#     # 1) Convert from dict to list
#     # updates_list = dict2list2(updates)
    
#     # 2) Which blocks and trial?
#     try:
#         blocks = updates["blocks_updated"]
#     except Exception as err:
#         print("HERE")
#         print(0, updates)
#         print(1, updates_list)
#         print(2, type(updates_list))
#         raise err
#     blocks = [int(b) for b in blocks] 
#     trial = int(updates["trial_of_update"])
    
#     # 3) Extract each update
#     print(updates)
#     print(updates.keys())
#     print(updates["block_field_val"])
#     assert False
#     updates_dict = {}
#     block_field_val = updates["block_field_val"]
#     nupdates = len(block_field_val)
#     for i in range(nupdates):
#         print("***", block_field_val)
#         f1 = block_field_val[i][0]
#         f2 = block_field_val[i][1]
#         vals = block_field_val[i][2]
#         assert len(vals)==len(blocks), "need one val per updated block"
#         assert len(block_field_val[i])==3, "did this include >2 fields?"
        
#         updates_dict[(f1, f2)] = vals
    
#     return trial, blocks, updates_dict

def _condition_bp_val(val, val_prev):
    if True:
        # new version, easier.
        assert type(val_prev)==type(val)
        val_new = val
    else:
        # Old version, where defualt BlockParams did not undergo
        # dict2list2(BLockParams), and was therefore still dict,
        # whereas updates_dict _did_ go thru dict2list2(updates_dict),
        # leading to incompatible types which are fixed in hacky manner below.
    
        # - make correct data structure for val
        if isinstance(val_prev, np.ndarray):
            val_new = np.array([[val]])

            if DEBUG:
                print(trialthis, bk, val_prev, val_new)

            # becuase is by default stored as [[val]]
#             assert isinstance(val_prev, list)
            assert isinstance(val_prev, np.ndarray)
            assert len(val_prev)==1
#             assert isinstance(val_prev[0], list)
            assert isinstance(val_prev[0], np.ndarray)
            assert len(val_prev[0])==1
#             assert type(val_prev[0][0]) == type(val)
#             assert type(val_prev[0]) == type(val)
#             assert type(val_prev[0], list)

            # check matching types
            if type(val_prev[0][0]) != type(val_new[0][0]):
                print(val_prev)
                print(val_new)
                print(type(val_prev[0][0]))
                print(type(val_new[0][0]))
                assert False, "why diff types?"

            # IF RUN FOLLOOWING:
#             print(val_prev)
#             print(type(val_prev))
#             print(type(val_prev[0]))
#             print(type(val_prev[0][0]))
            # GEt the following:
#                 array([[400.]])
#                 <class 'numpy.ndarray'>
#                 <class 'numpy.ndarray'>
#                 <class 'numpy.float64'>
        elif isinstance(val_prev, str):
            # e..g, val_prev = "manual"
            val_new = val
        # elif isinstance(val_prev, dict):
        #     # e.g., beheval features. just give the dict.
        #     val_new = val
        else:
           print(trialthis, bk)
           print("---", val_prev)
           print("===", val)
           print(type(val_prev), type(val))
           assert False
    return val_new

def getMultTrialsBlockParamsAllHotkeyUpdated_(fd, ver, DEBUG=False):
    """ Get blockparams across all blocks, post 11/28/22.
    Is clever about deciding whteehr to copy or to reference. accumuatetes
    changes over trials.
    RETURNS:
    - BlockParamsByTrial[ver][trial][block]...
    SAVES:
    - fd["BlockParamsByTrial"][ver]
    """
    from pythonlib.tools.monkeylogictools import dict2list2
    import copy

    if "BlockParamsByTrial" not in fd.keys():
        fd["BlockParamsByTrial"] = {}

    if ver in fd["BlockParamsByTrial"].keys():
        # Return pre-computed
        return fd["BlockParamsByTrial"][ver]

    # Which default blockparams to load?
    BlockParamsDef = getTrialsBlockParamsDefault_(fd, None, ver) 
    if ver=="RunParams":
        bpupdates = fd["TrialRecord"]["User"]["RunParams"]["BlockParamsHotkeyUpdatedv2"]
    elif ver=="TaskParams":
        bpupdates = fd["TrialRecord"]["User"]["BlockParamsHotkeyUpdatedv2"]
    else:
        assert False


    # Convert BlockParams to lists
    BlockParamsDef = dict2list2(BlockParamsDef, fix_problem_with_single_items=True)
    # convert to dict BlockParamsDef{bk} to stil use 1-indexing, compatibility with older code
    BlockParamsDef = {i+1:bp for i, bp in enumerate(BlockParamsDef)}

    # COnvert to list.
    if True:
        bpupdates_list = dict2list2(bpupdates, fix_problem_with_single_items=True)

        # First, collect all updates
        UpdatesDict = {}
        for updates in bpupdates_list:    
            assert isinstance(updates, dict)
            # For this update (i.e., trial) get what was updated
            # try:
            if len(updates)>0:
                trialthis, blocksdict, updates_dict, taskrundict, tpindsdict = _extract_updates_this_trial(updates, fd)
                UpdatesDict[trialthis] = (blocksdict, updates_dict, taskrundict, tpindsdict)
    else:
        # Dont convert to lists, beucase wnat to keep dat structures same as in default blockparams.
        if "blocks_updated" in bpupdates.keys():
            bpupdates = {"1":bpupdates}
        UpdatesDict = {}
        for updates in bpupdates.values():    
            assert isinstance(updates, dict)
            # For this update (i.e., trial) get what was updated
            # try:
            trialthis, blocks, updates_dict = _extract_updates_this_trial(updates, fd)
            UpdatesDict[trialthis] = (blocks, updates_dict)

    # for k, v in UpdatesDict.items():
    #     print(k, v)
    # assert False
    # Generate blockparams by trial
    BlockParamsByTrial = {}
    # list_trials = getIndsTrials(fd)
    list_trials = getIndsTrialsSimple(fd)
    trials_updates = list(UpdatesDict.keys())

    # 4/27/23 and on, saving the default blockparams. Solved minor problem.
    # Update the defualt. This is necessary since online in  matlab it updates
    # BlockParams and so hotkey params reflect their final value not iniutial.
    # Here are saved the defaults just for those that were updated online.
    # THis proble only affcets (probably) from whens tarted using hotkey v2 (iu.e,)
    # just saving updates, not entire BP.
    if ver=="RunParams" and "BlockParamsDefaults" in fd["TrialRecord"]["User"]["RunParams"].keys():
        bpd = fd["TrialRecord"]["User"]["RunParams"]["BlockParamsDefaults"] # Stores defaults
        bpd = dict2list2(bpd, fix_problem_with_single_items=False)
        for f1, tmp in bpd.items():
            for f2, dat in tmp.items():
                # print(f1, f2)
                # For this (key1, key2)
                blocks_updated = dat["blocks_updated"] 
                trial_of_update = dat["trial_of_update"]
                assert list(blocks_updated-1)==list(range(len(BlockParamsDef)))
                assert int(trial_of_update) == 1

                vals = dat["block_field_val"] # array of values one for each blcok
                # print(dat)
                # assign to BP.
                for block, val_new in zip(blocks_updated, vals):
                    # print(BlockParamsDef[int(block)][f1].keys())
                    val_orig = BlockParamsDef[int(block)][f1][f2]
                    val_new = _condition_bp_val(val_new, val_orig) # chaecking types, etc.
                    # print(val_new, val_orig)

                    # update blockparams
                    BlockParamsDef[int(block)][f1][f2] = val_new

    elif ver=="TaskParams" and "TaskParamsDefaults" in fd["TrialRecord"]["User"]["RunParams"].keys():
        # 12/22/23 and after.
        # saves here updated values of BlockParams(i).TaskParams(j), such as params_task. These
        # previously were in RunParams (so that nothing in TaskParams had to be updated).
        # Ignore that this is called RunParams.TaskParamsDefaults. Just a coding convenience.
        bpd = fd["TrialRecord"]["User"]["RunParams"]["TaskParamsDefaults"]
        bpd = dict2list2(bpd, fix_problem_with_single_items=False)
        for f1, tmp in bpd.items():
            for f2, dat in tmp.items():
                blocks_tps_updated = dat["blocks_tps_updated"] # list of (bk, tmpind)
                trial_of_update = dat["trial_of_update"]
                assert int(trial_of_update) == 1

                _blocks = [x[0] for x in blocks_tps_updated]
                _tpinds = [x[1] for x in blocks_tps_updated]
                assert False, "check that you have all blcoks and tpinds"
                # assert list(_blocks-1)==list(range(len(BlockParamsDef)))

                vals = dat["block_field_val"] # array of values one for each blcok

                # assign to BP.
                for (block, tpind), val_new in zip(blocks_tps_updated, vals):
                    # print(BlockParamsDef[int(block)][f1].keys())
                    val_orig = BlockParamsDef[int(block)]["TaskParams"][int(tpind)][f1][f2]
                    val_new = _condition_bp_val(val_new, val_orig) # chaecking types, etc.
                    # print(val_new, val_orig)

                    # update blockparams
                    BlockParamsDef[int(block)]["TaskParams"][int(tpind)][f1][f2] = val_new

                    assert False, "check that above worked"

    # If this is TaskParams, merge this trial's specific taskparams
    def _merge_this_trials_taskparams_into_blockparams(BLOCKPARAMS, t):
        # SImple, for just this trial's block, update its BP,
        # so that it just expose in BP keys that are references to 
        # keys in BP["TaskParams"][tp_index]
        # - The idea is that this runs on each trial, low computation.
        # PARAMS:
        # - BLOCKPARAMS, BlockParams across blcoks. keys are boocks..
        # RETURNS:
        # - Modifies BLOCKPARAMS[<this trials block>] to have keys that are references
        # to BLOCKPARAMS[<this trials block>]["TaskParams"].items(). 

        if ver=="TaskParams":

            # First, clear any previous TaskParams updates
            # Reason: mainly sanity check. I want to make sure that the updates that 
            # follow do NOT propogate to previous trials. If that were the case, then this
            # step would delete prvious trials tp info, which would surely cause error in
            # code elsewhere. 
            # CONFIRMED that this step catches such errors.
            _clear_taskparams_merged_keys_from_blockparams(BLOCKPARAMS)

            # get the single BP 
            bk = getTrialsBlock(fd, t)
            BP = BLOCKPARAMS[bk]

            # Get indices
            task = getTrialsTask(fd, t)
            tpindex = int(task["taskparams_index"][0][0])
            
            # merge, just for this blocks BP
            TP = BP["TaskParams"][tpindex-1]
            BP = merge_list_bp([BP, TP], assert_no_overlap=True) # copy keys of TP to BP.
            # Note: do assert_no_overlap=True as a sanity check that you indeed cleared the previous trial's info.

            # Put back into BLOCKPARAMS
            BLOCKPARAMS[bk] = BP

            # print(BP.keys())
            # print("--")
            # for k, v in BLOCKPARAMS.items():
            #     print(BLOCKPARAMS[k].keys())
            # assert False
        elif ver=="RunParams":
            pass
        else:
            assert False

    def _clear_taskparams_merged_keys_from_blockparams(BLOCKPARAMS):
        """
        Clear all TaskParams-realted fields from each bk in BLOCKPARAMS[bk].
        Important sanity.
        - Modifies each BLOCKPARAMS[bk] so that does NOT have the taskparams
        keys. 
        """
        if ver=="TaskParams":
            # First, what keys are TP-specific?
            keys_taskparams = BLOCKPARAMS[1]["TaskParams"][0].keys() # ok, since all blocks have same TP keys.
            for bk, BP in BLOCKPARAMS.items():
                BLOCKPARAMS[bk] = {k:v for k, v in BP.items() if k not in keys_taskparams}
        elif ver=="RunParams":
            # Irrelevant
            pass
        else:
            assert False

    # Iterate over all trials, and if there is an update, then do deepcopy and apply if.
    # If no update, then just reference to previous trial.
    for i, trial in enumerate(list_trials):
        # Get this trials hotkey updates.
        
        if i==0:
            # First trial, just default
            BlockParams = copy.deepcopy(BlockParamsDef)

            # _merge_this_trials_taskparams_into_blockparams(BlockParamsDef, trial)
            # BlockParamsByTrial[trial] = BlockParams # initialize default

        elif trial in trials_updates:
            BlockParams = copy.deepcopy(BlockParamsByTrial[list_trials[i-1]])
            # _merge_this_trials_taskparams_into_blockparams(BlockParams, trial)

            # Update this BlockParams
            # blocks, updates_dict, taskrundict, tpinds = UpdatesDict[trial]
            blocksdict, updates_dict, taskrundict, tpindsdict = UpdatesDict[trial]

            # is this modifying taskparams?
            # OLD VERSION... didnt actualy use.
            if False:
                if isinstance(blocks[0], int):
                    # this is either RunParams, or BlockParams but before 12/21/23-ish
                    VERSION = "flat"
                elif isinstance(blocks[0], tuple) and len(blocks[0])==2:
                    # each update is to a specific (blocknum, tpindex)
                    VERSION = "taskparams"
                    assert ver=="TaskParams", "sanity cehck"
                else:
                    print(blocks)
                    assert False, "what is it?"

            # apply to blockparams
            # - for each hotkey
            assert updates_dict.keys()==taskrundict.keys()
            for fields, vals in updates_dict.items():
                task_or_run = taskrundict[fields] # "task", or "run"
                f1 = fields[0]
                f2 = fields[1]

                blocks = blocksdict[(f1, f2)]
                tpinds = tpindsdict[(f1, f2)]

                # Determine which blocks were updated for these fields.
                # if task_or_run=="task":
                #     blocks_this = blocks_updated_for_taskparams
                # elif task_or_run=="run":
                #     blocks_this = blocks
                # else:
                #     print(task_or_run)
                #     assert False

                # Update each block
                assert len(blocks)==len(vals)
                assert len(blocks)==len(tpinds)
                for idx, val, idx_tp in zip(blocks, vals, tpinds):

                    # Extract BP
                    if task_or_run=="task":
                        # Then update TaskParams
                        BP = BlockParams[idx]["TaskParams"][idx_tp-1]
                    elif task_or_run=="run":
                        # Then update RunParams
                        BP = BlockParams[idx]
                        # assert idx_tp is None # This is only true bofore 12/22/23
                    else:
                        print(task_or_run)
                        assert False

                    # OLD VERSION... didnt actualy use.
                    if False:
                        if VERSION=="flat":
                            # Then you updated a key in surface level.
                            BP = BlockParams[idx]
                        elif VERSION=="taskparams":
                            bk = idx[0]
                            tp_index_1valued = idx[1]
                            BP = BlockParams[bk]["TaskParams"][tp_index_1valued-1]
                        else:
                            assert False

                    # Update BP
                    if f1 in BP.keys():
                        val_prev = BP[f1][f2]
                        val_new = _condition_bp_val(val, val_prev)

                        # replace
                        BP[f1][f2] = val_new
                    else:
                        HACK = fd["params"]["date"] in [231220,231221]
                        # Then 12/20/23 around then, until and including 12/21/23, applied hotkey to all taskparams,
                        # and didnt save it properpyl. Therefore just skip uypdating bp. This means that params might be
                        # incorrect on those days.
                        # HACK = False
                        if HACK:
                            # Then skip
                            pass
                        else:
                            print(val)
                            print(BP.keys())
                            print(f1, f2)
                            assert False, "Not sure why"

        else:
            # Just take reference to prev ious trial. the only thing that
            # changes is taskparam,s but merging tp with bp is fine even with
            # copy
            
            # NOTE: CANNOT use direct refrence to previous trial. CONFIREMD
            # must copy.
            # BlockParams = BlockParamsByTrial[list_trials[i-1]]
            BlockParams = copy.copy(BlockParamsByTrial[list_trials[i-1]])
            # _merge_this_trials_taskparams_into_blockparams(BlockParams, trial)
            
        # Save, for this trial
        _merge_this_trials_taskparams_into_blockparams(BlockParams, trial)
        BlockParamsByTrial[trial] = BlockParams

    # Save, and return.
    fd["BlockParamsByTrial"][ver] = BlockParamsByTrial

    return BlockParamsByTrial

def getMultTrialsBlockParamsAllHotkeyUpdated_COMBINED(fd, DEBUG=False):
    """ 
    RUn this once -- get blockparams for all trials. This is like 
    getMultTrialsBlockParamsAllHotkeyUpdated_, but does one time for
    both TaskParams and RunParams, which is necessary post 12/21/24, becuase
    now both BlockParams are potentialyl updated at the same time. Need to 
    run together since this acucalutes over trials, and the saved data refers back
    to both TaskParams and RunParams... Cannot do separately. 
    
    NOTE:
    - "task" is in BlockParams(i).TaskParams(j).
    - "run" is in BlockParams.
    
    RETURNS:
    - BlockParamsByTrial[trial][block int][field] --> params
    SAVES:
    - fd["BlockParamsByTrial"]["BOTH"]
    """
    from pythonlib.tools.monkeylogictools import dict2list2
    import copy

    assert int(fd["params"]["date"]) > 231221, "Otherwise use getMultTrialsBlockParamsAllHotkeyUpdated_"

    if "BlockParamsByTrial" in fd.keys() and len(fd["BlockParamsByTrial"])>0:
        return fd["BlockParamsByTrial"]["BOTH"]

    # - 5/3/24 - New version first merges blockparams. THis is important becuase new the updates 
    # saved in RunParams.BlockParamsHotkeyUpdatedv2 apply to both TaskParams and RunParams. So runs into
    # error as RunParams looks for TaskParams but cant find it. Solution is to merge TaskParams anbd
    # RunParams first.
    _BlockParamsDefTASK = getTrialsBlockParamsDefault_(fd, None, "TaskParams");
    _BlockParamsDefRUN = getTrialsBlockParamsDefault_(fd, None, "RunParams");

    # Convert BlockParams to lists
    # convert to dict BlockParamsDef{bk} to stil use 1-indexing, compatibility with older code
    _BlockParamsDefTASK = dict2list2(_BlockParamsDefTASK, fix_problem_with_single_items=True)
    _BlockParamsDefTASK = {i+1:bp for i, bp in enumerate(_BlockParamsDefTASK)}
    _BlockParamsDefRUN = dict2list2(_BlockParamsDefRUN, fix_problem_with_single_items=True)
    _BlockParamsDefRUN = {i+1:bp for i, bp in enumerate(_BlockParamsDefRUN)}

    ####################################    
    # REinitialize with default values

    # 4/27/23 and on, saving the default blockparams. Solved minor problem.
    # Update the defualt. This is necessary since online in  matlab it updates
    # BlockParams and so hotkey params reflect their final value not iniutial.
    # Here are saved the defaults just for those that were updated online.
    # THis proble only affcets (probably) from whens tarted using hotkey v2 (iu.e,)
    # just saving updates, not entire BP.
    
    bpd = fd["TrialRecord"]["User"]["RunParams"]["BlockParamsDefaults"] # Stores defaults
    bpd = dict2list2(bpd, fix_problem_with_single_items=False)
    for f1, tmp in bpd.items():
        for f2, dat in tmp.items():
            # print(f1, f2)
            # For this (key1, key2)
            blocks_updated = dat["blocks_updated"] 
            trial_of_update = dat["trial_of_update"]
            assert list(blocks_updated-1)==list(range(len(_BlockParamsDefRUN)))
            assert int(trial_of_update) == 1

            vals = dat["block_field_val"] # array of values one for each blcok
            # print(dat)
            # assign to BP.
            for block, val_new in zip(blocks_updated, vals):
                val_orig = _BlockParamsDefRUN[int(block)][f1][f2]
                val_new = _condition_bp_val(val_new, val_orig) # chaecking types, etc.
                # print(val_new, val_orig)

                # update blockparams
                _BlockParamsDefRUN[int(block)][f1][f2] = val_new

    # 12/22/23 and after.
    # saves here updated values of BlockParams(i).TaskParams(j), such as params_task. These
    # previously were in RunParams (so that nothing in TaskParams had to be updated).
    # Ignore that this is called RunParams.TaskParamsDefaults. Just a coding convenience.
    bpd = fd["TrialRecord"]["User"]["RunParams"]["TaskParamsDefaults"]
    bpd = dict2list2(bpd, fix_problem_with_single_items=False)
    for f1, tmp in bpd.items():
        for f2, dat in tmp.items():
            blocks_tps_updated = dat["blocks_tps_updated"] # list of (bk, tmpind)
            trial_of_update = dat["trial_of_update"]
            assert int(trial_of_update) == 1

            assert blocks_tps_updated.shape[0]==2

            if len(blocks_tps_updated.shape)>1:
                _blocks = blocks_tps_updated[0,:].astype(int)
                _tpinds = blocks_tps_updated[1,:].astype(int)
            else:
                _blocks = blocks_tps_updated[0].astype(int)
                _tpinds = blocks_tps_updated[1].astype(int)

            vals = dat["block_field_val"] # array of values one for each blcok
            assert len(_blocks)==len(_tpinds)==len(vals)
        
            # assign to BP.
            for block, tpind, val_new in zip(_blocks, _tpinds, vals):
                val_orig = _BlockParamsDefTASK[block]["TaskParams"][tpind-1][f1][f2]
                val_new = _condition_bp_val(val_new, val_orig) # chaecking types, etc.
                if DEBUG:
                    # Print the degfaults.
                    print(f1, f2, block, tpind, val_orig, val_new)
            
                # update blockparams
                _BlockParamsDefTASK[block]["TaskParams"][tpind-1][f1][f2] = val_new


    ## MERGE TaskParams and RunParams
    # from pythonlib.tools.checktools import check_objects_identical
    assert _BlockParamsDefTASK.keys() == _BlockParamsDefRUN.keys(), "identical blcoks."
    list_keys_same = [k for k in _BlockParamsDefTASK[1] if k in _BlockParamsDefRUN[1]]
    assert list_keys_same == ["GENERAL"], "did I add other fields that are same? why..."
    # ITs ok to take just the GENERAL for one of them, since this is mostly online params that I dont need 
    # to store. Example two instances oif GENERAL
    # _BlockParamsDefTASK {'block_transition_rule': 'manual', 'block_transition_params': [array(1.)], 'block_transition_minsuccess': array(0.), 'block_transition_minfailure': array(0.), 'block_transition_method': 'adaptive_success_rate', 'block_update_method': 'repeat_last_block', 'fixation_sufficient': array(0, dtype=uint8), 'PURSUIT_TIME_LIMIT': array(5000.), 'NDOTS_task': array(15.), 'reward_type': 'beh_eval'}
    # _BlockParamsDefRUN {'block_transition_rule': 'finish_tasks', 'block_transition_params': [array(1.)], 'block_transition_minsuccess': array(0.), 'block_transition_minfailure': array(0.), 'block_transition_method': 'adaptive_success_rate', 'block_update_method': 'repeat_last_block', 'fixation_sufficient': array(0, dtype=uint8), 'PURSUIT_TIME_LIMIT': array(5000.), 'NDOTS_task': array(15.), 'reward_type': 'beh_eval'}

    # for key_same in list_keys_same:
    #     for BP_TASK, BP_RUN in zip(_BlockParamsDefTASK.values(), _BlockParamsDefRUN.values()):
    #         if not check_objects_identical(BP_TASK[key_same], BP_RUN[key_same]):
    #             check_objects_identical(BP_TASK[key_same], BP_RUN[key_same], PRINT=True)
    #             assert False, "merge them correctly..."

    assert_no_overlap = False # can leave False, since I confirm above that the only overlap is ""GENERAL"
    BlockParamsDef = merge_list_blockparams([_BlockParamsDefTASK, _BlockParamsDefRUN], assert_no_overlap=assert_no_overlap)
    del _BlockParamsDefTASK
    del _BlockParamsDefRUN

    # print(_BlockParamsDefTASK[1].keys())
    # print(_BlockParamsDefRUN[1].keys())
    # print(BlockParamsDef[1].keys())
    # assert False

    ################ Load the updates
    bpupdates_RUN = fd["TrialRecord"]["User"]["RunParams"]["BlockParamsHotkeyUpdatedv2"]
    bpupdates_TASK = fd["TrialRecord"]["User"]["BlockParamsHotkeyUpdatedv2"]

    # COnvert to list.
    UpdatesDict_RUN = {}
    for updates in dict2list2(bpupdates_RUN, fix_problem_with_single_items=True):    
        assert isinstance(updates, dict)
        # For this update (i.e., trial) get what was updated
        if len(updates)>0:
            trialthis, blocksdict, updates_dict, taskrundict, tpindsdict = _extract_updates_this_trial(updates, fd)
            UpdatesDict_RUN[trialthis] = (blocksdict, updates_dict, taskrundict, tpindsdict)

    UpdatesDict_TASK = {}
    for updates in dict2list2(bpupdates_TASK, fix_problem_with_single_items=True):    
        assert isinstance(updates, dict)
        # For this update (i.e., trial) get what was updated
        if len(updates)>0:
            trialthis, blocksdict, updates_dict, taskrundict, tpindsdict = _extract_updates_this_trial(updates, fd)
            UpdatesDict_TASK[trialthis] = (blocksdict, updates_dict, taskrundict, tpindsdict)


    # print(UpdatesDict_RUN)
    # print(UpdatesDict_TASK)
    # assert False
    # If this is TaskParams, merge this trial's specific taskparams
    def _merge_this_trials_taskparams_into_blockparams(BLOCKPARAMS, t, ver):
        # SImple, for just this trial's block, update its BP,
        # so that it just expose in BP keys that are references to 
        # keys in BP["TaskParams"][tp_index]
        # - The idea is that this runs on each trial, low computation.
        # PARAMS:
        # - BLOCKPARAMS, BlockParams across blcoks. keys are int boocks..
        # RETURNS:
        # - Modifies BLOCKPARAMS[<this trials block>] to have keys that are references
        # to BLOCKPARAMS[<this trials block>]["TaskParams"].items(). 
        assert ver=="TaskParams", "this has the tasparams.."

        # First, clear any previous TaskParams updates
        # Reason: mainly sanity check. I want to make sure that the updates that 
        # follow do NOT propogate to previous trials. If that were the case, then this
        # step would delete prvious trials tp info, which would surely cause error in
        # code elsewhere. 
        # CONFIRMED that this step catches such errors.
        _clear_taskparams_merged_keys_from_blockparams(BLOCKPARAMS, ver)

        # get the single BP 
        bk = getTrialsBlock(fd, t)
        BP = BLOCKPARAMS[bk]

        # Get indices
        task = getTrialsTask(fd, t)
        tpindex = int(task["taskparams_index"][0][0])
        
        # merge, just for this blocks BP
        TP = BP["TaskParams"][tpindex-1]
        BP = merge_list_bp([BP, TP], assert_no_overlap=True) # copy keys of TP to BP.
        # Note: do assert_no_overlap=True as a sanity check that you indeed cleared the previous trial's info.

        # Put back into BLOCKPARAMS
        BLOCKPARAMS[bk] = BP

    def _clear_taskparams_merged_keys_from_blockparams(BLOCKPARAMS, ver):
        """
        Clear all TaskParams-realted fields from each bk in BLOCKPARAMS[bk].
        Important sanity.
        - Modifies each BLOCKPARAMS[bk] so that does NOT have the taskparams
        keys. 
        """
        assert ver=="TaskParams", "this has the tasparams"
        # First, what keys are TP-specific?
        keys_taskparams = BLOCKPARAMS[1]["TaskParams"][0].keys() # ok, since all blocks have same TP keys.
        for bk, BP in BLOCKPARAMS.items():
            BLOCKPARAMS[bk] = {k:v for k, v in BP.items() if k not in keys_taskparams}


    def _update_blockparams_this_trial(BlockParams, UpdatesDict, trial, ver):
        """
        Update BlockParams, modifies in place
        """
        blocksdict, updates_dict, taskrundict, tpindsdict = UpdatesDict[trial]

        # apply to blockparams
        # - for each hotkey
        assert updates_dict.keys()==taskrundict.keys()
        # for k in updates_dict:
        #     if taskrundict[k]=="task":
        #         # Then this requires integers for tpinds
        #         assert all([isinstance(_i, int) for _it in tpinds])
        #     else:
        #         # Then this is BlockParams, not TaskParams
        #         assert all([_i is None for _it in tpinds])

        for fields, vals in updates_dict.items():
            task_or_run = taskrundict[fields] # "task", or "run"
            f1 = fields[0]
            f2 = fields[1]
            blocks = blocksdict[(f1, f2)]
            tpinds = tpindsdict[(f1, f2)]

            if DEBUG:
                print("=========")
                print(blocks)
                print(updates_dict)
                print(taskrundict)
                print(tpinds)
                print("---- this key:", fields)
                print(type(tpinds[0]))
                # assert False

            # Sanity check that you have tp indices.
            if False: # dont do this anymore, since tpinds is general across any run and task that exist for the
                # same trial.
                if task_or_run=="task":
                    # Then this requires integers for tpinds
                    assert all([_i is not None for _i in tpinds])
                elif task_or_run=="run":
                    # Then this is BlockParams, not TaskParams
                    assert all([_i is None for _i in tpinds])
                else:
                    print(task_or_run)
                    assert False, "probably typo? what is this"

            # Update each block
            assert len(blocks)==len(vals)
            assert len(blocks)==len(tpinds)
            for idx, val_new, idx_tp in zip(blocks, vals, tpinds):

                # Extract BP
                if task_or_run=="task":
                    # Then update TaskParams
                    BP = BlockParams[idx]["TaskParams"][idx_tp-1]
                elif task_or_run=="run":
                    # Then update RunParams
                    BP = BlockParams[idx]
                    # assert idx_tp is None # This is only true bofore 12/22/23
                else:
                    print(task_or_run)
                    assert False

                # Update BP
                if f1 in BP.keys() and f2 in BP[f1].keys():
                    val_prev = BP[f1][f2]
                    val_new = _condition_bp_val(val_new, val_prev)

                    # replace
                    BP[f1][f2] = val_new
                else:
                    HACK = fd["params"]["date"] in [231220,231221]
                    # Then 12/20/23 around then, until and including 12/21/23, applied hotkey to all taskparams,
                    # and didnt save it properpyl. Therefore just skip uypdating bp. This means that params might be
                    # incorrect on those days.
                    # HACK = False
                    if HACK:
                        # Then skip
                        pass
                    else:
                        print(val)
                        print(BP.keys())
                        print(f1, f2)
                        assert False, "Not sure why"

    ########################################
    # Iterate over all trials, and if there is an update, then do deepcopy and apply if.
    # If no update, then just reference to previous trial.
    BlockParamsByTrial = {}
    # list_trials = getIndsTrials(fd)
    list_trials = getIndsTrialsSimple(fd)
    # trials_updates = list(UpdatesDict.keys())
    trials_updates = sorted(set(list(UpdatesDict_RUN.keys()) + list(UpdatesDict_TASK.keys())))

    print("HOTKEY updating these trials:",trials_updates)
    for i, trial in enumerate(list_trials):
        # Get this trials hotkey updates.
        
        if i==0:
            # First trial, just default
            BlockParams = copy.deepcopy(BlockParamsDef)

        elif trial in trials_updates:
            BlockParams = copy.deepcopy(BlockParamsByTrial[list_trials[i-1]])

            # Update this BlockParams
            # print("... Doing UpdatesDict_RUN")
            if trial in UpdatesDict_RUN:
                _update_blockparams_this_trial(BlockParams, UpdatesDict_RUN, trial, ver="RunParams")
            # print("... Doing UpdatesDict_TASK")
            if trial in UpdatesDict_TASK:
                _update_blockparams_this_trial(BlockParams, UpdatesDict_TASK, trial, ver="TaskParams")
        else:
            # Just take reference to prev ious trial. the only thing that
            # changes is taskparam,s but merging tp with bp is fine even with
            # copy
            
            # NOTE: CANNOT use direct refrence to previous trial. CONFIREMD
            # must copy.
            # BlockParams = BlockParamsByTrial[list_trials[i-1]]
            BlockParams = copy.copy(BlockParamsByTrial[list_trials[i-1]])
            # _merge_this_trials_taskparams_into_blockparams(BlockParams, trial)
        
        # Save, for this trial
        _merge_this_trials_taskparams_into_blockparams(BlockParams, trial, ver="TaskParams")
        BlockParamsByTrial[trial] = BlockParams

    # Save, and return.
    fd["BlockParamsByTrial"] = {"BOTH":BlockParamsByTrial}

    return BlockParamsByTrial
        
def getTrialsBlockParamsAllHotkeyUpdatedv2_(fd, trial, ver, DEBUG=False):
    """ GEt this trials' blockparams, using quick method that replaces 
    getTrialsBlockParamsAllHotkeyUpdated_, which is slow. 
    NOTE: There are 2 moethods to deal with post-11/28/22, either:
    1. getTrialsBlockParamsAllHotkeyUpdatedv2_ (HERE), which first caches all 
    blockapramns across all trials _OR_
    2. getTrialsBlockParamsAllHotkeyUpdated_ which computes a new each trial. This 
    uses deeepcopy, and is slow.
    RETURNS:
    - BP, a single item in BLockParams.
    """

    if int(fd["params"]["date"]) > 231221:
        # New version, tracking updates to each taskparam
        BlockParamsByTrial = getMultTrialsBlockParamsAllHotkeyUpdated_COMBINED(fd) 
    else:
        # Old version, this was failing, becuase you need to load up BOTH runbparams and taskparams
        # to construct BP for each trial
        BlockParamsByTrial = getMultTrialsBlockParamsAllHotkeyUpdated_(fd, ver) 

    # Which default blockparams to load?
    if ver=="RunParams":
        blockthis = getTrialsRunParamsBlock(fd, trial)
    elif ver=="TaskParams":
        blockthis = getTrialsBlock(fd, trial) 
    else:
        assert False
    if BLOCKPARAMS_ALREADY_CONVERTED_TO_LIST:
        BP = BlockParamsByTrial[trial][blockthis]
    else:
        BP = BlockParamsByTrial[trial][f"{blockthis}"]

    return BP


# def getTrialsBlockParamsAllHotkeyUpdated_(fd, trial, ver, DEBUG=False):
#     """ Extract blockparams with all accmulated hotkey updates, for 11/28/22 and onwards
#     when started saving only the changes (not the entire blockparams). 
#     PARAMS;
#     - ver, str, in {"TaskParams", "RunParams"}
#     RETURNS:
#     - BP, dict or params for a single block. e.g, BP[f1][f2] = val
#     NOTE: works by copying defualt and applying updates, staritng from first trial. This might
#     not be most efficient way (iknsetad start frm last trial, and stop if find an update). 
#     OBSOLETE: Takes to long likely becuae of (i) deepcopy and (ii) iterates over all trials on each call..
#     """
#     from pythonlib.tools.monkeylogictools import dict2list2
#     assert False, "obsolete, use getTrialsBlockParamsAllHotkeyUpdatedv2_"

#     # Which default blockparams to load?
#     if ver=="RunParams":
#         BlockParams = fd["TrialRecord"]["User"]["RunParams"]["BlockParams"]
#         # BlockParams = RunParams["BlockParams"]
#         blockthis = getTrialsRunParamsBlock(fd, trial)
#         bpupdates = fd["TrialRecord"]["User"]["RunParams"]["BlockParamsHotkeyUpdatedv2"]
#     elif ver=="TaskParams":
#         BlockParams = fd["TrialRecord"]["User"]["BlockParams"]
#         bpupdates = fd["TrialRecord"]["User"]["BlockParamsHotkeyUpdatedv2"]
#         blockthis = getTrialsBlock(fd, trial)
#     else:
#         assert False
    
#     # make copy, since you dont want to modify the default.
#     import copy
#     BP = copy.deepcopy(BlockParams[f"{blockthis}"])

#     # (1) No updates, return default.
#     if len(bpupdates)==0:
#         return BP

#     # print("1", bpupdates)
#     # print("1", bpupdates.keys())
#     # print("1", bpupdates["1"]) # dict.
#     # assert(False)
#     bpupdates_list = dict2list2(bpupdates, True)
#     # print("2", bpupdates_list)
#     # print("3", bpupdates_list[0])
#     # print("3", bpupdates_list[0])
#     # print("4", bpupdates_list[0].keys())

#     # assert False


#     # if "blocks_updated" in bpupdates.keys():
#     #     # for single item cell ararys, it msitakenly removes the list. undo that here.
#     #     bpupdates = {"1": bpupdates}
#     # else:
#     #     print("JHERE@")
#     #     print(bpupdates.keys())
#     #     for k, v in bpupdates.items():
#     #         print(k, ' --- ', v)
#     #     assert False
    
#     # (2) Apply all updates up to current trial.
#     for updates in bpupdates_list:    
#         if not isinstance(updates, dict):
#             print(updates)
#             print(type(updates))
#             assert False
#         # For this update (i.e., trial) get what was updated
#         # try:
#         trialthis, blocks, updates_dict = _extract_updates_this_trial(updates)
#         # except Exception as err:
#         #     print("JHERE@")
#         #     print(bpupdates.keys())
#         #     for k, v in bpupdates.items():
#         #         print(k, ' --- ', v)
#         #     raise err

#         if trialthis<=trial and blockthis in blocks:
#             # Then this update occured before or on the desired trial, and included this block.

#             # apply to blockparams
#             # - for each hotkey
#             for fields, vals in updates_dict.items():
#                 f1 = fields[0]
#                 f2 = fields[1]

#                 # - for each block
#                 if DEBUG:
#                     print("Trial", trialthis, "| Updating fields:", f1, f2, "| Over blocks:", blocks)

#                 assert len(blocks)==len(vals)
#                 for bk, val in zip(blocks, vals):
#                     if bk==blockthis:
#                         val_prev = BP[f1][f2]

#                         if isinstance(val_prev, np.ndarray):
#                             val_new = np.array([[val]])
            
#                             if DEBUG:
#                                 print(trialthis, bk, val_prev, val_new)

#                             # becuase is by default stored as [[val]]
#                 #             assert isinstance(val_prev, list)
#                             assert isinstance(val_prev, np.ndarray)
#                             assert len(val_prev)==1
#                 #             assert isinstance(val_prev[0], list)
#                             assert isinstance(val_prev[0], np.ndarray)
#                             assert len(val_prev[0])==1
#                 #             assert type(val_prev[0][0]) == type(val)
#                 #             assert type(val_prev[0]) == type(val)
#                 #             assert type(val_prev[0], list)

#                             # check matching types
#                             if type(val_prev[0][0]) != type(val_new[0][0]):
#                                 print(val_prev)
#                                 print(val_new)
#                                 print(type(val_prev[0][0]))
#                                 print(type(val_new[0][0]))
#                                 assert False, "why diff types?"

#                             # IF RUN FOLLOOWING:
#                 #             print(val_prev)
#                 #             print(type(val_prev))
#                 #             print(type(val_prev[0]))
#                 #             print(type(val_prev[0][0]))
#                             # GEt the following:
#                 #                 array([[400.]])
#                 #                 <class 'numpy.ndarray'>
#                 #                 <class 'numpy.ndarray'>
#                 #                 <class 'numpy.float64'>
#                         elif isinstance(val_prev, str):
#                             # e..g, val_prev = "manual"
#                             val_new = val
#                         else:
#                            print(trialthis, bk, val_prev, val_new, type(val_prev), type(val_new))
#                            assert False
                 

#                         # replace
#                         BP[f1][f2] = val_new

#     # Return the updated BlockParams
#     return BP

def merge_list_blockparams(list_blockparams, assert_no_overlap=False):
    """ 
    Merge blockparams, performing a copy at the dict-level, but refernce
    to the values within.
    list_blockparams, list of dicts, each dict is a dict holding a single block, 
    where the items in that dict are the field:params.
    If overlapping keys, will merge in order, so last bp will win.

    e.g.,:
    - list_blockparams = [BlockParams1 BlockParams2], where
    BlockParams1[1] = {field1:params, field2:params}
    BlockParams2[1] = {field3:params, field4:params}

    RETURNS:
    - BlockParams3[1] = dict with keys: {field1... field4}
    """

    # Check that they have same bk inds
    bkinds = None
    for blockparams in list_blockparams:
        if bkinds is not None:
            assert bkinds == blockparams.keys()
        bkinds = blockparams.keys()

    # combine the params
    BLOCKPARAMS = {}
    for bk in bkinds:
        list_bp = [blockparams[bk] for blockparams in list_blockparams]
        BLOCKPARAMS[bk] = merge_list_bp(list_bp, assert_no_overlap=assert_no_overlap)

    # for dict_of_bp in list_blockparams:
    #     for bk_int, bp_dict in dict_of_bp.items():
    #         if bk_int not in BLOCKPARAMS:
    #             BLOCKPARAMS[bk_int] = {}
    #         for field, params in bp_dict.items():
    #             if assert_no_overlap:
    #                 assert field not in BLOCKPARAMS[bk_int]
                
    #             BLOCKPARAMS[bk_int][field] = params

    return BLOCKPARAMS
    
def merge_list_bp(list_blockparams, assert_no_overlap=False):
    """ 
    Merge BP, which are dicts holding params for a SINGLE block
    e..g,:
    list_blockparams[0] = {field:params ..}
    perform a copy at the dict-level, but refernce
    to the values within.
    If overlapping keys, will merge in order, so last bp will win.
    """

    # combine the params
    # list_keys_union = list(set(bptask.keys() + bprun.keys()))
    BP = {}
    for bp in list_blockparams:
        for field, params in bp.items():
            if assert_no_overlap:
                assert field not in BP.keys()
            BP[field] = params
    return BP


def getTrialsTaskParams(filedata, trial):
    """ Return the TaskParams for this trials.
    Only works for data collected after ~9/19/22. 
    PReviously, all the fields in TaskParams were in 
    BlockParams. 
    RETURNS:
    - TaskParams, dict. 
    - or None, if this done for data pre-TaskPArams
    """

    task = getTrialsTask(filedata, trial)
    if "taskparams_index" in task.keys():
        index = int(task["taskparams_index"][0][0])
    else:
        return
    
    HACK = False # 12/21/23 - taskparams not correctly updated (hotkeys)
    bp = getTrialsBlockParamsHotkeyUpdated(filedata, trial, default_blockparams=HACK) 

    if BLOCKPARAMS_ALREADY_CONVERTED_TO_LIST:
        return bp["TaskParams"][index-1]
    else:
        return bp["TaskParams"][f"{index}"]


# def getTrialsBlockParams(filedata, trial, default_blockparams=False):
#     """finds this trials block and then
#     otuputs that blocks params
#     DEprecated - this calls hotkeyupdated version
#     """

#     if False:
#         # Old, getting default params.
#         bptask = getTrialsBlockParamsDefault(filedata, trial, "TaskParams")
#         bprun = getTrialsBlockParamsDefault(filedata, trial, "RunParams")

#         # combine the params
#         return merge_list_bp([bptask, bprun])
#     else:
#         # New, should always use hotkey updated
#         return getTrialsBlockParamsHotkeyUpdated(filedata, trial, default_blockparams=default_blockparams)

def getTrialsBehtype(filedata, trial):
    """get beh type (e.g., 'Trace (instant)')"""    
    return getTrialsBlockParamsHotkeyUpdated(filedata, trial)["behtype"]["type"]

def getTrialsUniqueTaskname(filedata, trial):
    """this is [blocknum]-[taskname],
    since diff blocks may have the same tasknames, this must
    be done to make sure thsi name is unqiue."""
    blocknum = getTrialsBlock(filedata, trial)
    taskname = getTrialsTask(filedata, trial)["str"]
    return "-".join([str(blocknum), taskname])

def getTrialsUniqueTasknameGood(filedata, trial, append_hash_even_for_fixed=True, 
        nhash=5, VERSION=2):
    """ this shoudlw ork well. better then non-good version, 
    which cannot deal with a block that is repeated, but sample a new
    task with same name. these should call different tasks. 
    here deal witht his by using bloque.
    - also figures out if is a fixed task. if so, then just the name +
    info about how is fixed will be appended.
    - append_hash_even_for_fixed should do, since somtimes for saved sets
    # combined multipel whne making, and 2 tasks might have same name.
    - VERSION, 2 means new version using TaskGeneral, which leads to names identical
    to those in Dataset.
    """
    isnew = getTrialsTaskVersionNew(filedata, trial)
    if not isnew:
        VERSION = 1

    if VERSION==2:
        # New version
        Task = getTrialsTaskClassGeneral(filedata, trial)
        taskname = Task.get_unique_name()
    elif VERSION==1:
        # OLD VERSION
        task = getTrialsTask(filedata, trial)
        info = getTrialsTaskProbeInfo(filedata, trial)
    #     print(info)
    #     print(getTrialsTaskIsFix(filedata, trial))

        def _hash(ndigits = 4, ver=1):
            # a number that should be unique, absed on task locations.
            # not perfecet...
            # ndigits is how long.
            # returns as a string.
            # NOTE only cares about the endpoints of each stroke.

            if ver==0:
                # Problem: np means can differ since the number of pts in the task
                # can differ even if the task is the same. solution is to just use the extremes.
                assert ndigits<8, "not sure if works properly..."
                tmp = []
                tmp.extend([[d[0], d[-1], np.mean(d)] for d in task["x"]])
                tmp.extend([[d[0], d[-1], np.mean(d)] for d in task["y"]])
                

                tmp = [xx for x in tmp for xx in x]
                tmp = np.product(tmp) # will be <1, since coordinates are between 0 and 1.
                assert tmp<1, "otherwise cant guarantee output will be only integers. could have period."
                tmp = 1/tmp # this forces output >1, and small diff in tmop will be magnified.
                tmp = tmp*10**ndigits # to amke sure enough digits to left of decimal pt.

                h = str(tmp)[1:ndigits+1] # exclude first, since first digit is often less random
                assert "." not in h
                return h 
            elif ver==1:
                # only use endpoints, since the in between points might differ but
                # tasks still the same.
                # and uses sum, seems probably less prone to rounding errors.

                tmp = []

                # Collect endpoints over all strokes
                tmp.extend([[d[0], d[-1]] for d in task["x"]])
                tmp.extend([[d[0], d[-1]] for d in task["y"]])
                tmp = [xx for x in tmp for xx in x]

                # take sum
                tmp = [xx*10**ndigits for xx in tmp]
                tmp = np.sum(tmp)

                # take first ndigits as string.
                h = str(tmp)[1:ndigits+1] # exclude first, since first digit is often less random
                if "." in h:
                    tmp = h
                    ind = tmp.find(".")
                    tmp = tmp[:ind] + "0" + tmp[ind+1:]
                    h=tmp

                # tmp = np.product(tmp) # will be <1, since coordinates are between 0 and 1.
                # assert tmp<1, "otherwise cant guarantee output will be only integers. could have period."
                # tmp = 1/tmp # this forces output >1, and small diff in tmop will be magnified.
                # tmp = tmp*10**ndigits # to amke sure enough digits to left of decimal pt.

                # h = str(tmp)[1:ndigits+1] # exclude first, since first digit is often less random
                # assert "." not in h

                
                return h 


        def _name(actuallyfixed=False):
            # is randomly sampled. need to make it unique based on bloque and coordiantes.
            bq = getTrialsBloque(filedata, trial)
            tk = task["str"]
            # to make even more unique
            # since before 8/4/2020 did not guarantee that each task (wiothin a bloque) 
            # has unique name.

            x = str(task["x"][0][0]*100)[0:2] 
            y = str(task["y"][0][0]*100)[0:2]

            try:
                x+=str(task["x"][0][1]*100)[-1]
                y+=str(task["y"][0][1]*100)[-1]
            except:
                pass

            x+=str(task["x"][-1][-1]*100)[0:2]
            y+=str(task["y"][-1][-1]*100)[0:2]

            # x+=str(task["x"][-1][-1]*100)[0:2]
            # y+=str(task["y"][-1][-1]*100)[0:2]

            if actuallyfixed:
                # then is acutlaly fixed
                return f"{tk}-{x}{y}"
            else:
                return f"{tk}-{bq}-{x}{y}"
            # taskname = f"{tk}-{bq}-{x}{y}"
            return taskname


        if getTrialsTaskIsFix(filedata, trial):
            # is this resynthesized task?
            # print([_hash(n) for n in range(1,8)])
            # # print(_hash(1))
            # # print(_hash(2))
            # # print(_hash(3))
            # # print(_hash(4))
            # # print(_hash(5))
            # # print(_hash(6))

            # assert False
            if append_hash_even_for_fixed:
                h = f"-{_hash(nhash)}"
            else:
                h = ""


            if info["resynthesized"]==1:
                taskname = f"{task['str']}-resynth-{info['resynthesized_setname']}-{info['resynthesized_setnum']}-{info['resynthesized_trial']}{h}"
            elif info["saved_setnum"] is not None:
                taskname = f"{task['str']}-savedset-{info['saved_setnum']}{h}"
                # print(taskname)
            elif info["prototype"]==1:
                taskname = f"{task['str']}-protype{h}"
            elif task["stage"]=="triangle_circle":
                # then OK, hard coded to maek this a fixed task.
                taskname = _name(actuallyfixed=True)
            else:
                assert False, "i am confyused - not sure in what way this task is 'fixed'"

        else:
            taskname = _name()

        # === hard coded cases
        if task['str']=="triangle_circle_1" and filedata["params"]["expt"]=="arc2":
            # these are same, but problem is that in figures9 I saved this as a set, while
            # previously it was protytpye.
            if append_hash_even_for_fixed:
                h = f"-{_hash(nhash)}"
            else:
                h = ""
            taskname = f"{task['str']}-savedset-1{h}"
    else:
        assert False

    return taskname



def getTrialsTaskClass(filedata, trial):
    """ Get task as TaskClass (ml2-specific)
    This is usually first step before convert to general class 
    NOTE: This is identical to what is done in ProbeDat (and Dataset)
    """
    from pythonlib.drawmodel.tasks import TaskClass
    task = getTrialsTask(filedata, trial)
    # try:
    #     spad = getTrialsSketchpad(filedata, trial)
    # except Exception as err:
    #     print(err)
    #     spad = []
    spad = getTrialsSketchpad(filedata, trial)
    fix = getTrialsFix(filedata, trial)
    strokes_task = getTrialsTaskAsStrokes(filedata, trial)

    task["strokes"] = strokes_task
    task["sketchpad"] = spad
    task["fixpos"] = fix

    return TaskClass(task)

def getTrialsTaskClassGeneral(filedata, trial, convert_coords_to_abstract=False):
    """ Get task represetned as TaskGeneral class
    Same as Dataset representation
    """
    from pythonlib.drawmodel.taskgeneral import TaskClass
    
    isnew = getTrialsTaskVersionNew(filedata, trial)
    assert isnew

    task = getTrialsTaskClass(filedata, trial)
    Tnew = TaskClass()
    Tnew.initialize("ml2", task, 
        convert_coords_to_abstract=convert_coords_to_abstract)
    return Tnew

def getTrialsTaskIsFix(filedata, trial):
    """ is this task fixed or random?
    - returns True if is fixed task, False 
    if random"""
    
    if getTrialsTask(filedata, trial)["stage"]=="triangle_circle":
        # since was random, but actualyl only small set of fixed tasks.
        return True
    
    info = getTrialsTaskProbeInfo(filedata, trial)
    if info is None:
        # thne this is old version. they were never "fixed"
        return False
        # print(info)
        # for k, v in getTrialsTask(filedata, trial).items():
        #     print(k, v)

    if info["prototype"]==0 and info["saved_setnum"] is None and info["resynthesized"]==0:
        # then this is new random task sampled each block.
        return False
    else:
        return True


def getTrialsTaskVersionNew(filedata, trial):
    """ is this new (MakeDrawTasks) or old? 
    only after 8/30/20 could they be new.
    RETURNS:
    - True, new
    - False, old
    """
    task = getTrialsTask(filedata, trial)

    isnew = False
    if "TaskNew" in task.keys():
        if len(task["TaskNew"])>0:
        # print("length")
        # print(len(task["TaskNew"]))
            if len(task["TaskNew"]["Objects"])>0: # quirk of transition b/w v1, v2: if Objects==[], v1; else v2
                isnew=True
            else:
                isnew=False
    return isnew


def getTrialsTaskProbeKind(filedata, trial):
    """ high-level wrapper to get task category,
    , probe category, etc. .e,g, kinds are like:
    "task", "probe1" , ...
    """

    if not getTrialsTaskVersionNew(filedata, trial):
        # then is old, these would all count as "train"
        kind = "train"
        return kind
    
    probe = getTrialsTaskProbeInfo(filedata, trial)
    task = getTrialsTask(filedata, trial)

    # - is this a fixed or random task?
    if getTrialsTaskIsFix(filedata, trial):
        randomtask=False
    else:
        randomtask = True
        
#     if probe["prototype"]==0 and probe["saved_setnum"] is None and probe["resynthesized"]==0:
#         # then this is new random task sampled each block.
#         randomtask = True
#     else:
#         randomtask = False

    def fbver_same_as_task(probe):
        """ Returns True if fb ver for this probe is same as task.
        False for anything else
        """
        if probe["feedback_ver"] == "same_as_task":
            return True
        elif probe["feedback_ver"] in ["thresh_active", "mid_reward", "same_except_model", "subset_features", "subset_features_ignorebinary"]:
            return False
        elif "same_except" in probe["feedback_ver"]:
            # e.g., same_except_posterior, same_except_posterior_numstrokesfrac
            return False
        else:
            print(probe)
            assert False, "what is this?"

    if probe["probe"]==0:
        kind = "train"
    else:
        if probe["feedback_ver"]=="same_as_task" and randomtask==False:
            if probe["constraints_to_skip"]=={}:
                kind = "probe1_liketrain"
            elif "strokes" in probe["constraints_to_skip"].values():
                kind = "probe1_nostrokeconstraint"
        elif probe["feedback_ver"]=="same_as_task" and randomtask==True and probe["constraints_to_skip"]=={}:
            kind = "probe2_liketrain"
        elif probe["feedback_ver"]=="same_as_task" and randomtask==True and "strokes" in probe["constraints_to_skip"].values():
            kind = "probe2_nostrokeconstraint"
        elif probe["feedback_ver"] in ["thresh_active", "mid_reward"] and randomtask==False and "strokes" in probe["constraints_to_skip"].values():
            kind = "probe3"
        elif probe["feedback_ver"] in ["same_except_model"] and randomtask==False and "strokes" in probe["constraints_to_skip"].values():
            kind = "probe3_hdpos"
        elif probe["feedback_ver"] in ["thresh_active", "mid_reward"] and randomtask==True and "strokes" in probe["constraints_to_skip"].values():
            kind = "probe4"

        # -- below, more liberal to capture all else. (9/22/20)
        # -- I decided the follwoing: 
        # probe1 (liek task, fixed); probe2 (like task fb, random); probe3 (no model fb, fixed); probe4 (no model fb, random)
        elif fbver_same_as_task(probe) and randomtask==False:
            kind="probe1"
        elif fbver_same_as_task(probe) and randomtask==True:
            kind="probe2"
        elif not fbver_same_as_task(probe) and randomtask==False:
            kind="probe3"
        elif not fbver_same_as_task(probe) and randomtask==True:
            kind="probe4"


        else:
            print(probe)
            assert False, "what kind is this?"

    # ad hoc stuff
    if task["stage"]=="triangle_circle":
        kind = "probe3" # since was random, but actualyl only small set of fixed tasks.

    return kind

def getTrialsTaskProbeInfo(filedata, trial):
    """ gets infor related to probe task parameters.
    only works for 8/30 and after,, when starting using probes.
    NOTE: this applies not just for probes. Useful info for figuring
    out what kind of task this is (e.g., random/fixed, prototyupe, etc)
    - if is not 8/30 or after, then returns None"""
    from pythonlib.drawmodel.tasks import _get_task_probe_info

    task = getTrialsTask(filedata, trial)

    if not getTrialsTaskVersionNew(filedata, trial):
        # then this is old version...
        return None

    # if "TaskNew" not in task.keys():
    #     # then this is old version...
    #     return None
    
    if len(task["TaskNew"])==0:
        return None

    probe = _get_task_probe_info(task)

    # Run this here (not in pythonlib) since needs getTrialsBlockParamsHotkeyUpdated
    if "constraints_to_skip" not in task.keys():
        # then this was when I used general version in block params
        assert False, "need to use Blockparams.TaskParams.probes"
        if "constraints_to_skip" not in getTrialsBlockParamsHotkeyUpdated(filedata, trial)["probes"]:
            co = ""
        else:
            co = getTrialsBlockParamsHotkeyUpdated(filedata, trial)["probes"]["constraints_to_skip"]
    else:
        co = task["constraints_to_skip"]
    probe["constraints_to_skip"] = co

    return _get_task_probe_info(task)



# def getTrialsSelectBlockParams(filedata, trial, paramnames=None):
#     """ Print and return select values for this trials BlockParams.
#     - paramsnames is list of 2-element tuples f, where f[0] and f[1] 
#     are "fields" in the matlab code, to index this item. if None
#     then will use a defayult version I set up[ for Red/Pancho on
#     8/5/20 (copy strategy 12.8).
#     """
#     assert False, " need to change to use each trials specific param. will do that by saving blockparams at each hotkey."
    
#     if paramnames is None:
#         paramnames = [
#             ("params_task", "MinDistancePeanutToInk"),
#             ("params_task", "PnutSampCollisExt"),
#             ("params_task", "rewardmax_success"),
#             ("sizes", "INK_SIZE"),
#             ("donebutton", "frac_touch"),     
#         ]

#     print("value === paramname")
#     vals = []
#     for f in paramnames:
#         v = getTrialsBlockParamsHotkeyUpdated(filedata, trial)[f[0]][f[1]]
#         print(f"{v} === {f[0]} - {f[1]}")
#         vals.append(v)
        
#     return vals

# ==== score vs. error
def getTrialsBehEvaluation(filedata, trial, exclude_strings=True):
    """ returns None if this trial was fixation error (and therefore 
    doesnt have beh evaluation.
    - note: only saved if there was not fixation error, so
    index numbers DO NO correpsond to trial...
    - note, if before I saved beh eval, then this will also output None.
    - if doesn't find anything, then returns None
    """

    fd = filedata

    if "behEvaluation" not in fd["TrialRecord"]["User"].keys():
        return None
    
    for val in fd["TrialRecord"]["User"]["behEvaluation"].values():
        if int(val["trialnum"][0][0])==trial:
            assert len(val)>0

            # Clean up
            output = {}
            # example key, vals in val["output"]
                # dist_total {'rescale': array([[1.]]), 'value': array([[-0.24350323]])}
                # frac_touched {'rescale': array([[nan]]), 'value': array([[nan]])}
                # shortness {'rescale': array([[0.]]), 'value': array([[0.24350323]])}
                # hausdorff {'rescale': array([[0.]]), 'value': array([[-2.91434463]])}
                # frac_overlap {'rescale': array([[0.]]), 'value': array([[0.26408451]])}
                # frac_strokes {'rescale': array([[0.]]), 'value': array([[0.]])}
                # ft_decim {'rescale': array([[0.]]), 'value': array([[0.2]])}
                # ft_minobj {'rescale': array([[0.]]), 'value': array([[0.]])}
                # numstrokes_frac {'rescale': array([[1.]]), 'value': array([[-0.61538462]])}
                # posterior {'rescale': array([[0.]]), 'value': array([[-6.42449189]])}
                # pacman {'rescale': array([[1.]]), 'value': array([[-0.]])}
                # numstrokesorig {'rescale': array([[0.]]), 'value': array([[-2.]])}
                # circleness_obj {'rescale': array([[nan]]), 'value': array([[nan]])}
                # pacman_minobj {'rescale': array([[1.]]), 'value': array([[-0.]])}
                # feature_picked ft_minobj
            # 1. make sure all values are (1,1) arrays
            # 2. make sure "feature_picked" is not returned unless asked for (exclude_strings)
            for k, v in val["output"].items():
                if isinstance(v, str):
                    if exclude_strings:
                        pass    
                    else:
                        output[k] = v
                else:
                    if len(v["value"])==0:
                        v["value"] = np.array([np.nan])[:,None]
                    output[k] = v
            val["output"] = output
            return val

    # if did not find trial, then must be because this trial was a fixation error
    return None

def getTrialsBlockCategory(fd, trial, block_categories=None):
    """
    - blockcategory is a name assigned to set of blokcs. e.g., if two blovsk:
    {"practice", "probe"} but repeat N times (so N x 2 blocks total), should have 
    block_categories = {
    "practice":[1, 3, 5, ...],
    "probe":[2, 3, 6, ...]
    - if dont pass in, then will find it automatically.
    }
    - 
    """
    if block_categories is None:
        block_categories = getMultTrialsBlockCategories(fd)
    if block_categories is None:
        print("problem, cannot automatically figure out block_categories, niot sufficient data saved in matlab code")
        return None

    # get this trial's block, and from that figure out its category.
    b = getTrialsBlock(fd, trial)
    for k, v in block_categories.items():
        if b in v:
            return k
    assert False, "did not find category for some reason..."


def getTrialsFadeValues(filedata, trial):
    """ get fade values for this trial. could be
    sampled independentlyu for this trial.
    0 is full fade.
    (this acts by controlling alphas)
    """
    if "fade" in filedata["TrialRecord"]["User"]["Params"][f"{trial}"].keys():
        fadevals = filedata["TrialRecord"]["User"]["Params"][f"{trial}"]["fade"]["this_trial_fade"]
        
        fadedict = {
            "guide1_task": fadevals[0],
            "guide1_fix": fadevals[1],
            "samp1":fadevals[2],
            "samp2":fadevals[3],
            "pnut1":fadevals[4],
            "pnut2":fadevals[5]
        }
        return fadedict
    else:
        print("cannot get fade, since was not saved...")
        print("This could be that around 8/1 to 8/12/20 bug and did not save fade.")
        return None


def getTrialsGuideDots(filedata, trial):
    """ get positions of guide dots.
    in coords identical to those for task (i.e., pixel, monkey")
    - returns None if doesnt exist.
    - takes into account if uses subset of dots, as specified in 
    metadat
    - returns None if cant find.
    """

    task = getTrialsTask(filedata, trial)
    
    if "guide_dot_coords" in task.keys():
        if len(task["metadat"])==0:
            return None
        else:
            gdpos = task["guide_dot_coords"].T
            if len(gdpos)==0:
                return None
            if getTrialsBlockParamsHotkeyUpdated(filedata, trial)["guide_dots"]["method"] == \
                "guide_dot_coords_metadat_subset":
                # then using subset of dots, as specified in task metadat
                idx = list(task["metadat"]["guide_dots_to_display"])
                # idx = int(idx)
                idx = [int(i)-1 for i in idx]
                gdpos = gdpos[idx, :]

        #     taskpos = np.array([task["x"], task["y"]]).squeeze().T

        #     print(gdpos)
        #     print(taskpos)
            # convert to centered monkey 
            gdpos = convertCoords(filedata, gdpos, "norm2centeredmonkeypix")
        #     taskpos = convertCoords(filedata, taskpos, "norm2centeredmonkeypix")
        #     print(gdpos)
        #     print(taskpos)
        #     print(task["x_rescaled"])
            return gdpos
    else:
        return None

def getTwoTrialsScore(filedata, trialbeh, trialtask, ver="HD_pos", replaynum=None):
    """ score behavior for trialbeh relative ground truth stimulus for 
    trialtask. 
    - ver, what metric to use. 
    - replaynum=None, take peanut pos.
    if 1, means take first replay (if replay happend) or take final
    peanutpos
    """

    if getTrialsFixationSuccess(filedata, trialbeh) is False:
        return np.nan

    strokes_task = getTrialsTaskAsStrokes(filedata, trialtask)
    strokes_beh = getTrialsStrokesByPeanuts(filedata, trialbeh, replaynum=replaynum)
    if len(strokes_beh)==0:
        return np.nan
        
    if ver=="HD_pos":
        # then take coordinates (ignore all time dimension) and 
        # do modHausdorff dist
        from pythonlib.tools.vectools import modHausdorffDistance
        
        pos_beh = np.concatenate(strokes_beh)
        pos_task = np.concatenate(strokes_task)

        d = modHausdorffDistance(pos_beh, pos_task, dims = [0,1])
    elif ver=="DTW_min":
        # uses strokes, takes all permtuation of task strokes, returns
        # minimum dsitance over all permtuations.
        from itertools import permutations
        from pythonlib.tools.stroketools import distanceDTW
        scores =[]
        for s in permutations(strokes_task):
            # print(distanceDTW(strokes_beh, s, ver="segments", asymmetric=False))
            scores.append(distanceDTW(strokes_beh, s, ver="segments", asymmetric=False)[0])
        d = min(scores)
    elif ver=="DTW_min_minus_max":
        # returns negative score, which is how muchn lower "min" compared to "max", taken oer
        # all permutations for task.
        from itertools import permutations
        from pythonlib.tools.stroketools import distanceDTW
        scores =[]
        for s in permutations(strokes_task):
            # print(distanceDTW(strokes_beh, s, ver="segments", asymmetric=False))
            scores.append(distanceDTW(strokes_beh, s, ver="segments", asymmetric=False)[0])
        d = min(scores) - max(scores)

    else:
        assert False, "not coded"

    return d


def getTrialsErrorCode(filedata, trial):
    """ error codes as saved online, e..g,
    0 - success
    5 - OK
    6 - fail,...
    RETURNS
    - int.
    """
    return int(filedata["TrialRecord"]["TrialErrors"][trial-1][0])

def getTrialsScoreRecomputed(filedata, trial, ver="HD_pos", replaynum=None, normalize=False,
    norm_min = -100, norm_max = 0):
    """ recompute score offline for beh relative to ground truth.
    - is wrapper for getTwoTrialsScore
    """

    s = getTwoTrialsScore(filedata, trial, trial, ver=ver, replaynum=replaynum)
    if normalize:
        if ver!="compositonal_min_minus_max": # since for this, more negative is better.
            if norm_min < norm_max:
                s = -s # i.e. if larger is worse, then flip in sign.  
        else:
            s = s/2
        return (s-norm_min)/(norm_max - norm_min)
    else:
        return s


def getTrialsScoreOnline(filedata, trial, ver="hausdorff"):
    """ wrapper to quickly get score (or np.nan if for whatever
    reason score doesnt exist)"""
    if getTrialsFixationSuccess(filedata, trial) is False:
        return np.nan
    
    if getTrialsBehEvaluation(filedata, trial) is None:
        return np.nan
    
    if ver=="hausdorff":
        return getTrialsBehEvaluation(filedata, trial)["output"]["hausdorff"]["value"][0][0]
    else: 
        assert False, "not coded"

    if False:
        # == SANITY CHECK - 
        scores_online = [getTrialsScoreOnline(fd, t) for t in getIndsTrials(fd)]
        scores_offline = [getTrialsScoreRecomputed(fd, t) for t in getIndsTrials(fd)]

        for t, (a, b) in enumerate(zip(scores_online, scores_offline)):
            try:
                assert np.isnan(a)==np.isnan(b)
            except:
                print("this trial, online -- ofline scores")
                print(t)
                print(f"{a} - {b}")
        print("if mismatch, likely since online when there is no pnut placed, I give it a minimum score")
                
        # == PLOT TO COMPARE OFFLINE AND ONLINE
        plt.figure(figsize=(15,15))
        plt.plot(scores_offline, scores_online, 'ok')
        plt.xlabel("offline")
        plt.ylabel("online")


def getTrialsDateTime(filedata, trial, fmt=None):
    """ get datetime, returns a datetime object.
    - can also choose to return in a string format. do this
    by entiering format as:
    - None, then datetime
    - "str", then uses default string format (%y%m%d-%H%M%S)
    - any string, then uses that.
    """
    from datetime import datetime
    y = int(filedata["trials"][trial]["TrialDateTime"][0][0])
    mo = int(filedata["trials"][trial]["TrialDateTime"][1][0])
    d = int(filedata["trials"][trial]["TrialDateTime"][2][0])
    h = int(filedata["trials"][trial]["TrialDateTime"][3][0])
    mi = int(filedata["trials"][trial]["TrialDateTime"][4][0])
    s = int(filedata["trials"][trial]["TrialDateTime"][5][0])
    
    dt = datetime(y, mo, d, h, mi, s)
    
    if fmt is None:
        return dt
    elif fmt=="str":
        return dt.strftime("%y%m%d-%H%M%S")
    elif isinstance(fmt, str):
        return dt.strftime(fmt)
    else:
        print(fmt)
        assert False, "dont knwo this one"




#############################################################################
def getIndsTrialsSimple(filedata):
    """ Get trials, wihtout worrying about any filter. Useful to avoid circuiliatry
    in function calls, which can happen if this function calls another fucntion that
    Requreis this function
    """
    n_trials = filedata["params"]["n_trials"]
    trials_list = list(range(1,n_trials+1))
    return trials_list

def getIndsTrials(filedata, targ=None, rand_subset=None, order="chron",
    keep_only_unique_tasks=False, keep_only_if_go=False, 
    keep_only_if_clean_strokes=False):
    import random
    """extracts trials that pass criteria in the targ,
    a filter dict. 
    To get all trials in a list, leave targ empty dict
    rand_subset, gets random subset (no replacement), this many.
    - keep_only_if_go, will ony keep trials that passed go cue.

    """
    
    if targ is None:
        targ = {}

    # -- sanity check
    for key in targ.keys():
        if key not in ("task_stage", "block", "fracsuccess_min", "behtype", 
            "blockcategory"):
            assert False, "have not coded this..."
        assert(isinstance(targ[key], list)), "need for targ items to all be lists"

    # --- run
    n_trials = filedata["params"]["n_trials"]

    trials_list = list(range(1,n_trials+1))
    
    # ---- COLLECT TRIALS
    # start with all trials and only keep if passes each filter. this way 
    # gets only trials that pass every filter. 
    if "task_stage" in targ.keys():
        trials_list = [t for t in trials_list if 
        getTrialsTask(filedata, t)["stage"] in targ["task_stage"]]

    if "block" in targ.keys():
        trials_list = [t for t in trials_list if 
        getTrialsBlock(filedata, t) in targ["block"]]

    if "fracsuccess_min" in targ.keys():
        trials_list = [t for t in trials_list if 
        getTrialsOutcomesAll(filedata, t)["fracinkgotten"]>=targ["fracsuccess_min"][0]]

    if "behtype" in targ.keys():
        trials_list = [t for t in trials_list if getTrialsBehtype(filedata, t) in targ["behtype"]]

    if "blockcategory" in targ.keys():
        assert len(targ["blockcategory"])==1, "have not coded for multiple categories yet. easy to do"

        blockcategories = getMultTrialsBlockCategories(filedata)
        assert blockcategories is not None, "not part of matlab saved code?"
        blocktrials = getMultTrialsBlockTrials(filedata)

        # -- get all trials for this category
        blocks = blockcategories[targ["blockcategory"][0]]
        trials_good = []
        for b in blocks:
            trials_good.extend(blocktrials[b])
        trials_list = [t for t in trials_list if t in trials_good]

    if keep_only_if_go:
        trials_list = [t for t in trials_list if 20 in getTrialsBehCodes(filedata, t)["num"]]

    if keep_only_if_clean_strokes:
        def has_clean_strokes(t):
            if len(getTrialsStrokesClean(filedata, t))>0:
                return True
            else:
                return False
        trials_list = [t for t in trials_list if getTrialsStrokesClean(filedata, t)]

    # ---- get random subset?
    if rand_subset and len(trials_list)>rand_subset:
        print('[getIndsTrials] getting random subset of trials')
        trials_list = random.sample(trials_list, rand_subset)

    # ---- chronologtical order?
    if order=="chron":
        # then in trial number order
        trials_list.sort()
    else:
        assert False, "not coded"

    # ---- only keep unique tasks?
    if keep_only_unique_tasks:
        # if a task has multipel trials, takes the first trial
        assert False, "not coded yet!"
        # pass

    # last trial might often be bad, so check that
    try: 
        getTrialsFixationSuccess(filedata, trials_list[-1])
    except IndexError:
        trials_list = trials_list[:-1]

    # if last trials is fixation failure, then dont include it. this is becauase some data not 
    # saved if is failure.
    if getTrialsFixationSuccess(filedata, trials_list[-1]) is False:
        trials_list = trials_list[:-1]
    # print(f"got {len(trials_list)} total trials")

    return trials_list

def removeRedundantTrials(filedata, trials_list, method = "keep_latest"):
    """
    NOTE: currently works by removing trials that are followed by trial in
    one ind. i.e, if there are a sequence of trials, 565 566 567, that are all the
    same block and same taskname, then will only keep 567. the output
    trials_list will have those trials removed
    - must resort to this since a given block can repeat, and so even with unique block and 
    task name is not sure if this is really a unique task.
    
    """
    print("THIS NOT OPTIMAL - see notes. is ok if same tasks always occur in direct succession chron order.")
    tasknamedict = {}
    print(f"started with {len(trials_list)} trials")
    for t in trials_list:
        if True:
            taskname = getTrialsUniqueTasknameGood(filedata, t)
        else:    
            taskname = getTrialsUniqueTaskname(filedata, t)

        # -- keep track of tasks
        if taskname in tasknamedict.keys():
            tasknamedict[taskname].append(t)
        else:
            tasknamedict[taskname]=[t]

    # -- if multiple, keep one
    trials_to_remove = []
    for task, trials in tasknamedict.items():
        if method=="keep_latest":
#             print(trials)
#             print((np.diff(trials)==1).nonzero())
            tmp = (np.diff(trials)==1).nonzero()[0]
    #         trials[tmp]
            trials_to_remove.extend([trials[int(t.reshape(-1,))] for t in tmp]) # inds that are followed by ind+1. those are removed.
    #         for t in tmp:
    #             trials_to_remove

    #         trials_to_keep.append(max(trials))
        else:
            assert False, "not coded"

    trials_list = [t for t in trials_list if t not in trials_to_remove]
    print(f"ended with {len(trials_list)} trials")

    return trials_list


######################### THINGS THAT OPERATE ON FILEDATA (may not care about trialsd)
def getMultTrialsBehEvalFeatures(filedata):
    """ get list of features that are present
    in any of the blocks, across allb locks.
    """
    feature_list = []
    for t in getIndsTrials(filedata):
        for v in getTrialsBlockParamsHotkeyUpdated(filedata, t)["behEval"]["beh_eval"]:
            feature_list.append(v["feature"])
    return list(set(feature_list))

def getMultTrialsBlockTrials(filedata):
    """gets dict, where dict[block] = list of trials in this block
    if a block has noncontyinuous trials, will still include all. will be in order"""
    
    block_trials = {}
    BlockParams = getTrialsBlockParamsDefault_(filedata, None, "TaskParams")
    nblocks = len(BlockParams)
    for i in range(nblocks):
        block = i+1
        block_trials[int(block)] = [trial+1 for trial, bl in enumerate(filedata["TrialRecord"]["BlocksPlayed"]) if int(bl[0])==int(block)]
        
    return block_trials




def getMultTrialsTouchData(filedata, trial_list, post_go_only = False):
    """given list of trials, gets data. if list is empty, then gets all"""
    assert False, "Not yet ready!"
    dat = []
    for t in trial_list:
        dat.append({
            "trial":t,
            "dat":getTrialsTouchData(filedata, trial, post_go_only)
        })
    return dat


# get list of all task types
def getMultTrialsTaskStages(filedata, trials_list=None):
    """gets list of all names of task stages
    outputs as keys in dict, with num trials for each as entry
    [default] Leave trials_list empty to get all trials"""
    from pythonlib.tools.listtools import tabulate_list

    if trials_list is None:
        trials_list = getIndsTrials(filedata)
    
    stages = []
    for t in trials_list:
        stages.append(getTrialsTask(filedata, t)["stage"])
    
    stages_dict = tabulate_list(stages)
    
    return stages_dict


def getTrialsTaskAsStrokes(filedata, trial, fake_timesteps=None, 
    chunkmodel = None, chunkmodel_default = "eachstroke", chunkmodel_idx=0):
    """wrapper to extract and process trial task
    - fake_timesteps is how to append times as 3rd col.
    if None, then just goes from 1,2,  ....
    - chunkmodel is string telling me which model to use to find
    chunk info. will reassign output strokes so that 
    all strokes in a chunks are concatenated. this is done before 
    assigning fake timesteps.
    - chunkmodel_default, what to do if dont find the chunkmodel. this
    ignored if chunkmodel si None.
    - chunkmodel_idx if multiple parses, then which one to use? if this idx 
    too large, then returns None
    """

    task = getTrialsTask(filedata, trial) # task dict
    strokes = convertTask2Strokes(task, concat_timesteps=True, interp=25) # list of nparay

    if chunkmodel is not None:
        chunkdict = getTrialsTaskChunks(filedata, trial)
        if chunkmodel not in chunkdict.keys():
            if chunkmodel_default=="eachstroke":
                parse = [[a] for a in range(len(strokes))]
            else:
                print(chunkmodel_default)
                assert False, "not coded"
            if 0<chunkmodel_idx:
                return None
        else:
            if len(chunkdict[chunkmodel])-1<chunkmodel_idx:
                return None
            parse = chunkdict[chunkmodel][chunkmodel_idx]
        # strokes in same chunk, concatenate them.
        strokes_chunked = []
        for chunk in parse:
            strokthis = [strokes[i] for i in chunk]
            strokthis = np.concatenate(strokthis, axis=0)
            strokes_chunked.append(strokthis)
        strokes = strokes_chunked

    return strokes


def getMultTrialsBlockCategories(fd, names = None):
    """ 
    gets block categories for this dataset.
    must be on or after July 8th 2020, beucause must have
    fd["TrialRecord"]["User"]["BlockParams"]["1"]["progression"],
    where progression tells me what this block's level and category is.
    - unless you tell me, I will outputs categories named as
    1, 2, 3, ... based on order on the first level.
    - if want to assign names to each block category (number), pass in dict
    e.g.: names = {1:"practice", 2:"probe", ...}
    - blocks will be sorted 
    - returns None if data is from before July 8th 2020.
    """

    blockparams = getTrialsBlockParamsDefault_(fd, None, "RunParams")

    if "progression" not in blockparams["1"].keys():
        print("cannot infer block params, did not save progression params in matlab code...")
        return None
    
    block_categories = {}
    for block, bp in blockparams.items():
        if len(bp)>0:
            if len(bp["progression"]["orig_block_num"])==0:
                # then this was old code, I fogot to save block number for ignored blocks./
                # assign all of them to 0 by default./
                bcat = 0
            else:
                bcat = int(bp["progression"]["orig_block_num"][0][0])
        else:
            bcat = 0
        b = int(block)
        # -- add it to output
        if bcat in block_categories.keys():
            block_categories[bcat].append(b)
        else:
            block_categories[bcat] = [b]

    # -- sort blocks within each category
    for c, blocks in block_categories.items():
#         print(blocks)
#         print(sorted(blocks))
#         block_categories[c] = sorted(blocks)
        blocks.sort()
    
    # --- assign names
    if names is not None:
        # sanity checks
        assert sorted(block_categories.keys()) == sorted(names.keys()), "category numbers dont match..."
        x = [v for _, v in names.items()]
        assert len(x) == len(set(x)), "new names are not unique?"
                
        block_categories_new = {}
        for nold, nnew in names.items():
            block_categories_new[nnew] = block_categories[nold]
        block_categories = block_categories_new
        
    return block_categories
        

def getTrialsTaskChunks(filedata, trial):
    """ gets chunking structure for this task.
    only works for new makeDrawTask Tasks (around
    sep 2020 started).
    - note: output will be structured as:
    chunkdict[modelname] = [parse1, parse2, ..], where
    parse1 = [chunk1, chunk2, ..], where
    chunk1 = [snum1, snum2, ...]. where snum are the 
    original stroke numbers, starting from 0, e.g.,:
    chunkdict = {'linePlusL': [[[0, 1], [2]], [[0], [1, 2]]], 
    '3line': [[[0], [1], [2]]]}.
    - note: cannot go deeper level of heirarchy than that (and
    neither can the matlab code, if I recall..)
    - note: returns None if cant find this inforamtion.
    """
    
    task = getTrialsTask(filedata, trial)
    if "TaskNew" not in task.keys():
        print("skipping getTrialsTaskChunks, since task doesnt have chunks encopded")
        return None
    else:
        from .tasks import task2chunklist
        chunkdict = task2chunklist(task)
    #     chunks = task["TaskNew"]["Task"]["chunks"]

    #     # convert dict
    #     chunks = [v for v in chunks.values()]
    #     models = chunks[::2]
    #     stroke_assignments = chunks[1::2]
    # #     stroke_assignments = [s for s in stroke_assignments]

    #     chunkdict = {m:[] for m in models} 
    #     for m, s in zip(models, stroke_assignments):
    #         chunkdict[m].append([[int(vv[0]-1) for vv in v] for _, v in s.items()]) # append, since a model may have multiple chunks
        return chunkdict


def getTrialsOnsOffsAllowFail(filedata, trial, return_strokes=False):
    """ Same, but if fail necase not peanut strokes, just returns an empty array
    """
    
    strokes = getTrialsStrokesByPeanuts(filedata, trial)
    ons = []
    offs = []
    for s in strokes:
        ons.append(s[0,2])
        offs.append(s[-1,2])
    if return_strokes:
        return ons, offs, strokes
    else:
        return ons, offs

    # if len(strokes)==0:
    #     if return_strokes:
    #         return np.empty(0), np.empty(0), np.empty(0)
    #     else:
    #         return np.empty(0), np.empty(0)
    # else:
    #     ons = [s[0,2] for s in strokes]
    #     offs = [s[-1,2] for s in strokes]
        
    #     if return_strokes:
    #         return ons, offs, strokes
    #     else:
    #         return ons, offs

def getTrialsOnsOffs(filedata, trial, return_strokes=False):
    """ get time (rel trail onset) of all
    stroke onset and offsets. (in seconds)
    - by default gets strokes by peanuts
    RETURNS:
    - (ons, offs)
    - if return_strokes, then (ons, offs, strokes)
    """
    
    strokes = getTrialsStrokesByPeanuts(filedata, trial)
    
    assert len(strokes)>0, "no peanuts.."
    ons = [s[0,2] for s in strokes]
    offs = [s[-1,2] for s in strokes]
    
    if return_strokes:
        return ons, offs, strokes
    else:
        return ons, offs


def getTrialsTimesOfMotorEvents(fd, t, return_strokes=False):
    """ 
    [GOOD, wrapper] Get times of major events in a trial.
    go_cue and done_triggered are based on event codes.
    Rest are based on actual motor timing.
    - return_strokes, then returns strokes (good, peanuts)
    RETURNS:
    - out, dict with values
    - out, strokes, if return_strokes
    NOTE: if something not defined, then is nan
    """
    
    ons, offs, strokes_draw = getTrialsOnsOffs(fd, t, return_strokes=True)
    strokes_go = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)")
    if strokes_go is not None:
        if len(strokes_go)>1:
            # problem...
            print("---")
            print(strokes_go)
            strokes_orig = getTrialsStrokes(fd, t, 
                          only_if_part_of_stroke_postgo=False) # standard
            x = getTrialsTimeOfBehCode(fd, t, "go (draw)")
            s = _get_strokes_overlapping_pnuttimes(fd, strokes_orig, np.array(x))
            for i, strok in enumerate(strokes_orig):
                for tmp in strok:
                    print("STROKE ", i, tmp)
            print(x)
            print(s)
            print("trial:", t)
            assert False, "hmmm?"
        time_of_raise = strokes_go[0][-1,2]
    else:
        if len(ons)>0:
            # must have been bug, failed to find sterok overlaping go.
            # Not possibelk to have strokes during draw, but not overlaping go cue.

            ######## TRY EDGE CASES

            #### Weird edge case #1. two strokes are stradling the go cue, becuase of 
            # touchscrern detection issues, but is really one long hold. should find time
            # that the second stroke raised.
            # - Try again with more lenient time win
            strokes_go = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
                append_sec_stroke_onset=-0.1) # find stroke that follows the go 
            tmp = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
                append_sec_stroke_offset=0.1) # find stroke preceding...
            # If only one strokes AFTER go is returned, then this is ok. 
            # Make sure no strokes immediately preceding go, no particular reason.
            # this just is the pattern I'd expect.
            EDGE_CASE_1 = strokes_go is not None and len(strokes_go)==1 and tmp is None

            #### Weird edge case #2. Raise in anticipation, so is raised before go cue, but 
            # somehow sometimes monkeylogic doesnt fail you... This is becuase the "guide" image
            # can disappear before the "go cue". meaning the actual go cue (from monkeys persepctive)
            # is earlier. E.g. Foudn a case that was 
            # raise 536MS before go cue... Diego , 12/18/23 - trail 313.
            # Time of go:  [4.676028799996857]
            # Time of offset of last stroke before go: STROKE  0 [  -8.         -235.00000082    4.14      ]

            # find stroke that follows the go 
            # strokes_post = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
            #     append_sec_stroke_onset=-0.35) # 10/1/23 - used to be this
            # strokes_post = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
            #     append_sec_stroke_onset=-0.11) # changed to this, since found a case where
            # # stroke onset occured soon after "go cue"
            strokes_post = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
                append_sec_stroke_onset=-0.105) # changed to this, since found a case where
            # stroke onset occured soon after "go cue", for Luca.
            
            # find stroke preceding...
            # strokes_pre = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
            #     append_sec_stroke_offset=0.34) 
            strokes_pre = getTrialsStrokesOverlappingEventcode(fd, t, "go (draw)",
                append_sec_stroke_offset=0.537)
            
            # Diagnostic: pre stroke exists, no post stroke...
            EDGE_CASE_2 = strokes_pre is not None and len(strokes_pre)==1 and strokes_post is None

            #### CHECK IF ONLY ONE OF THE EDGE CASES IS CORRECT
            if EDGE_CASE_1:
                assert EDGE_CASE_2==False
                time_of_raise = strokes_go[0][-1,2]
            elif EDGE_CASE_2:
                assert EDGE_CASE_1==False
                time_of_raise = strokes_pre[0][-1,2] # time of raise, before go.
                x = getTrialsTimeOfBehCode(fd, t, "go (draw)")
                assert time_of_raise<x[0], "just sanity check. this must be right."
            else:
                # Then not sure...
                # problem...
                print(strokes_go)
                print(tmp)
                print("---")
                strokes_orig = getTrialsStrokes(fd, t, 
                              only_if_part_of_stroke_postgo=False) # standard
                x = getTrialsTimeOfBehCode(fd, t, "go (draw)")
                s = _get_strokes_overlapping_pnuttimes(fd, strokes_orig, np.array(x))
                for i, strok in enumerate(strokes_orig):
                    for tmp in strok:
                        print("STROKE ", i, tmp)
                print("Time of go: ", x)
                print(s)
                print("trial:", t)
                assert False, "cant have ons, but no rraise..."


            # if strokes_go is not None and len(strokes_go)==1 and tmp is None:
            #     # If only one strokes AFTER go is returned, then this is ok. 
            #     # Make sure no strokes immediately preceding go, no particular reason.
            #     # this just is the pattern I'd expect.
            #     # Good
            #     time_of_raise = strokes_go[0][-1,2]
            # else:
            #     # problem...
            #     print(strokes_go)
            #     print(tmp)
            #     print("---")
            #     strokes_orig = getTrialsStrokes(fd, t, 
            #                   only_if_part_of_stroke_postgo=False) # standard
            #     x = getTrialsTimeOfBehCode(fd, t, "go (draw)")
            #     s = _get_strokes_overlapping_pnuttimes(fd, strokes_orig, np.array(x))
            #     for i, strok in enumerate(strokes_orig):
            #         for tmp in strok:
            #             print("STROKE ", i, tmp)
            #     print(x)
            #     print(s)
            #     print("trial:", t)
            #     assert False, "cant have ons, but no rraise..."
        else:
            # No strokes found overalpping go but that's fine, since no draw strokes found either
            time_of_raise = np.nan

    ### TIME OF DONE BUTTON TOUCH.
    if False: # NOTE: This is the time that DragAroundv2.m ends. Not related to done button.
        # time that the stroke (which will get "done" button) first touches screen is
        # taken as done time.
        strokes_done = getTrialsStrokesOverlappingEventcode(fd, t, "done")
        if strokes_done is not None:
            time_of_done_touch = strokes_done[-1][0, 2]
        else:
            time_of_done_touch = np.nan
    else:
        # 1/13/24 - New method, must get a result, or fails...
        strokes_done = getTrialsStrokesOverlappingEventcode(fd, t, "DoneButtonTouched")
        if getTrialsOutcomesWrapper(fd, t)["trial_end_method"] in ["pressed_done_button", "postscene_hotkey_abort"]:
            if strokes_done is None:
                # Then a time MUST exist.
                print("trial:", t)
                print(getTrialsOutcomesWrapper(fd, t)["trial_end_method"])
                done_button_trigger_time = getTrialsTimeOfBehCode(fd, t, "DoneButtonTouched")
                print("New method, done time: ", done_button_trigger_time)
                done_button_trigger_time2 = getTrialsTimeOfBehCode(fd, t, "done")
                print("Old method, done time: ", done_button_trigger_time2)
                print(getTrialsStrokesOverlappingEventcode(fd, t, "done")) # Older method..
                assert False, "why cant detect when touched screen, surrounding eventcode for done button?"
            else:
                time_of_done_touch = strokes_done[-1][0, 2]
        else:
            # Is ok, this is not a done button trial...
            time_of_done_touch = np.nan
            if strokes_done is not None:
                print("trial:", t)
                print(getTrialsOutcomesWrapper(fd, t)["trial_end_method"])
                print(strokes_done[0][:,2])
                done_button_trigger_time = getTrialsTimeOfBehCode(fd, t, "DoneButtonTouched")
                print("New method, done time: ", done_button_trigger_time)
                done_button_trigger_time2 = getTrialsTimeOfBehCode(fd, t, "done")
                print("Old method, done time: ", done_button_trigger_time2)
                print(getTrialsStrokesOverlappingEventcode(fd, t, "done")) # Older method..
                assert False, "Sanity check, should not have..."

    go_time = getTrialsTimeOfBehCode(fd, t, "go (draw)")
    assert len(go_time)==1
    done_time = getTrialsTimeOfBehCode(fd, t, "done")
    assert len(done_time)==1
 
    out = {
        "go_cue":go_time[0],
        "raise":time_of_raise,
        "ons":ons,
        "offs":offs,
        "done_touch":time_of_done_touch,
        "done_triggered":done_time[0]
    }
    if return_strokes:
        return out, strokes_draw
    else:
        return out
    


def getTrialsReactionTime(fd, t, ver = "go2touch"):
    """ get reaction time, different versions.
    - ver, 
    --- "go2touch", go cue to first touch. go is based on event code. 
    touch is based on first stroke by peanuts.
    """
    
    if ver=="go2touch":
        # difference of # get time of first stroke minus # get time of go
        rt = getTrialsOnsOffs(fd, t)[0][0] - getTrialsTimeOfBehCode(fd, t, "go (draw)")[0]
    else:
        assert False, "not coded"

    if rt is None:
        assert False, "why"
    return rt

def getTrialsFixDur(fd,t):
    """ get duration of fix that is enforced (i.e the params)
    """

    if int(fd["params"]["date"])>210423:
        assert False, "shoudnt use this method. instead should have saved the appropriate event code for fixation (guide)"

    # use hack, based on duration of first touch tio go...
    assert False, "not coded, get the updated version using event code"



def getTrialsMotorTimingStats(fd, t):
    """ get trials movemnet timing stats, in a dixct,
    (rt: rt [go to first touch],
    nstrokes: nstrokes,
    sdur: total duration of strokes,
    isi: totla duration of inter-stroke intervals (only between storkes)
    endtime: time from end of last stroke to touch of done button,
    disttravel_strokes: cumulative distance 
    disttravel_gaps ((ignoring go-->on and off-->done)
    )

    """
    from pythonlib.drawmodel.features import computeDistTraveled

    # ==== TIMING STATS
    timestats, strokes = getTrialsTimesOfMotorEvents(fd, t, return_strokes=True)
    ons = timestats["ons"]
    offs = timestats["offs"]
    time_go2raise = timestats["raise"] - timestats["go_cue"]
    # donetime = timestats["done_touch"]
    donetime = timestats["done_triggered"]
    time_raise2firsttouch = ons[0] - timestats["raise"]

    sdur = 0
    for a, b in zip(ons, offs):
        sdur+= b-a

    isi = 0
    if len(ons)>1:
        for a, b in zip(ons[1:], offs[:-1]):
            isi+=a-b

    time_touchdone = donetime - offs[-1]

    # ==== DISTANCE STATS
    nstrokes = len(strokes)

    # movent distnaces
    origin = getTrialsFix(fd, t)["fixpos_pixels"]
    donepos = getTrialsDoneButtonPos(fd, t)

    dist_strokes_plus_gaps = computeDistTraveled(strokes, origin, doneloc=donepos, include_lift_periods=True, 
        include_origin_to_first_stroke=False, include_transition_to_done=False)
    dist_strokes = computeDistTraveled(strokes, origin, include_lift_periods=False)

    if True:
        dist_onset = computeDistTraveled(strokes, origin, doneloc=donepos, include_lift_periods=True, 
            include_origin_to_first_stroke=True, include_transition_to_done=False) - dist_strokes_plus_gaps

        if getTrialsDoneButtonMethod(fd, t)=="skip":
            # Then no done button
            dist_offset = np.nan
        else:
            dist_offset = computeDistTraveled(strokes, origin, doneloc=donepos, include_lift_periods=True, 
                include_origin_to_first_stroke=False, include_transition_to_done=True) - dist_strokes_plus_gaps

        # print("dist_onset", "dist_offset")
        # print(dist_onset, dist_offset)

    return {
        "nstrokes":nstrokes,
        "time_go2raise":time_go2raise,
        "time_raise2firsttouch":time_raise2firsttouch,
        "dist_raise2firsttouch":dist_onset,
        "sdur":sdur,
        "isi":isi,
        "time_touchdone": time_touchdone,
        "dist_touchdone":dist_offset,
        "dist_strokes":dist_strokes,
        "dist_gaps":dist_strokes_plus_gaps-dist_strokes}


def getTrialsDoneButtonMethod(fd, t):
    """ returns a string, indicating method.
    e.g, "skip" means did not use.
    RERTURNS:
    - (2,) arrary
    """
    return getTrialsBlockParamsHotkeyUpdated(fd, t)["params_task"]["donebutton_criterion"]

def getTrialsDoneButtonPos(fd, t):
    """ get position of done button, in 
    pixel monkey coords, If did not use done button, then returns None"""

    if getTrialsDoneButtonMethod(fd, t)=="skip":
        # then did not use done button
        return None

    if int(fd["params"]["date"])>210427:
        # included in params after this dayte
        prms =  getTrialsParams(fd, t)
        # print(prms)
        pos_norm = prms["done_pos"]
    else:
        # assert False, "shoudnt use this method. instead should have saved into Params online"
            
        # use this method, generally correct, since I 
        # use the pos given in BP, but not generlaly corect. 
        BP = getTrialsBlockParamsHotkeyUpdated(fd, t)
        assert BP["donebutton"]["posmethod"]=="use_pos", "then cant use the pos as truth"
        pos_norm = BP["donebutton"]["pos"]
        
    # convert from norm to monkey
    pos_pix = convertCoords(fd, pos_norm, ver="norm2centeredmonkeypix")[0] # 0, since is [1,2]
    
    return pos_pix
       
def getTrialsHoldTimeActual(fd,t):
    """ if not yet saved in fd, then returns nan
    """
    prms = getTrialsParams(fd, t)
    if "fix_new" in prms.keys():
        return prms["fix_new"]["hold_time_actual"][0]
    else:
        return np.nan

def getTrialsStimDelayTime(fd, t):
    """ Delya between first toiuch fix, and when stim is shown.
    Returns nan if not yet saved for this expert
    """
    tmp = getTrialsTaskMod(fd, t)
    if tmp is None:
        return np.nan
    elif "delay_before_show_guide" not in tmp.keys():
        return np.nan
    else:
        # print(tmp.keys())
        return tmp["delay_before_show_guide"]



    
def getTrialsTaskMod(fd, t):
    """ Returns ParamsMod item.
    Only works for 4/25/21 and after.
    
    RETURNS:
    - dict, with keys param names, and values the params
    - If before, or is empty, then returns None
    """
    task = getTrialsTask(fd,t)
    if "ParamsMod" not in task.keys():
        return None
    if len(task["ParamsMod"].keys())==0:
        return None
    
    mod = task["ParamsMod"]
    # structured as {key, value, key, value..}, so here is:
    # {'1': 'hold_time', '2': array([[1000.]])}
    
    n = len(mod.keys())
    assert n%2==0
    keyinds = list(range(1, n+1, 2)) # [1, 3, 5, ...]
    out = {}
    for i in keyinds:
        k = mod[f"{i}"]
        v = mod[f"{i+1}"].squeeze()
        out[k]=v
    return out


def getTrialsAbortParams(fd, t):
    """ returns dict, 
    dict[on], bool, holding whether is abort
    dict[modes] list of strings, what abort modes are on"""
    
    bb = getTrialsAdapterParams(fd, t)

    out = {}
    out["on"] = bool(bb["enableAbort"])
    out["modes"] = [k for k, v in bb["abortmodes"].items() if bool(v)]
    
    return out

def getTrialsSupervisionParams(fd, t):
    """ returns dict with params related to supervision.
    Wrapper for all kinds of supervision.
    RETURNS:
    - dict, where each item is either bool (on or off) or string (ver)
    """
    
    def _interpret_sequence_module(bb):
        """ summarize sequence module into:
        - on (bool)
        - ver (string, version)
        NOTE:
        fails if cant figure out.
        """
        S = bb["sequence"]
        ALPHA_MAX = 0.8 # consider any alpha higher than this to be visible. (no supervision)
        on = bool(S["on"])

        # If not on
        if on==False:
            ver = ""
            alpha=1
            return on, ver, alpha

        # Figure out what ver.
        if "turn_off_done_chunks" not in S.keys():
            turn_off_done_chunks = False
        else:
            turn_off_done_chunks = bool(S["turn_off_done_chunks"].squeeze()) # NOTE: this occurs even if manipulations is empty
        manips = [v for v in S["manipulations"].values()]        

        if manips==["mask"]:
            if turn_off_done_chunks:
                # then is old version. this means go from alpha 0 to 1.
                ver = "v3_remove_and_show"
                alpha = 0
                return on, ver, alpha
            else:
                # show one by one, but dont remove previously done chunks.
                ver = "v3_noremove_and_show"
                alpha = 0
                return on, ver, alpha

        if len(S["ordering_params"])==1:
            print(S)
            assert False

        alpha = S["ordering_params"]["2"].squeeze()

        if manips ==["alpha", "disappear"] and turn_off_done_chunks:
            if alpha==0:
                # disapper current when get.
                ver = "v4_remove_when_touch"
            elif alpha < ALPHA_MAX:
                # disapper current when get.
                ver = "v4_fade_when_touch"
            else:
                assert False, "not sure"

        elif manips ==["alpha", "mask"]:
            if alpha==0:
                if turn_off_done_chunks:
                    # one by one, remove done chunk and show next in order
                    ver = "v3_remove_and_show"
                else:
                    ver = "v3_noremove_and_show"
            elif alpha < ALPHA_MAX:
                if turn_off_done_chunks:
                    # one by one, show next in order
                    ver = "v3_remove_and_fadein"  
                else:
                    ver = "v3_noremove_and_fadein"  
            elif len(alpha)==0:
                ver = "unknown" # Probably after starting using ObjectClass, stopped using sequence module.
            else:
                print(S)
                print(fd["params"])
                print(t)
                assert False, "not sure..."
        elif len(manips)==0:
            if turn_off_done_chunks:
                # Then no sequential, but still turn off chunks
                print(S)
                print(fd["params"])
                print(t)
                assert False, "not sure"
                ver = "v4_remove_when_touch"
            else:
                on = False
                ver = ""
        elif manips ==["alpha", "active_chunk"]:
                ver = 'objectclass_active_chunk'
        else:
            [print(k,v) for k,v in S.items()]
            print(S)
            print(fd["params"])
            print(t)
            assert False
            
        return on, ver, alpha


    bb = getTrialsAdapterParams(fd, t)
    out = {}
    
    # keep ink on
    out["keep_ink_on"] = bool(bb["keep_ink_on"][0])
    
    # sound
    out["play_hit_sound"] = not bool(bb["suppress_hit_sound"][0])
    
    # sequential presentation
    on, ver, alpha = _interpret_sequence_module(bb)
    out["sequence_on"] = on
    out["sequence_ver"] = ver
    out["sequence_alpha"] = alpha

    # colored strokes
    out["draw_colored_strokes"]=False
    if len(bb["InkColorsByPt"])>0:
        # if colors are different across pts
        # print(bb["InkColorsByPt"])
        # print(bb["InkColorsByPt"].shape)
        # print(np.diff(bb["InkColorsByPt"], axis=1))
        cols_unique = np.unique(bb["InkColorsByPt"], axis=1)
        # print(cols_unique.shape)
        if cols_unique.shape[1]>1:
            out["draw_colored_strokes"]=True

    ################## GUIDE
    guide = getTrialsAdapterParams(fd, t, "guide")

    # colored strokes
    out["guide_colored_strokes"]=False
    if bool(guide["strokes"]["on"]):
        ncols = np.unique(np.r_[[v for v in guide["strokes"]["colors"].values()]].squeeze(2), axis=0).shape[0]
        if ncols>1:
            # Then strokes are diff colored
            out["guide_colored_strokes"]=True

    # dynamic guide.
    if "on" not in guide["dynamic"].keys():
        # Then this was not invented yet
        out["guide_dynamic_strokes"]=False
        out["guide_dynamic_strokes_ver"] = ""
    else:
        if bool(guide["dynamic"]["on"]):
            out["guide_dynamic_strokes"]=True
            out["guide_dynamic_strokes_ver"] = guide["dynamic"]["version"]
        else:
            out["guide_dynamic_strokes"]=False
            out["guide_dynamic_strokes_ver"] = ""
    
    return out
    

# ======================= CONVERTING THINGS

def convertCoords(filedata, xy, ver):
    """ wrapper for converting coordiantes, relevant for moenky task"""
    
    def _monkeynorm2centeredmonkeypix(xy):
        # xy is single row vector.

        res = filedata["params"]["resolution"]
        
        x0 = -res[1]/2
        x1 = res[1]/2
        y0 = -res[0]/2
        y1 = res[0]/2
        
        out = [((x1-x0)*xy[0])+x0, ((y1-y0)*xy[1])+y0]
        return out


    if ver=="deg2pix":
        return convertDeg2Pix(filedata, xy)
    elif ver=="centeredpix2monkeynorm":
        # using pix, but just shifting so center of screen is set to 0,0
        return convertPix2Relunits(filedata, xy)
    elif ver=="monkeynorm2centeredmonkeypix":
        # is the inverse of centeredpix2monkeynorm
        # - monkeynorm: (0,0) bottom left; (1,1) top right, monkey perspective
        # - centeredmonkeypix: center is (0,0), bottom left is (-resx/2, -resy/2). 
        # - xy can be single row shape (2,) single row shape (1,2),
        # or multipel rows shape (N, 2)
        if len(xy.shape)==1:
            # convert to row vector
            assert xy.shape[0]==2
            xy = xy.reshape(1,2)
        if xy.shape[1]!=2:
            print(xy)
            print(xy.shape)
            assert False
        out = np.zeros_like(xy)
        for i in range(out.shape[0]):
            out[i,:] = _monkeynorm2centeredmonkeypix(xy[i,:]) 
        return out

    elif ver=="centerize":
        # simple, just centerize
        xy[:,0] = xy[:,0] - np.mean(xy[:,0])
        xy[:,1] = xy[:,1] - np.mean(xy[:,1])
        return xy
    elif ver=="pix2monkey":
        return convertPix2Monkey(filedata, xy, centerize=True)
    elif ver=="norm2centeredmonkeypix":
        # THIS is identical to the transformation for task, from x-->x_rescaled.
        if xy.shape[1]==1 and xy.shape[0]>1:
            xy = xy.T
        out = np.zeros_like(xy)
        out[:,0] = -(xy[:,1] - 0.5) * filedata["params"]["resolution"][1]
        out[:,1] = -(xy[:,0] - 0.5) * filedata["params"]["resolution"][0]
        return out
    else:
        assert False, "not coded yet"

def convertDeg2Pix(filedata, xydeg):
    """xydeg should be [x,y], and just 2 values"""
    
    xout = xydeg[1]*filedata["params"]["pix_per_deg"][0] 
    yout = xydeg[0]*filedata["params"]["pix_per_deg"][1]
    
    return np.array((xout[0], yout[0]))

def convertDeg2PixArray(filedata, xydeg):
    """ convert from deg to pixels
    PARAMS:
    - xydeg, (N,2) array
    RETURNS:
    - xypix, (N,2) array
    """
    xout = xydeg[:,1]*filedata["params"]["pix_per_deg"][0] 
    yout = xydeg[:,0]*filedata["params"]["pix_per_deg"][1]
    return np.stack((xout, yout), axis=1)


def convertPix2Relunits(filedata, xypix):
    """xypix should be [x,y], and just 2 values
    - pix, actually means centerized pix (monkey perspective).
    I use this for plots
    - rel units, just rescales from 0 to 1, basicalyl
    'monkey' units as called in matlab (I think?)"""
    res = filedata["params"]["resolution"]
    
    x0 = -res[1]/2
    x1 = res[1]/2
    y0 = -res[0]/2
    y1 = res[0]/2
    
    if True:
        out = [(xypix[0]-x0)/(x1-x0), (xypix[1]-y0)/(y1-y0)]
    else:
        print(res)
        xypix[:,0] = (xypix[:,0]-x0)/(x1-x0)
        xypix[:,1] = (xypix[:,1]-y0)/(y1-y0)
        out = xypix
    return out



def convertPix2Monkey(filedata, pos, centerize=True):
    """ 
    - pos, N x 2 array, in matlab pixel units (i.e., monkey top
    right is 0,0, and x increases as go downwards).
    - centerize, True, then screen center is 0,0. otherwise bottom-left
    is 0,0. Note that True is default for my plotting and analysis code.
    RETURNS:
    - posnew, N x 2 array, in pixel units, but monkey pers[pective, so
    bottom x increasea as gor giht, and y as go up.
    """
    
    res = filedata["params"]["resolution"] # (1024, 768)
    if centerize:
        y = -(pos[:,0] - res[0]/2)
        x = -(pos[:,1] - res[1]/2)
    else:
        y = -(pos[:,0] - res[0])
        x = -(pos[:,1] - res[1])
    return np.stack((x,y)).transpose()

def compareTasks(task1, task2, input_strokes_directly=False):
    """ returns True is identical, False otherwise.
    bases this on actual positions of coords. uses np.isclose
    - input_strokes_directly, then tasks should be strokes objects.
    """

    if input_strokes_directly:
        return all([np.all(np.isclose(s1, s2)) for s1, s2 in zip(task1, task2)])
    else:
        return all([np.all(np.isclose(task1["x_rescaled"], task2["x_rescaled"])),
           np.all(np.isclose(task1["y_rescaled"], task2["y_rescaled"]))])
