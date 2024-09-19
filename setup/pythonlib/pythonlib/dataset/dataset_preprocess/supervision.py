""" To extract supervision params implemented online on each trial, and related tings.

"""

import numpy as np

def extract_supervision_params(D, ind):
    """ Extract the supervision params for this trial index in this dataset
    NOTE: some of these requires objectclass version of tasks, and so may break. 
    RETURNS:
    - params, dict of features. or None (if this is old task version)
    """
    
    IS_NEW_TASK_VER = D.taskclass_is_new_version(ind)
    # if not D.taskclass_is_new_version(ind):
    #     # Then this is old task version, many things are not defined. just return None./
    #     return None

    # Get the allparams and allparams
    # taskparams = D.blockparams_extract_single_blockparams(ind)
    # blockparams = D.blockparams_extract_single(ind)
    allparams = D.blockparams_extract_single_combined_task_and_block(ind)

    ########### ABORT MODES
    ABORT_ON = D.Dat.iloc[ind]["abort_params"]["on"]
    ABORT_MODES = list(D.Dat.iloc[ind]["abort_params"]["modes"])
    ABORT_PUNISH = allparams["params_task"]["enableAbort_punish_if_abort"]==1
    if "failed_rule" in ABORT_MODES:
        # Then also include aborts due to failing ObjectClass rules
        if "RuleList" in allparams["task_objectclass"].keys():
            found_rule=False
            for rule in allparams["task_objectclass"]["RuleList"]:
                if False:
                    assert len(rule[1])==0, "this rule has a param, have to incorportae this info into ABORT_MODES somehow"
                    # Just take the first item (the kind of rule)
                if len(rule)>0:
                    ABORT_MODES.append(rule[0])
                    found_rule = True
            if not found_rule:
                # happened for gridlinecirlce sometime
                ABORT_MODES.append("not_sure")
        else:
            # skip.
            ABORT_MODES.append("not_sure")

    ############## SEQUENCE SUPERVISION
    if IS_NEW_TASK_VER and "Params" in allparams["task_objectclass"].keys():
        seq = allparams["sequence"]

        if seq["ver"]=="objectclass" and seq["on"]==1 and seq["manipulations"] == ['alpha', 'active_chunk'] and allparams["task_objectclass"]["Params"]["ChunksDone"]["order"]=="any_order":
            SEQUENCE_SUP = "active_chunk"
            assert allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
            SEQUENCE_ALPHA = allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
            
            # chunk active abort [usually with chunk active]
            assert ['chunks_any_order', {}] in allparams["task_objectclass"]["RuleList"]
            assert "failed_rule" in ABORT_MODES
            
        elif seq["ver"]=="objectclass" and seq["on"]==1 and seq["manipulations"] == ['alpha', 'mask'] and allparams["task_objectclass"]["Params"]["ChunksDone"]["order"]=="in_order":
            #SEQUENCE_SUP = "mask"
            assert allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
            SEQUENCE_ALPHA = allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
            
            # Sequence abort (in order) [usually with mask]
            if ['chunks_in_order', {}] in allparams["task_objectclass"]["RuleList"] and "failed_rule" in ABORT_MODES:
                SEQUENCE_SUP = "mask"
            else:
                SEQUENCE_SUP = "char_strokes"

        elif seq["on"]==0:
            SEQUENCE_SUP = "off"
            #assert allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
            if "Params" in allparams["task_objectclass"].keys():
                # Newer version using tasksequencer on objectlcass
                SEQUENCE_ALPHA = allparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
            else:
                # older vesrion (e.g., gridlinecirlce).
                SEQUENCE_ALPHA = "not_sure" # ignore

        else:
            print(seq)
            print(allparams["task_objectclass"]["Params"])
            assert False
    else:
        SEQUENCE_SUP = "not_sure"
        SEQUENCE_ALPHA = "not_sure"

    ############# COLOR
    ### COLOR
    # for k in allparams.keys():
    #     print(k in allparams.keys())
    # assert False
    # # if "colormod" in allparams
    # print(allparams.keys())
    # print(allparams.keys())
    COLOR_ON = allparams["colormod"]["strokes"]["on"]==1
    if COLOR_ON==1:
        COLOR_ON=True

    if "color_method" not in allparams["colormod"]["strokes"].keys():
        if "randomize_each_stroke" in allparams["colormod"]["strokes"].keys():
            if allparams["colormod"]["strokes"]["randomize_each_stroke"]==1:
                COLOR_METHOD = 'randomize_each_stroke'
            else:
                COLOR_METHOD = ''
        else:
            # before added this feature
            COLOR_METHOD = ''
            
    elif len(allparams["colormod"]["strokes"]["color_method"])>0:
        COLOR_METHOD = allparams["colormod"]["strokes"]["color_method"]
    elif allparams["colormod"]["strokes"]["randomize_each_stroke"]==1:
        COLOR_METHOD = 'randomize_each_stroke'
    else:
        COLOR_METHOD = ''

    # SOmetimes colors can be selectgibely applied to any combination of {fixation, strokes_guide, strokes_draw, post}. Assumes
    # colors applied to all, unless specified otherwise
    # COLORS_OBJECTS_APPLIED_TO = 'fgdp' 
    # f = fixation cue
    # g = taskstroke (stim) during fixation(guide)
    # d = taskstrokes (stim) during drawing (after go)
    # p = taskstrokes (stim) during post scene (after done).

    def _get_fade_for_each_item(applyto, fade_val):
        """
        PARAMS;
        - applyto, list of str, which items to apply this to
        - fade_val, np scalar, the fade value nbeing applied. 
        0 means fully fade to defualt
        RETURNS:
        - dict, i.e.,mg out[item] = num
        """
        out = {}
        for item in ["fixcue", "strokes_guide", "strokes_draw", "strokes_post"]:
            if item in applyto:
                # then the fade is being applied
                out[item] = fade_val
            else:
                # then no fade. 
                out[item] = np.array(1.)
        return out

    COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT = {
        "fixcue":np.array(1.),
        "strokes_guide":np.array(1.),
        "strokes_draw":np.array(1.),
        "strokes_post":np.array(1.)
    }

    if "extra_methods" in allparams["colormod"].keys():
        if len(allparams["colormod"]["extra_methods"])>0:
            for i in range(len(allparams["colormod"]["extra_methods"])):
                meth = allparams["colormod"]["extra_methods"][i]
                prms = allparams["colormod"]["extra_methods_params"][i]
                applyto = allparams["colormod"]["extra_methods_applyto"][i]

                if meth=="fade_to_default_unless_ranked_colormethod":
                    # then fade out color, unless COLOR_METHOD=='rank'
                    fade_val = prms[0]
                    if COLOR_METHOD=='rank':
                        items_to_check = ["strokes_draw", "strokes_post"] # this method never modifies fix cue or strokes guide.
                        applyto = [item for item in applyto if item in items_to_check]
                    COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT = _get_fade_for_each_item(applyto, fade_val)
                elif meth =="fade_to_default":
                    # fade out
                    fade_val = prms[0]
                    COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT = _get_fade_for_each_item(applyto, fade_val)
                elif meth == "randomly_replace_with_probe_color":
                    prob = prms[0] # array(0.08)
                    color_to_replace_with = prms[1] # array([0.4, 0.4, 0.4])
                    # SKIP THIS FOR NOW.
                    # Should extract, for each trial, the actual color that was used (from TrialRecord.User.AdapterParams)
                else:
                    print(meth, prms, applyto)
                    assert False, "dont know this"

    ##### what phase in trial is color shown?
    # Need separate parameters for color method during:
    # - fixation cue
    # - guide
    # - samp
    # - post.
    # on 10/18/22, hacked so that guide and samp were faded out over sets of blocks. see gslides for details. this was hakced in 
    # drag.m, but is saved in bb or guide ink colors.

    COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST = []    
    COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST_BINARY = '' # 1001 means fixation and post have color. strokes guide and strokes draw are faded fully to defualt.
    for item in ["fixcue", "strokes_guide", "strokes_draw", "strokes_post"]:
        fadeval = COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT[item] # 0 means faded fully to default

        # 1) list of fade values
        COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST.append(fadeval)
    
        # 2) Encode this as binary string (either color visible (val>0) or fully faded (val=0))
        if np.isclose(fadeval, 0.):
            COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST_BINARY += '0'
        else:
            COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST_BINARY += '1'


    ################ ONLINE VISUAL FB
    if IS_NEW_TASK_VER:
        # Newere methods (2022)
        if "AllPtsVisCriteria" not in allparams["task_objectclass"].keys() or "CtrlPtsVisCriteria" not in allparams["task_objectclass"].keys():
            VISUALFB_METH = "not_sure"
        else:
            p = allparams["task_objectclass"]["AllPtsVisCriteria"]
            c = allparams["task_objectclass"]["CtrlPtsVisCriteria"]

            if p == [['always_vis', {}, 'and', 0]] and c == [['always_vis', {}, 'and', 0]]:
                VISUALFB_METH = "none"
            elif p == [['not_touched', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0]]:
                VISUALFB_METH = "all"
            elif p == [['always_vis', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0], ['no_isolated_pts', {}, 'or', 1, 1]]:
                VISUALFB_METH = "control_pts"
            elif p == [['not_touched', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0], ['no_isolated_pts', {}, 'or', 1, 1]]:
                # Unusual. 
                VISUALFB_METH = "all_and_control_pts_respawn"
            elif p == [['always_vis', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0]]:
                VISUALFB_METH = "control_pts_norespawn"
            else:
                print(p)
                print(c)
                assert False
    else:
        # Older., liolke gridlinecirlce.
        VISUALFB_METH = "not_sure"


    ################# guide dynamic
    GUIDEDYN_ON = allparams["guide_dynamic"]["on"]==1
    GUIDEDYN_VER = allparams["guide_dynamic"]["version"]
    GUIDEDYN_DURATION = allparams["guide_dynamic"]["duration"] ==1;
    if "chunks_fly_in_together" in allparams["guide_dynamic"].keys():
        GUIDEDYN_CHUNKSFLY = allparams["guide_dynamic"]["chunks_fly_in_together"] ==1;
    else:
        GUIDEDYN_CHUNKSFLY = "not_sure"


    ############### suonds
    # print(allparams["DonenessTracker"])
    # if IS_NEW_TASK_VER:
    #     print(allparams["DonenessTracker"])
    #     assert allparams["DonenessTracker"]["make_sound_on_chunk_switch"]==0
    
    SOUNDS_TRIAL_ONSET = allparams["sounds"]["play_sound_trial_onset"]==1
    if "play_done_sound_on" in allparams["sounds"].keys():
        SOUNDS_ALLPTSDONE = allparams["sounds"]["play_done_sound_on"]==1
    else:
        # Before around Jun 2022 - was default on.
        SOUNDS_ALLPTSDONE = True

    if IS_NEW_TASK_VER and "Feedback" in allparams["task_objectclass"].keys():
        SOUNDS_HIT_VER = allparams["task_objectclass"]["Feedback"]["hit_sound_ver"]
        if allparams["task_objectclass"]["Feedback"]["chunk_active_change_sound_multiplier"]>0:
            SOUNDS_CHUNK_CHANGE = allparams["task_objectclass"]["Feedback"]["chunk_active_change_sound_ver"]
        else:
            SOUNDS_CHUNK_CHANGE = "none"

        # Older method, overwrites.
        if allparams["DonenessTracker"]["make_sound_on_chunk_switch"]==1:
            SOUNDS_CHUNK_CHANGE = "any_change"

        SOUNDS_STROKES_DONE = allparams["task_objectclass"]["Feedback"]["stroke_done_sound_ver"]
    else:
        SOUNDS_HIT_VER = "not_sure"
        SOUNDS_CHUNK_CHANGE = "not_sure"
        SOUNDS_STROKES_DONE = "not_sure"



    ############# CONTROL PTS
    if "control_pts" in allparams.keys():
        CONTROL_SIZE = allparams["control_pts"]["size_mult"]
        CONTROL_COLOR_DIFF = allparams["control_pts"]["color_diff"]
        CONTROL_VISIBLE = CONTROL_SIZE>1 or np.any(CONTROL_COLOR_DIFF!=0)
        if "ControlPts" in allparams["task_objectclass"]["Params"].keys():
            CONTROL_GEN_METHOD = allparams["task_objectclass"]["Params"]["ControlPts"]["generate_method"]
        else:
            CONTROL_GEN_METHOD = "not_sure"
    else:
        CONTROL_SIZE = "not_sure"
        CONTROL_COLOR_DIFF = "not_sure"
        CONTROL_VISIBLE = "not_sure"
        CONTROL_GEN_METHOD = "not_sure"


    ################ PEANUT
    PEANUT_ALPHA = allparams["sizes"]["PeanutAlpha"]



    ############## REPORT DONE KMETHOD
    def extract_taskdone_criteria(allparams):
        # allparams(ind).task_objectclass.TaskDoneCriteria = struct;
        # allparams(ind).task_objectclass.TaskDoneCriteria(1).ver = 'ctrl_pts_all';
        # allparams(ind).task_objectclass.TaskDoneCriteria(1).params = {};
        # allparams(ind).task_objectclass.TaskDoneBool = 'and';
        
        if isinstance(allparams["task_objectclass"]["TaskDoneCriteria"], dict):
            ver = allparams["task_objectclass"]["TaskDoneCriteria"]["ver"]
            assert len(allparams["task_objectclass"]["TaskDoneCriteria"]["params"])==0, "incorproate this info into ver..."
        else:
            print(allparams["task_objectclass"])
            assert False, "is this multiple criteria?"
        return ver

    if allparams["params_task"]["donebutton_criterion"]:
        # Used done button
        meth = allparams["params_task"]["donebutton_criterion"]
        DONE_METHOD = f"donebutton-{meth}"
    #     DONE_PARAMS = meth
    else:
    #     allparams(ind).params_task.reportDoneMethod = ...
    #     {'taskobject', 'numstrokes_rel_task'}; % nstrokes, so that can end if keeps trying and failing.
    #     allparams(ind).params_task.reportDoneParam= ...
    #         {{}, 4};

        DONE_METHOD = []
    #     DONE_PARAMS = []
        rdm = allparams["params_task"]["reportDoneMethod"]
        rdp = allparams["params_task"]["reportDoneParam"]
        for meth, par in zip(rdm, rdp):
            if meth=="taskobject":
                # then look into allparams to figure out
                methodthis = extract_taskdone_criteria(allparams)
                assert len(par)==0
            else:
                print(meth)
                print(par)
                assert False, "convert (meth, par) into tuple?"
            DONE_METHOD.append(methodthis)
    #         DONE_PARAMS.append(paramsthis)

    
    ##### SCREEN POST VER
    SCREENPOST_ON = allparams["scenes"]["sceneSchedule"]["samp1"][3]==1
    bpthis = allparams["postscene"]
    SCREENPOST_ALPHA = bpthis["samp1_force_alpha_to"]
    SCREENPOST_SIZE = bpthis["samp1_force_size_to"]
    if "dynamic" in bpthis.keys():
        spd = bpthis["dynamic"]
        spd_ver = bpthis["dynamic_ver"]

        if IS_NEW_TASK_VER:
            assert bpthis["samp1_only_show_untouched"]==0

        assert bpthis["pnut1_interp"]["on"]==1
        assert bpthis["pnut1_interp"]["params"] == ['upsample', 3.]
        
        # print(spd)
        # print(bpthis.ke)
        # print(spd["ver"])
        if spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'all_strokes']:
            SCREENPOST_DYNAMIC_VER = "all_strokes"
        elif spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'active_chunk']:
            SCREENPOST_DYNAMIC_VER = "flash_active_chunk"
        elif spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100.]:
            # Previuosly, default was active_chunk
            SCREENPOST_DYNAMIC_VER = "flash_active_chunk"
        elif spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'strokes_notdone']:
            SCREENPOST_DYNAMIC_VER = "flash_missed_taskstrokes"
        elif spd["on"]==0:
            SCREENPOST_DYNAMIC_VER = "off"
        else:
            print(1)
            print(spd)
            print(2)        
            print(spd_ver)
            print(3)
            print(bpthis["dynamic_params"])
            assert False
    else:
        SCREENPOST_DYNAMIC_VER = "not_sure"
        

    # Whtehr cue and stim are flipped
    if "fix_tp" in allparams.keys():
        CUESTIM_FLIP = allparams["fix_tp"]["flip_cue_image_order"]==1
    else:
        assert int(D.Dat.iloc[ind]["date"]) <= 231001
        CUESTIM_FLIP = False

    params = {
        "ABORT_ON":ABORT_ON,
        "ABORT_MODES":ABORT_MODES,
        "ABORT_PUNISH":ABORT_PUNISH,
        
        "SEQUENCE_SUP":SEQUENCE_SUP,
        "SEQUENCE_ALPHA":SEQUENCE_ALPHA,
        
        "COLOR_ON":COLOR_ON,
        "COLOR_METHOD":COLOR_METHOD,
                
        "COLOR_ITEMS_FADE_TO_DEFAULT": COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT,  # 0 means default.
        "COLOR_ITEMS_FADE_TO_DEFAULT_LIST": COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST,  # 0 means default.
        "COLOR_ITEMS_FADE_TO_DEFAULT_BINSTR": COLORS_ITEMS_FADE_ZERO_MEANS_DEFAULT_LIST_BINARY,  #

        "VISUALFB_METH":VISUALFB_METH,
        
        "GUIDEDYN_ON":GUIDEDYN_ON,
        "GUIDEDYN_VER":GUIDEDYN_VER,
        "GUIDEDYN_CHUNKSFLY":GUIDEDYN_CHUNKSFLY,
        "GUIDEDYN_DURATION":GUIDEDYN_DURATION,

        "SOUNDS_HIT_VER":SOUNDS_HIT_VER,
        "SOUNDS_STROKES_DONE":SOUNDS_STROKES_DONE,
        "SOUNDS_TRIAL_ONSET":SOUNDS_TRIAL_ONSET,
        "SOUNDS_ALLPTSDONE":SOUNDS_ALLPTSDONE,

        "CONTROL_SIZE":CONTROL_SIZE,
        "CONTROL_COLOR_DIFF":CONTROL_COLOR_DIFF,
        "CONTROL_VISIBLE":CONTROL_VISIBLE,
        "CONTROL_GEN_METHOD":CONTROL_GEN_METHOD,

        "PEANUT_ALPHA":PEANUT_ALPHA,

        "DONE_METHOD":DONE_METHOD,

        "SCREENPOST_ON":SCREENPOST_ON,
        "SCREENPOST_ALPHA":SCREENPOST_ALPHA,
        "SCREENPOST_SIZE":SCREENPOST_SIZE,
        "SCREENPOST_DYNAMIC_VER":SCREENPOST_DYNAMIC_VER,

        "CUESTIM_FLIP":CUESTIM_FLIP
    }



    return params 
            
                