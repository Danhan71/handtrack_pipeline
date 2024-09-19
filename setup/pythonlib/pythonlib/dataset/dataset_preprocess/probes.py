""" To classify probes, etc
"""

import numpy as np


def extract_epochs(D):
    """ Return list of epochs in D, taking into account:
    # Ignore these because baseline has no correct seq, and is not testing anything, and so
    # shouldn't contribute to the probe kind label.
    """
    list_epochs_ignore = ["baseline", "baseline|0", "baseline|1", "base", "base|0", "base|1"]
    def _ignore(ep):
        # Ignore it if it contains the string base (returns True)
        return "base" in ep

    epochs = sorted(D.Dat["epoch"].unique().tolist())
    # epochs = [ep for ep in epochs if ep not in list_epochs_ignore]
    epochs = [ep for ep in epochs if not _ignore(ep)]
    return epochs

def _extract_concurrent_train_tasks(task_probe, D):
    """ for a given probe task, get list of all non-probe tasks presented in each epoch
    RETURNS:
    - tasks_per_epoch, dict[epoch] = list_character_nbames
    """
    
#     blocks = D.Dat[D.Dat["character"]==task_probe]["block"] # blocks that has this 
    epochs = D.Dat[D.Dat["character"]==task_probe]["epoch"] # epochs that this task is in
    epochs = [ep for ep in epochs if ep in extract_epochs(D)]
    tasks_per_epoch = {}
    for ep in epochs:
        inds = (D.Dat["epoch"]==ep) & (D.Dat["probe"]==0)
        tasks_per_epoch[ep] = sorted(D.Dat[inds]["character"].unique())
    return tasks_per_epoch    
    
def _check_same_spatial_config(task1, task2, D, mapper_taskname_epoch_to_taskclass):
    """ given a pair of tasks, test whether has same spatial config.
    - i..e, have prims in the same grid locations
    PARAMS:
    - task1 and task2 are string names.
    RETURNS:
    - bool, True if prims are in same locaiton (disregarding order)
    """
    
    # 1) get taskclass objects 
    try:
        Task1 = _extract_taskclass_from_taskname(task1, mapper_taskname_epoch_to_taskclass)
        Task2 = _extract_taskclass_from_taskname(task2, mapper_taskname_epoch_to_taskclass)
        out = Task1.compare_prims_on_same_grid_location(Task2)
    except Exception as err:
        Task1.plotStrokes()
        Task2.plotStrokes()
        raise err
    return out

def _extract_n_prims(taskname, mapper_taskname_epoch_to_taskclass):
    """ Return dict mapping from prims to number times they are present
    """
    Task = _extract_taskclass_from_taskname(taskname, mapper_taskname_epoch_to_taskclass,
        epoch=None) 


def _extract_taskclass_from_taskname(taskname, mapper_taskname_epoch_to_taskclass,
        epoch=None):
    """ Return a single taskclass for this taskname. by default doesnt care about epoch,
    unless epoch is not None
    """

    if True:
        for taskname_epoch, taskclass in mapper_taskname_epoch_to_taskclass.items():
            if epoch is None:
                # Then ignore epoch
                if taskname_epoch[0]==taskname:
                    return taskclass
            else:
                if taskname_epoch == (taskname, epoch):
                    return taskclass

        # fail if got here and didnt find
        print(taskname)
        print(epoch)
        assert False, "didnt find it"
    else:
        # too slow?
        return D.Dat[D.Dat["character"]==taskname]["Task"].values[0]
    
# def _check_equidistant_from_first_stroke(taskname, epoch):
#     """ Returns True if strokes 2 and 3 are the same distance (euclidain)
#     from stroke 1. looks at centers on grid.
#     """
#     asdsad
#     Task = _extract_taskclass_from_taskname(taskname)

def _check_equidistant_from_first_stroke(ind, D):
    """ Returns True if strokes 2 and 3 are the same distance (euclidain)
    from stroke 1. looks at centers on grid.
    PARAMS:
    - ind, index into D.Dat
    """
    
    # 1) get ground truth sequencing order (using tasksequencer)
    sdict = D.grammarmatlab_extract_beh_and_task(ind)
#     {'taskstroke_inds_beh_order': [2, 0, 1],
#          'taskstroke_inds_correct_order': [2, 0, 1],
#          'active_chunk': <pythonlib.chunks.chunksclass.ChunksClass at 0x7f2c5f9b1cd0>,
#          'supervision_tuple': 'off|0||0',
#          'epoch': 'D|0'}
    if False: # To get rulstring, like:
        # ('preset-null-AnBmCk2RndFlx1', {'categ_matlab': None, 'params_matlab': None, 'params_good': 'AnBmCk2RndFlx1', 'categ': 'preset', 'subcat': 'null', 'params': 'AnBmCk2RndFlx1', 'rulestring': 'preset-null-AnBmCk2RndFlx1'})
        try:
            print(D.grammarparses_ruledict_rulestring_extract(ind))
        except Exception as e:
            pass
    taskstroke_inds_correct_order = sdict["taskstroke_inds_correct_order"]
    
    # 2) get locations of taskinds
    T = D.Dat.iloc[ind]["Task"]
    if not T.get_grid_ver()=="on_grid":
        # THen cannot compute this...
        return False

    dseg = D.taskclass_tokens_extract_wrapper(ind, "task")
    # dseg2 = T.tokens_generate()
    # assert dseg == dseg2
    locations = [d["gridloc"] for d in dseg]
    if False:
        print("LOCATIONS:", locations)
        print("taskstroke_inds_correct_order:", taskstroke_inds_correct_order)
        print("sdict:", sdict)
    if len(locations)<3:
        # then cant compute
        return False
    else:
        # 3) get distances between pairs of stropkes.
        def _distance_between_strokes(ind1, ind2):
            """Euclidian dist from center of taskstroke ind1 to ind2, where
            inds are default indices into strokes
            """
            loc1 = np.array(locations[ind1])
            loc2 = np.array(locations[ind2])
            return np.linalg.norm(loc2 - loc1)
        def _distance_between_strokes_using_tasksequencer_rank(rank1, rank2):
            """ same, but using indices after ordering using
            tasksequenccer """
            
            ind1 = taskstroke_inds_correct_order[rank1]
            ind2 = taskstroke_inds_correct_order[rank2]
            return _distance_between_strokes(ind1, ind2)
        
        # Sanity check
        for loc in locations:
            if loc is None:
                print("--", locations)
                print("task ver:", T.get_grid_ver())
                D.plotSingleTrial(ind)
                assert False, "should not get to this point if this task is not on_grid"

        dist1 = _distance_between_strokes_using_tasksequencer_rank(0, 1)
        dist2 = _distance_between_strokes_using_tasksequencer_rank(0, 2)
        
        return np.isclose(dist1, dist2)
        

def _classify_probe_task(novel_location_config, equidistant, novel_location_shape_combo, 
        more_n_strokes, detailed=True):
    """ map from params to name that describes this probe task.
    Can use name as taskgroup
    """
    
    if novel_location_config or novel_location_shape_combo or more_n_strokes:
        s = "E"
    else:
        s = "I"

    if detailed:
        if novel_location_config:
            s += "_cfig"

        if equidistant:
            s += "_eq"

        if novel_location_shape_combo:
            s += "_locsh"

        if more_n_strokes:
            s += "_nstrk"

        # if novel_location_config and equidistant:
        #     return "E_cfig_eq"
        # elif novel_location_config and not equidistant:
        #     return "E_cfig"
        # elif not novel_location_config and equidistant:
        #     return "I_eq"
        # else:
        #     return "I"

    return s
    
def _generate_map_taskclass(D):
    """ Generate dict mapping from (taskname, epochname) [both strings] to 
    TaskClass [a single example]. Useful for speeding up subsequent stuff
    RETURNS:
    - mapper_taskname_epoch_to_taskclass, dict,(see above
    """
    # epochs = D.Dat["epoch"].unique().tolist()
    epochs = extract_epochs(D)
    tasks = D.Dat["character"].unique().tolist()
    mapper_taskname_epoch_to_taskclass = {}
    for ep in epochs:
        for ta in tasks:

            # find inds for this
            list_tc = D.Dat[(D.Dat["epoch"]==ep) & (D.Dat["character"]==ta)]["Task"].values
            if len(list_tc)>0:
                # take the first
                TaskClass = list_tc[0]
                mapper_taskname_epoch_to_taskclass[(ta, ep)] = TaskClass

    return mapper_taskname_epoch_to_taskclass




# def preprocess_extract_features_all(D):
#   """ First run this to save relevant features for all tasks. SPeeds up subsequent stuff 
#   dramatically
#   """

#   for ind in range(len(D.Dat)):



def compute_features_each_probe(D, only_do_probes = True, CLASSIFY_PROBE_DETAILED=True,
    PRINT=False):
    """ For each probe task, compute its "features" which indicate ways that it is 
    different from train tasks. This is compuited spearately for each epoch, becuase
    features will depend on what sequence rule is being trained 
    PARAMS:
    - only_do_probes, then if not probe, skips you. this defualt.
    RETURNS:
    - dict_probe_features, dict, with keys (epoch, probe_task_name), mapping to tuple of
    features.
    - dict_probe_kind, dict, with keys (epoch, probe_task_name), mapping to string name
    defining this kind of probe.
    NOTE: only includes probes actually present in that epoch.
    NOTE: generally uses tasks taht were actually poresented, nopt what is in blockparams
    """

    # First, ignore any "baseline" epochs


    # 1) get names of all probes
    if only_do_probes:
        list_tasks_probe = sorted(D.Dat[D.Dat["probe"]==1]["character"].unique())
    else:
        list_tasks_probe = sorted(D.Dat["character"].unique())

    mapper_taskname_epoch_to_taskclass = _generate_map_taskclass(D)
    list_epochs = extract_epochs(D)

    # 2) For each probe, collect its features
    dict_probe_features = {}
    dict_probe_kind = {}
    for task_probe in list_tasks_probe:
        train_tasks_per_epoch = _extract_concurrent_train_tasks(task_probe, D) 
        epochs_that_have_train_tasks = [ep for ep, tasks in train_tasks_per_epoch.items() if len(tasks)>0]

        if len(train_tasks_per_epoch)==0:
            # Then this task was never presented with train tasks (e..g, just in pretest)
            for ep in list_epochs:
                dict_probe_features[(ep, task_probe)] = {
                    "novel_location_config":False, 
                    "equidistant_from_first_stroke":False,
                    "novel_location_shape_combo":False,
                    "more_n_strokes":False                    
                    }
                dict_probe_kind[(ep, task_probe)] = "undefined"
        else:
            ########### IF Same correct beh across rules, then ignore classifying this probe.
            list_correct_sequence = []
            for ep in list_epochs:
                list_inds = D.Dat[(D.Dat["character"]==task_probe) & (D.Dat["epoch"]==ep)].index.tolist()
                if len(list_inds)>0:
                    # list_equidistant = [] # check all inds, and confirm they are identical
                    sdict = D.grammarmatlab_extract_beh_and_task(list_inds[0])
                    taskstroke_inds_correct_order = sdict["taskstroke_inds_correct_order"]
                    list_correct_sequence.append(tuple(taskstroke_inds_correct_order))
            list_correct_sequence_unique = list(set(list_correct_sequence))

            if len(list_correct_sequence)==0:
                # Then no train tasks found...
                for ep in list_epochs:
                    dict_probe_features[(ep, task_probe)] = {
                        "novel_location_config":False, 
                        "equidistant_from_first_stroke":False,
                        "novel_location_shape_combo":False,
                        "more_n_strokes":False                    
                        }
                    dict_probe_kind[(ep, task_probe)] = "none"
            elif len(list_correct_sequence_unique)==1 and len(epochs_that_have_train_tasks)>1:
                # Then multipel epochs require the same beh sequence...
                for ep in list_epochs:
                    dict_probe_features[(ep, task_probe)] = {
                        "novel_location_config":False, 
                        "equidistant_from_first_stroke":False,
                        "novel_location_shape_combo":False,
                        "more_n_strokes":False                    
                        }
                    dict_probe_kind[(ep, task_probe)] = "same_beh"
            else:
                for ep, tasks_this_epoch in train_tasks_per_epoch.items():
                        
                    if PRINT:
                        print("probe task in epoch: ", task_probe, ep)
                    ### COLLECT FEATURES
                    # 1) probe is a unique location config?
                    list_same_config = []
                    for task_train in tasks_this_epoch:
                        same_config = _check_same_spatial_config(task_probe, task_train, 
                            D, mapper_taskname_epoch_to_taskclass)
                        list_same_config.append(same_config)
                    novel_location_config = not any(list_same_config)
                        
                    # 2) taskstrokes 2 and 3 (in sequence) are same distance from 1?
                    # get inds that have this task and epoch
                    list_inds = D.Dat[(D.Dat["character"]==task_probe) & (D.Dat["epoch"]==ep)].index.tolist()
                    list_equidistant = [] # check all inds, and confirm they are identical
                    for ind in list_inds:
                        eq = _check_equidistant_from_first_stroke(ind, D)
                        list_equidistant.append(eq)
                    if "rndstr" in ep:
                        # then there is no consistent sequence, it is randomiozed, so cant compute thisd
                        equidistant = False
                    else:
                        # all trials for this probe taskk shoudl have same seuqence.
                        if len(set(list_equidistant))>1:
                            # This is p ossible, e.g, for rank color (rand) where each trial is a different sequence
                            # Give this false.
                            equidistant = False
                        else:
                            assert len(set(list_equidistant))==1, "got different ones? how is that posibe..."
                            equidistant = list_equidistant[0]

                    # 3) How many instances of each prim? (n)
                    _extract_n_prims(task_probe, mapper_taskname_epoch_to_taskclass)

                    # 4) Any shape-location combo never seen during training?
                    # get list of all shape-location combos for tasks
                    list_shape_loc_in_train = []
                    for task_train in tasks_this_epoch:
                        Task = _extract_taskclass_from_taskname(task_train, 
                            mapper_taskname_epoch_to_taskclass)
                        tokens = Task.tokens_generate(assert_computed=True)
                        sl_this = [(t["shape"], t["gridloc"]) for t in tokens]
                        list_shape_loc_in_train.extend(sl_this)
                    list_shape_loc_in_train = set(list_shape_loc_in_train)

                    # for each probe, check if any of its items are in that list
                    TaskProbe = _extract_taskclass_from_taskname(task_probe, 
                        mapper_taskname_epoch_to_taskclass)
                    tokens = TaskProbe.tokens_generate(assert_computed=True)
                    sl_this = [(t["shape"], t["gridloc"]) for t in tokens]
                    novel_location_shape_combo = any([x not in list_shape_loc_in_train 
                        for x in sl_this])        

                    # 5) Diff n strokes
                    # list_shape_l    oc_in_train = []
                    list_nstrok_in_train = []
                    for task_train in tasks_this_epoch:
                        Task = _extract_taskclass_from_taskname(task_train, 
                            mapper_taskname_epoch_to_taskclass)
                        tokens = Task.tokens_generate(assert_computed=True)
                        n_tok = len(tokens)
                        list_nstrok_in_train.append(n_tok)
                    list_nstrok_in_train = set(list_nstrok_in_train)

                    # for each probe, check if any of its items are in that list
                    TaskProbe = _extract_taskclass_from_taskname(task_probe, 
                        mapper_taskname_epoch_to_taskclass)
                    tokens = TaskProbe.tokens_generate(assert_computed=True)
                    n_tok = len(tokens)
                    try:
                        more_n_strokes = n_tok > max(list_nstrok_in_train)
                    except Exception as err:
                        raise err
                    ### GIVE PROBE TASK CATEGORY A NAME
                    c = _classify_probe_task(novel_location_config, equidistant, 
                        novel_location_shape_combo, more_n_strokes, detailed=CLASSIFY_PROBE_DETAILED)

                    ### SAVE
                    dict_probe_features[(ep, task_probe)] = {
                        "novel_location_config":novel_location_config, 
                        "equidistant_from_first_stroke":equidistant,
                        "novel_location_shape_combo":novel_location_shape_combo,
                        "more_n_strokes":more_n_strokes
                        }
                    dict_probe_kind[(ep, task_probe)] = c

    if False:
        # Sort by name of (epoch, probe).
        keys = sorted(dict_probe_kind.keys())
        tmp = {}
        for k in keys:
            tmp[k] = dict_probe_kind[k]
        dict_probe_kind = tmp

    return dict_probe_features, dict_probe_kind, list_tasks_probe

def taskgroups_assign_each_probe(D, only_give_names_to_probes=True, CLASSIFY_PROBE_DETAILED=True):
    """ Rename each probe's taskgroup based on autoamtically detecting what is
    special about each probe task. Does this in clever fashion by comparing
    to the actual training tasks presaented in each epoch.
    NOTE: takes into account that probe might have different features and thus be
    different category across epochs. will name it based on conjuction of those categoryse.
    RETURNS:
    - map_task_to_taskgroup, dict mapping from character name of probe --> category.
    """

    # Then extract. need this to know what is ground truth tasksequencer sequence
    D.behclass_preprocess_wrapper()

    # 1) compute_features_each_probe
    print("Computing features each probe")
    dict_probe_features, dict_probe_kind, list_tasks_probe = compute_features_each_probe(D,
        only_do_probes=only_give_names_to_probes, CLASSIFY_PROBE_DETAILED=CLASSIFY_PROBE_DETAILED)

    # print(dict_probe_features)
    # print(dict_probe_kind)
    # print(list_tasks_probe)
    # assert False
    # 2) map from probe_task to categeroy. main point of this step is to combine catgegory names
    # across epochs if they are different names.
    # epochs = D.Dat["epoch"].unique().tolist()
    epochs = extract_epochs(D)
    map_task_to_taskgroup = {}
    for task_probe in list_tasks_probe:
        
        # a) collect names across epochs
        epochs_used = []
        probe_kind_used = []
        for ep in epochs:
            key = (ep, task_probe)
            if key in dict_probe_kind.keys():
                epochs_used.append(ep)
                probe_kind_used.append(dict_probe_kind[key])

        if len(probe_kind_used)==0:
            print(dict_probe_kind)
            print(task_probe)
            print(ep, ' -- ', epochs)
            assert False, "not sure how this possible."

        
        # b) combine names across epochs   
        # - if a given probe task has same cartegoryt acropss all epochs, give it that name
        # - if has different categories across epochs, then give it long name that combines all that info
        if len(set(probe_kind_used))>1:
            # Then diff name under diff epochs.
            taskgroup = ""
            for ep, probe_kind in zip(epochs_used, probe_kind_used):
                if len(taskgroup)>0:
                    taskgroup += "+"    
                taskgroup += ep + ":"
                taskgroup += probe_kind            
        else:
            # same probe kind across all epochs.
            taskgroup = probe_kind_used[0]
        
        # c) save
        map_task_to_taskgroup[task_probe] = taskgroup

    return map_task_to_taskgroup