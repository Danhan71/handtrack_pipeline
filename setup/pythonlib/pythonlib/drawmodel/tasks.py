""" stuff to do with tasks...
specifically for tasks in monkeylogic
"""

import numpy as np
from pythonlib.tools.stroketools import fakeTimesteps
from pythonlib.chunks.chunks import chunk_strokes, chunks2parses
import sys
class TaskClass(object):
    """ Holds a single task object.
    """

    def __init__(self, task):
        """ task can be formated in several ways:
        (1) ML2 task, which is a dict. 
        i.e., task = getTrialsTask(fd, 1), or
        _, tasklist = Probedat.pd2strokes(Probedat.Probedat) [this
        is same, but has also "strokes" and "fixpos"
        ... (nothing else coded so far)
        NOTE: leave task as None if just want to initialize blank.
        """

        if task is None:
            return

        self.Task = task
        if "strokes" not in self.Task.keys():
            assert False, "see Probedat.pd2strokes for how to get this."
        self.Strokes = self.Task["strokes"]
        assert len(self.Strokes)>0, "maybe is novel prim or somethign?"
        self.Program = None
        self.Objects = None
        self.PlanDat = None

        if isinstance(task, dict):
            keys_to_check = ["x", "y", "stage"] # simply keys that are diagnostic of this being ml2 task
            if all(k in task.keys() for k in keys_to_check):
                # then this is ml2 task
                self._cleanMonkeylogic()

        # Default for affine transfomations of objects.
        self._features_default()

    def _features_default(self):
        """ default affine features"""
        self.FEATURES_DEFAULT = {
            "x":0.,
            "y":0.,
            "th":0.,
            "sx":1., 
            "sy":1.,
            "order":"trs"}
        for k, v in self.FEATURES_DEFAULT.items():
            if not isinstance(v, str):
                self.FEATURES_DEFAULT[k] = np.asarray(v)

    def _cleanMonkeylogic(self):
        """ clean monkeylogic tasks
        """

        # 1) remove redundant information
        _list_keys_to_ignore = ["x_before_sketchpad", "y_before_sketchpad", "x", "y", "x_rescaled", "y_rescaled"]
        self.Task = {k:v for k,v in self.Task.items() if k not in _list_keys_to_ignore}



    ################## NAMING THE TASK (CATEGORIES)
    def info_is_new(self):
        """ true if this is a new task version
        """
        """ is this new (MakeDrawTasks) or old? 
        only after 8/30/20 could they be new.
        RETURNS:
        - True, new
        - False, old
        """
        task = self.Task
        isnew = False
        if "TaskNew" in task.keys():
            if len(task["TaskNew"])>0:
                isnew=True
        return isnew

    def info_is_new_objectclass(self, strict=True):
        """ Is even newer, like early 2022, using ObjectClass 
        to define tasks. Returns True if it is.
        """

        if not hasattr(self, "ObjectClass"):
            # Try extracting...
            self.objectclass_extract_all()

        if self.ObjectClass is None:
            # Try extracting...
            self.objectclass_extract_all()

        if len(self.PlanDat) == 0:
            # Try extracting...
            self.planclass_extract_all()

        # now that you've extracted, do a second check
        if self.ObjectClass is None:
            return False
        if len(self.PlanDat) == 0:
            return False
        if strict==True:
            if "Info" not in self.PlanDat.keys() or "TaskSetClass" not in self.PlanDat["Info"].keys():
                return False
        
        return True
        #return self.ObjectClass is not None

    def info_is_fixed(self):
        """ if True, then fixed, False, then random
            [Copied from drawmonkey utils
        """
        if self.info_summarize_task()["probe"]["stage"]=="triangle_circle":
            # since was random, but actualyl only small set of fixed tasks.
            return True
        
        info = self.info_summarize_task()
        if info is None:
            # thne this is old version. they were never "fixed"
            return False

        probe = info["probe"]
        if probe["prototype"]==0 and probe["saved_setnum"] is None and probe["resynthesized"]==0:
            # then this is new random task sampled each block.
            return False
        else:
            return True



    def info_summarize_task(self):
        """ helper to extract summary information
        RETURNS:
        dict, holding:
        - probe, Copied from drawmonkey utils
        - los_info, tuple, (setname, setnum, index), if this is fixed task.
        - tsc_info, tasksetclass info.
        """

        if self.info_is_new()==False:
            return None

        task = self.Task

        probe = _get_task_probe_info(task)

        probe["stage"] = self.Task["stage"]

        # LOS (load old set) info
        if probe["los_setindthis"] is not None:
            los_info = (probe["los_setname"], int(probe["los_setnum"]), int(probe["los_setindthis"]))
        else:
            los_info = None

        # TSC, TaskSetClass info
        tsc_info = None
        if self.PlanDat is not None:
            if "Info" in self.PlanDat.keys():
                if "TaskSetClass" in self.PlanDat["Info"].keys():
                    tsc_info = self.PlanDat["Info"]["TaskSetClass"]
                    if len(tsc_info)==0:
                        tsc_info = None

        info = {
            "probe":probe,
            "los_info":los_info,
            "tsc_info":tsc_info
        }

        return info
        
    def info_generate_unique_name(self, strokes=None, nhash = 6, 
        include_taskstrings=True, include_taskcat_only=False):
        """ This is simialr to in utils, but here is more confident that each task
        is unique. 
        INPUTS:
        - strokes, uses self.Strokes unless pass in strokes.
        - nhash, num digits to use.
        NOTE:
        - here uses all pts, not just those at endpoints, since that important for sme
        tasks. Differs from tools.utils, where ignores inside since same task might
        have diff num pts at different times.
        """
        MIN = 1000

        if strokes is None:
            strokes = self.Strokes

        from pythonlib.tools.stroketools import strokes_to_hash_unique
        _hash = strokes_to_hash_unique(strokes, nhash)

        # # Collect each x,y coordinate, and flatten it into vals.
        # vals = []
        # # for S in self.Strokes:
        # #     vals.extend(S[0])
        # #     vals.extend(S[-1])
        # for S in strokes:
        #     for SS in S:
        #         vals.extend(SS)
        #         # vals.extend(SS[1])
        # # print(np.diff(vals))
        # # tmp = np.sum(np.diff(vals))
        # # vals = np.asarray(vals)

        # vals = np.asarray(vals)
        # # vals = vals+MIN # so that is positive. taking abs along not good enough, since suffers if task is symmetric.

        # # Take product of sum of first and second halves.
        # # NOTE: checked that np.product is bad - it blows up.
        # # do this splitting thing so that takes into account sequence.
        # tmp1 = np.sum(vals[0::4])
        # tmp2 = np.sum(vals[1::4])
        # tmp3 = np.sum(vals[2::4])
        # tmp4 = np.sum(vals[3::4])

        # # rescale to 1
        # # otherwise some really large, some small.
        # # divie by 10 to make sure they are quite large.
        # tmp1 = tmp1/np.max([np.floor(tmp1)/10, 1])
        # tmp2 = tmp2/np.max([np.floor(tmp2)/10, 1])
        # tmp3 = tmp3/np.max([np.floor(tmp3)/10, 1])
        # tmp4 = tmp4/np.max([np.floor(tmp4)/10, 1])


        # # tmp1 = 1+tmp1-np.floor(tmp1)
        # # tmp2 = 1+tmp2-np.floor(tmp2)
        # # tmp3 = 1+tmp3-np.floor(tmp3)
        # # print(tmp1, tmp2, tmp3, tmp4)
        # # assert False

        # # tmp1 = np.sum(vals)
        # # tmp = np.sum(vals)

        # # Take only digits after decimal pt.
        # if True:
        #     tmp = tmp1*tmp2*tmp3*tmp4
        #     # print(tmp)
        #     tmp = tmp-np.floor(tmp)
        #     tmp = str(tmp)
        #     # print(tmp)
        #     # assert False
        # else:
        #     # This doesnt work well, all tmps end up lookgin the same. 
        #     tmp = np.log(np.abs(tmp1)) + np.log(np.abs(tmp2)) + np.log(np.abs(tmp3))
        #     print(tmp)
        #     tmp = str(tmp)
        #     ind = tmp.find(".")
        #     tmp = tmp[:ind] + tmp[ind+1:]
        # _hash = tmp[2:nhash+2]

        # Compose the task name
        taskcat = self.info_name_this_task_category()
        tasknum = self.info_summarize_task()["probe"]["tasknum"]

        # If LOS exists, then use that
        los_info = self.info_summarize_task()["los_info"]
        if los_info is not None:
            taskcat = f"{los_info[0]}-{los_info[1]}"
            tasknum = los_info[2]
        if include_taskstrings:
            assert include_taskcat_only is False, "choose one or other"
            return f"{taskcat}-{tasknum}-{_hash}"
        elif include_taskcat_only:
            return f"{taskcat}-{_hash}"
        else:
            return str(_hash)


    # SCRATCH - to compare tasks names before and after update hash code.
    # answers whether task that are diff (but only one name before) now succesflyll 
    # split inot diff names.
    # dat = {}
    # dat2 = {}
    # for i in range(len(dfthis)):
    #     name_new = dfthis.iloc[i]["Task"].Params["input_params"].info_generate_unique_name()
    # #     name_old = dfthis.iloc[i]["unique_task_name"]
    #     name_old = dfthis.iloc[i]["character"]
        
    #     if name_old in dat.keys():
    #         dat[name_old].append(name_new)
    #     else:
    #         dat[name_old] = [name_new]
        
    #     if name_new in dat2.keys():
    #         dat2[name_new].append(name_old)
    #     else:
    #         dat2[name_new] = [name_old]
        
    # dat = {k:list(set(v)) for k, v in dat.items()}
    # dat2 = {k:list(set(v)) for k, v in dat2.items()}


    # task = "mixture2-ss-52_1-376117"
    # row_variable = "date"
    # figb, figt = plot_beh_grid_singletask_alltrials(D, task, row_variable)
    # # figb.savefig(f"{sdirthis}/{task}-beh.pdf");
    # # figt.savefig(f"{sdirthis}/{task}-task.pdf");
    # # plt.close("all")

    

    def info_name_this_task_category(self):
        """ 
        [GOOD] Wrapper to name the task cartegory in smart way, based on 
        the kind of task.
        RETURNS:
        - taskcat, string name
        """
        info =  self.info_summarize_task()
        probe = info["probe"]
        los_info = info["los_info"]
        tsc_info = info["tsc_info"]
        task = self.Task

        if self.info_is_fixed():
            # First, see if is latest versions where saved los
            if los_info is not None:
                # then use this
                taskcat = f"{los_info[0]}-ss-{los_info[1]}"
            elif tsc_info is not None:
                # Then this is fixed task generated by TaskSetClass
                taskcat = f"tsc-{tsc_info['UniqueID']}"
            else:
                # Old versions
                if probe["resynthesized"]==1:
                    taskcat = f"{task['stage']}-rsyn-{probe['resynthesized_setname']}-{probe['resynthesized_setnum']}-{probe['resynthesized_trial']}"
                elif probe["saved_setnum"] is not None:
                    taskcat = f"{task['stage']}-ss-{probe['saved_setnum']}"
                elif probe["prototype"]==1:
                    taskcat = f"{task['stage']}-pt"
                elif task["stage"]=="triangle_circle":
                    # then OK, hard coded to maek this a fixed task.
                    assert False, "not coded"
                else:
                    assert False, "i am confyused - not sure in what way this task is 'fixed'"
        else:
            # New version, is the TaskSetClass generative model saved?
            if tsc_info is not None:
                taskcat = f"tsc-{tsc_info['UniqueID']}-random"
            else:
                # Old version
                taskcat = f"{task['stage']}-random"
        return taskcat


    ####################### ASSIGN EACH TASK STROKE A LABEL
    def assignLabelToEachStroke(self, method):
        """ label is generally string, but is like a class
        - method is either string, or a function, i..e,
        func(self.Task) --> <list of labels>.
        """
        if isinstance(method, str):
            assert False, "not coded yet"
        else:
            # assume method is a function
            lab = method(self.Task)
            if len(lab)==0:
                assert False
            self.TaskLabels = lab


    ####################### HELPERS
    def get_tasknew(self):
        """ extract tasknew, which is the one from makeDrawTasks
        RETURNS:
        - NOne, if doesnt exist (older tasks) or TaskNew, a dict.
        """
        if "TaskNew" not in self.Task:
            return None
        if len(self.Task["TaskNew"])==0:
            return None
        return self.Task["TaskNew"]

    ####################### GET FEATURES
    # (see efficiency cost model)


    ###################### GET CHUNKS
    def chunks_extract_models(self):
        """ what models were pre-entered when constructed this task?
        OUT:
        - dict, where each item is a single model, and value is list of chunks
        e.g.:
        {'linePlusL': [[[0, 1], [2, 3]]], '3line': [[[0], [1], [2], [3]]]}
        """
        return task2chunklist(self.Task)


    def chunks_extract(self, model="eachstroke"):
        """ 
        OUT:
        - list_of_chunks, where each chunk corresponds to a single parse (or strokes).
        --- e.g,, list_of_chunks = [
            [[0, 1], 2], 
            [[0, 2], 1]]
        (doesnt do permutations or whatever, just waht is entered in task)
        """        
        task = self.Task
        if isinstance(model, str):
            strokestask = self.Strokes # only used for extracting default, if usinbg eachstroke.
            chunklist = task2chunklist(task)
            chunks = chunklist2chunks(chunklist, strokestask, model)
        elif callable(model):
            # then model is a fucntion
            chunks = model(task)
        else:
            print(model)
            assert False, "not sure what is"
            
        return chunks


    def chunks_convert_to_strokes(self, list_of_chunks, reorder=False, thresh=10, sanity_check=False):
        """
        INPUT:
        - list_of_chunks, where each chunk corresponds to a single parse (or strokes).
        --- e.g,, list_of_chunks = [
            [[0, 1], 2], 
            [[0, 2], 1]]
        - reorder, then returns so strokes ordered in space. fails if there are branching.
        OUT:
        - list_of_strokes, same strcuture as loc.
        """
        return chunks2parses(list_of_chunks, self.Strokes, reorder=reorder,
            thresh=thresh, sanity_check=sanity_check)



    ####################### GET PARSES
    # (see taskmodel and below)


    ####################### TAKE IN A BEHAVIORAL TRIAL AND DO SOMETHING
    def behAssignClosestTaskStroke(self, strokes_beh):
        """ for each beh stroke, it is "assigned" a task stroke (not vice versa)
        assign each stroke the task stroke that is the closest
        """
        from pythonlib.tools.vectools import modHausdorffDistance
        if len(strokes_beh)==0:
            assert False, "strokes_beh is empty" 
        strokes_task = self.Task["strokes"]
        assignments =[]
        for s_beh in strokes_beh:
            
            # get distnaces from this behavioal stroke to task strokes
            distances = [modHausdorffDistance(s_beh, s_task) for s_task in strokes_task]

            # if len(distances)==0:
            #     asdfasfsdf
            # assign the closest stroke
            assignments.append(np.argmin(distances))
        return assignments

    ######################## PLOTS
    def plotTaskOnAx(self, ax=None, plotkwargs = None):
        """ plot task on axes"""
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        import matplotlib.pyplot as plt

        if plotkwargs is None:
            plotkwargs = {}
        if ax is None:
            fig, ax = plt.subplots(1,1)

        strokes = self.Strokes
        
        plotkwargs["clean_unordered"] = True
        plotDatStrokes(strokes, ax, **plotkwargs)

        # limits
        spad = self.Task["sketchpad"]
        ax.set_xlim(spad[:,0])
        ax.set_ylim(spad[:,1])


    ############# CONVERT TO OBJECTS
    # Objects is list of primitives, each with its own independent transofmation (locaiton, etc)

    def objects_extract(self, fail_if_no_match=False):
        """ Initial extraction of objects (and their transforms).
        RETURNS:
        - modified self.Objects, which is list of dicts, one for each object.
        or None, if TaskNew doesnt exist.
        - fail_if_no_match, then is fail if any subprograms not converted succesfulyl into object.
        NOTE:
        - There are multiple sources of this information. Get all, and make sure they corroborate.
        Also useful for me so here saving notes that might be useful later.
        """

        T = self.get_tasknew()
        if T is None:
            return None

        # V1: based on saved Objects
        if "Objects" in T.keys():
            if len(T["Objects"])>0:
                # New version, like 6/2021 
                if "Features" in T["Objects"]:
                    Feat = T["Objects"]["Features"]
                elif "Features_Active" in T["Objects"]:
                    Feat = T["Objects"]["Features_Active"]
                else:
                    print(T["Objects"])
                    assert False, "not sure why..."
                if isinstance(Feat, dict):
                    # print(Feat)
                    # print(type(Feat))
                    # for k, v in Feat.items():
                    #     print(k, v)
                    shapes1 = Feat["shapes"] # dict
                    shapes1 = self._program_line_dict2list(shapes1)
                else:
                    # print("---")
                    # print(self.Task)
                    # print("---")
                    # for k, v in T.items():
                    #     print(k, v)
                    # assert False
                    shapes1 = None
            else:
                shapes1 = None
        else:
            shapes1 = None

        # V2: if this is "mixture2" task, then the mixture2 params
        Task = T["Task"]
        if Task["stage"]=="mixture2":
            # Then the input params for this (makeDrawTasks params) explicitly defines
            # objects in order.
            shapes = Task["info"]["params"]["objectsInOrder"]
            shapes = self._program_line_dict2list(shapes)
            
            tforms_subprog = Task["info"]["params"]["tformsSubprograms"]
            tforms_subprog = self._program_line_dict2list(tforms_subprog)

        # V3: analyze program, extract the primitives.
        self.program_extract()
        Objects = []
        # print("------")
        # print(shapes1)
        # print(self.Program)
        for i in range(len(self.Program)):
            out = self.program_interpret_subprog(i, fail_if_no_match=fail_if_no_match)
            if out["obj"] is None:
                # replace it with the saved name
                if shapes1 is not None:
                    out["obj"] = shapes1[i]
                    assert len(shapes1)==len(self.Program), "Objects and Program don't match, not sure why"
            # print()
            Objects.append(out)
        self.Objects = Objects



    ############# PROCESS MATLAB PROGRAMS TO USEFUL FORMAT
    def _program_line_dict2list(self, line):
        """ given dict, where each item is indexed by number (in string format),
        converts to list, where order is maintained, using values from dict,
        and taking care of convering numbers to squeezed arrays, 
        NOTE: Is recursive.
        e.g., 
        INPUT:
        - line={'1': 'transform', '2': array([[0.]]), '3': array([[0.]]), '4': array([[0.]]), '5': array([[1.]]), '6': array([[1.]]), '7': 'trs'}
        OUTPUT:
        - line_list = ['transform', array(0.), array(0.), array(0.), array(1.), array(1.), 'trs']"""
        from pythonlib.tools.monkeylogictools import dict2list, dict2list2
        return dict2list2(line)

    def program_extract(self, skip_if_already_done=True):
        """ Initial extraction of program into useful format.
        OUTPUT:
        - assigns to self.Program
        NOTE: Skips running if self.Program is already present
        """

        if skip_if_already_done:
            if hasattr(self, "Program"):
                if self.Program is not None:
                    if len(self.Program)>0:
                        return

        if self.get_tasknew() is None:
            self.Program = None
            return None

        TaskNew = self.get_tasknew()["Task"]

        # --- Tasknew doesnt exist (older code)
        if TaskNew is None:
            self.Program = None
            return None

        # --- TaskNew exists...
        if (isinstance(self.get_tasknew()["Info"], dict)) and ("MorphParams" in self.get_tasknew()["Info"]):
            # Then this is morphed, it will not have program
            self.Program = None
        else:
            # print(self.get_tasknew()["Task"].keys())
            program = self.get_tasknew()["Task"]["program"]
            if True:
                # use recursive function
                program_list = self._program_line_dict2list(program)
                if isinstance(program_list[0][0], list):
                    # program = [subprog1, subprog2]
                    # subprog = [['line', array(0.), array(0.), array(0.), array(1.2), array(1.2), 'trs'], ['line', array(1.5), array(0.), array(1.57079633), array(1.54285714), array(1.54285714), 'trs'], ['line', array(1.5), array(1.5), array(3.14159265), array(1.2), array(1.2), 'trs'], ['line', array(-1.5), array(1.5), array(0.), array(1.42857143), array(1.42857143), 'trs']]
                    # good
                    pass
                else:
                    # program is actually a subprogram...
                    # convert it to program..
                    program_list = [program_list]
            else:
                # OLD, 
                # convert to nested lists.
                nlev1 = len(program)
                program_list = []
                for i in range(nlev1):
                    idx = f"{i+1}"
                    if doprint:
                        print("subprog:", idx)
                    subprog = program[idx]
                    nlev2 = len(subprog)
                    
                    # print(subprog)
                    # assert False
                    # Convert subprogram to list
                    subprog_list = []
                    for ii in range(nlev2):
                        idx = f"{ii+1}"
                        if doprint:
                            print("-- line:", idx)
                        line = subprog[idx]
                        line_list = self._program_line_dict2list(line)
                        if doprint:
                            print(line_list)
                        subprog_list.append(line_list)
                        
                    program_list.append(subprog_list)
            
            self.Program = program_list

    def program_get_line(self, ind_subprog, ind_line):
        """ returns line from extracted Program (self.Program)
        e.g., program[ind_subprog][ind_line]
        - inds in set, {-1, 0, 1, ...}
        - EG:
        --- prog_get_line(program_list, 0, -1)
        """
        self.program_extract()
        program = self.Program

        if ind_line != -1 and len(program[ind_subprog])<=ind_line:
            print(len(program[ind_subprog]))
            print(ind_line)
            assert False, "you are asking for a line that doesnt exist"
            
        return program[ind_subprog][ind_line]

    def program_get_tform_dict(self, x=None, y=None, th=None, sx=None, sy=None, order=None):
        if not hasattr(self, "FEATURES_DEFAULT"):
            self._features_default()
        tform = self.FEATURES_DEFAULT.copy()
        if x:
            tform["x"] = x
        if y:
            tform["y"] = y
        if th:
            tform["th"] = th
        if sx:
            tform["sx"] = sx
        if sy:
            tform["sy"] = sy
        if order:
            tform["order"] = order
        return tform

    def program_compare_tform_dicts(self, dict1, dict2):
        keys = self.program_get_tform_dict().keys()
        for k in keys:
            if isinstance(dict1[k], str):
                if not dict1[k]==dict2[k]:
                    return False
            else:
                if not np.isclose(dict1[k], dict2[k]):
                    return False
        return True

    def program_interpret_line(self, line, return_none_if_tform_static=True):
        """ helper to interpret a single subprogram line, (list),
        into a dict with relevant features labeled. does things like: fills
        in missing variables using default features, etc.
        - pass in a single line (e.g, output of program_get_line)
        - return_none_if_tform_static, returns None if a line says to do a tformation
        that has no affect.
        NOTE:
        Assumptions: 
        - first iutem in a line is the kind, like "line", or "tform"
        - remaining items are tforms in order as defined by default params.
        """
        
        # Default things
        FEATURES_NAMES = ["x", "y", "th", "sx", "sy", "order"]
        if not hasattr(self, "FEATURES_DEFAULT"):
            self._features_default()
        FEATURES_DEFAULT = self.FEATURES_DEFAULT
        assert FEATURES_NAMES==list(FEATURES_DEFAULT.keys()), "order maintined for python 3.7+"
        
        # Break down this line into features
        kind = line[0]
        feats = line[1:]
        out = {} 

        def _affine_feats(feats):
            """ feats is variable length list, where position
            in list indicates what kind of feature, in this order:
            [x, y, th, sx, sy, order]. Can be empty. 
            RETURNS:
            - dict[feat] = val
            """
            outthis = {}
            for name, val in zip(FEATURES_NAMES, feats):
                outthis[name] = val
            return outthis
            
        prims_list = ["line", "circle", "dot"]
        if kind in prims_list:
            # Is a stroke 
            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

        elif kind=="transform":
            # Then this line is an affine transform
            out["type"] = "transform"
            out["kind"] = "affine"
            out["affine"] = self.program_get_tform_dict(*feats)
            if return_none_if_tform_static:
                # Check if this tformation doesnt do anything
                if out["affine"]==self.FEATURES_DEFAULT:
                    return None
        elif kind=="repeat":
            # Then this line is an affine transform

            N = feats[0]
            feats = feats[1:]

            out["type"] = "repeat"
            out["kind"] = "default"
            out["affine"] = self.program_get_tform_dict(*feats)
            out["N"] = N
            if return_none_if_tform_static:
                # Check if this tformation doesnt do anything
                if out["affine"]==self.FEATURES_DEFAULT:
                    return None

        elif kind=="reflect":
            if len(feats)>0:
                theta = feats[0]
                try:
                    if isinstance(theta, list):
                        if len(theta)==0:
                            from math import pi
                            theta = np.asarray(pi/2)
                    # if len(theta)==0:
                    #     from math import pi
                    #     theta = np.asarray(pi/2)
                except Exception as err:
                    print(feats)
                    raise err
            else:
                from math import pi
                theta = pi/2
            if len(feats)>1:
                doreflect = feats[1]    
            else:
                doreflect = np.asarray(1.)


            out["type"] = "reflect"
            out["kind"] = "default"
            out["theta"] = theta
            out["doreflect"] = doreflect

        elif kind=="arc":
            # Is a stroke, but arc has 2 additional params before features
            arclength = feats[0] # in radians.
            ccw = feats[1] # if do ccw
            feats = feats[2:]

            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

            # additional params for arc
            out["arclength"] = arclength
            out["ccw"] = ccw

        elif kind=="curve":
            # Is a stroke, but has 1 additional params before features
            ctrlpts = feats[0] # in radians.
            feats = feats[1:]

            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

            # additional params for arc
            out["ctrlpts"] = ctrlpts

        else:
            # Generic
            out["type"] = None
            out["kind"] = None
            out["affine"] = None

        # else:
        #     print(kind)
        #     print(feats)
        #     print(prims_list)
        #     assert False, "confirm that is correct assumption that the 1: items are features in order."
        return out

    def program_interpret_subprog(self, ind_subprog, fail_if_no_match=True):
        """ Decides if a subprogram is a "standard" object, like an 
        object primitive.
        And what is the final transformation in space
        NOTE:
        Assumptions:
        - each subprogram is one and only one object.
        - if is N lines, first N-2 or N-1 define the object, and the last lines define the flexible tform.
        """

        self.program_extract()
        program = self.Program

        def _compare_to_primitive_templates(lines_good, fail_if_no_match=fail_if_no_match):
            """ checks if this matches predefined templates for action primitives.
            INPUTS:
            - lines_good, 
            --- these are lines are the first N-1 lines of a subprogram - assuming that the last
            --- line is the affine transofmr. 
            --- These are line dicts. i..e, output of prog_interpret_line
            - fail_if_no_match, if False, then returns None if no match.
            RETURNS:
            - either a string defining the primitive {"line", "circle", ...} or None, which means did not
            match (or fails, see above).
            """

            assert isinstance(lines_good, list)

            if len(lines_good)==1:
                # Check if this is a simple prim
                L = lines_good[0]
                if L["type"]=="stroke" and L["affine"]==self.FEATURES_DEFAULT:
                    return L["kind"]
            elif len(lines_good)==2:
                if lines_good[0]["kind"]=="arc":
                    if lines_good[0]["affine"] == self.program_get_tform_dict() \
                        and lines_good[1]["type"]=="transform" and lines_good[1]["affine"]==self.program_get_tform_dict(y=0.25, sx=0.5, sy=0.5, order='trs'):
                        return "arc"

                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(y=0.28867513)) and \
                    lines_good[1]["type"]=="repeat" and lines_good[1]["N"]==3 and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(th = 2.0943951, order="rst")):
                    return "triangle1"

                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(y=0.5)) and \
                    lines_good[1]["type"]=="repeat" and lines_good[1]["N"]==4 and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(th = 1.57079633, order="rst")):
                    return "square1"

            elif len(lines_good)==3:
                from math import pi

                if lines_good[0]["kind"]=="line" and lines_good[0]["affine"]==self.program_get_tform_dict(x=0.5) and \
                    lines_good[1]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(y=0.5, th=1.57079633, order="str")) and \
                    lines_good[2]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(sx=0.5, sy=0.5, order="rst")):
                    return "L"

                if lines_good[0]["kind"]=="curve" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict()) and \
                    lines_good[1]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(order="tsr")) and \
                    lines_good[2]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(y=-0.5, sx=0.71428571, sy=0.71428571, order="str")):
                    return "squiggle1"

            elif len(lines_good)==4:
                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(x=0.5)) and \
                    lines_good[1]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(x=0.5, th=0.78539816, order="srt")) and \
                    lines_good[2]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(x=0.20710678, y=0.70710678, order="str")) and \
                    lines_good[3]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[3]["affine"], self.program_get_tform_dict(x=-0.35355339, y=-0.35355339, order="rst")):
                    return "zigzag1"

            if fail_if_no_match:
                print("here")
                # print(lines_good)
                print(len(lines_good))
                [print(l) for l in lines_good]
                assert False, "failed to find match..."
            else:
                return None

        def _compose_transformations(lines_good):
            """ given multiple lines, extract final transformation, 
            INPUT:
            - lines_good, list of dicts
            RETURNS:
            - dict
            - 
            """
            assert isinstance(lines_good, list), "maybe you passed in dict?"
            if len(lines_good)==1:
                L = lines_good[0]
                return L["affine"]
            else:
                for l in lines_good:
                    print(l)
                assert len(lines_good)==1, "not yet coded for >1 lines, need to compose them"
            

        # 1) get all lines
        nlines = len(program[ind_subprog])
        lines_good = []
        for i in range(nlines):
            line = self.program_get_line(ind_subprog, i) 
            try:
                line_good = self.program_interpret_line(line)
            except:
                print(ind_subprog)
                print(i)
                print(self.Program)
                print(line)
                assert False, "why empty?"

            if line_good is not None:
                lines_good.append(line_good)

        # 2) Ask whether is object :
        # first N-1 lines combine to define the object
        # Last line defines transformation.
        if len(lines_good)==1:
            # move the affine to a new line
            lines_tform = [{}] 
            lines_tform[0]["type"] = "transform"
            lines_tform[0]["kind"] = "affine"
            lines_tform[0]["affine"] = lines_good[0]["affine"].copy()
            lines_obj = lines_good
            lines_obj[0]["affine"] = self.program_get_tform_dict() # use default (static)
        elif len(lines_good)==2:
            # then assume it's [obj, tform]
            lines_obj = lines_good[:1]
            lines_tform = lines_good[1:]
        elif lines_good[0]["kind"] in ["arc"] and lines_good[0]["type"]=="stroke": 
            # primitives are defined by 2 lines.
            lines_obj = lines_good[:2]
            lines_tform = lines_good[2:]
        elif len(lines_good)==4 and lines_good[0]["kind"]=="line" and lines_good[1]["kind"]=="line" and lines_good[2]["type"]=="transform":
            # primitives defined by 3 lines
            lines_obj = lines_good[:3]
            lines_tform = lines_good[3:]
        elif len(lines_good)==4 and lines_good[0]["kind"]=="curve":
            # could be squiggles.
            lines_obj = lines_good[:3]
            lines_tform = lines_good[3:]
        elif len(lines_good)==5 and self.Task["stage"] in ["_zigzag"]:
            lines_obj = lines_good[:4]
            lines_tform = lines_good[4:]
        elif len(lines_good)==6 and self.Task["stage"] in ["_zigzag2"]:
            lines_obj = lines_good[:5]
            lines_tform = lines_good[5:]
        elif len(lines_good)==3 and self.Task["stage"] in ["_triangle1", "_square"]:
            lines_obj = lines_good[:2]
            lines_tform = lines_good[2:]
        else:
            if fail_if_no_match:
                print(self.Task)
                [print(l) for l in lines_good]
                assert False, "coudl be either (1) obj needs multipel lines, like square, or (2) multiple lines of tform (e.g., global). solution: try both cases (where take last 1, 2, 3,,..) lines as tform. stop when get that first N-1 lines are a valid primitive"
            else:
                out = {
                    "obj":None,
                    "tform":None}
                return out

        # print(len(lines_good))
        # print(lines_tform)
        # asfdsaf
        # print('1')
        # print(lines_obj)
        # print(lines_tform)
        # print(self)
        # print('2')
        obj = _compare_to_primitive_templates(lines_obj)
        if len(lines_tform)>0:
            tform = _compose_transformations(lines_tform)
        else:
            tform = None

        # --- Return a dict
        out = {
            "obj":obj,
            "tform":tform}
        
        return out

    ####################### LATEST (FINAL) EXTRACTION OF OBJECTS/PLANS
    # Supercedes previous code on programs extract.
    # Here looks directly into "Plan" representation, which is the latest way that tasks
    # are defined in matlab side.
    def objectclass_extract_all(self, auto_hack_if_detects_is_gridlinecircle_lolli=True):
        """ Extract TaskNew.Objects into self.ObjectClass. Preprocesses to corerctly extract
        chunks, etc.
        - auto_hack_if_detects_is_gridlinecircle_lolli, bool, autmatocally fix issue with
        one day of code (concatenaitng) (see belw0.)
        RETURNS:
        - modifies self.ObjectClass
        """

        from pythonlib.tools.monkeylogictools import dict2list2
        
        if self.get_tasknew() is None:
            self.ObjectClass = None
            return 

        if "Objects" not in self.get_tasknew().keys():
            self.ObjectClass = None
            return

        Objects = self.get_tasknew()["Objects"]

        # get all keys, except explicitly ignored
        _list_keys_to_ignore = ['StrokesFinalNorm_', 'StrokesFinalNorm_Active', 'RuleStates', 
            'StrokesFinalSketchpadPix'] # ignore since redundant with elsewhere.
        _list_keys_to_ignore_2 = ["StrokesVisible"] # not used. e..g, some are in older versions of
        # dragmonkey, like StrokesVisible for gridlinecirce.
        _list_keys = [k for k in Objects.keys() if k not in _list_keys_to_ignore]

        # list_keys_check = ["StrokesFinalNorm", "StrokesMaskTouched", 
        #     "StrokesVisible", "StrokindsDone"]
        # print(Objects.keys())
        # for k in list_keys_check:
        #     if k in Objects.keys():
        #         print(k, Objects[k], type(Objects[k]))
        # assert False

        # Extract things
        dat = {}
        for k in _list_keys:
            dat[k] = dict2list2(Objects[k])

        # Process chunks into ChunksClassList
        from pythonlib.chunks.chunksclass import ChunksClassList
        for k in dat:
            if "Strokinds" in k:
                if type(dat[k]) == np.ndarray:
                    # print(dat[k])
                    # print(dat[k].shape)
                    if len(dat[k].shape) == 0:
                        dat[k] = [dat[k]]
                    else:
                        dat[k] = list(dat[k])
                        

        # Save number of strokes
        if "StrokesFinalNorm" in dat.keys():
            nstrokes = len(dat['StrokesFinalNorm'])
        elif "StrokindsDone" in dat.keys():
            nstrokes = len(dat['StrokindsDone'])
        elif "StrokesMaskTouched" in dat.keys():
            nstrokes = len(dat['StrokesMaskTouched'])
        elif "Seq" in dat.keys(): 
            nstrokes = len(dat["Seq"])
        else:
            for k, v in dat.items():
                print(k, ' -- ' , v)
            print(dat.keys())
            # print(dat["StrokesAll"])
            # print(dat["StrokesObj"])
            # print(dat["Seq"])
            assert False

        list_keys_check = ["StrokesFinalNorm", "StrokesMaskTouched", 
            "StrokindsDone"]
        for k in list_keys_check:
            if k in dat.keys():
                # print(k, dat[k], type(dat[k]))
                if dat[k] is not None and len(dat[k])>0:
                    # only if not empty
                    assert len(dat[k]) == nstrokes, "possibly somewhere concatenated strokes? find the _actual_ nstrokes"
        dat["actual_nstrokes"] = nstrokes
        
        ######## chunklist (included within task)
        if "ChunkList" in dat.keys():
            # Autoatmicalyl get CHunksList based on matlab tasksequencer mods.
            chunkslist = []
            if isinstance(dat["ChunkList"], dict):
                # shoudl be list of dicts...
                dat["ChunkList"] = [dat["ChunkList"]]

            for this in dat["ChunkList"]:
                # convert this item into the proper format
                # print(type(this))
                # print(type(dat["ChunkList"]))
                # print("adsad")
                # print(dat["ChunkList"])

                try:
                    flips = [x["Flipped"] for x in this["StrokeSequence"]]
                except Exception as err:
                    print(1)
                    print(dat)
                    print(2)
                    print(dat["ChunkList"])
                    print(3)
                    print(this)
                    print(4)
                    print(type(this))
                    print(this.keys)
                    raise err

                if "chunks_" in this.keys():
                    # Newer code, where chunks_ is just a readout.
                    hier = [x-1 for x in this["chunks_"]] # convert to 0-index. 
                elif "chunks" in this.keys():
                    # Old code, should be the samet hing
                    hier = [x-1 for x in this["chunks"]] # convert to 0-index
                else:
                    # oldest.
                    # this should be identical to chunks.
                    hier = []
                    for x in this["StrokeSequence"]:
                        tmp = x["Index"]-1 # array of ints
                        if tmp.size==1:
                            # is a nuber
                            hier.append([int(tmp)])
                        else:
                            # make into list of ints
                            hier.append([int(xx) for xx in tmp])
                
                # HACK!!    
                if auto_hack_if_detects_is_gridlinecircle_lolli and "chunks_" not in this.keys():
                    # (Checking if chunks_ exists is just hacky way of ehcking that this is older experiment)
                    # check if hier is like [[0,1], [2,3]] but nstrokes = 2, becuase in this code
                    # (matlab) I concatenated. After this expt, I did not do that...
                    # SOLUTION: convert hier to [0,1]
                    vals = []
                    for x in hier:
                        if isinstance(x, list):
                            for xx in x:
                                vals.append(xx)
                        else:
                            vals.append(x)
                    # print(this)
                    # print("nstrokes", dat["actual_nstrokes"])
                    # print("va;s", vals)
                    # print("hier", hier)
                    try:
                        max_index_in_hier = max(vals)
                    except Exception as err:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        self.plotTaskOnAx(ax)
                        fig.savefig("/tmp/test.pdf")
                        print(list(hier[1]))
                        raise err

                    if nstrokes < max_index_in_hier+1:
                        print("Tasks.objectclass_extract_all - HACKY FIX, only should happen for gridlinecirlce..")
                        hier = [i for i in range(nstrokes)]

                index = int(this["ind"])

                if len(flips)!=len(hier):
                    print(this)
                    assert False
                chunkslist.append([this["modelname"], hier, flips, index, this["color"]])

            # shapes = dat["Features_Active"]["shapes"] # these are the useless ones.
            CL = ChunksClassList("chunkslist_entry", 
                {"chunkslist":chunkslist, "nstrokes":nstrokes, 
                "shapes":None})
            dat["ChunksListClass"] = CL
        else:
            # Else, leave empty. older code (e.g,, gridlinecircle), to be 
            # replaced by GrammarDat post-hoc extraction of ChunksListClass
            dat["ChunkList"] = None
            dat["ChunksListClass"] = None
            dat["ChunkState"] = None 
        
        # Remove strokes, only needed above
        if "Strokes" in dat.keys():
            del dat["Strokes"]

        if len(dat["RuleFailureTracker"].shape)==0:
            # then is a scalar, put into an array.
            # shape from () --> (1,)
            dat["RuleFailureTracker"] = np.array([dat["RuleFailureTracker"]])

        # store
        self.ObjectClass = dat

    def objectclass_extract_active_chunk(self):
        """ Return the chunk (tasksequencer) that was active
        during matlab task, by looking into ObjectClass
        RETURNS:
        - NOne, if dosnt find
        - or a ChunksClass object
        """
        
        self.objectclass_extract_all()
        O = self.ObjectClass

        if O["ChunksListClass"] is None:
            return None
        else:
            # which are active chunk
            model = O["ChunkState"]["active_chunk_model"]
            index = O["ChunkState"]["active_chunk_ind"]

            # find it
            CLC = O["ChunksListClass"]
            C = CLC.find_chunk(model, index)
            return C

    def planclass_inputed_plan_extract(self, ind):
        """ GIven the final index of this prim (e.g,, after concatting)
        return the original inputted prims and rels (i.e, in the plan
        cell array) inputed to PlanClass() in matlab
        RETURNS:
        - prim_entry, list holding info for this prim
        - rel_entry, list, info for the rel that mods this prim
        """
        # Extract the original direct entry of this prim
        prim_entry = self.PlanDat["Plan"][2*ind]
        rel_entry = self.PlanDat["Plan"][1+2*ind]
        return prim_entry, rel_entry

    def planclass_inputed_plan_extract_novelprims(self):
        """ For each novelprim in this task, extract its low-level 
        prim entry
        RETURNS:
        - dict[index for prim]: prim_entry, where 
        prim_entry = 
            ['novel_prim',
             ['direct_chunk_entry',
              [['line', ['prot', array(9.), array(2.)]],
               ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
               ['line', ['prot', array(9.), array(3.)]],
               ['translate_xy', array(-1.), ['end2', 'end1', array([0., 0.])]],
               ['line', ['prot', array(9.), array(1.)]],
               ['translate_xy', array(-1.), ['end2', 'end1', array([0., 0.])]]]]]
        """

        # Find novelprims
        inds_novel = []
        ShapesAfterConcat = self.PlanDat["ShapesAfterConcat"]
        dict_novelprims = {}
        for i in range(len(ShapesAfterConcat)):
            if ShapesAfterConcat[i][:9]=="novelprim":
                inds_novel.append(i)
                # Extract the original direct entry of this prim

                prim_entry, _ = self.planclass_inputed_plan_extract(i)
                assert prim_entry[0] == "novel_prim", "sanity check, I assume this data structure"
                assert prim_entry[1][0] == "direct_chunk_entry", "sanity check, I assume this data structure"
                list_subprims = prim_entry[1][1][::2] # e..g, [['line', ['prot', array(9.), array(2.)]], ['line', ['prot', array(9.), array(3.)]], ['line', ['prot', array(9.), array(1.)]]]
                list_subrels = prim_entry[1][1][1::2] # e..g, [['translate_xy', array(0.), ['center', 'center', array([0., 0.])]], ['translate_xy', array(-1.), ['end2', 'end1', array([0., 0.])]], translate_xy', array(-1.), ['end2', 'end1', array([0., 0.])]]]

                dict_novelprims[i] = (list_subprims, list_subrels)
        return dict_novelprims

    def planclass_extract_all(self, DEBUG=False):
        """ Wrapper to extract all planclass info.
        Also extracts "Objects" based on plan (e.g,, motifs that are touching are 
        actually a single object)
        NOTE: if old version (no planclass) should fail gracefully, return None
        RETURNS:
        - dat, dict holding all plan things, including processed.
        -- or None, if no Plan dat
        - self.PlanDat will be replaced with dat
        - self.Objects modified.
        """
        from pythonlib.tools.monkeylogictools import dict2list2, dict2list
        from pythonlib.primitives.primitiveclass import PrimitiveClass

        if self.get_tasknew() is None:
            self.PlanDat = None
            return None

        if "Plan" not in self.get_tasknew().keys():
            self.PlanDat = None
            return

        Plan = self.get_tasknew()["Plan"]
        if len(Plan)==0:
            self.PlanDat = None
            return

        if True:
            # get all keys, except explicitly ignored
            _list_keys_to_ignore = ['PlanSpatial', 'StrokesBasePrims', 'CentersBasePrims', 
                'PrimsAsTasks', 'PrimsAsTasksNoMods', 'TaskClass', 'InputGridname', 'TaskGridname']
                # note: removing ChunksList since the updated one on each trial is actually
                # in Objects
            _list_keys = [k for k in Plan.keys() if k not in _list_keys_to_ignore]
        else:
            # tell what keys to get. this doesnt keep up with new versions
            _list_keys = ['Plan', 'CentersActual', 'Rels', 'Prims', 'ChunksList', 'Strokes']
            # older version didnt hold TaskGridClass
            if 'TaskGridClass' in Plan.keys():
                _list_keys.append('TaskGridClass')
        dat = {}
        # Extract things
        # _list_keys = {'Plan', 'CentersActual', 'Rels', 'TaskGridClass', 'Prims', 'ChunksList', 'Strokes'}
        for k in _list_keys:
            dat[k] = dict2list2(Plan[k])
        # Extract useful summaries
        # - (Base) prims and their locations
        # - Chunks
        # from pythonlib.chunks import ChunksClassList
        # CL = ChunksClassList():
        # dat["ChunksList"]
        # for chunk in dat["ChunksList"]:


        # Convert to hier and flips to list of list. This solves issue where they can be list of arrays with different dimensinoalitys, () or (1), which fails 
        # when try to index into array latter. Solve by converting all to list(array of shape (N,))
        # i.e., Chunkslist:
        # ChunksList [['default', [array(1.), array(2.), array(3.), array([4., 5., 6.])], [array(0.), array(0.), array(0.), array([0., 0., 0.])], {'color': [array([0.07714286, 0.07714286, 0.44214286]), array([0.655, 0.205, 0.205]), array([0.365, 0.   , 0.365]), array([0.53332996, 0.40376795, 0.44890529])]}, array([0., 0., 0., 1.])]]
        # convert this ([array(1.), array(2.), array(3.), array([4., 5., 6.])]) to :
        # ... [[1], [2], [3], [4,5,6]] ...
        # THis solves indexing problem that might arise below.
        # Fix so that each ch in chunks are same shape

        def _arr_to_list(x):
            if len(x.shape)==0:
                # array(1.) --> array([1])
                return np.array([int(x)])
            else:
                # array([1,2]) --> array([1,2])
                return np.array([int(xx) for xx in x])
        
        for chunkset in dat["ChunksList"]:
            chname = chunkset[0] # str
            hier = chunkset[1] # list of arr
            flips = chunkset[2]

            if DEBUG:
                print("Starting hier:", hier)
            hier = [_arr_to_list(x) for x in hier]
            flips = [_arr_to_list(x) for x in flips]
            if DEBUG:
                print("Ending hier:", hier)

            # Reassign
            chunkset[1] = hier
            chunkset[2] = flips

        #     for i, x in hier:

        #     print(hier)
        #     print(flips)
        #     assert False
        #     print("CHUNKS FOR ", chname)
        #         # print(x)
        #         # print(x.shape)
        #         # print(list(x))
        # assert False
        ############################################################
        # Extract shapes and their params
        dat["shapes"] = [x[0] for x in dat["Prims"]]      
        dat["primitives"] = []

        # Check whether prims were concatted.
        if len(dat["Prims"])==len(self.Strokes):
            # # dat["ChunksList"][0][4] is what triggers concatenatsion in dragmonkey. 
            # assert dat["ChunksList"][0][4]==0
            # NO concats!
            DID_CONCAT = False
            dat["CentersAfterConcat"] = dat["CentersActual"]

            # Get reflects. this is becuase i was stupid and didnt save them in 
            # Objectclass in dragmonkye. This should be correct here, as when
            # there is concats, then this doesnt mater.
            do_reflects = []
            for prim in dat["Prims"]:
                params = prim[1]
                if len(params)>=5:
                    reflect = params[4] # 1 means reflect.
                else:
                    reflect = np.array(0.)
                do_reflects.append(reflect)
            dat["ReflectsAfterConcat"] = do_reflects
            dat["ShapesAfterConcat"] = [p[0] for p in dat["Prims"]]
            dat["PrimsAfterConcat"] = dat["Prims"]
        else:
            # One itme per concat.
            DID_CONCAT = True

            # Extract features (just those that will use later on due to legacy code)
            # for each chunk.
            chunks =  dat["ChunksList"][0][1]

            centers = []
            do_reflects = []
            list_shapes = []
            list_prims = []

            if DEBUG:
                for k, v in dat.items():
                    print(k, v)
                print(dat["Prims"])
                print(len(self.Strokes))
                for ch in chunks:
                    print("here", ch)

            for ch in chunks:
                
                # Get the prim index for the first prim in this concat. many features are tied
                # to this first prim, by convention.
                idx_base1 = ch[0]
                idx_base0 = int(idx_base1 - 1)

                # center of the chunk, this is by default the center of the first stroke.
                cen = dat["CentersActual"][idx_base0]
                centers.append(cen)

                # Get reflects. this is becuase i was stupid and didnt save them in 
                # Objectclass in dragmonkye. This should be correct here, as when
                # there is concats, then this doesnt mater.
                prim = dat["Prims"][idx_base0]
                params = prim[1]
                if len(params)>=5:
                    reflect = params[4] # 1 means reflect.
                else:
                    reflect = np.array(0.)
                do_reflects.append(reflect)

                # get shapes
                # assert isinstance(ch, list), "the line len(ch)>1 assumes this is a list, such that len of 1 means singleton..."
                if len(ch)>1:
                    # then this is concatted. append "novel"
                    list_shapes.append("novelprim")
                    list_prims.append("novelprim")
                else:
                    list_shapes.append(prim[0])
                    list_prims.append(prim)

                # if len(ch)>1:
                #     # then this is concatted. append "novel"
                # else:

            dat["CentersAfterConcat"] = centers
            dat["ReflectsAfterConcat"] = do_reflects
            dat["ShapesAfterConcat"] = list_shapes
            dat["PrimsAfterConcat"] = list_prims
        assert len(dat["ReflectsAfterConcat"])==len(self.Strokes), 'bug above'
        assert len(dat["CentersAfterConcat"])==len(self.Strokes), 'bug above'

        # - nstrokes here should match nstrokes extracted elsewhere
        if False:
            _list_keys_check_nstrokes = ["Prims", "CentersActual", "Rels"]
            for k in _list_keys_check_nstrokes:
                assert len(dat[k]) == len(self.Strokes)

        # Get chunks in ChunksListClass format
        # NOTE on kinds of chunks, in order that they are applied in dragmonkey:
        # 1) chunks in dat["ChunksList"] which apply concat [index 4 equals 1]. Excluded
        # from extraction below.
        # 2) chunks in dat["ChunksList"] which define hierarchies. after around July 
        # 2022 I stopped using these. These are extracted below.
        # 3) Chunks defined in ObjectClass. This is good version. Extracted in
        # self.objectclass_extract_all()
        from pythonlib.chunks.chunksclass import ChunksClassList
        chunkslist = []
        for this in dat["ChunksList"]:
            # "this" represents all chunks for this task (e..g, "default" chunks)

            # print(len(this))
            # for x in this:
            #     print(x)
            # print(this[4])
            if len(this)>4 and np.any(this[4]==1):
                # Then skip, since this was used for concatting, and iwll not match the 
                # num strokes. 
                # .e.g, this[0:4] = 
                    # default
                    # [array([1]), array([2]), array([3]), array([4, 5, 6])]
                    # [array([0]), array([0]), array([0]), array([0, 0, 0])]
                    # {'color': [array([0.07714286, 0.07714286, 0.44214286]), array([0.655, 0.205, 0.205]), array([0.365, 0.   , 0.365]), array([0.53332996, 0.40376795, 0.44890529])]}
                    # [0. 0. 0. 1.]
                # The problem is that this refers to strokes 4 5 and 6, which don't exist now, since they were already cocnated, (ie..
                # 4,5,6 were the matlab instructiuons there for how to concat.

                continue
            ch = []

            # just giving variables here note-taking purpose

            # 1) name of model
            ch.append(this[0]) 

            # 2) hier (chunks)
            hier = list(this[1])
            ch.append([ch-1 for ch in hier]) # - change to 0-index

            # 3) flips
            ch.append(this[2]) # usually [0, 0, ...]

            # 4) index
            ch.append(None) # give it an index... (None, since only one. objectclass uses ints)
            
            # 5) color
            ch.append(this[3]["color"]) 
            # Note: color is actually better in objectclass, acurate online)

            chunkslist.append(ch)

        # if DOPRINT:
        #     print("HERE", chunkslist)
        #     for c in chunkslist:
        #         print(c)
        #     assert False

        del dat["ChunksList"]

        # Genreate default chunks, in case therea re no chunks for som reason. eg., all concats?
        nstrokes = len(self.Strokes)
        if len(chunkslist)==0:
            col_default = np.array([0.5, 0.5, 0.5])
            chunkslist.append([
                "default", 
                [i for i in range(nstrokes)], 
                [0 for _ in range(nstrokes)],
                None,
                [col_default for _ in range(nstrokes)],
            ])


        # ChunksClassList
        # CL = ChunksClassList("chunkslist_entry", 
        #     {"chunkslist":chunkslist, "nstrokes":nstrokes, 
        #     "shapes":dat["shapes"]})
        CL = ChunksClassList("chunkslist_entry", 
            {"chunkslist":chunkslist, "nstrokes":nstrokes})
        dat["ChunksListClass"] = CL
        
        # Remove strokes, only needed above
        if "Strokes" in dat.keys():
            del dat["Strokes"]

        # Centers of the first strokes in each chunk(hier)
        # Use the first chunk, by convnetion
        # NOTE: wont be accurate forthat one glc lolli day
        Ch = dat["ChunksListClass"].ListChunksClass[0]
        centers = dat["CentersAfterConcat"]
        centers = [centers[h[0]] for h in Ch.Hier] # take first stroke in each chunk(hier)
        dat["CentersAfterConcat_FirstStrokeInChunk"] = centers

        # store
        self.PlanDat = dat

        # NOVEL PRIMS
        # If any novel prims, assign them hash (unique shape)
        dict_novel_prims = self.planclass_inputed_plan_extract_novelprims()
        from pythonlib.tools.listtools import stringify_list

        if False: # I tried this, but never implemented, since I realied this
            #  doesnt work, since you need to run this inputing all tasks for the day. 
            # strategy, check that all hash unique. if so, then just use index...
            map_hashnum_indprim = {}
            for indprim, params in dict_novel_prims.items():
                list_subprims, list_subrels = params[0], params[1]
                hashnum = hash(tuple(stringify_list(list_subprims) + stringify_list(list_subrels)))
                if hashnum in map_hashnum_indprim:
                    map_hashnum_indprim[hashnum].append(indprim)
                else:
                    map_hashnum_indprim[hashnum] = [indprim]
            
            # Then, give each unique prim its unqiue index
            map_indprim_indnovel = {}
            for inovel, (hashnum, indprims) in enumerate(map_hashnum_indprim.items()):
                for iprim in indprims:
                    map_indprim_indnovel[iprim] = inovel

        for indprim, params in dict_novel_prims.items():
            list_subprims, list_subrels = params[0], params[1]
            _code = tuple(stringify_list(list_subprims) + stringify_list(list_subrels))
            if False:
                # PROBLEM: hash is not determistic across Python runs
                hashnum = hash(_code)
                hashnum += sys.maxsize + 1 # To make it positive (important, to avoid dash between novelprim and hash, which will be parsed incorrectly when doign novelprim-x-x-x)
            else:
                # Use task as string to seed.. this is detemrinstic.
                import random
                _code = "".join(_code) # convert to stroiung
                r = random.Random(_code); # seed with this unique string
                ndigs = 12
                hashnum = int("".join([str(r.randrange(10)) for _ in range(ndigs)]))

            assert hashnum>=0

            shapenew = f"novelprim{hashnum}"

            assert dat["ShapesAfterConcat"][indprim][:9]=="novelprim"
            dat["ShapesAfterConcat"][indprim] = shapenew

        # GENERATE PRIMS
        # for i, (shapenew, prim, loc) in enumerate(zip(dat["ShapesAfterConcat"], dat["Prims"], dat["CentersActual"])):
        #     shape = prim[0]

        #     if not shape==shapenew:
        #         # shapenew should only be different if this is a novel prim,m it is novelprim-<hashnum>
        #         print(shape, shapenew)
        #         assert shapenew[:9]=="novelprim", "shape or shapenew, which is correct?"
        for i, (prim, loc) in enumerate(zip(dat["Prims"], dat["CentersActual"])):
            shape = prim[0]
            params = prim[1]
            primkind = params[0] # e.g., prot, abstract, motif
            scale = params[1]
            rotation = params[2]
            if len(params)>=4:
                col = params[3] # not using here... (color)
            else:
                col = np.nan
            if len(params)>=5:
                reflect = params[4] # 1 means reflect.
            else:
                reflect = np.array(0.)

            # if i>len(self.Strokes)-1:
            #     for k, v in dat.items():
            #         print("---")
            #         print(k)
            #         print(v)
            #     print(dat["Prims"])
            #     print("===", params)
            #     print(len(self.Strokes))
            #     print(i)
            #     print("****************8")
            #     self.objectclass_extract_all()
            #     print(self.ObjectClass)
            #     assert False

            if True:
                # Stop collecting traj, beacuse self.Strokes is after concating, while
                # i iterates over each base prim (before concat). Look in to ObjectClass for
                # infor about each stroke.

                # look into objectClass for strokes.
                traj = None
            else:
                traj = self.Strokes[i]

            assert params[0]=="prot", "I assume everything in dat[prims] is baseprim..."
            # tform = {"x":loc[0], "y":loc[1], "th":rot, "sx":scale, "sy":scale, "order":

            Prim = PrimitiveClass()
            Prim.input_prim("prototype_prim_abstract", {
                    "shape":shape,
                    # "shape":shapenew,
                    "scale":scale,
                    "rotation":rotation,
                    "reflect":reflect,
                    "x":loc[0],
                    "y":loc[1]}, 
                    traj = traj)
            dat["primitives"].append(Prim)

        # Sanity checks
        # - Base prims should correspond to objects
        if False:
            if self.Objects is not None:
                for o, p in zip(self.Objects, dat["primitives"]):
                    if False:
                        # actually these could differ, since obj can be L while p.Shape is more accurately Lcentered
                        assert o["obj"] == p.Shape
                    assert np.isclose(o["tform"]["x"], p.ParamsConcrete["x"])
                    assert np.isclose(o["tform"]["y"], p.ParamsConcrete["y"])

        return dat

    ############ TASKSETCLASS STUFF
    def tasksetclass_summary(self):
        """ Return summayr of the tasksetclass used in dragmonkey to generate this
        task. only appliues for newer tasks. 
        These are general params used for the TSC, _NOT_ params speciric to this task.
        RETurns dict holding TSC params.
        """
        if self.info_is_new_objectclass():
            # TSC = self.PlanDat["Info"]["TaskSetClass"]["tsc_params"]["quick_sketchpad_params"]
            if len(self.PlanDat) == 0:
                return None
            # print(self.PlanDat)
            # print(self.PlanDat.keys())
            #print(self.PlanDat["Info"])
            #print(self.PlanDat["Info"].keys())
            TSC = self.PlanDat["Info"]["TaskSetClass"]
            return TSC
        else:
            return None

######################################## OTHER STUFF
def task2chunklist(task):
    """
    """

    chunks = task["TaskNew"]["Task"]["chunks"]

    # convert dict
    chunks = [v for v in chunks.values()]
    models = chunks[::2]
    stroke_assignments = chunks[1::2]
#     stroke_assignments = [s for s in stroke_assignments]

    chunkdict = {m:[] for m in models} 
    for m, s in zip(models, stroke_assignments):
        chunkdict[m].append([[int(vv[0]-1) for vv in v] for _, v in s.items()]) # append, since a model may have multiple chunks
    return chunkdict


def convertTask2Strokes(task, concat_timesteps=False, interp=None, fake_timesteps=None):
    """ given one task object, converts to a strokes (list of nparary)
    - concat_timesteps, then added 0,1,2,3 ... as a third dimension, (T x 3 output)
    - splits into multiple strokes based on assumption that NDOTS (points in a stroke)
    - 7/16/20 - FIXED so that accurately splits into strokes, instead of asuming that NDOTS is 
    15. Now doesn't use NDOTS, instead uses onsets saved by matlab.
    """

    if "strokes" in task.keys():
        return task["strokes"]

    if False:
        # OLD: does not split into multiple strokes.
        strokes = np.concatenate((task["x_rescaled"], task["y_rescaled"]), axis=0).T
        if concat_timesteps:
            strokes = np.concatenate((strokes, np.arange(strokes.shape[0]).reshape(-1,1)), axis=1)
            strokes = [strokes]
        if not interp is None:
            from pythonlib.tools.stroketools import strokesInterpolate
            strokes = strokesInterpolate(strokes, N=interp)
        return strokes
    else:

        strokes_flat = np.concatenate((task["x_rescaled"], task["y_rescaled"]), axis=0).T
        strokes = []

        if False:
            # -- split by NDOTS
            NDOTS = 15 # TODO: replace this with TrialRecord.User.GENERAL... (now saving this param)
            assert(len(strokes_flat)%NDOTS==0), "assumption of 15 dots is wrong?"
            # OLD METHOD - using hard codede NDOTS
            edges = np.linspace(0, len(strokes_flat), len(strokes_flat)/NDOTS+1)
            for e1, e2 in zip(edges[:-1], edges[1:]):
            #     print([e1, e2])
                strokes.append(strokes_flat[int(e1):int(e2),:])
        else:
            # NEW MOETHOD - using "onsets", which is coded in MATLAB.
            onsets = task["onsets"] # e..g, [1, 16, ...]
            if len(onsets)==0:
                onsets = [1]
            onsets = [int(o-1) for o in onsets]
            onsets.append(len(strokes_flat))
            for o1, o2 in zip(onsets[:-1], onsets[1:]):
                # print(o1)
                # print(o2)
                # print(strokes_flat)
                strokes.append(strokes_flat[o1:o2])

        # -- append fake timesteps
        strokes = fakeTimesteps(strokes, [], "in_order")
         
        # -- interpolate
        if not interp is None:
            from pythonlib.tools.stroketools import strokesInterpolate
            strokes = strokesInterpolate(strokes, N=interp)       

        if fake_timesteps=="from_orig":
            orig = getTrialsFix(filedata, trial)["fixpos_pixels"]
            strokes = fakeTimesteps(strokes, point=orig, ver="from_point")
        elif not fake_timesteps is None:
            assert False, "dont know this one"
        else:
            strokes = fakeTimesteps(strokes, point=[], ver="in_order")


        return strokes

# def chunklist2parses(chunklist, strokes, model, default="eachstroke"):
#     """ for this chunklist and model, extract
#     each way to chunking strokes. e.g.
#     [[0,1], 2] leads to chuning of strokes 0 and 1
#     Returns one strokes object for each way of chunking.
#     (note, can do:
#     chunklist = getTrialsTaskChunks(...))
#     - NOTE: 3rd dim (time) might not make sense).
#     - parses_list is len of num chunks, each element a strokes.
#     - default, what to do if dont find chunk? 
#     --- eachstroke, then will treat each stroke as chunk,
#     """
#     if model not in chunklist.keys():
#         if default=="eachstroke":
#             # then each stroke is a chunk
#             chunks = [[[i] for i in range(len(strokes))]]
#     else:
#         chunks = chunklist[model]


#     parses_list = []
#     for c in chunks:
#         strokesnew = [np.concatenate([strokes[i] for i in s], axis=0) for s in c]
#         parses_list.append(strokesnew)

#     # === remove temporal inforamtion from parses (since innacuarte)
#     parses_list = [[strok[:,[0,1]] for strok in strokes] for strokes in parses_list]

#     return parses_list


# def task2parses(task, model):
#     """ given task and model, get parses_list,
#     which is list of strokes, each strokes a 
#     permutation of strokes consistent with chuking for 
#     model in drawmodel.analysis."""
#     from tools.tasks import task2chunklist, convertTask2Strokes
#     from pythonlib.tools.stroketools import getAllStrokeOrders
    

#     chunklist = task2chunklist(task)
#     strokestask = convertTask2Strokes(task)
#     parses_list = chunklist2parses(chunklist, strokestask, model)

#     # === get all permutations for parses (output will be list of strokes)
#     parses_allperms = []
#     for strokes in parses_list:
#         parses_allperms.extend(getAllStrokeOrders(strokes)[0])

#     # == convert to format model wants
#     parses = []
#     for strokes in parses_allperms:
#         orders = None
#         parses.append({
#             "strokes":strokes, # is not using extra memory.
#             "order":orders,
#             })
#     return parses


def flatten_hier(hier):
    """ returns [x,y , ..., ..
    Where x, y, .. are in hierarchy of same shape as chunks (see
    chunk_strokes)
    """
    out = []
    for x in hier:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def chunklist2chunks(chunklist, strokes, model, default="eachstroke"):
    """ for this chunklist and model, extract
    each way to chunking strokes. e.g.
    [[0,1], 2] leads to chuning of strokes 0 and 1
    Returns one strokes object for each way of chunking.
    - default, what to do if dont find chunk? 
    --- eachstroke, then will treat each stroke as chunk,
    OUT:
    --- e.g,, chunks = [
        [[0, 1], 2], 
        [[0, 2], 1]]
    """
    if model not in chunklist.keys():
        if default=="eachstroke":
            # then each stroke is a chunk
            chunks = [[[i] for i in range(len(strokes))]]
    else:
        chunks = chunklist[model]
    return chunks

def task2parses(task, model):
    """ given task and model, get parses_list,
    which is list of strokes, each strokes a 
    permutation of strokes consistent with chuking for 
    model in drawmodel.analysis.
    - model, either string, or a function: task--> chunks"""

    # from tools.tasks import task2chunklist, convertTask2Strokes
    from pythonlib.tools.stroketools import getAllStrokeOrders
    strokestask = convertTask2Strokes(task)
    
    if isinstance(model, str):
        chunklist = task2chunklist(task)
        chunks = chunklist2chunks(chunklist, strokestask, model)
    elif callable(model):
        # then model is a fucntion
        chunks = model(task)
    else:
        print(model)
        assert False, "not sure what is"
        
    parses_list = chunks2parses(chunks, strokestask)
    print(chunks)
    print(parses_list)
    assert False

    # === get all permutations for parses (output will be list of strokes)
    parses_allperms = []
    for strokes in parses_list:
        parses_allperms.extend(getAllStrokeOrders(strokes)[0])

    # == convert to format model wants
    parses = []
    for strokes in parses_allperms:
        orders = None
        parses.append({
            "strokes":strokes, # is not using extra memory.
            "order":orders,
            })
    return parses



def _get_task_probe_info(task):
    """
    task is not class, is dict from ml2.
    Replaces getTrialsTaskProbeInfo in drawmonkey.utils, since this is used both here (pythonlib)
    and there.
    NOTE:
    part of this will not work for old tasks (long time ago) since needs getTrialsBlockParamsHotkeyUpdated.
    See below. If needed, import that 
    """

    if "constraints_to_skip" not in task.keys():
        # Just putting None, since I don't use this ever (only used a long time ago, and those
        # cases I am not analyzing using Dataset). See drawmonkey.utils.getTrialsTaskProbeInfo if 
        # want to extract this.
        co = None
        # # then this was when I used general version in block params
        # assert False, "need to bring in getTrialsBlockParamsHotkeyUpdated somehow"
        # if "constraints_to_skip" not in getTrialsBlockParamsHotkeyUpdated(filedata, trial)["probes"]:
        #     co = ""
        # else:
        #     co = getTrialsBlockParamsHotkeyUpdated(filedata, trial)["probes"]["constraints_to_skip"]
    else:
        co = task["constraints_to_skip"]

    # === get task number.
    # This identifies task if is prototype or savedsetnum.
    # This may not identify, if is random task, or resynthesized task.
    taskcat = task["TaskNew"]["Task"]["stage"]
    taskstr = task["TaskNew"]["Task"]["str"]
    idx = taskstr.find(taskcat)
    if idx<0:
        # COuld be because is hybrid
        idxthis = taskcat.find("-")
        if idxthis>0:
            # taskcat = <cat1>-<cat2>
            # taskstr = <cat1>_num1-<cat2>_num2
            assert len([x for x in taskcat if x=="-"])==1, "multiple hyphens..."

            cat1 = taskcat[:idxthis]
            cat2 = taskcat[idxthis+1:]

            idxhyphen = taskstr.find("-")
            assert len([x for x in taskstr if x=="-"])==1, "multiple hyphens..."

            num1 = taskstr[len(cat1)+1:idxhyphen]
            num2 = taskstr[idxhyphen+len(cat2)+2:]

            # new num is just concatenate these nums
            tasknum = int(num1 + num2)

            # print(taskcat, taskstr, cat1, cat2, num1, num2, tasknum)
            # assert False

        else:
            print(taskstr)
            print(taskcat)
            print(idx)
            assert False, "tascat not in taskstr"
    else:
        # then is good
        tasknum = int(taskstr[idx+len(taskcat)+1:])


    if "TaskNew" not in task.keys():
        p = 0
        saved_setnum = []
    else:
        INFO = task["TaskNew"]["Task"]["info"]
        try:
            if "prototype" not in INFO.keys():
                p = 0
            else:
                p = int(INFO["prototype"][0][0])
        except Exception as err:
            print(err)
            print(task)
            print(task['TaskNew'])
            print("This is old task version. how to handle?")
            assert False

        if "saved_setnum" in INFO.keys():
            if len(INFO["saved_setnum"])==0:
                saved_setnum = None
            else:    
                saved_setnum = int(INFO["saved_setnum"][0][0])
        else:
            saved_setnum = None

        # NEwer vesrion, directly saving load old set
        if "load_old_set_setnum" in INFO.keys():
            if len(INFO["load_old_set_setnum"])>0:
                los_setname = INFO["load_old_set_ver"]
                los_setnum = int(INFO["load_old_set_setnum"][0])
                los_setinds = [int(x) for x in INFO["load_old_set_inds"][0]]
                if "load_old_set_indthis" in INFO.keys():
                    los_setindthis =  int(INFO["load_old_set_indthis"][0])
                    # assert los_setindthis==los_setinds[tasknum]
                    tasknum = los_setindthis
                else:
                    los_setindthis = None

                # Replace old indices
                saved_setnum=los_setnum
                tasknum = los_setindthis 
            else:
                los_setname = None
                los_setnum = None
                los_setinds = None
                los_setindthis = None              
        else:
            los_setname = None
            los_setnum = None
            los_setinds = None
            los_setindthis = None

    # was this resynthesized?
    resynthesized = 0
    rpath = None
    rtrial = None
    rsetnum = None
    rsetname = None
    if "savedTaskSet" in task.keys():
        if len(task["savedTaskSet"])>0:
            if task["savedTaskSet"]["reloaded"][0][0]==1:
                resynthesized = 1
                rpath = task["savedTaskSet"]["path"]
                rtrial = int(task["savedTaskSet"]["trial"][0][0])

                idx = task["savedTaskSet"]["path"].find("set")
                rsetnum = int(task["savedTaskSet"]["path"][idx+3:])
                rsetname = task["savedTaskSet"]["path"][:idx-1]
    else:
        resynthesized = 0
        rpath = None
        rtrial = None
        rsetnum = None
        rsetname = None


    if "feedback_ver_prms" not in task.keys():
        fp = None
    else:
        fp = task["feedback_ver_prms"]


    probe = {
        "probe":task["probe"][0][0], 
        "feedback_ver":task["feedback_ver"], 
        "feedback_ver_prms":fp,
        "constraints_to_skip":co, 
        "prototype":p,
        "saved_setnum":saved_setnum,
        "tasknum":tasknum, 
        "resynthesized":resynthesized, 
        "resynthesized_path":rpath, 
        "resynthesized_trial":rtrial, 
        "resynthesized_setnum":rsetnum, 
        "resynthesized_setname":rsetname, 
        "los_setname": los_setname,
        "los_setnum": los_setnum,
        "los_setinds": los_setinds,
        "los_setindthis":los_setindthis}

    return probe