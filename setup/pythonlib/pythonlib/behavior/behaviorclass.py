""" 
Each instance is a single beh trial.
Consolidating the following:
- beh labeling, discretization, and processing
- motor timing, etc.
- lots of stuff that already done in Dataset, but here inlcude all the deeper analyses.
- and so on...
"""
import numpy as np
import matplotlib.pyplot as plt

class BehaviorClass(object):
    """docstring for BehaviorClass"""
    
    def __init__(self, params, ver="dataset", reset_tokens=False):
        """ 
        PARAMS:
        - params. Different methods (ver) for params entry
        if ver =="dataset", then enter a single row in Dataset object.
        params["ind"], int, row in Dataset
        params["dataset"], Dataset instance
        if ver=="raw", then enter all these things by hand.
        params["strokes_beh"]
        params["strokes_task"]
        params["TaskClass"]
        In any case, can have the following keys:
        -- "shape_dist_thresholds", value is dict, specifying max distance for calling something a given label.
        - reset_tokens, bool, if True, then deletes the tokens in Task, so that you reset it.
        """

        # self.Dataset = None
        self.Datrow = None
        # self.IndDataset = None
        self.Expt = None
        self.Strokes = None # holds strokes as strokeclass
        self.StrokesVel = None
        self.StrokesOrdinal = None
        self._TokensLocked = False

        if ver=="dataset":
            if False:
                # no need to cpy, since not storing it
                D = params["D"].copy()
            else:
                D = params["D"]
                
            ind = params["ind"]
            Task = D.Dat.iloc[ind]["Task"]

            if reset_tokens:
                Task._tokens_delete()

            assert "expt" in params.keys() and isinstance(params["expt"], str), "you really should pass in the expt, becaause it can define diff steps in parts of the code -- e.g,, alignsim_extract_datsegs for gridlinecircle."
            # if "expt" in params.keys():]
            self.Expt = params["expt"]
            strokes_task = D.Dat.iloc[ind]["strokes_task"]

            # 2) Get the "names" for all task strokes
            list_shapes = [s[0] for s in Task.Shapes]
            assert(len(list_shapes)==len(strokes_task))
            assert(all([isinstance(s, str) for s in list_shapes]))

            # 3) Compute best alignemnt with beh
            from pythonlib.tools.stroketools import assignStrokenumFromTask
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            
            # if False:
            #     strokes_beh = [np.copy(strokes_beh[0]), np.copy(strokes_beh[1])]
            # else:
            #     strokes_beh = [np.copy(s) for s in strokes_beh] + [np.copy(strokes_beh[0]), np.copy(strokes_beh[1])]
            if False:
                inds_task = assignStrokenumFromTask(strokes_beh, strokes_task, ver="stroke_stroke_lsa_usealltask")

            # 4) Plot 
            # D.plotSingleTrial(ind);
            # # reshuffle task strokes then plot in order
            if False:
                strokes_task_aligned = [strokes_task[i] for i in inds_task]
                D.plotMultStrokes([strokes_beh, strokes_task, strokes_task_aligned]);

            # Try abnother version
            inds_task, list_distances = assignStrokenumFromTask(strokes_beh, strokes_task, 
                ver="each_beh_stroke_assign_one_task")

            # 6) For each beh stroke, give it a label
            labels = []
            for i in inds_task:
                if len(i)==1:
                    labels.append(list_shapes[i[0]])
                elif len(i)==2:
                    # then concatenate into a string
                    tmp = sorted([list_shapes[ii] for ii in i])
                    labels.append("-".join(tmp))

            # Timing distributions
            motordat = D.get_motor_stats(ind)

            # - gap durations
            if False:
                gap_durations = np.array(motordat["ons"])[1:] - np.array(motordat["offs"])[:-1]
                stroke_durations = np.array(motordat["offs"]) - np.array(motordat["ons"])

            ##### MASKS
            # mask, to pick out only the first sucesful getting of each prim.
            inds_task_done_sofar = []
            inds_task_mask = []
            for indthis in inds_task:
                if all([i in inds_task_done_sofar for i in indthis]):
                    # then this is not new
                    inds_task_mask.append(False)
                else:
                    # then you got something new
                    inds_task_mask.append(True)
                    inds_task_done_sofar.extend(indthis)

            # mask, true only if beh stroke matches only a single task stroke
            inds_task_single = np.ndarray(len(inds_task))
            for i, val in enumerate(inds_task):
                if len(val)>1:
                    inds_task_single[i] = np.nan
                else:
                    inds_task_single[i] = val[0]
            maskinds_single = ~np.isnan(inds_task_single)
            if False:
                print("here")
                print(inds_task)
                print(inds_task_single)
                print(maskinds_single)
                print(type(maskinds_single))

            # mask based on distance to match. if worse then threshold, then mask this off
            if False:
                shape_dist_thresholds = params["shape_dist_thresholds"]
                mask_shapedist = []
                for sh, dist in zip(labels, list_distances):
                    if isempty(dist):
                        # then always allow this to pass
                        mask_shapedist.append(True)
                    else:
                        if dist>shape_dist_thresholds[sh]:
                            mask_shapedist.append(False)
                        else:
                            mask_shapedist.append(True)
                mask_shapedist = np.array(mask_shapedist)
            else:
                if "shape_dist_thresholds" in params.keys():
                    assert False, "this not yet coded"
                mask_shapedist = None


            self.Dat = {
                "strokes_task":strokes_task,
                "shapes_task":list_shapes,
                "strokes_beh":strokes_beh,
                "taskstrokeinds_beh":inds_task, # list of list
                "taskstrokeinds_dists":list_distances.reshape(-1,), # nparray
                "taskstrokeinds_beh_singleonly":inds_task_single, # np array, either single task ind, or nan
                "shapes_beh":labels,
                # "gap_durations":gap_durations,
                # "stroke_durations":stroke_durations,
                # "maskinds_repeated_beh": np.array(inds_task_mask, dtype=int)
                "maskinds_repeated_beh": np.array(inds_task_mask), # true if this beh mathces taskind that has not been gotetn yuet
                "maskinds_singlematch_beh": maskinds_single, # True if this beh matches only one task stroke
                "maskinds_shapedist_beh": mask_shapedist, #
                "stroke_dists": np.array(motordat["dists_stroke"]),
                "gap_dists": np.array(motordat["dists_gap"]),
                "trialcode":D.Dat.iloc[ind]["trialcode"],
                "character":D.Dat.iloc[ind]["character"]}

            # self.Dataset = D.copy() # must copy, or else if modify D, the ind doesnt match anymore.
            if True:
                # stop saving dat. this forces me to copy it whch takes a while
                self.Task = D.Dat.iloc[ind]["Task"]
            else:
                self.Datrow = D.Dat.iloc[ind]

            #### easier access to some variables
            # self.Strokes = self.Dat["strokes_beh"]
            from .strokeclass import StrokeClass
            self.Strokes = [StrokeClass(S) for S in self.Dat["strokes_beh"]]
            self.MotorStats = motordat

        else:
            assert False, "not coded"

    ##################### SHAPES, LABELS, etc
    def shapes_nprims(self, ver="task", must_include_these_shapes=None):
        """ count how oftne each shape occurs in the beh or task.
        PARAMS:
        - ver, whether to use "beh" or "task" shapes.
        - must_include_these_shapes, can be 0.
        RETURNS:
        - dict, num times each shape occured. e.g., {"circle":5, ...}
        """

        if must_include_these_shapes is None:
            must_include_these_shapes =["circle", "line"]

        out = {}
        for sh in must_include_these_shapes:
            out[sh] = 0

        if ver=="task":
            list_shapes = self.Dat["shapes_task"]
        elif ver=="beh":
            list_shapes = self.Dat["shapes_beh"]
        else:
            assert False

        for sh in list_shapes:
            if sh in out.keys():
                out[sh]+=1
            else:
                out[sh]= 1
        return out


    ##################### MOTOR STATS
    def motor_extract_durations(self, list_inds, ver="stroke"):
        """ Extract either gap or stroke durations, given a list of beh stroke inds.
        PARAMS:
        - ver, str, either "stroke" or "gap"
        - list_inds, list of ints. interpretations depends on ver.
        --- if ver=="stroke", returns list of stroke durs. straighforwad
        --- if ver=="gap", then returns gaps between adjacent inds. asserts that inds are monotinic
        incresaing. e..g, [1,2,3], then returns gap betwen [1-2, 2-3]
        RETURNS:
        - array_durs, see above.
        """

        if ver=="stroke":
            return self.Dat["stroke_durations"][list_inds]
        elif ver=="gap":
            assert(np.all(np.diff(list_inds)==1.)), "need to be monotinoc increasing"
            return self.Dat["gap_durations"][list_inds[:-1]]

    def motor_extract_dists(self, list_inds, ver="stroke"):
        """ Extract dists in pixels for stroke or gaps. See 
        motor_extract_durations for params
        """
        if ver=="stroke":
            return self.Dat["stroke_dists"][list_inds]
        elif ver=="gap":
            assert(np.all(np.diff(list_inds)==1.)), "need to be monotinoc increasing"
            return self.Dat["gap_dists"][list_inds[:-1]]


    ##################### HELPERS TO EXTRACT
    def extract_strokes_ordinal(self, cache=True):
        """ For each timepoint in strokes, assigns it ordinal (0,1,2,..)
        # By convention, during gap will return ordinal for the preceding stroke.
        # starts at 0 (before first stroke). So betweeen onset of first and 2nd stroke
        # is 1.
        RETURNS:
        - strokes_ordinal, list of np array, each (N,2), with col 1 ordina, col2 times ( same as strokes);
        - saves in self.StrokesOrdinal, if cache==True
        """

        if self.StrokesOrdinal is None:
            motor = self.MotorStats

            # Funct applies to single timpoint. 
            onsarray = np.array(motor["ons"])
            def fun(x):
                """
                Counts how many "ons" this time (x) is chronoltically after
                """
                return np.sum(x >= onsarray)
            vfun = np.vectorize(fun)

            strokes_ordinal = []
            for s in self.Strokes:
                times = s[:,2]
                strokes_ordinal.append(np.c_[vfun(times), times])

            if cache:
                self.StrokesOrdinal = strokes_ordinal
        else:
            strokes_ordinal = self.StrokesOrdinal
        return strokes_ordinal



    def extract_strokes_vels(self, Dataset, IndDataset, cache=True):
        """ Extract stroke velocities using defgault params
        PARAMS:
        - Dataset, uses this to get fs
        - IndDataset, index into Dataset (for this trial)
        - cache, bool (True), if Ture, saves in self.StrokesVel
        RETURNS:
        - strokes_vels, same shape as strokes.
        NOTE: only runs if self.StrokesVel is None
        """
        assert False, 'input trialcode, not ind'

        if self.StrokesVel is None:
            from pythonlib.tools.stroketools import strokesVelocity
            # fs = self.Dataset.get_sample_rate(self.IndDataset)
            fs = Dataset.get_sample_rate(IndDataset)
            strokes_vel, strokes_speeds = strokesVelocity(self.Strokes, fs, clean=True)
            if cache:
                self.StrokesVel = strokes_vel
        else:
            strokes_vel = self.StrokesVel
        return strokes_vel

    def extract_strokes(self, ver="beh", list_inds=None):
        """ Extract strokes for this trial.
        PARAMS:
        - ver, str, from {'beh', 'task'}
        - list_inds, 
        --- None, then returns entire strokes
        --- list of ints, then returns just those trajectories in strokes. in the order
        provided
        RETURNS:
        - strokes, list of np arrays
        """

        if ver=="beh":
            strokes = self.Dat["strokes_beh"]
        elif ver=="task":
            strokes = self.Dat["strokes_task"]
        elif ver=="task_after_alignsim":
            # THIS IS first touch
            strokes = self.Dat["strokes_task"]
            idxs = self.Alignsim_taskstrokeinds_sorted
            strokes = [strokes[i] for i in idxs]
        else:
            assert False

        if list_inds is None:
            return strokes
        else:
            return [strokes[i] for i in list_inds]

    def extract_taskstrokes_aligned_to_this_beh(self, ind, concat_task_strokes=False):
        """
        PARAMS:
        - ind, int, index into beh strokes
        - concat_task_strokes, then returns a single traj (concatted). might not be in a
        a good order...
        RETURNS:
        - inds_task, list of ints, the task inds matching this beh stroke
        - strokes_task, the strokes matching inds_task
        - dist, distance for this match (higher the worse)
        """

        taskinds = self.Dat["taskstrokeinds_beh"]
        taskinds_this = taskinds[ind]

        strokes_task = self.Dat["strokes_task"]
        strokes_task = [strokes_task[i] for i in taskinds_this]

        distances = self.Dat["taskstrokeinds_dists"]
        distances = distances[ind]

        if concat_task_strokes:
            strokes_task = [np.concatenate(strokes_task, axis=0)]


        return taskinds_this, strokes_task, distances

    #################### Task stuff
    def task_extract(self):
        """ Return the TaskClass object
        """

        # print(self.Dataset)
        if False:
            return self.Datrow["Task"]
        else:
            return self.Task



    ###################### FIND THINGS
    def find_labelmotif(self, labelmotif, ver="shapes", good_strokes_only=False):
        """ find cases where this labelmotif is present in this beh trial
        PARAMS:
        - labelmotif, list of tokens, either int or strings, depending on ver
        - ver, which beh representation to care about.
        --- "shapes", then is list of shape names
        --- "inds" then is list of task stroke inds.
        - good_strokes_only, then applies masks to get only good strokes. will not allow skips over
        bad storkes.
        RETURNS:
        - list_matches, list of list, where each inner list are the indices into list_beh which 
        would match motif. e.g., [[0,1], [4,5]]
        """
        assert False, "use motifs_search.py instead"
        if ver=="shapes":
            list_beh = self.Dat["shapes_beh"]
        elif ver=="inds":
            list_beh = self.Dat["taskstrokeinds_beh_singleonly"]
        else:
            assert False

        if good_strokes_only:
            # Generate mask, so only conisder good strokes
            maskinds = self.Dat["maskinds_repeated_beh"] & self.Dat["maskinds_singlematch_beh"]
        else:
            maskinds = None

        return self._find_motif_in_beh(list_beh, labelmotif, list_beh_mask=maskinds)




    ##################################### Using similarity matrix to get beh-task alignment
    def alignsim_compute(self, remove_bad_taskstrokes=False, 
        taskstrokes_thresh=0.4, ploton=False):
        """ Compute the aligned similarity matrix between beh and task strokes.
        PARAMS:
        - remove_bad_taskstrokes, bool, will not include taskstrokes whos max sim (across
        beh strokes) is less that taskstrokes_thresh. This will prune smat_sorted and idxs, but
        not smat, which is always the input order of taskstrokes.
        RETURNS:
        - self.Alignsim_taskstrokeinds_sorted, the best ordering of taskstrokes to match
        behstrokes, where each taskstorke is used max 1 time. can be 0 times if it is removed.
        and so is possibly shorter than num taskstrokes if remove_bad_taskstrokes
        is on, but the indices will still index into the original list.
        - self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices, the "FINAL" list of taskstroke inds,
        one for each beh stroke. IMPORANT: this does NOT index into original taskstrokes, but insteas
        into the pruned and sorted list of taskstrokes (pruend only if any not gotten). 
        NOTE: this differs from idxs, since the latter only uses each taskstroke
        a single time, since it is the best way to sort taskstrokes. idxs_beh is the "FINAL" version,
        since it allows for a given taskstroke to be matched >1 time (e.g. got it with 2 beh strokes.)
        NOTE: in RARE cases, it is possible for an index to exist in Alignsim_taskstrokeinds_sorted, but not 
        in Alignsim_taskstrokeinds_foreachbeh_sorted_origindices, because in the latter it independently
        tries to match each beh stroke to its best task stroke.
        """
        from pythonlib.drawmodel.behtaskalignment import aligned_distance_matrix
        import numpy as np

        strokes_beh = self.extract_strokes()
        strokes_task = self.extract_strokes("task")
        smat_sorted, smat, idxs = aligned_distance_matrix(strokes_beh, strokes_task, ploton)

        # print("unpruned", smat)
        # print("unpruned", idxs)
        if remove_bad_taskstrokes:
            maxsims = np.max(smat, axis=0)
            indstokeep = np.where(maxsims > taskstrokes_thresh)[0]
            # smat = smat[:, indstokeep]
            # print(smat)
            # print(indstokeep)
            idxs = [i for i in idxs if i in indstokeep] # Only keep taskstrokes that are gotten.
            smat_sorted = smat[:, idxs]

        if smat_sorted.shape[1]==0:
            # due to thresholding this datapoint is empty. 
            # i..e, no beh stroke matches even one task stroke
            self.Alignsim_taskstrokeinds_sorted = idxs
            self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices = []
            self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices = []
            self.Alignsim_simmat_sorted = smat_sorted
            self.Alignsim_simmat_unsorted = smat
            self.Alignsim_Datsegs = None
        else:
            # Compute for each beh stroke the best matching task stroke, based on the max similarity
            # [1,2,2,0] means behstroke 0 is aligned to takstroke 0, etc... after taskstrokes are sorted.
            # NOTE: this differs from idxs, since the latter only uses each taskstroke
            # a single time, since it is the best way to sort taskstrokes. idxs_beh is the "FINAL" version,
            # since it allows for a given taskstroke to be matched >1 time (e.g. got it with 2 beh strokes.)
            try:
                idxs_beh = np.argmax(smat_sorted, axis=1)
                idxs_beh_into_original_taskstrokeinds = [idxs[i] for i in idxs_beh]
            except Exception as err:
                self.plotStrokes()
                self.plotTaskStrokes()
                print("--")
                print(strokes_beh)
                print(smat)
                print(idxs)
                print("----")
                print(smat_sorted)
                print(len(smat_sorted))
                print(smat_sorted.shape)
                raise err

            self.Alignsim_taskstrokeinds_sorted = idxs
            self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices = idxs_beh
            self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices = idxs_beh_into_original_taskstrokeinds
            self.Alignsim_simmat_sorted = smat_sorted
            self.Alignsim_simmat_unsorted = smat
            self.Alignsim_Datsegs = None

        # sanity checks, to confirm my udnerstanding of what shoudl happen.
        assert len(set(idxs))==len(idxs), "should be unique indices, since they are sort indices."


    def alignsim_extract_datsegs(self, expt=None, plot_print_on=False, recompute=False,
            include_scale=True, input_grid_xy=None, reclassify_shape_using_stroke=False,
            reclassify_shape_using_stroke_version = "default", tokens_gridloc_snap_to_grid=False,
            list_cluster_by_sim_idx=None):
        """
        [GOOD! This is the only place where datsegs are generated]
        Generate datsegs, sequence of tokens. Uses alignement based on similairty matrix.
        Will only compute the first time run. then will save and reload cached every time run again
        (unless recompute=True)
        PARAMS:
        - expt, str, name of expt which is used for defining params for extracting stroke labels. 
        e..g, things like size of sketchpad grid. None, to use self.Expt
        - reclassify_shape_using_stroke, see Dataset()
        RETURNS:
        - datsegs, list of dicts, each w same keys, each a single token, length of beh strokes
        MODIFIES:
        - self.Alignsim_Datsegs, length of the taskstrokes that are NOT pruned, in the order
        that sorts them so that they best match beahvior. No taskstroke is used more than once.
        This means Alignsim_Datsegs[2] doesnt necesasiyl match strokes_beh[2]. Can think of this
        as taskstroke inds ordered by their "first touich"
        - self.Alignsim_Datsegs_BehLength, legnth of beh stroke, where each index is corresponds to
        its same index in beh. This means can have a tasktrokes used multipe times. THis is based on
        indepednetly matching each beh stroke to its best-matching task stroke.

        NOTE: This is the ONLY place that Task.tokens_generate is run.
        """
        import copy

        if not hasattr(self, "_TokensLocked"):
            self._TokensLocked = False

        if self._TokensLocked:
            assert False, "tried to generate new toekns, but it is locked... bug. you should be loading tokens directly from D."

        # THought about having datsegs always dynamically computed, but decided that might be
        # too expensive.
        # if not recompute:
        #     # Then load already-computed from Task
        #     Task = self.task_extract()
        #     # datsegs = Task.tokens_generate(assert_computed=True)
        #     # # Now use the aligned task inds
        #     inds_taskstrokes = self.Alignsim_taskstrokeinds_sorted
        #     datsegs = Task.tokens_reorder(inds_taskstrokes)
        #     # # Saved cached datsegs
        #     # self.Alignsim_Datsegs = datsegs

        if not recompute:
            # Load cached datsegs
            if self.Alignsim_Datsegs is not None:
                return self.Alignsim_Datsegs
            else:
                assert False, "you expect it to already be computed"

        # If you are computing, then you should pass in input_grid_xy
        if input_grid_xy is None:
            print("recompute:", recompute)
            print("self.Alignsim_Datsegs:", self.Alignsim_Datsegs)
            print("SOLUTION: run D.taskclass_preprocess_wrapper() first")
            assert False, "you need to pass in input_grid_xy if you are recompujting... to make sure all tasks in this dataset have same grid..."

        if expt is None:
            expt = self.Expt
            assert expt is not None

        params = {
            "expt":expt}

        # Genrate tokens, taskstroke inds order.
        Task = self.task_extract()
        hack_is_gridlinecircle = params["expt"] in ["gridlinecircle", "chunkbyshape2"]
        Task.tokens_generate(hack_is_gridlinecircle=hack_is_gridlinecircle, 
            assert_computed=False,
            include_scale=include_scale, input_grid_xy=input_grid_xy,
                             reclassify_shape_using_stroke=reclassify_shape_using_stroke,
                             reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
                             tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid,
                             list_cluster_by_sim_idx=list_cluster_by_sim_idx) # generate the defualt order

        # Now use the aligned task inds
        inds_taskstrokes = self.Alignsim_taskstrokeinds_sorted
        # Saved cached datsegs
        # datsegs = Task.tokens_reorder(inds_taskstrokes)
        self.Alignsim_Datsegs = Task.tokens_reorder(inds_taskstrokes)

        if plot_print_on:
            for x in self.Alignsim_Datsegs:
                print(x)
            self.alignsim_plot_summary()

        # Extract best guess for behavior-length datsegs
        datsegs_behlength = [copy.copy(self.Alignsim_Datsegs[i]) for i in self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices]
        self.Alignsim_Datsegs_BehLength = datsegs_behlength
 
        # Sanity cehcek, confirming that I am sure what is coming out.
        tmp = Task.tokens_reorder(self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices)
        assert len(tmp)==len(self.Alignsim_Datsegs_BehLength)
        for t1, t2 in zip(tmp, self.Alignsim_Datsegs_BehLength):
            assert t1["ind_taskstroke_orig"] == t2["ind_taskstroke_orig"], "bug somewhere"

        # Sanity check
        assert [t["ind_taskstroke_orig"] for t in self.Alignsim_Datsegs] == self.Alignsim_taskstrokeinds_sorted, "no idea. mistake somewhere"
        assert [t["ind_taskstroke_orig"] for t in self.Alignsim_Datsegs_BehLength] == self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices, "no idea. mistake somewhere"
        return self.Alignsim_Datsegs

    def alignsim_extract_datsegs_both_beh_task(self, DEBUG=False):
        """
        [GOOD] get summary of beh and task strokes, including their alignemnet.
        RETURNS:
        - out_combined. Combined representaion, list (length num taskstrokes, but might be less
        if any were pruned!) of tuples, and in order that they are gotten by first touch (based on alignsim). Each tuple: 
        (inds_beh, strokesbeh, dseg_task), where inds_beh are indices into self.Strokes,
        strokesbeh are those sliced strokes, and dseg_task is the single dseg for thsi taskstroke,
        - datsegs_behlength, see notes within
        - datsegs_firsttouch, see notes within. matches out_combined.
        """        

        # task datsegs (get in both (i) length of task and (ii) length of beh.
        datsegs_firsttouch = self.alignsim_extract_datsegs() 

        if DEBUG:
            print(len(datsegs_firsttouch))
            print(self.Alignsim_taskstrokeinds_sorted)
            print(self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices)
            print(self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices)
            assert False

        # datsegs_behlength = [datsegs_firsttouch[i] for i in self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices]
        datsegs_behlength = self.Alignsim_Datsegs_BehLength

        # Combined representaion, list (length num taskstrokes) of tuples, each tuple:
        # Combined representaion, list (length num taskstrokes) of tuples, each tuple:
        # (inds_beh, strokesbeh, dseg_task), where inds_beh are indices into self.Strokes,
        # strokesbeh are those sliced strokes, and dseg_task is the single dseg for thsi taskstroke,
        def find_inds_behstroke_aligned_to_this_taskstroke(indtask_get):
            return [indbeh for indbeh, indtask in enumerate(self.Alignsim_taskstrokeinds_foreachbeh_sorted_newindices) if indtask==indtask_get]

        out_combined = []
        for i, dseg_task in enumerate(datsegs_firsttouch):

            # get all the beh that are aligned with this task
            inds_beh = find_inds_behstroke_aligned_to_this_taskstroke(i)

            # sanity check
            indtask_orig = dseg_task["ind_taskstroke_orig"]
            tmp = [indbeh for indbeh, indtask in enumerate(self.Alignsim_taskstrokeinds_foreachbeh_sorted_origindices) if indtask==indtask_orig]
            if not tmp == inds_beh:
                print(inds_beh)
                print(tmp)
                print(dseg_task)
                print(datsegs_firsttouch)
                assert False

            # beh = [{"indbeh":i, "behstroke":self.Strokes[i]} for i in inds_beh]
            strokesbeh = [self.Strokes[i] for i in inds_beh]
            out_combined.append((inds_beh, strokesbeh, dseg_task))

        return out_combined, datsegs_behlength, datsegs_firsttouch

    def alignsim_plot_summary(self):
        """Plot results of alignments
        """
        from pythonlib.dataset.dataset import Dataset
        D = Dataset([])
        strokes_beh = self.extract_strokes()
        strokes_task = self.extract_strokes("task_after_alignsim")
        # idxs = self.Alignsim_taskstrokeinds_sorted
        # strokes_task_aligned = [strokes_task[i] for i in idxs]
        strokes_task_orig =  self.extract_strokes("task")

        # fig1, _ = D.plotMultStrokes([strokes_beh, strokes_task_aligned], number_from_zero=True) # plot, crude, shwoing task after alignemnt.
        fig1, _ = D.plotMultStrokesByOrder([strokes_beh, strokes_task],
            titles=["beh", "task"],
            plotkwargs = {"number_from_zero":True}) # plot, crude, shwoing task after alignemnt.

        if hasattr(self, "Alignsim_simmat_sorted"):
            fig2 = plt.figure()
            smat_sorted = self.Alignsim_simmat_sorted
            plt.imshow(smat_sorted, cmap="gray_r", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("task stroke inds")
            plt.ylabel("beh stroke inds")
            plt.title("after sorting task strokes")
        else:
            fig2 = None
        
        return fig1, fig2


    #################################### FEATURE TIMECOURSES
    # in all cases, signature is time_wind --> feature vector
    def timeseries_extract_features_all(self):
        """ For each timepoint in this trial, extract a feature vector.
        """
        pass

    def timepoint_extract_features_all(self, twind):
        """ Extract all features for this time window, including
        - continuos (related to strokes motor features)
        - categorical (related to chunks, etc)
        - progression (continuos and categor, related to progression along
        trial or strokes, etc)
        PARAMS:
        - twind, [t1, t2], inclusive, secs
        RETURNS:
        - Either:
            - featuredict, dict feature: vals, where vals are np arrays, shape depends on feature.
            e.g., {'cont_pos_mean_xy': array([-115.4969943 ,   45.23218542]),
                 'cont_vel_mean_xy': array([ -8.39052243, 481.03840076]),
                 'catg_ordinal': array([6.])}
            - None, if all values are none  
        """
        from pythonlib.tools.stroketools import timepoint_extract_features_continuous

        # Preprocess
        self.extract_strokes_vels(cache=True)

        # Collect
        featuredict = {} # Collects all variables for this timewindow

        # 1. Extract cotinuous motor features
        list_feature = ["mean_xy"]
        for strokeskind in ["pos", "vel"]:
            if strokeskind=="pos":
                strokes = self.Strokes
            elif strokeskind=="vel":
                strokes = self.StrokesVel
            else:
                assert False
            
            vals = timepoint_extract_features_continuous(strokes, twind, list_feature)
            for f,v in zip(list_feature, vals):
                featuredict[f"cont_{strokeskind}_{f}"] = v     

        # 2. Extract categorical variables
        list_feature = ["ordinal", "shape_oriented"]
        vals = self.timepoint_extract_features_categorical_(twind, list_feature)
        for f,v in zip(list_feature, vals):
            featuredict[f"catg_{f}"] = v   
            
        # 3. Extract continuous variables about task progression
        if False:
            # Havent coded
            list_feature = ["doneness_trial"]
            vals = timepoint_extract_features_progression_(self, twind, list_feature)
            for f,v in zip(list_feature, vals):
                featuredict[f"catg_{f}"] = v   

        # SAnity checks
        # - if any vals are None, then they must all be None
        x = [np.any(v==None) for k, v in featuredict.items()] # list of bools
        if any(x):
            assert all(x)
            featuredict = None

        return featuredict


    def timepoint_extract_features_categorical_(self, twind, 
            list_feature=None):
        """ Extract categorical feature for this stroke at this timepoint
        PARAMS:
        - twind, [t1, t2], where feature uses data within this windopw (inclusinve)
        - list_feature, list of string, name of feature
        RETURNS:
        - features, list of values for features. Type for each feature can vary
        -- returns list of None (saem len as list_feature) if this window has no data...
        """
        if list_feature is None:
            list_feature = ["ordinal", "shape_oriented"]

        from pythonlib.tools.stroketools import sliceStrokes
        from scipy import stats

        # Preprocess
        self.extract_strokes_ordinal()

        # Get motor timing stats
        # Extract time of events
    #     motor = D.get_motor_stats(idx)
        motor = self.MotorStats
        
    #     # go cue
    #     motor["go_cue"]

    #     # raise
    #     motor["raise"]

    #     # stroke onsets
    #     motor["ons"]

    #     # stroke offsets
    #     motor["offs"]

    #     # touch done
    #     motor["done_touch"]

        # First, get this time slice for ordinal.
        # then all categorical variables will be indexed by ordinal
        strokes_ordinal_sliced = sliceStrokes(self.StrokesOrdinal, twind, False)
        if len(strokes_ordinal_sliced)==0:
            # no data, return None
            return [None for _ in range(len(list_feature))]

        pts_ordinal = np.concatenate(strokes_ordinal_sliced) # (N,2)
        mode_ordinal = stats.mode(pts_ordinal[:,0]).mode

        
        # take mean within this slice
        features = []
        for f in list_feature:
            if f=="ordinal":
                # which stroke number?
                # By convention, during gap will return ordinal for the preceding stroke.
                # starts at 0 (before first stroke). So betweeen onset of first and 2nd stroke
                # is 1.
                # - If time window spans an onset, then will assign it to ordinal for which 
                # more timepoints are overlaping.
                val = mode_ordinal
            elif f=="chunk ordinal":
                pass
            elif f=="shape_oriented":
                # Uses datsegs from alignment
                datsegs = self.alignsim_extract_datsegs()
                ind_stroke = int(mode_ordinal)-1
                val = datsegs[ind_stroke]["shape_oriented"]
            else:
                print(f)
                assert False, "code it"
            features.append(val)

        return features

    ##################################### PLOTS
    def plotStrokes(self, ax=None):
        """ Quick plot of this task
        """
        import matplotlib.pyplot as plt
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(2.5,2.5))

        strokes = [S() for S in self.Strokes]
        plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, number_from_zero=True)
        return fig, ax

    def plotTaskStrokes(self, ax=None):
        """ Quick plot of this task
        """
        T = self.task_extract()
        T.plotStrokes(ax, ordinal=True)
