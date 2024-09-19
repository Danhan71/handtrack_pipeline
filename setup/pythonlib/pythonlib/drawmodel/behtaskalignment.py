""" Tools for aligning behavior with task
Generally works with strokes (list of np array) separate for beh and task.
"""
import numpy as np
import matplotlib.pyplot as plt

def aligned_distance_matrix(strokes_beh, strokes_task, ploton=False, 
        do_dtw_refinement=False, sort_ver="max", cap_dist=150, squared_sim=True,
        distancever_forsorting = "hausdorff_mins", 
        distancever_foroutput="hausdorff_mins", cap_sim=0.7,
        similarity_method="divide_by_maxcap"):
    """ Instead of taking single strokes task ind that matches each beh ,
    return a distance matrix where tasks (columns) are sorted so that one can
    read out the order of columns as the best-aligned task sequence.
    Does not explicitly map one-to-one strokes between beh and task.
    PARAMS:
    - strokes_beh, strokes_task, list of np.array, the usual.
    - do_dtw_refinement, bool, if refine, then repeatedly swaps pairs of adjacent columns
    and keeps the best resulting sim matrix, based on dtw of strokes as cost function.
    Generally avoid this, since it really wants for the last beh stroke to be paired with
    last task stroke, but monkeys often redraw a stroke for the last storke.
    - sort_ver, str, how to sort the columns (task inds) of smat. 
    --- "max", (default), for each col, finds max beh stroke ind. sorts task cols so the
    inds are in incresaing order. breaks ties using the distance.
    --- "mean", similar to above, but not max, instead takes mean (of squared sim).
    - cap_dist, see inside.
    - squared_sim, then squares the sim scores. this is useful for emphasizing peaks in sim
    - cap_sim, then clamps at this value, the sim matrix. only for if sort_ver=="max". then
    subtracts very small value to break equality, so that sims for beh strokes earlier in time
    are higher. This is useful so that if a task stroke similar to two diff beh strokes, will
    use the one that is earlier in time. use None to skip
    RETURNS:
    - smat_sorted, similarity matrix (nbehstrokes, ntaskstrokes), where columns
    have been sorted already to maximize alignemtn
    - smat, same but unsorted (input order of task strokes)
    - idxs, idxs for sorting columns. smat_sorted = smat[:, idxs]
    """
    from pythonlib.tools.stroketools import distanceDTW
    from pythonlib.drawmodel.strokedists import distStrokWrapperMult

    # 1) Get similarity matrix using the inputed strokes order (for now)
    if False:
        # Test out different versio of asymetry for hd.
        for av in [None, "A", "B"]:
            smat = distMatrixStrok(strokes_beh, strokes_task, convert_to_similarity=True, 
                                   ploton=True, similarity_method=similarity_method,
                                  distStrok_kwargs={"asymmetric_ver":av}) # (nbeh, ntask)

    # Decision: Use symmetric version. 
    # - use this for deciding sorting
    smat = distStrokWrapperMult(strokes_beh, strokes_task, distancever=distancever_forsorting,
                                convert_to_similarity=True, ploton=ploton, asymmetric_ver= None,
                                similarity_method=similarity_method, cap_dist=cap_dist)  # (nbeh, ntask)
    # - use this for actual output similarity scores.
    smat_output = distStrokWrapperMult(strokes_beh, strokes_task, distancever=distancever_foroutput,
                                       convert_to_similarity=True, ploton=ploton,
                                       asymmetric_ver= None, similarity_method=similarity_method,
                                       cap_dist=cap_dist)  # (nbeh, ntask)

    # smat_output = smat
    # 2) any normalizations?
    # TODO, any other normalizations before compute? Specifically norm along columns
    # or rows?

    if squared_sim:
        smat=smat**2
        smat_output=smat_output**2

    # smat_orig = smat.copy()

    # 3) sort columns (task labels) of smat to maximize the DTW - the sum along diagonal path basicallt
    # - method, for each column, get its center of mass. Sort columns by COM.
    if sort_ver=="mean":
        nbeh, ntask = smat.shape
        bins = np.linspace(0, nbeh-1, nbeh)
        bins = np.repeat(bins[:, None], ntask, axis=1)
        if not squared_sim:
            weights = smat**2 # square, to emphasize peaks
        ranks = np.average(bins, 0, weights)
        idxs = np.argsort(ranks)
    elif sort_ver=="max":
        # alternative method, for each column, get its beh stroke that is max. sort columns by beh stroke, breaking ties using the cost.
        def _process_smat(smat, use_cap=True):
            nbeh, ntask = smat.shape
            if use_cap:
                if cap_sim is not None:
                    smat[smat>cap_sim] = cap_sim + 0.01
                    tmp = np.repeat(np.linspace(0, 0.01, nbeh)[:, None], ntask, 1)
                    smat = smat - tmp
            max_sims = np.max(smat, axis=0)
            max_inds = np.argmax(smat, axis=0)
            return smat, max_sims, max_inds
        
        smat, max_sims, max_inds = _process_smat(smat)

        # Add secondary and tertiary indices for sorting, by splitting beh strokes.
        # This enforces temporal ordering for cases where single beh stroke covers multiple
        # task strokes. The first one (split 2) helps not have small strokes, more accurate
        # the second one (split 4) is a tie breaker, and important for longer strokes.
        # use hausdorff_means, because want to really care about the shape match.
        from pythonlib.tools.stroketools import splitStrokesOneTime
        strokes_beh_split = splitStrokesOneTime(strokes_beh, num=2, reduce_num_if_smallstroke=True)
        smat_split = distStrokWrapperMult(strokes_beh_split, strokes_task, distancever="hausdorff_means",
                                          convert_to_similarity=True, ploton=ploton,
                                          asymmetric_ver=None,
                                          similarity_method=similarity_method, cap_dist=cap_dist)  # (nbeh, ntask)
        smat_split, max_sims_split, max_inds_split = _process_smat(smat_split, use_cap=False)

        strokes_beh_split = splitStrokesOneTime(strokes_beh, num=4, reduce_num_if_smallstroke=True)
        smat_split = distStrokWrapperMult(strokes_beh_split, strokes_task, distancever="hausdorff_means",
                                          convert_to_similarity=True, ploton=ploton,
                                          asymmetric_ver=None,
                                          similarity_method=similarity_method, cap_dist=cap_dist)  # (nbeh, ntask)
        smat_split2, max_sims_split2, max_inds_split2 = _process_smat(smat_split, use_cap=False)

        # list_to_sort = [(i, s, i2, s2, indtaskstroke) for indtaskstroke, (i, s, i2, s2) 
        #     in enumerate(zip(max_inds, max_sims, max_inds_split, max_sims_split))]
        list_to_sort = [(i, s, i2, s2, i3, s3, indtaskstroke) for indtaskstroke, (i, s, i2, s2, i3, s3) 
            in enumerate(zip(max_inds, max_sims, max_inds_split, max_sims_split, max_inds_split2, max_sims_split2))]
        list_to_sort = sorted(list_to_sort, key=lambda x:(x[0], x[2], x[4], -x[1], -x[3], -x[5]))

        # print(max_sims)
        # print(max_inds)
        # print(list_to_sort)
        # print(smat>0.7)
        # print(smat>0.7)
        # list_to_sort = sorted(list_to_sort) # sort first by beh inds, then break ties by sim score.
        idxs = [l[6] for l in list_to_sort] # recover original task inds
    else:
        assert False

    if do_dtw_refinement:
        # 4) Refine, by switching pairs of adjacent taskstrokes and computing dtw. Keep doing
        # this until converges.
        # - do dtw splitting beh segments. This makes task strokes in temporal order. e.g., if a 
        # single beh stroke gets multipel task strokes.
        def _refine_idxs(idxs, strokes_task):
            """ Get distance each time after swapping pairs of adjacent task strokes.
            RETURNS:
            - idxs, list of ints, the best-matching ordering of task storkes.
            - dist, scalar, the distance assigned for this ordering.
            """
            list_out = []
            for i in range(len(strokes_task)):
                idxs_swapped = [i for i in idxs]
                if i<len(strokes_task)-1:
                    # do swap of adjacent task strokes.
                    idxs_swapped[i] = idxs[i+1]
                    idxs_swapped[i+1] = idxs[i]
                strokes_task_this = [strokes_task[i] for i in idxs_swapped]
                dist = distanceDTW(strokes_beh, strokes_task_this, 
                    ver="split_segments", splitnum1=2, splitnum2=1, asymmetric=False)
                list_out.append({"idxs":idxs_swapped, "dist":dist[0]})

            # Keep the idxs with lowest distance
            list_out = sorted(list_out, key=lambda x:x["dist"])
            idxs = list_out[0]["idxs"]
            dist = list_out[0]["dist"]
            return idxs, dist

        idxs_new, dist = _refine_idxs(idxs, strokes_task)
        # while any([i1!=i2 for i1, i2 in zip()])
        while not np.all(idxs_new==idxs):
            # print("inds refinement:", idxs_new, idxs, dist)
            idxs = idxs_new
            idxs_new, dist = _refine_idxs(idxs, strokes_task)

    # Return sorted dmat and task stroke inds
    smat_sorted = smat_output[:, idxs]
    return smat_sorted, smat_output, idxs



def assignStrokenumFromTask(strokes_beh, strokes_task, ver="pt_pt", sort_stroknum=False, 
    doprint = False):
    """ different ways of assigning a stroke_task id to each strokes_beh.
    Different motehods. None are model based. all simple
    ver:
    - pt_pt, each beh pt is assigned to a stroke based on pt-pt distances.
    - stroke_stroke
    sort_stroknum, then the strok num is arbitrary, so will sort so that the earliest touched 
    is 0, and so on.
    """
    from pythonlib.drawmodel.strokedists import distStrokWrapperMult
    from pythonlib.tools.stroketools import splitTraj
    
    if len(strokes_beh)==0:
        return []

    if ver=="stroke_stroke":
        # 1) stroke(beh) assigned a stroke(task).
        # - every real stroke must be "assigned" a task stroke (not vice versa)
        # - assign each stroke the task stroke that is the closest
        from pythonlib.tools.vectools import modHausdorffDistance
        stroke_assignments = [] # one for each stroke in behavuiopr
        distances_all = []
        for s_beh in strokes_beh:
            # get distnaces from this behavioal stroke to task strokes
            distances = []
            for s_task in strokes_task:
                distances.append(modHausdorffDistance(s_beh, s_task))

            # assign the closest stroke
            stroke_assignments.append(np.argmin(distances))
            # just for debugging
            distances_all.append(sorted(distances))

        return stroke_assignments, distances_all
    elif ver=="stroke_stroke_lsa":
        """ Use linear sum assignment to find optimal alignment.
        Note, will only use num strokes equal to the min num strokes across strokes beh and task.
        """
        from scipy.optimize import linear_sum_assignment as lsa
        dmat = distStrokWrapperMult(strokes_beh, strokes_task, convert_to_similarity=False)
        inds1, inds2 = lsa(dmat)
        stroke_assignments = inds2.tolist()
        return stroke_assignments

    elif ver=="stroke_stroke_lsa_usealltask":
        """ Like stroke_stroke_lsa, but if is strokes_task is longer than strokes_beh,
        then figures out how to use the "excess" strokes in strokes_task. Does this in
        greedy fashion. for each excess stroke, tries all possible insertion locations. 
        Takes the location that allows max similarity to strokes_beh. Does this until
        uses up all excess strokes. Note, if not all beh strokes are used up, then doesnt 
        do this - so is assymetric
        RETURNS:
        - inds, same legnth as task strokes, telling where each task stroke should go.
        Note, if more beh strokes than task strokes, this is not that informative.
        Note, if more task storkes, than beh storkes, also not that informative.
         """

        inds_assigned = assignStrokenumFromTask(strokes_beh, strokes_task, 
            "stroke_stroke_lsa")
        strokes_assigned = [strokes_task[i] for i in inds_assigned]

        # any unassigned task strokes, find minimum over all ways of sticking it into the sequence
        nstrokes_task = len(strokes_task)
        inds_unassigned = [i for i in range(nstrokes_task) if i not in inds_assigned]
        for i in inds_unassigned:
            # find where this extra strok should go
            # print("this unassigned ind:", i)
            strok_extra = strokes_task[i]
            slot = insert_strok_into_strokes_to_maximize_alignment(strokes_beh, 
                strokes_assigned, strok_extra)

            # update
            inds_assigned.insert(slot, i)
            strokes_assigned.insert(slot, strok_extra)
        return inds_assigned



    elif ver=="each_beh_stroke_assign_one_task":
        """ simple, good. for each beh stroke, find its closest task stroke. assign that task stroke to
        that beh stroke. 

        RETURNS:
        - inds, list of list, same length as beh strokes, item being index in task strokes. each inner list
        are the stroke inds that are assigned to this beh stroke. e.g, [[0], [1,2], [3]]...
        - list_distances, distnace for the match. if match to >1 strok, then concats task before
        computing distance.
        
        NOTE:
        - not guaranteed to use up all task strokes. but is useful since works no matter how many beh 
        strokes exist.
        """

        THRESH_FORCE_SINGLE = 0.5 # if ratio of dist of best match to 2nd best match is less than this
        # then will force that beh is assigned only the best match (i.e., no change of getting 2 matches)

        # 2) Check if any beh stroke is better matched to 2 task strokes.
        if len(strokes_task)>1:

            list_indstask_mindist = [] # holding final output.
            list_distances = [] # the distances. if match >1 stroke, then is distance after concat the strokes.
            for i, strok in enumerate(strokes_beh):
                strok = strokes_beh[i]

                # 1) dist from this beh strok to all task strokes
                dmat1 = distStrokWrapperMult([strok], strokes_task, convert_to_similarity=False)  # (1, ntask)
                inds_mindist = np.argsort(dmat1).squeeze()[:2] # [3,4] means task strokinds 3, then 4, are closest to this beh stroke.

                # 2) ratio of min to 2nd min
                # i..e, if this beh stroke is already a really good match to one task stroke. then dont bother.
                ratiothis = dmat1[0, inds_mindist[0]]/dmat1[0, inds_mindist[1]]
                if ratiothis<THRESH_FORCE_SINGLE:
                    list_indstask_mindist.append(list(np.argmin(dmat1, axis=1)))
                    list_distances.append(np.min(dmat1, axis=1))

                    if doprint:
                        print(i, "-----------")
                        print("Skipping, since ratio: ", ratiothis)
                    # continue, since the min has clearly won out.
                    continue

                # 3) Split this beh stroke. if both of the 2 mindist task strokes now are closer to eitehr of the new split storkes, then 
                # infer that this beh stroke is spanning those 2 task strokes.
                list_stroksplit = splitTraj(strok)
                dmat2 = distStrokWrapperMult(list_stroksplit, strokes_task, convert_to_similarity=False)
                dmat2_mins = np.min(dmat2, axis=0) # TODO: ideally use both beh strokes...
                inds = dmat2_mins<dmat1.squeeze()
                inds = inds.squeeze() # [True, False, ...]
                if np.all(inds[inds_mindist]):
                    # then the two taskstrokes that were closest to this beh stroke both got even closer after
                    # splitting this beh stroke. Therefore assign both these taskstrokes to this beh stroke.
                    inds_task = list(inds_mindist)

                    tmp = [strokes_task[i] for i in inds_task]
                    tmp = [np.concatenate(tmp)]
                    dthis = distStrokWrapperMult([strok], tmp, convert_to_similarity=False)
                    list_distances.append(dthis[0])

                else:
                    inds_task = list(np.argmin(dmat1, axis=1))
                    list_distances.append(np.min(dmat1, axis=1))


                list_indstask_mindist.append(inds_task)
                if doprint:
                    print(i, "-----------")
                    print(dmat1)
                    print(np.argsort(dmat1))
                    print(dmat2)
                    print(inds_mindist, inds)
                    print(list_indstask_mindist)
        else:
            # 1) Simple, for each beh stroke get min dist task stroke.
            dmat = distStrokWrapperMult(strokes_beh, strokes_task, convert_to_similarity=False)
            inds_task_mindist = np.argmin(dmat, axis=1)
            list_indstask_mindist = [[i] for i in inds_task_mindist]
            list_distances = np.min(dmat, axis=1)

        return list_indstask_mindist, np.array(list_distances)

    # elif ver=="each_beh_stroke_assign_one_task_wrapper":
    #     """ Decides which version to use based on match in num strokes between task and bhge.
    #     if same: uses stroke_stroke_lsa, so that guaranteed is all beh and tasks are used up.
    #     if nbeh > ntask: uses 


    elif ver=="pt_pt":
        # 2) Each pt (beh) assigned a stroke(task)
        from scipy.spatial.distance import cdist
        # flatten strokes
        sb = np.concatenate(strokes_beh, axis=0)[:,:2]
        st = np.concatenate(strokes_task, axis=0)[:,:2]
        distmat = cdist(sb, st, "euclidean")
        closest_task_pts = np.argmin(distmat, axis=1)
        closest_task_dist = np.min(distmat, axis=1)

        # given task pt, figure out which stroke it is.
        closest_task_stroknum = []
        for idx in closest_task_pts:
        #     print([idx, np.sum(idx > np.cumsum([len(s) for s in strokes_task]))])
            closest_task_stroknum.append(np.sum(idx > np.cumsum([len(s) for s in strokes_task])))

        if sort_stroknum:
            indexes = np.unique(closest_task_stroknum, return_index=True, return_inverse=True) # get location of first index for each unique snum
            tmp2 = np.argsort(indexes[1]) # for each unique ind, figure out which ordianl position.
            D = {}
            for i, t in enumerate(tmp2):
                D[t]=i # wnat to replace t with i
            closest_task_stroknum = [D[i] for i in indexes[2]]

        # return to strokes format
        closest_task_stroknum_unflat = convertFlatToStrokes(strokes_beh, closest_task_stroknum)

        return closest_task_stroknum_unflat
    else:
        assert False