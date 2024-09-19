""" based around parsing in pyBPL package (Reuben Feinman)
Img --> parses (where parses are strokes)
"""
import numpy as np
# import torch
import matplotlib.pyplot as plt

def extractJunctions(strokes, sketchpad_edges, image_edges):
    """ automatically converts strokes to image coordinates, then gets posiotions of junctions
    - strokes, in monkey sketchpad space
    - can first run:
    --- sketchpad_edges, image_edges = getSketchpadEdges(canvas_max_WH, image_WH)
    # This section deals with issue that corners are in general not used as junctions in 
    # the undirected graph that represents the drawing during parsing.
    # For monkey, they often break down L-shaped things into two lines, so we want to consider
    # parses where corners can have a junction.

    # Solution: determine coordinates of all "segment" endpoints, which will generally include
    # corners (since things like L's are represnted as two lines usually). 
    # Will just pass in all endpoints, since anything that is redundant with the 
    # BPL junctions will be automaticalyl discarded.
    from pythonlib.drawmodel.parsing import extractJunctions

    if stim_ver=="task_image":
        extra_junctions = extractJunctions(strokes, sketchpad_edges, image_edges)
        print(extra_junctions)
    elif stim_ver=="monkey_strokes":
        # then dont need to do this, since if monkey raises finger at corner, that will be
        # detected by BPL, since it will not be a perfectly clean corner, and so will
        # be considered a "cross". [have not verified this by testing]
        pass
        

    """
    from pythonlib.drawmodel.image import convStrokecoordToImagecoord, getSketchpadEdges
    
    # use this to convert strokes to image space
    extra_junctions = []
    for pts in strokes:
        pts_image_inds = convStrokecoordToImagecoord(pts, sketchpad_edges, image_edges)
        extra_junctions.append(pts_image_inds[0])
        extra_junctions.append(pts_image_inds[-1])

    extra_junctions = np.stack(extra_junctions, axis=0)
    return extra_junctions


def get_parses_from_strokes(strokes, canvas_max_WH, image_WH=105, k=20,     
    score_fn = lambda parses: score_function(parses, ver="travel", normalization="negative", use_torch=True), 
    use_extra_junctions=False, plot=False, return_in_strokes_coords=False,
    configs_per = 10, trials_per=50, max_ntrials=75, max_nwalk=75, max_nstroke=100):
    """ 
    - return_in_strokes_coords, then first converts back to origianl canvas size
    - Note, params for (generate randm prase) and (seach) have been reduced to reduce process
    time. Tested that for monlkey characters, 1,2,3,4,5 strokes, generally is pretty comparable 
    to if use large values for these params. so use this for now.
    """
    from pythonlib.drawmodel.parsing import extractJunctions
    from pythonlib.drawmodel.image import getSketchpadEdges
    from pythonlib.drawmodel.image import strokes2image
    # from gns.inference.parsing.top_k import get_topK_parses    

    # convert to images
    I = strokes2image(strokes, canvas_max_WH, image_WH, plot=plot)

    # Extract junctions
    if use_extra_junctions:
        sketchpad_edges, image_edges = getSketchpadEdges(canvas_max_WH, image_WH)
        extra_junctions = extractJunctions(strokes, sketchpad_edges, image_edges)
    else:
        extra_junctions = None

    # Parse
    grp_kwargs = {"ver":"lucas", "extra_junctions":extra_junctions, "max_ntrials":max_ntrials, 
    "max_nwalk":max_nwalk, "max_nstroke":max_nstroke}
    # grp_kwargs = {"ver":"lucas", "extra_junctions":extra_junctions, "max_ntrials":5, 
    # "max_nwalk":5, "max_nstroke":5}
    pp_kwargs = {"do_fit_spline":False}
    parses_k, log_probs_k = get_topK_parses(I, k, score_fn, 
        configs_per=configs_per, trials_per=trials_per, pp_kwargs=pp_kwargs, **grp_kwargs)

    # Conver to np
    def _torch2np(strokes):
        return [s.numpy() for s in strokes]
    parses_k = [_torch2np(strokes) for strokes in parses_k]
    log_probs_k = log_probs_k.numpy()

    if return_in_strokes_coords:
        parses_k = [convertCoordParseToStrokes(strokes, canvas_max_WH) for strokes in parses_k]

    return parses_k, log_probs_k

def score_function(parses, ver="ink", normalization = "inverse", test=False,
                  use_torch=False, origin=None):
    """ 
    - ver, str, determines what score to use
    --- "ink", then total distnace traveled on page
    --- "travel", then total distance traveled, including
    gaps, starting from position of first touch.
    - normalization, how to normalize raw distnace. distance will
    be that more positive is more worse. 
    --- inverse take inverse, so that now less positive is worse.
    --- negative, ...
    """
    from pythonlib.drawmodel.features import strokeDistances, computeDistTraveled
    if use_torch:
        import torch

    if test:
        # then just return random number, one for each parse
        return torch.tensor([random.random() for _ in range(len(parses))])    
    
    if ver=="ink":
        # === Total ink used
        distances = [np.sum(strokeDistances(strokes)) for strokes in parses]
    elif ver=="travel":
        # conisder origin to be onset of first storke.
        # Note: has issue in that a single stroke task, flipped, is idnetical cost to the same task unflipped.
        # leads to problems later since unique(score) is used to throw out redundant parses.
        distances_traveled = [computeDistTraveled(strokes, origin=strokes[0][0,[0,1]]) for strokes in parses]
        distances = distances_traveled
    elif ver=="travel_from_orig":
        # pass in origin. 
        assert origin is not None, " must pass in coordinate for origin"
        distances_traveled = [computeDistTraveled(strokes, origin=origin) for strokes in parses]
        distances = distances_traveled

    elif ver=="nstrokes":
        # num strokes
        # == plit histogram of num strokes
        nstrokes = [len(p) for p in parses]        
    else:
        print(ver)
        assert False, "not codede"
        
    if use_torch:
        distances = torch.tensor(distances)
    else:
        distances = np.array(distances)
        
    if normalization=="inverse":
        return 1/distances
    elif normalization=="negative":
        return -distances
    else:
        print(normalization)
        assert False, "not coded"
    

def convertCoordParseToStrokes(parse, canvas_max_WH, image_WH=105):
    """ Convert parses back into monkey sketchpad coords
    parses, output of pyPBL based parsing, convert back to monkey sketchpad
    - parse, is list of np arraus, (like "strokes")
    """
    from pythonlib.drawmodel.image import convCoordGeneral

    parse_edges = np.array([[1, image_WH-1], [-(image_WH-1), -1]]) # since parsing has translated y to negatives.
    sketchpad_edges = np.array([[-canvas_max_WH, canvas_max_WH], [-canvas_max_WH, canvas_max_WH]])
    strokes = [convCoordGeneral(S, parse_edges, sketchpad_edges) for S in parse]

    return strokes


################# COPIES FROM GNS LIBRARY, BUT ADDING PARAMS
def process_parse(parse, device=None, do_fit_spline=True):
    """ same as gns.inference.parsing.top_k, but allowing to skip fitting 
    spline """
    from pybpl.data import unif_space
    from gns.omniglot.minimal_splines import fit_minimal_spline
    import torch

    parse_ = []
    for stk in parse:
        # for ntraj = 1, set spline as the original stroke
        if len(stk) == 1:
            spl = torch.tensor(stk, dtype=torch.float, device=device)
            parse_.append(spl)
            continue
        # for ntraj > 1, fit minimal spline to the stroke
        stk = unif_space(stk)
        stk = torch.tensor(stk, dtype=torch.float, device=device)
        if do_fit_spline:
            spl = fit_minimal_spline(stk, thresh=0.7, max_nland=50)
        else:
            spl = stk
        parse_.append(spl)
    return parse_


def get_topK_parses(img, k, score_fn, configs_per=100, trials_per=800,
                    device=None, seed=3, pp_kwargs=None, **grp_kwargs):
    """ copied from
    gns.inference.parsing.top_k, except allwoing for pp_kwargs, which is useful 
    if want to skip spline fit optimiziation step
    """
    from pybpl.matlab.bottomup import generate_random_parses
    from gns.inference.parsing.top_k import search_parse    
    import torch

    # generate random walks (the "base parses")
    base_parses = generate_random_parses(I=img, seed=seed, **grp_kwargs)
    # convert strokes to minimal splines
    base_parses = [process_parse(parse, device, **pp_kwargs) for parse in base_parses]

    # search for best stroke ordering & stroke direction configurations
    np.random.seed(seed)
    n = len(base_parses)
    parses = []; log_probs = []
    for i in range(n):
        parses_i, log_probs_i = search_parse(
            base_parses[i], score_fn, configs_per, trials_per)
        parses.extend(parses_i)
        log_probs.append(log_probs_i)
    log_probs = torch.cat(log_probs)

    # refine to unique & sort
    log_probs, idx = np.unique(log_probs.cpu().numpy(), return_index=True)
    log_probs = torch.from_numpy(log_probs).flip(dims=[0])
    idx = torch.from_numpy(idx).flip(dims=[0])
    parses = [parses[i] for i in idx]

    return parses[:k], log_probs[:k]



# ==== PLOTS
def plotParses(parses, plot_timecourse=True, titles=None, ignore_set_axlim=False,
    jitter_each_stroke=True):
    from ..tools.stroketools import fakeTimesteps
    from .strokePlots import plotDatStrokes, plotDatStrokesTimecourse
    n = len(parses)
    ncols = 4
    nrows = int(np.ceil(n/ncols))

    if jitter_each_stroke:
        print("NOTE: jittering each stroke!!!")
    # = plot timecourses
    if plot_timecourse:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
        for p, ax in zip(parses, axes.flatten()[::-1]):
            p = fakeTimesteps(p, p[0], "in_order")
            plotDatStrokesTimecourse(p, ax, plotver="raw")

    # = plot images
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    for i, (p, ax) in enumerate(zip(parses, axes.flatten()[::-1])):
        if not ignore_set_axlim:
            ax.set_xlim([0, 105])
            ax.set_ylim([-105, 0])
        plotDatStrokes(p, ax, clean_ordered=True, 
            jitter_each_stroke=jitter_each_stroke, alpha=0.5)
        if titles is None:
            ax.set_title(i)
        else:
            ax.set_title(titles[i])
            
def summarizeParses(parses, plot_timecourse=False, ignore_set_axlim=False, titles=None):
    # Get total distance traveled (i.e., amount of ink)
    from pythonlib.drawmodel.features import strokeDistances, computeDistTraveled

    # === single parse
    if False:
        ind = 0
        strokes = parses[ind]
        strokeDistances(strokes)
        print("parses ink:")
        print(distances[0])

    # === Total ink used
    distances = [np.sum(strokeDistances(strokes)) for strokes in parses]
    # conisder origin to be onset of first storke.
    distances_traveled = [computeDistTraveled(strokes, origin=strokes[0][0,[0,1]]) for strokes in parses]
    # num strokes
    # == plit histogram of num strokes
    nstrokes = [len(p) for p in parses]

    # get total expected ink
    try:
        distance_actual = np.sum(strokeDistances(strokes_imagecoord))
    except:
        distance_actual = 0
    
    
    # ==== PLOTS
    fig, axes = plt.subplots(2, 3, figsize=(10,6))
    
    ax = axes.flatten()[0]
    ax.hist(distances)
    ax.axvline(distance_actual)
    ax.set_xlabel("ink used (vertical line = actual, bars = parses")
    
    ax = axes.flatten()[1]
    ax.hist(distances_traveled)
    ax.set_xlabel("distance traveled")
    
    ax = axes.flatten()[2]
    ax.hist(nstrokes)
    ax.set_xlabel("n strokes")
    
    ax = axes.flatten()[3]
    ax.plot(distances, distances_traveled, "xk")
    ax.set_xlabel("ink used")
    ax.set_ylabel("distnace traveled")
    
    ax = axes.flatten()[4]
    ax.plot(nstrokes, distances_traveled, "xk")
    ax.set_xlabel("nstrokes")
    ax.set_ylabel("distnace traveled")
    
    ax = axes.flatten()[5]
    ax.plot(nstrokes, distances, "xk")
    ax.set_xlabel("nstrokes")
    ax.set_ylabel("distnaces")
    
    # === sort by distances traveled
    tmp = [[d, trav, p] for d, trav, p in zip(distances, distances_traveled, parses)]
    tmp = sorted(tmp, key= lambda x: x[0])

    # - replot, but sorted by distance
    distances_sorted = [t[0] for t in tmp]
    traveled_sorted = [t[1] for t in tmp]
    parses_sorted = [t[2] for t in tmp]
    titles = [f"dist_{d:.2f}-trav_{t:.2f}" for d, t, in zip(distances_sorted, traveled_sorted)]
    plotParses(parses_sorted, plot_timecourse=plot_timecourse, titles=titles, ignore_set_axlim=ignore_set_axlim)
    print(f"parses, sorted by distance traveled")


if __name__=="__main__":
    from pythonlib.dataset.dataset import Dataset
    from pythonlib.drawmodel.parsing import *
    import numpy as np
    import torch

    # Parse a single datset, but iter over multiple datsetes. 
    # Parses are one for each unique task, this is more efficiecnt that each row of datsaet.
    # Default is to add extra_junction to make sure to get corners.

    ##################### PARAMS
    # OK version (good balance between fast and variety of parses)
    # params_parse = {
    #     "configs_per":10,
    #     "trials_per":50,
    #     "max_ntrials":75, 
    #     "max_nwalk":75,
    #     "max_nstroke":100
    # }

    # # FAST VERSION
    # params_parse = {
    #     "configs_per":5,
    #     "trials_per":20,
    #     "max_ntrials":50, 
    #     "max_nwalk":50,
    #     "max_nstroke":50
    # }
    # VERY FAST
    params_parse = {
        "configs_per":5,
        "trials_per":10,
        "max_ntrials":25, 
        "max_nwalk":25,
        "max_nstroke":25
    }
    return_in_strokes_coords = True
    kparses = 10
    # animal = "Red"

    use_extra_junctions=True
    # score_ver = "travel"
    score_ver = "travel_from_orig" # better, since differentiates 2 tasks thjat are just flipped (and so will not throw one of them out)
    score_norm = "negative"
    image_WH = 105

    #################### RUN
    for animal in ["Pancho"]:
        # Load datasets
        if animal == "Red":
            path_list = [
                "/data2/analyses/database/Red-lines5-formodeling-210329_005719",
                "/data2/analyses/database/Red-arc2-formodeling-210329_005550",
                "/data2/analyses/database/Red-shapes3-formodeling-210329_005200",
                "/data2/analyses/database/Red-figures89-formodeling-210329_005443"
            ]
            # path_list = [
            #     "/data2/analyses/database/Red-figures89-formodeling-210329_005443"
            # ]
        elif animal=="Pancho":
            path_list = [
                "/data2/analyses/database/Pancho-lines5-formodeling-210329_014835",
                "/data2/analyses/database/Pancho-arc2-formodeling-210329_014648",
                "/data2/analyses/database/Pancho-shapes3-formodeling-210329_002448",
                "/data2/analyses/database/Pancho-figures89-formodeling-210329_000418"
            ]
            # path_list = [
            #     "/data2/analyses/database/Pancho-shapes3-formodeling-210329_002448",
            #     "/data2/analyses/database/Pancho-figures89-formodeling-210329_000418"
            # ]
            
        append_list = None

        for paththis in path_list:
            # Load single dataset
            D = Dataset([paththis], append_list)

            # No need to preprocess data, since can just apply any preprocess afterwards to both task and beahviro

            # Get sketchpad edges
            maxes = []
            for k, v in D.Metadats.items():
                maxes.append(np.max(np.abs(v["sketchpad_edges"].flatten())))
            canvas_max_WH = np.max(maxes)

            # origin for tasks, is generally around (0, 50) to (0,100). so just hard code here.
            origin = torch.tensor([image_WH/2, -(0.45*image_WH)])
            # alternatively, could look into D.Dat["origin"], and convert to image coords [(1,105), (-105, -1)]

            # For each row, parse its task
            score_fn = lambda parses: score_function(parses, ver=score_ver, 
                                                     normalization=score_norm, use_torch=True, origin=origin)

            if False:
                # Just testing, pick a random trial
                import random
                ind = random.sample(range(len(D.Dat)), 1)[0]
                strokes = D.Dat["strokes_task"].values[ind]

                if False:
                    from pythonlib.tools.stroketools import strokesInterpolate2
                    strokes = strokesInterpolate2(strokes, N=["npts", 100])
                else:
                    pass
                parses, log_probs_k = get_parses_from_strokes(strokes, canvas_max_WH, 
                                                              use_extra_junctions=use_extra_junctions, plot=True,
                                                             return_in_strokes_coords=True, k=5)

            # save params
            params_parse["canvas_max_WH"] = canvas_max_WH
            params_parse["use_extra_junctions"] = use_extra_junctions
            params_parse["return_in_strokes_coords"] = return_in_strokes_coords
            params_parse["score_ver"] = score_ver
            params_parse["score_norm"] = score_norm

            # OLD VERSION - GET EACH ROW
        #     # Collect parses
        #     PARSES = []
        #     for row in D.Dat.iterrows():
        #         if row[0]%100==0:
        #             print(row[0])
        #         strokes = row[1]["strokes_task"]
        #         index = row[0]
        #         trial_id = row[1]["trialcode"]
        #         unique_task_name = row[1]["unique_task_name"]
        #         character = row[1]["character"]

        #         parses, log_probs = get_parses_from_strokes(strokes, canvas_max_WH, 
        #                                                   use_extra_junctions=use_extra_junctions, plot=False,
        #                                                  return_in_strokes_coords=return_in_strokes_coords, k=k,
        #                                                    configs_per = params_parse["configs_per"],
        #                                                    trials_per = params_parse["trials_per"],
        #                                                    max_ntrials = params_parse["max_ntrials"],
        #                                                    max_nstroke = params_parse["max_nstroke"],
        #                                                    max_nwalk = params_parse["max_nwalk"],
        #                                                    )

        #         PARSES.append(
        #             {"strokes_task":strokes,
        #              "index_dat":index,
        #              "trial_id":trial_id,
        #              "unique_task_name":unique_task_name,
        #              "character":character,
        #              "parses":parses,
        #              "parses_log_probs":log_probs}
        #         )

            # NEW VERSION - only do once for each unique task
            tasklist = sorted(list(set(D.Dat["unique_task_name"])))
            # Collect parses
            PARSES = []

            for i, task in enumerate(tasklist):
                # find the first row that has this task
                row = D.Dat[D.Dat["unique_task_name"]==task].iloc[0]
                assert row["unique_task_name"]==task

        #         if i%20==0:
        #             print(i, "-",  task)
                print(i, "-",  task)
                strokes = row["strokes_task"]

                parses, log_probs = get_parses_from_strokes(strokes, canvas_max_WH, 
                                                          use_extra_junctions=use_extra_junctions, 
                                                          score_fn=score_fn,
                                                            plot=False, image_WH=image_WH,
                                                            return_in_strokes_coords=return_in_strokes_coords, 
                                                            k=kparses, configs_per = params_parse["configs_per"],
                                                           trials_per = params_parse["trials_per"],
                                                           max_ntrials = params_parse["max_ntrials"],
                                                           max_nstroke = params_parse["max_nstroke"],
                                                           max_nwalk = params_parse["max_nwalk"],
                                                           )
                assert len(parses)>0, "why?"

                PARSES.append(
                    {"strokes_task":strokes,
                     "unique_task_name":task,
                     "parses":parses,
                     "parses_log_probs":log_probs}
                )

            # === save as dataframe
            import os
            import pandas as pd
            import pickle
            
            DIR, FNAME = os.path.split(paththis)
            fname_parse = f"{DIR}/{FNAME}/parses.pkl"
            print("Saving at:")
            print(fname_parse)

            PARSES  = pd.DataFrame(PARSES)
            PARSES.to_pickle(fname_parse)

            fname_params = f"{DIR}/{FNAME}/parses_params.pkl"
            with open(fname_params, "wb") as f:
                pickle.dump(params_parse, f)