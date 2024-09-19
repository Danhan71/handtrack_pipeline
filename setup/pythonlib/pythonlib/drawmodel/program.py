""" stuff to work with programs. programs are list of numpy arrays that are output from the draw grammar.
NOTE: programs are differnt from strokes in that programs dont have regularly spaced strokes and dont have 
time as a column """
import numpy as np
from ..tools.stroketools import strokesInterpolate
from .segmentation import getSegmentation

def program2strokesWrapper(programs, get_datsegs=False):
    """ this converts from program (dreamcoder) to datflat useful in behaivorla analysis.
    Also can run segmentation automatically. 
    NOTE: datflat[0]["trialstrokes"] will give you cleaned up strokes useful for beahvior.
    NOTE: I have carefulyl walked thorugh and this should be doing all the steps previously did for 
    drawgood analysis.
    - One note is that it centers drawings which may or may not be what you want. """

    datflat = parses2datflat(programs, finalize_strokes=True)
    if get_datsegs:
        datsegs = getSegmentation(datflat, unique_codes=True, dosplits=True, 
                             removebadstrokes=True, removeLongVertLine=False, 
                           include_npstrokes=True, do_not_modify_strokes=True)
        return datflat, datsegs
    else:
        return datflat, None



def program2strokes(program):
    """given program convert into "strokes", so that can pass into same behavioral aalysis as for subjects"""
    # program is list of numpy arrays (flattened)
    # for nowwill put down times in order. will be in fake milliseconds. 
    
    on = 1
    off = 300
    strokes = []
    for p in program:
        times = np.linspace(on, off, p.shape[0])
        p = np.concatenate((p, times[:,None]), axis=1)
        on+=500
        off+=500
        strokes.append(p)
    return strokes
    
    # strokes = program2strokes(dreams[1].evaluate([]))
    # strokes = program2strokes(result.allFrontiers[tasks[1]].bestPosterior.program.evaluate([]))
    # strokes = [s.tolist() for s in strokes]


def parses2datflat(programs, stimname="", condition="", randomsubsample=[],
    finalize_strokes=False, Ninterp=15):
    """Given a list of flattened program (i.e, list of numpy) converts to datflat.
    this is basically a wrapper for program2strokes. it outputs a list of dicts, where
    each dict has a key 'trialstrokes' that is just strokes. is not necssaery, but was 
    useful for the transfomations: 
    [BEHAVIOR] datall--> datflat--> datseg
    [MODEL] programs(i.e., strokes, no time) --> datflat (strokes, interpolated + time) --> datseg """
    # ideally stimname is the name of stim for all parses (i.e progs)
    # ideally condition indicates what the training schedule was.

    # === A HACK, to make this work with code written fro beahviroal analysis, 
    # convert this list of programs --> list of strokes --> one datflat object
    # this datflat can then be treated just like a human subject's data.

    datflat = []
    from pythonlib.tools.listtools import permuteRand
    if isinstance(randomsubsample, int)>0:
        # takes a random subsample of all programs, without repalcement
        if len(programs)>randomsubsample:
            import random
            print("(parses --> datflat) doing random subsample of {}{} from {} to {}".format(stimname, condition, len(programs), randomsubsample))
            programs = random.sample(programs, randomsubsample)
    else: assert isinstance(randomsubsample, list), "expect either int or empty list..."


    for i, prog in enumerate(programs):
        # print("prog {}".format(i))
        if False:
            # this needs to import the Parse object...
            if isinstance(prog, Parse):
                prog = prog.flatten()

        # 1) append fake timestamps, so that prog become a strokes list:
        strokes = program2strokes(prog)

        if finalize_strokes:
            # 2) center, update timesteps/convert to seconds. [confirmed this is what does]
            strokes = strokes2nparray(strokes, recenter_and_flip=True, combinestrokes=False, 
                sec_rel_first_stroke=False, sec_rel_task_onset=True)
            # 3) interpolate [so that each stroke is same num pts]
            strokes = strokesInterpolate(strokes, N=Ninterp) # do interpolation

        # create a new entry in datflat, 
        datflat.append({
            "parsenum": i, # this is unique to dreamcoder; each program has multiple parses
            "trialstrokes":strokes,
            "trialonset": 0,
            "stimname": stimname, 
            "trialprimitives":[],
            "trialcircleparams":[],
            "condition":condition
        })
    return datflat



def strokes2nparray(strokes_from_datflat, recenter_and_flip=False, justdorecenter=True, combinestrokes=True, sec_rel_first_stroke=True,
    sec_rel_task_onset=False):
    """ formatting, specific for dataset for drawgood (mturk). goes from strokes(i.e., trialstrokes in datflat object)
    to strokes (similar, but orientation correct, added time, centered, etc)"""
    # justdorecenter, if True and recenter_and_flip is True, then will only do recenter.
    # this is becuase I modified earlier analyess so that flips during preprocessig...
    # sec_rel_task_onset, then overwrites sec_rel_first_stroke(which initiates each stroke at 0...
    # the latter was default for osme reason...)

    def A(strokes_from_datflat, center, sec_rel_first_stroke=sec_rel_first_stroke, sec_rel_task_onset=sec_rel_task_onset):
        strokearray = np.array([ss for s in strokes_from_datflat for ss in s])
        if recenter_and_flip and len(center)>0:
            # median x and y as center of canvas. flip y so that positive is up
            strokearray[:,:2] = strokearray[:,:2] - center
            if not justdorecenter:
                strokearray[:,1] = -strokearray[:,1]
        if sec_rel_task_onset==False and sec_rel_first_stroke==True:
            strokearray[:,2] = (strokearray[:,2] - strokearray[0,2])/1000
        return strokearray

    # -- get center of canvas.
    strokearray=A(strokes_from_datflat, center=[], sec_rel_first_stroke=False) # TODO: rewrite to not have to run A just to get center?
    center = np.median(strokearray[:,:2],axis=0)

    if combinestrokes:
        strokearray=A(strokes_from_datflat, center)
        if sec_rel_task_onset:
            t0 = strokearray[0,2]
            strokearray[:,2] = strokearray[:,2]-t0
    else:
        strokearray=[A([stroke], center) for stroke in strokes_from_datflat]
        if sec_rel_task_onset:
            t0 = strokearray[0][0,2]
            for S in strokearray:
                S[:,2] = (S[:,2]-t0)/1000

    return strokearray




